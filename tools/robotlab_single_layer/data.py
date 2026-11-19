"""FP16 raw single-layer dataset; original pairs/loss keys, bounded read-only maps."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .baseline import CAMERAS, IMAGES, POLICY_REVISION, POLICY_WEIGHT, SHAPES, digest, historical_record
from .manifest import validate_plan
from .metadata import conditioning_source


class MapPool:
    """Per-process LRU: 454 episodes must not retain thousands of open files."""
    def __init__(self, maximum=16):
        if maximum < 2:
            raise ValueError('Need at least a visual and action map')
        self.maximum, self.pid, self.maps = maximum, os.getpid(), OrderedDict()

    def close(self):
        for value in self.maps.values():
            value._mmap.close()
        self.maps.clear()

    def get(self, path):
        if os.getpid() != self.pid:
            self.close()
            self.pid = os.getpid()
        key = str(Path(path).resolve())
        if key not in self.maps:
            if len(self.maps) >= self.maximum:
                _, previous = self.maps.popitem(last=False)
                previous._mmap.close()
            self.maps[key] = np.load(key, mmap_mode='r', allow_pickle=False)
        self.maps.move_to_end(key)
        return self.maps[key]


POOL = MapPool()


def checked_array(path, shape, dtype=None):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if value.shape != shape or (dtype is not None and value.dtype != dtype):
            raise ValueError('Shape/dtype mismatch: ' + str(path))
        return np.array(value, copy=True) if len(shape) <= 2 else None
    finally:
        value._mmap.close()


def verify_episode(spec, layer):
    if layer not in SHAPES:
        raise ValueError('Only S2/S8, not J fusion or a new layer')
    raw, deep = Path(spec['raw_path']), Path(spec['connector_path'])
    if raw.resolve() == deep.resolve():
        raise ValueError('Raw and Connector input directories must be distinct')
    marker = json.loads((raw / 'COMPLETE.json').read_text())
    meta = json.loads((deep / 'metadata.json').read_text())
    n, ep = spec['frames'], spec['episode']
    complete, complete_path, _ = conditioning_source(deep, ep, n)
    expected = dict(format='multilayer_raw_fp16_v1', episode=ep, num_frames=n, fps=10.,
        pooling='none', dtype='float16', layer_indices_zero_based=[2, 8],
        camera_keys=CAMERAS, source_image_keys=IMAGES,
        shape_per_layer=[n, 2, 1024, 768], policy_revision=POLICY_REVISION,
        policy_weight_sha256=POLICY_WEIGHT, k_data_frames=10,
        deep_cache_complete_sha256=digest(complete_path),
        frame_index_and_timestamp_source='deep_cache_path',
        action_source='deep_cache_path/actions.npy; checkpoint_processor_normalized')
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError('Raw protocol mismatch: ' + key)
    checks = dict(format='siglip_frame_cache_v2', split=spec['split'], num_frames=n,
        fps=10., policy_revision=POLICY_REVISION, camera_keys=CAMERAS, image_keys=IMAGES,
        action_representation='checkpoint_processor_normalized')
    if any(meta.get(k) != v for k, v in checks.items()):
        raise ValueError('Connector metadata/split/action protocol mismatch')
    sources = meta['raw_provenance']['sources']
    if marker['source_parquet_sha256'] not in [r['sha256'] for r in sources]:
        raise ValueError('Raw and action/image provenance do not match')
    raw_file = raw / (layer + '.fp16.npy')
    raw_entry = marker['files'][raw_file.name]
    if (raw_file.is_symlink() or raw_file.stat().st_size != raw_entry['size_bytes']
            or digest(raw_file) != raw_entry['sha256']):
        raise ValueError('Raw layer SHA/size mismatch')
    checked_array(raw_file, (n, 2, 1024, 768), np.dtype('float16'))
    files = {layer + '.fp16.npy': raw_entry['sha256'], 'raw_COMPLETE': digest(raw / 'COMPLETE.json')}
    frozen = historical_record(ep)
    for name in ('actions.npy', 'frame_index.npy', 'episode_index.npy', 'timestamp.npy', 'metadata.json'):
        if digest(deep / name) != complete['files'][name] or complete['files'][name] != frozen['files'][name]:
            raise ValueError('Action/timebase/metadata SHA mismatch: ' + name)
        files[name] = complete['files'][name]
    frames = checked_array(deep / 'frame_index.npy', (n,))
    episodes = checked_array(deep / 'episode_index.npy', (n,))
    times = checked_array(deep / 'timestamp.npy', (n,))
    actions = checked_array(deep / 'actions.npy', (n, 7), np.dtype('float16'))
    if (not np.issubdtype(frames.dtype, np.integer) or not np.issubdtype(episodes.dtype, np.integer)
            or not np.array_equal(frames, np.arange(n)) or not np.all(episodes == ep)
            or not np.isfinite(times).all() or not np.isfinite(actions).all()
            or not np.allclose(times, np.arange(n) / 10., atol=1e-4, rtol=0)):
        raise ValueError('Episode/frame/timestamp alignment mismatch')
    return files


class EpisodeDataset(Dataset):
    """Only current feature + current action are inputs; future feature is target-only."""
    def __init__(self, spec, layer):
        self.spec, self.layer = dict(spec), layer
        self.ep_idx = np.full(spec['frames'], spec['episode'], dtype=np.int64)
        self.pairs = [(t, t + 10) for t in range(2, spec['frames'] - 10)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        t, due = self.pairs[index]
        raw = POOL.get(Path(self.spec['raw_path']) / (self.layer + '.fp16.npy'))
        current = np.array(raw[t], dtype=np.float32, copy=True).reshape(2048, 768)
        future = np.array(raw[due], dtype=np.float32, copy=True).reshape(2048, 768)
        actions = POOL.get(Path(self.spec['connector_path']) / 'actions.npy')
        action = np.array(actions[t], dtype=np.float32, copy=True)
        if not all(np.isfinite(v).all() for v in (current, future, action)):
            raise FloatingPointError('Nonfinite sample; no filtering or rescaling')
        z = torch.from_numpy(current)
        return dict(z_t=z, z_hist=z.unsqueeze(0), a_t=torch.from_numpy(action),
                    delta_z=torch.from_numpy(future - current))


def datasets(plan, layer):
    validate_plan(plan)
    grouped, records = {'train': [], 'validation': []}, []
    for spec in sorted(plan['records'], key=lambda r: r['episode']):
        files = verify_episode(spec, layer)
        files['single_layer_data_adapter'] = digest(__file__)
        ds = EpisodeDataset(spec, layer)
        grouped[spec['split']].append(ds)
        records.append(dict(episode=spec['episode'], frames=spec['frames'], split=spec['split'],
            pairs=len(ds), files=files, feature_layer=layer,
            feature_contract=dict(raw_shape=[2, 1024, 768], model_shape=[2048, 768],
                storage='float16', pooling='none', inputs='current visual, current single action',
                target='future[t+10] minus current[t]', normalization='none')))
    if any(not values for values in grouped.values()):
        raise ValueError('Both train and validation episodes are required')
    return {k: ConcatDataset(v) for k, v in grouped.items()}, records, (2048, 768, 7, 10.)

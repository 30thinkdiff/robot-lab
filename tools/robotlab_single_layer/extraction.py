"""Resumable raw S2/S8 extraction using the verified frozen encoder helpers.

This module has no automatic entry point. Formal extraction must be invoked
explicitly after the stage A-C receipts pass. Interrupted attempts are retained,
never reused as complete data or silently deleted.
"""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .baseline import (CAMERAS, IMAGES, POLICY_REVISION, POLICY_WEIGHT,
                       digest, load_reference)
from .data import verify_episode
from .extraction_plan import EXPORTER_SHA256
from .manifest import validate_plan, write_new
from .metadata import (check_connector_frame, conditioning_source,
                       validate_conditioning)
from .training import require_gates
from .progress import estimate


def atomic_status(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def exclusive_worker(root):
    """Lock only this new experiment, not a shared dataset or frozen source."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def next_attempt(root, episode):
    work = Path(root) / '.work'
    work.mkdir(exist_ok=True)
    for number in range(1, 10000):
        candidate = work / ('episode_%d_attempt_%04d' % (episode, number))
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError('Unexpected number of retained attempts')


def boundary(root, remaining_bytes, stop_file, free_bytes=None):
    """One complete episode is the recovery/STOP unit; reserve remains 5 GiB."""
    if Path(stop_file).exists():
        return 'stop_requested'
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    if free < remaining_bytes + 5 * 2**30 + 64 * 2**20:
        return 'insufficient_disk'
    return None


def validate_rows(rows, episode, frames, timestamps):
    if (len(rows) != frames
        or [r['frame_index'] for r in rows] != list(range(frames))
        or any(r['episode_index'] != episode for r in rows)):
        raise ValueError('Parquet episode/frame alignment mismatch')
    actual = np.asarray([r['timestamp'] for r in rows], dtype=np.float64)
    if (not np.isfinite(actual).all()
        or not np.allclose(actual, timestamps, rtol=0, atol=1e-4)):
        raise ValueError('Parquet vs original cache timestamp mismatch')


def publish(attempt, final, spec):
    """Only verified complete episodes can be atomically published."""
    final, attempt = Path(final), Path(attempt)
    if final.exists():
        raise FileExistsError('Never overwrite a published episode: ' + str(final))
    staged = dict(spec, raw_path=str(attempt))
    for layer in ('block_02', 'block_08'):
        verify_episode(staged, layer)
    attempt.rename(final)


def load_helpers(exporter):
    if digest(exporter) != EXPORTER_SHA256:
        raise ValueError('Successful frozen exporter SHA mismatch')
    sys.path.insert(0, str(Path(exporter).parent))
    try:
        spec = importlib.util.spec_from_file_location('_slc_frozen_export_helpers', exporter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def configure_runtime(base):
    base=Path(base)
    os.environ.update(HF_HOME=str(base/'runtime/hf'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        LEROBOT_SRC=str(base/'runtime/lerobot/src'))
    # _ensure_lerobot_importable inside the original extractor consults this
    # variable and reorders sys.path again at construction time.
    sys.path.insert(0,str(base/'runtime/vla-corrector-reference/src'))
    sys.path.insert(0,str(base/'runtime/lerobot/src'))


def run(manifest, *, base, data_root, gates, output, microbatch=32):
    """Explicit full job. Reuses helpers, not the old pooling diagnostics loop."""
    import torch
    import pyarrow.parquet as pq

    manifest, base, output, data_root = map(Path, (manifest, base, output, data_root))
    plan = json.loads(manifest.read_text())
    validate_plan(plan)
    if output.resolve() != Path(plan['records'][0]['raw_path']).parent.resolve():
        raise ValueError('Output is not the explicit new manifest raw root')
    if any(Path(r['raw_path']).parent.resolve() != output.resolve() for r in plan['records']):
        raise ValueError('Episode raw roots differ')
    reference = load_reference(base/'night-code-1e80242789', base/'runtime/vla-corrector-reference')
    for layer in ('block_02', 'block_08'):
        require_gates(gates[layer], layer, reference, manifest, microbatch)
    # Validate all real conditioning files BEFORE allocating the GPU or raw NPYs.
    for rec in plan['records']:
        validate_conditioning(rec['connector_path'], rec['episode'], rec['frames'], rec['split'])
    exporter = base/'dev/j36m-35ecdd7488bf/scripts/corrector_multilayer_pilot.py'
    helper = load_helpers(exporter)
    configure_runtime(base)
    from corrector_data_workbench import cached_policy_snapshot, cached_visual_extractor, decode_rgb
    from siglip_dynamics.extract import SmolVLAExtractor
    from corrector_multilayer import ReadOnlyVisionTap, quantize_per_token
    import subprocess
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPU already occupied; no extraction started: ' + busy)
    identity = dict(manifest_sha256=digest(manifest), extractor_sha256=digest(__file__),
                    source_exporter_sha256=EXPORTER_SHA256, microbatch=microbatch)
    with exclusive_worker(output):
        identity_path = output/'JOB_IDENTITY.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Resume code/manifest differs; use an independent job')
        else:
            write_new(identity_path, identity)
        stop = output/'STOP'
        history=[]
        journal=output/'completion_history.jsonl'
        if journal.exists():
            lines=journal.read_text().splitlines(keepends=True)
            for index,line in enumerate(lines):
                if not line.strip():continue
                try:history.append(json.loads(line))
                except json.JSONDecodeError:
                    if index==len(lines)-1 and not line.endswith('\n'):continue
                    raise
        state = dict(status='running', completed=0, planned=len(plan['records']),
                     pid=os.getpid(), frames_completed=0, frames_total=plan['frames'],
                     current_frames_completed=0, full_resolution=True, stage='feature_extraction')
        def update():
            state.update(estimate(state['frames_total'],state['frames_completed'],state['current_frames_completed'],history))
            state.update(data_free_gib=shutil.disk_usage(output).free/2**30,
                         system_free_gib=shutil.disk_usage('/').free/2**30,
                         updated_unix=time.time())
            atomic_status(output/'status.json',state)
        update()
        encoder, hashes = None, {}
        try:
            for rec in plan['records']:
                if stop.exists():
                    state.update(status='paused', reason='stop_requested')
                    update()
                    return state
                final = Path(rec['raw_path'])
                if final.exists():
                    state.update(phase='verifying_existing', episode=rec['episode'])
                    update()
                    for layer in ('block_02', 'block_08'):
                        verify_episode(rec, layer)
                    state['completed'] += 1
                    state['frames_completed'] += rec['frames']
                    update()
                    continue
                reason = boundary(output, rec['frames']*6*2**20, stop)
                if shutil.disk_usage('/').free<5*2**30:reason='insufficient_system_disk'
                if reason:
                    state.update(status='paused', reason=reason)
                    update()
                    return state
                episode_start=time.perf_counter()
                state.update(episode=rec['episode'], phase='extracting',current_frames_completed=0)
                update()
                if encoder is None:
                    snapshot = Path(cached_policy_snapshot(False))
                    weight = snapshot/'model.safetensors'
                    if digest(weight) != POLICY_WEIGHT:
                        raise ValueError('Frozen policy weight SHA mismatch')
                    encoder = cached_visual_extractor(SmolVLAExtractor, snapshot, 'cuda', allow_network=False)
                    import inspect
                    if not Path(inspect.getfile(type(encoder.policy))).resolve().is_relative_to((base/'runtime/lerobot').resolve()):
                        raise ValueError('Actual SmolVLA class imported from wrong frozen source')
                    encoder.policy.eval().requires_grad_(False)
                    if list(encoder.policy.config.image_features) != CAMERAS:
                        raise ValueError('Camera order differs from original cache')
                    torch.set_num_threads(4)
                path, _ = helper.locate_parquet(data_root, rec['episode'], plan)
                if str(path) not in hashes:
                    hashes[str(path)] = digest(path)
                deep = Path(rec['connector_path'])
                meta = json.loads((deep/'metadata.json').read_text())
                if hashes[str(path)] not in [s['sha256'] for s in meta['raw_provenance']['sources']]:
                    raise ValueError('Source Parquet provenance changed')
                _, source_marker, refs = conditioning_source(deep, rec['episode'], rec['frames'])
                if refs is None:
                    raise ValueError('Explicit original per-frame Connector references required')
                timestamps = np.load(deep/'timestamp.npy', allow_pickle=False)
                table = pq.read_table(path, columns=['episode_index','frame_index','timestamp',*IMAGES])
                rows = sorted((r for r in table.to_pylist() if r['episode_index']==rec['episode']),
                              key=lambda r:r['frame_index'])
                validate_rows(rows, rec['episode'], rec['frames'], timestamps)
                attempt = next_attempt(output, rec['episode'])
                maps = {name:np.lib.format.open_memmap(attempt/(name+'.fp16.npy'),mode='w+',
                        dtype=np.float16,shape=(rec['frames'],2,1024,768))
                        for name in ('block_02','block_08')}
                started = time.perf_counter()
                wrapper = encoder.model.vlm_with_expert
                try:
                    with torch.inference_mode(), ReadOnlyVisionTap(wrapper,(2,8)) as tap:
                        for fi,row in enumerate(rows):
                            views=[]
                            for camera,image in zip(CAMERAS,IMAGES,strict=True):
                                prepared=encoder._prepare_image(decode_rgb(row[image]['bytes']))
                                with tap.capture(camera) as captured:
                                    connector=wrapper.embed_image(prepared)
                                if not torch.equal(connector,captured['connector']):
                                    raise ValueError('Readonly hook changed Connector output')
                                views.append({k:t[0].float().cpu().numpy().copy() for k,t in captured.items()})
                            connector=np.stack([v['connector'] for v in views]).reshape(128,960)
                            check_connector_frame(quantize_per_token,connector,refs,fi)
                            for name,arr in maps.items():
                                value=np.stack([v[name] for v in views])
                                if value.shape!=(2,1024,768) or not np.isfinite(value).all():
                                    raise ValueError('Raw feature shape/numerics changed')
                                value=value.astype(np.float16)
                                if not np.isfinite(value).all():raise FloatingPointError('FP16 overflow')
                                arr[fi]=value
                            if fi%50==0:
                                state.update(frame=fi,episode_frames=rec['frames'],current_frames_completed=fi+1)
                                update()
                finally:
                    for arr in maps.values():
                        arr.flush(); arr._mmap.close()
                if digest(weight)!=POLICY_WEIGHT:raise ValueError('Frozen weight changed')
                files={p.name:dict(sha256=digest(p),size_bytes=p.stat().st_size)
                       for p in attempt.iterdir() if p.suffix=='.npy'}
                marker=dict(format='multilayer_raw_fp16_v1',episode=rec['episode'],num_frames=rec['frames'],
                    fps=10.,layer_indices_zero_based=[2,8],camera_keys=CAMERAS,source_image_keys=IMAGES,
                    dtype='float16',pooling='none',shape_per_layer=[rec['frames'],2,1024,768],
                    source_parquet=str(path),source_parquet_sha256=hashes[str(path)],
                    policy_revision=POLICY_REVISION,policy_weight_sha256=POLICY_WEIGHT,
                    deep_cache_path=str(deep),deep_cache_complete_sha256=digest(source_marker),
                    frame_index_and_timestamp_source='deep_cache_path',
                    action_source='deep_cache_path/actions.npy; checkpoint_processor_normalized',
                    k_data_frames=10,files=files,connector_exact_frames=rec['frames'],
                    elapsed_seconds=time.perf_counter()-started)
                write_new(attempt/'COMPLETE.json',marker)
                publish(attempt,final,rec)
                state['completed']+=1;state['frames_completed']+=rec['frames']
                state['current_frames_completed']=0
                row=dict(episode=rec['episode'],frames=rec['frames'],wall_seconds=time.perf_counter()-episode_start,
                         completed_unix=time.time())
                with journal.open('a') as file:file.write(json.dumps(row)+'\n');file.flush()
                history.append(row)
                update()
                print(json.dumps(dict(episode=rec['episode'],completed=state['completed'],planned=state['planned'],
                    frames_completed=state['frames_completed'],frames_total=state['frames_total'],
                    eta_hours=state['eta_seconds']/3600,frames_per_second=state['frames_per_second']),
                    ensure_ascii=False),flush=True)
            state.update(status='completed',phase='all_episodes_verified')
            update()
            return state
        except BaseException as e:
            state.update(status='error',error=type(e).__name__+': '+str(e))
            update()
            raise

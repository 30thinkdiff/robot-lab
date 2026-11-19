"""Original conditioning bytes, with an explicit metadata-only cache format."""
import hashlib
import json
from pathlib import Path

from .baseline import digest, historical_record


def conditioning_source(directory, episode, frames):
    directory = Path(directory)
    subset = directory / 'METADATA_COMPLETE.json'
    if not subset.exists():
        original = directory / 'COMPLETE.json'
        return json.loads(original.read_text()), original, None
    marker = json.loads(subset.read_text())
    if marker.get('format') != 'single_layer_conditioning_metadata_v1' or (marker['episode'], marker['frames']) != (episode, frames):
        raise ValueError('Conditioning metadata identity mismatch')
    for name, sha in marker['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe metadata marker path')
        p = directory / name
        if p.is_symlink() or digest(p) != sha:
            raise ValueError('Conditioning metadata SHA mismatch: ' + name)
    original = directory / 'SOURCE_COMPLETE.json'
    if digest(original) != marker['source_complete_sha256']:
        raise ValueError('Original completion manifest binding mismatch')
    refs = json.loads((directory / 'connector_frame_hashes.json').read_text())
    frozen = historical_record(episode)
    if (refs['format'] != 'connector_quantized_frame_sha256_v1'
        or (refs['episode'], refs['frames']) != (episode, frames)
        or len(refs['frame_sha256']) != frames
        or refs['source_q_npy_sha256'] != frozen['files']['z_q.npy']
        or refs['source_scale_npy_sha256'] != frozen['files']['z_scale.npy']
        or refs['source_complete_sha256'] != digest(original)):
        raise ValueError('Per-frame Connector hashes are not bound to original tensors')
    if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in refs['frame_sha256']):
        raise ValueError('Invalid frame hash')
    return json.loads(original.read_text()), original, refs


def check_connector_frame(quantize_per_token, connector, references, frame):
    """Exact old quantization bytes, not a tolerance or a changed feature space."""
    q, scale = quantize_per_token(connector)
    if q.shape != (128, 960) or scale.shape != (128,) or q.dtype != 'int8' or scale.dtype != 'float16':
        raise ValueError('Original Connector quantization shape/dtype changed')
    actual = hashlib.sha256(q.tobytes(order='C') + scale.tobytes(order='C')).hexdigest()
    if actual != references['frame_sha256'][frame]:
        raise ValueError('Original Connector byte parity failed at frame ' + str(frame))
    return actual


def validate_conditioning(directory, episode, frames, split):
    """Validate real actions/time before GPU feature extraction; no raw tensors needed."""
    import numpy as np
    directory = Path(directory)
    complete, _, refs = conditioning_source(directory, episode, frames)
    frozen = historical_record(episode)
    if (frozen['frames'], frozen['split']) != (frames, split):
        raise ValueError('Conditioning split differs from original training')
    for name in ('actions.npy', 'raw_actions.npy', 'frame_index.npy', 'episode_index.npy', 'timestamp.npy', 'metadata.json'):
        if complete['files'][name] != frozen['files'][name] or digest(directory/name) != frozen['files'][name]:
            raise ValueError('Original conditioning bytes changed: '+name)
    meta = json.loads((directory/'metadata.json').read_text())
    if (meta['num_frames'], meta['split'], meta['fps'], meta['action_representation']) != (frames, split, 10., 'checkpoint_processor_normalized'):
        raise ValueError('Conditioning metadata/timebase mismatch')
    for name, shape, expected in (
        ('actions.npy', (frames,7), None), ('frame_index.npy',(frames,),np.arange(frames)),
        ('episode_index.npy',(frames,),np.full(frames,episode)), ('timestamp.npy',(frames,),np.arange(frames)/10.)):
        array = np.load(directory/name,mmap_mode='r',allow_pickle=False)
        try:
            if array.shape != shape or not np.isfinite(array).all(): raise ValueError('Invalid '+name)
            if name=='actions.npy' and array.dtype!=np.float16: raise ValueError('Original action dtype changed')
            if expected is not None:
                valid = np.allclose(array,expected,atol=1e-4,rtol=0) if name=='timestamp.npy' else np.array_equal(array,expected)
                if not valid: raise ValueError('Episode/frame alignment changed: '+name)
        finally:array._mmap.close()
    return dict(episode=episode,frames=frames,split=split,exact_connector_frames=len(refs['frame_sha256']) if refs else None)

"""Export original actions/time metadata and exact per-frame Connector hashes.

Stdlib only. Reads original caches, never changes them. This avoids transferring
8 GiB of Connector tensors while retaining every-frame byte parity checks for
the new raw-layer extractor. It does not extract images or use any GPU.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


class NpyFrames:
    def __init__(self, path, shape, dtype):
        self.file = Path(path).open('rb')
        try:
            magic = self.file.read(8)
            if magic[:6] != b'\x93NUMPY' or magic[6:] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
                raise ValueError('Invalid NPY')
            size = 2 if magic[6:] == b'\x01\x00' else 4
            length_bytes = self.file.read(size)
            length = struct.unpack('<H' if size == 2 else '<I', length_bytes)[0]
            if length > 65536: raise ValueError('Oversized NPY header')
            header = self.file.read(length)
            meta = ast.literal_eval(header.decode('utf8' if magic[6:] == b'\x03\x00' else 'latin1'))
            if meta['shape'] != shape or meta['descr'] != dtype or meta['fortran_order']:
                raise ValueError('Connector shape/dtype/order differs from original cache')
            self.whole_sha = hashlib.sha256(magic + length_bytes + header)
        except BaseException:
            self.file.close()
            raise

    def frame(self, size):
        data = self.file.read(size)
        if len(data) != size: raise ValueError('Truncated NPY frame')
        self.whole_sha.update(data)
        return data

    def finish(self):
        if self.file.read(1): raise ValueError('Unexpected trailing NPY bytes')
        return self.whole_sha.hexdigest()


def export(features, baseline, output):
    features, baseline, output = Path(features), Path(baseline), Path(output)
    protocol = json.loads(baseline.read_text())
    records = protocol['records']
    if (len(records) != 454 or len({r['episode'] for r in records}) != 454
            or Counter(r['split'] for r in records) != {'train': 364, 'validation': 90}):
        raise ValueError('Original full Object10 split required')
    if output.exists(): raise FileExistsError('New export directory required')
    output.mkdir(parents=True)
    exports = []
    copied = ('actions.npy', 'raw_actions.npy', 'frame_index.npy', 'episode_index.npy', 'timestamp.npy', 'metadata.json')
    for i, rec in enumerate(sorted(records, key=lambda r: r['episode']), 1):
        ep, n = rec['episode'], rec['frames']
        source = features / ('episode_' + str(ep))
        target = output / source.name
        target.mkdir()
        complete_bytes = (source / 'COMPLETE.json').read_bytes()
        complete = json.loads(complete_bytes)
        for name in copied:
            if complete['files'][name] != rec['files'][name] or digest(source / name) != rec['files'][name]:
                raise ValueError('Original metadata/action SHA mismatch: %s/%s' % (ep, name))
            shutil.copyfile(source / name, target / name)
        meta = json.loads((target / 'metadata.json').read_text())
        if (meta['num_frames'], meta['split'], meta['fps']) != (n, rec['split'], 10.):
            raise ValueError('Original frame/split/timebase mismatch')
        q = NpyFrames(source / 'z_q.npy', (n, 128, 960), '|i1')
        scale = None
        try:
            scale = NpyFrames(source / 'z_scale.npy', (n, 128), '<f2')
            refs = [hashlib.sha256(q.frame(128 * 960) + scale.frame(128 * 2)).hexdigest() for _ in range(n)]
            q_sha, scale_sha = q.finish(), scale.finish()
        finally:
            q.file.close()
            if scale is not None: scale.file.close()
        if q_sha != rec['files']['z_q.npy'] or scale_sha != rec['files']['z_scale.npy']:
            raise ValueError('Full original Connector NPY SHA mismatch: ' + str(ep))
        (target / 'SOURCE_COMPLETE.json').write_bytes(complete_bytes)
        write_new(target / 'connector_frame_hashes.json', dict(
            format='connector_quantized_frame_sha256_v1', episode=ep, frames=n,
            source_q_npy_sha256=q_sha, source_scale_npy_sha256=scale_sha,
            source_complete_sha256=hashlib.sha256(complete_bytes).hexdigest(),
            formula='sha256(C-order INT8 z_q[t] bytes + little-endian FP16 z_scale[t] bytes)',
            frame_sha256=refs))
        files = {p.name: digest(p) for p in target.iterdir()}
        marker = dict(format='single_layer_conditioning_metadata_v1', episode=ep,
            frames=n, split=rec['split'], files=files,
            source_complete_sha256=hashlib.sha256(complete_bytes).hexdigest(),
            omitted=['z_q.npy', 'z_scale.npy', 'reference_images'],
            purpose='Original actions/time only plus exact Connector frame hashes; not a complete Connector tensor cache')
        write_new(target / 'METADATA_COMPLETE.json', marker)
        exports.append(dict(episode=ep, split=rec['split'], frames=n,
            metadata_marker_sha256=digest(target / 'METADATA_COMPLETE.json')))
        if i % 25 == 0:
            print(json.dumps({'phase': 'reading_original_cache_cpu_only', 'completed': i, 'planned': 454}), flush=True)
    receipt = dict(format='conditioning_metadata454_export_v1', status='completed', episodes=454,
        frames=sum(r['frames'] for r in records), baseline_protocol_sha256=digest(baseline),
        original_data_manifest_sha256=protocol['data_manifest_sha256'], records=exports,
        original_source_writes=False, gpu_used=False)
    write_new(output / 'EXPORT_COMPLETE.json', receipt)
    archive = output.with_suffix('.zip')
    if archive.exists(): raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(output).as_posix())
    result = dict(zip=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
                  episodes=454, frames=receipt['frames'], source_writes=False, gpu_used=False)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--features', required=True, type=Path)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    export(a.features, a.baseline, a.output)


if __name__ == '__main__': main()

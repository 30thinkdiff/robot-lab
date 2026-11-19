"""CPU-only audit of migrated Parquet against original conditioning files."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .baseline import digest
from .extraction import validate_rows
from .manifest import validate_plan, write_new
from .metadata import validate_conditioning


def audit(manifest, data_root, output):
    import pyarrow.parquet as pq
    manifest, data_root, output = map(Path, (manifest, data_root, output))
    if output.exists():
        raise FileExistsError('Use a new audit receipt; retain earlier audits')
    plan = json.loads(manifest.read_text())
    validate_plan(plan)
    files = defaultdict(list)
    for rec in plan['records']:
        if len(rec['files']) != 1:
            raise ValueError('Original single-shard episode protocol required')
        name = rec['files'][0]['path']
        path = (data_root/name).resolve()
        if data_root.resolve() not in path.parents or path.is_symlink():
            raise ValueError('Parquet outside the explicit raw dataset root')
        files[name].append(rec)
    reports = []
    started = time.perf_counter()
    for name, records in sorted(files.items()):
        path = data_root/name
        sha = digest(path)
        rows = pq.read_table(path, columns=['episode_index', 'frame_index', 'timestamp', 'action']).to_pylist()
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['episode_index']].append(row)
        for rec in records:
            deep = Path(rec['connector_path'])
            validate_conditioning(deep, rec['episode'], rec['frames'], rec['split'])
            meta = json.loads((deep/'metadata.json').read_text())
            if sha not in [s['sha256'] for s in meta['raw_provenance']['sources']]:
                raise ValueError('Migrated Parquet differs from original source SHA')
            selected = sorted(grouped[rec['episode']], key=lambda r:r['frame_index'])
            times = np.load(deep/'timestamp.npy', allow_pickle=False)
            validate_rows(selected, rec['episode'], rec['frames'], times)
            actions = np.asarray([r['action'] for r in selected], dtype=np.float32)
            original = np.load(deep/'raw_actions.npy', allow_pickle=False)
            if actions.shape != (rec['frames'], 7) or original.dtype != np.float32 or not np.isfinite(actions).all():
                raise ValueError('Original raw action shape/dtype/numerics changed')
            if not np.array_equal(actions, original):
                raise ValueError('Real Parquet actions differ from original raw_actions.npy')
            reports.append(dict(episode=rec['episode'], frames=rec['frames'], split=rec['split'],
                                task_id=rec['task_id'], parquet_sha256=sha,
                                raw_actions_exact=True, time_frame_alignment=True))
    result = dict(status='completed', scope='CPU original Parquet/action/time/split consistency only',
        manifest_sha256=digest(manifest), episodes=len(reports), parquet_files=len(files),
        frames=sum(r['frames'] for r in reports), splits=dict(Counter(r['split'] for r in reports)),
        tasks=dict(Counter(r['task_id'] for r in reports)), seconds=time.perf_counter()-started,
        gpu_used=False, images_extracted=False, formal_training_started=False, records=reports)
    write_new(output, result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=audit(a.manifest,a.data_root,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

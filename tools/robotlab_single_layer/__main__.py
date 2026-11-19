"""Development commands only. No extraction/fit/rollout command is enabled."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--old-manifest', required=True, type=Path)
    plan.add_argument('--baseline-protocol', required=True, type=Path)
    plan.add_argument('--raw-root', required=True, type=Path)
    plan.add_argument('--connector-root', required=True, type=Path)
    plan.add_argument('--output', required=True, type=Path)
    model = sub.add_parser('inspect-model')
    model.add_argument('--baseline-dir', required=True, type=Path)
    model.add_argument('--source-root', required=True, type=Path)
    data = sub.add_parser('inspect-cache')
    data.add_argument('--manifest', required=True, type=Path)
    data.add_argument('--layer', choices=['block_02', 'block_08'], required=True)
    a = p.parse_args()
    if a.mode == 'plan':
        from .manifest import build_manifest, write_new
        value = build_manifest(a.old_manifest, a.raw_root, a.connector_root, a.baseline_protocol)
        write_new(a.output, value)
        print(json.dumps({k: v for k, v in value.items() if k != 'records'}, ensure_ascii=False, indent=2))
    elif a.mode == 'inspect-model':
        import torch
        from .baseline import load_reference, make_predictor
        from .model import LayerShapeAdapter
        reference = load_reference(a.baseline_dir, a.source_root)
        with torch.device('meta'):
            predictor = make_predictor(reference, 'meta')
            model = LayerShapeAdapter(predictor)
            shape = list(model(torch.empty(1, 2, 1024, 768), torch.empty(1, 7)).shape)
        print(json.dumps(dict(status='cpu_meta_shape_checked', parameters=sum(p.numel() for p in predictor.parameters()),
            output_shape=shape, real_gpu_run=False, training_started=False), indent=2))
    else:
        from .data import datasets
        groups, records, dims = datasets(json.loads(a.manifest.read_text()), a.layer)
        print(json.dumps(dict(episodes=len(records), samples={k: len(v) for k, v in groups.items()},
            model_shape=list(dims), training_started=False), indent=2))


if __name__ == '__main__':
    main()

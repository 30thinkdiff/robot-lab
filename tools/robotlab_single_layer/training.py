"""Dataset-only injection into the frozen historical trainer; no new optimizer/loss."""
from argparse import Namespace
import json
from pathlib import Path
from unittest.mock import patch

from .baseline import TRAINING, digest
from .data import datasets
from .manifest import validate_plan


def arguments(manifest, source_root, output, *, microbatch=32, seed=42,
              device='cuda', resume=False, stop_after_epoch=0):
    if microbatch < 1 or 512 % microbatch or seed not in TRAINING['seeds']:
        raise ValueError('Microbatch must divide effective batch512; seeds42/43/44')
    return Namespace(mode='fit', features=Path(manifest).parent, manifest=Path(manifest),
        local_cache=None, source_root=Path(source_root), output=Path(output),
        model_type='mlp', h_window=1, anchor_history=3, profile='paper-width',
        implementation='reference', split_mode='episode', val_ratio=.1,
        k_step=10, epochs=150, patience=5, batch_size=microbatch,
        accumulate=512 // microbatch, lr=.0003, weight_decay=.0001,
        loss_type='both', beta=1., seed=seed, predictor_precision='amp-bfloat16',
        cpu_threads=4, loader_workers=2, pin_memory=device == 'cuda', resume=resume,
        stop_file=Path(output) / 'STOP', snapshot_every=0,
        stop_after_epoch=stop_after_epoch, device=device)


def require_gates(path, layer, reference, manifest, microbatch=32):
    """No executable long training until real GPU/data gates have been recorded."""
    gates = json.loads(Path(path).read_text())
    expected = dict(layer=layer, manifest_sha256=digest(manifest),
        data_adapter_sha256=digest(Path(__file__).with_name('data.py')),
        training_adapter_sha256=digest(__file__),
        baseline_train_sha256=digest(reference['train'].__file__), microbatch=microbatch,
        effective_batch=512)
    if any(gates.get(k) != v for k, v in expected.items()):
        raise ValueError('Gate receipt belongs to different data/code')
    required = ('cpu_tests', 'real_gpu_forward_backward', 'checkpoint_reload',
                'lvm_state_parity', 'ogg_action_gradient', 'capture_on_off_parity')
    if any(gates.get(k) is not True for k in required) or gates.get('device') != 'cuda':
        raise ValueError('Stages A-C incomplete; full extraction/training remains disabled')
    if microbatch != 32 and not gates.get('microbatch_change_reason'):
        raise ValueError('Changing the historical microbatch requires a documented GPU resource reason')


def fit(reference, manifest, layer, output, *, gates, **kwargs):
    """Explicit future entry; not called by this development-stage CLI."""
    manifest = Path(manifest)
    plan = json.loads(manifest.read_text())
    validate_plan(plan)
    if len(plan['records']) != 454 or plan['train_pairs'] != 49447 or plan['validation_pairs'] != 12089:
        raise ValueError('Full formal baseline sample set required')
    args = arguments(manifest, kwargs.pop('source_root'), output, **kwargs)
    require_gates(gates, layer, reference, manifest, args.batch_size)
    groups, records, dims = datasets(plan, layer)
    # The frozen run stores these records in protocol/checkpoints and rejects
    # resume if a layer, raw SHA, data adapter or training adapter has changed.
    for row in records:
        row['files']['single_layer_training_adapter'] = digest(__file__)
    output = Path(output)
    for spec in plan['records']:
        for key in ('raw_path', 'connector_path'):
            if output.resolve() == Path(spec[key]).resolve() or Path(spec[key]).resolve() in output.resolve().parents:
                raise ValueError('Training output must not be inside historical cache')
    if output.exists() and not args.resume:
        raise FileExistsError('Use new output; resume must be explicit')
    output.mkdir(parents=True, exist_ok=args.resume)
    train = reference['train']
    # Single synchronous process, scoped replacement of the input-data adapter
    # only. Frozen forward, loss, accumulation, clipping, optimizer, scheduler,
    # best selection, RNG checkpoints, early stopping and STOP stay unchanged.
    with patch.object(train, 'datasets', return_value=(groups, records, dims)):
        with train.log_to(output / 'train.log'):
            train.run(args)

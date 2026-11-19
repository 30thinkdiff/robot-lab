"""Explicit small real-data GPU gate; never starts full extraction or training.

The frozen successful exporter validates the first 13 real frames of episode813.
One t2->t12 pair is repeated only for a production-shape microbatch32 memory and
backward check. Models are discarded; no probe checkpoint is a formal model.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .baseline import load_reference, model_config, digest, historical_record
from .extraction_plan import EXPORTER_SHA256


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def probe_layer(reference, raw, deep, layer, output):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    array = np.load(raw / (layer + '.fp16.npy'), mmap_mode='r', allow_pickle=False)
    if array.shape != (13, 2, 1024, 768) or array.dtype != np.float16:
        raise ValueError('Real probe raw feature resolution/dtype mismatch')
    z = torch.from_numpy(np.array(array[2], dtype=np.float32)).reshape(1, 2048, 768).cuda()
    future = torch.from_numpy(np.array(array[12], dtype=np.float32)).reshape(1, 2048, 768).cuda()
    array._mmap.close()
    actions = np.load(deep / 'actions.npy', mmap_mode='r', allow_pickle=False)
    action = torch.from_numpy(np.array(actions[2], dtype=np.float32)).reshape(1, 7).cuda()
    actions._mmap.close()
    if digest(deep / 'actions.npy') != historical_record(813)['files']['actions.npy']:
        raise ValueError('Actual action differs from historical frozen data')
    cfg = model_config(reference)
    cls = reference['models'].precision_class(reference['model_cls'], 'amp-bfloat16')
    model, optimizer, scheduler = reference['train'].reference_components(cls, cfg, 'cuda', .0003, .0001, 150)
    loss_type = reference['config'].LossType.BOTH
    model.train(); optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter()
    # No fallback to smaller batches. An OOM preserves the error and stops.
    prediction = model(z.repeat(32, 1, 1), action.repeat(32, 1))
    loss, mse, cosine = reference['loss'](prediction, (future - z).repeat(32, 1, 1), loss_type, 1.)
    norm = reference['train'].reference_update(model, optimizer, loss, True)
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError('Real full-shape backward loss/gradient')
    report = dict(layer=layer, input_shape=[32, 2048, 768], parameters=sum(p.numel() for p in model.parameters()),
        microbatch=32, intended_formal_effective_batch=512, formal_epochs_started=False,
        probe_optimizer_steps=1, loss=float(loss.detach()), mse=float(mse.detach()), cosine_loss=float(cosine.detach()),
        grad_norm_before_clip=float(norm), forward_backward_seconds=time.perf_counter() - begin,
        sample_definition='one real demo t2->t12 pair repeated32 for GPU memory only; not a training epoch or validation metric')
    model.eval().requires_grad_(False)
    with torch.no_grad(): expected = model(z, action).detach()
    checkpoint = output / (layer + '_discarded_probe_weights.pt')
    torch.save(dict(state_dict=model.state_dict(), config=json.loads(json.dumps(asdict(cfg))), scope='discarded GPU engineering probe, not a formal trained predictor'), checkpoint)
    reloaded = cls(cfg).cuda().eval().requires_grad_(False)
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'], strict=True)
    with torch.no_grad(): actual = reloaded(z, action)
    if not torch.equal(expected, actual):
        raise ValueError('GPU checkpoint reload changed the same forward')
    x, velocity, t = torch.randn(1, 50, 32, device='cuda'), torch.randn(1, 50, 32, device='cuda'), torch.tensor([.5], device='cuda')
    correction = expected - (future - z)
    captured = []
    def observe(module, inputs):
        if inputs[1].requires_grad: inputs[1].register_hook(lambda g: captured.append(g.detach().clone()))
    handle = reloaded.register_forward_pre_hook(observe)
    try:
        guided, ogg_loss = reference['runtime'].guided_velocity(reloaded, future, x, velocity, t, correction, 1.)
    finally:
        handle.remove()
    difference = guided - velocity
    if (not torch.isfinite(guided).all() or not captured or not torch.isfinite(captured[0]).all()
        or float(captured[0].norm()) <= 0 or float(difference.norm()) <= 0
        or not torch.equal(guided[:, 1:], velocity[:, 1:])
        or not torch.equal(guided[:, 0, 7:], velocity[:, 0, 7:])):
        raise ValueError('OGG gradient/support is not original first-action7 behavior')
    if any(p.requires_grad or p.grad is not None for p in reloaded.parameters()):
        raise ValueError('OGG predictor parameters must remain frozen')
    report.update(checkpoint_reload_verified=True, checkpoint_sha256=digest(checkpoint),
        ogg_loss=ogg_loss, actual_candidate_action_gradient_l2=float(captured[0].norm()),
        velocity_update_l2=float(difference.norm()), ogg_first_action7_only=True,
        ogg_candidate_source='synthetic flow tensor for interface/gradient check, not an executed robot command',
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--raw-data-root', required=True, type=Path)
    p.add_argument('--old-manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--original-raw-reference', required=True, type=Path)
    p.add_argument('--execute-real-gpu', action='store_true')
    a = p.parse_args()
    if not a.execute_real_gpu: raise RuntimeError('Explicit real GPU probe flag required')
    if a.output.exists(): raise FileExistsError('New probe output required')
    a.output.mkdir(parents=True)
    status = dict(status='running', phase='preflight', full_extraction_started=False, formal_training_started=False)
    write(a.output/'status.json', status)
    try:
        base = a.base.resolve()
        exporter = base/'dev/j36m-35ecdd7488bf/scripts/corrector_multilayer_pilot.py'
        if digest(exporter) != EXPORTER_SHA256: raise ValueError('Wrong frozen exporter')
        raw_output = a.output/'raw_prefix_probe'
        env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', HF_HOME=str(base/'runtime/hf'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, str(exporter), '--runtime', str(base/'runtime'),
            '--source-root', str(base/'runtime/vla-corrector-reference'),
            '--deep-root', str(base/'object10/features'), '--data-root', str(a.raw_data_root),
            '--manifest', str(a.old_manifest), '--output', str(raw_output),
            '--episodes', '813', '--layers', '2', '8', '--save-fp16', '--max-frames', '13', '--min-free-gib', '5']
        status.update(phase='real_13_frame_export', command=command); write(a.output/'status.json', status)
        with (a.output/'raw_probe.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        raw = raw_output/'episode_813'
        marker = json.loads((raw/'COMPLETE.json').read_text())
        if marker['num_frames'] != 13 or marker['pooling'] != 'none': raise ValueError('Probe raw protocol mismatch')
        for name, row in marker['files'].items():
            if digest(raw/name) != row['sha256']: raise ValueError('Probe raw SHA mismatch')
        import numpy as np
        old = json.loads(a.original_raw_reference.read_text())
        if (old['episode'], old['frames_checked'], old['format']) != (813, 13, 'original_raw_fp16_frame_sha256_v1'):
            raise ValueError('Original raw reference identity mismatch')
        for layer in ('block_02', 'block_08'):
            values = np.load(raw/(layer+'.fp16.npy'), mmap_mode='r', allow_pickle=False)
            try:
                for i in range(13):
                    if hashlib.sha256(values[i].tobytes(order='C')).hexdigest() != old['layers'][layer]['prefix_frame_sha256'][i]:
                        raise ValueError('Original raw layer byte parity failed: %s frame%d' % (layer, i))
            finally: values._mmap.close()
        # This 13-frame prefix is never published into raw454 or used by fit.
        reference = load_reference(base/'night-code-1e80242789', base/'runtime/vla-corrector-reference')
        reports = []
        for layer in ('block_02', 'block_08'):
            status.update(phase='gpu_backward_and_ogg', layer=layer); write(a.output/'status.json', status)
            reports.append(probe_layer(reference, raw, base/'object10/features/episode_813', layer, a.output))
            gc.collect()
        result = dict(status='completed', scope='small real-data GPU engineering probe only',
            original_connector_parity='frozen exporter requires exact requantized bytes for all13 frames',
            original_raw_layer_parity='Block2/Block8 FP16 bytes equal original cache for all13 frames',
            reports=reports, full_extraction_started=False, formal_training_started=False,
            real_closed_loop_evaluation_completed=False, motion_capture_on_off_gpu_parity=False,
            all_stage_C_gates_passed=False)
        write(a.output/'result.json', result)
        status.update(status='completed', phase='gpu_probe_complete_remaining_gates_pending')
        write(a.output/'status.json', status)
    except BaseException as error:
        status.update(status='error', error=type(error).__name__+': '+str(error))
        write(a.output/'status.json', status)
        raise


if __name__ == '__main__': main()

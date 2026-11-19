"""CPU-only verification of an explicitly supplied read-only audit snapshot.

Only the two upstream threshold definitions are compiled. No model weights,
feature tensor payloads, training or simulation are loaded.
"""
import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(snapshot):
    d = json.loads(snapshot.read_text())
    upstream = next(v for v in d['sources'].values() if 'threshold_ast_source' in v)
    tree = ast.parse(upstream['threshold_ast_source'])
    if [n.name for n in tree.body] != ['RobustThresholdState', 'robust_threshold_step']:
        raise ValueError('Unexpected upstream threshold definitions')
    scope = {'np': np, 'dataclass': dataclass, 'field': field}
    exec(compile(tree, '<audited-upstream-threshold>', 'exec'), scope)
    State, step = scope['RobustThresholdState'], scope['robust_threshold_step']
    out = defaultdict(lambda: dict(trials=0, scored_steps=0, eligible_steps=0,
        threshold_mismatches=0, trigger_mismatches=0, max_threshold_error=0.,
        max_extra_gap=0, cancellations_logged=0, predictions_scored_across_replans=0,
        first_score_steps=set(), successes=0))
    for label in ('formal_j', 'historical_four'):
        for row in d[label]['rows']:
            method = row['case']['method']
            if not row['trace']:
                continue  # H50/H20 have no monitor; absence is not zero coverage.
            result, trace = row['metrics'], row['trace']
            stats, state = out[method], State()
            stats['trials'] += 1
            stats['successes'] += int(result['success'])
            stats['eligible_steps'] += result['env_steps'] - 24
            stats['cancellations_logged'] += row['prediction_cancellations_logged']
            stats['first_score_steps'].add(trace[0]['step'])
            steps = [r['step'] for r in trace]
            gaps = [trace[0]['step'] - 24, result['env_steps'] - 1 - steps[-1]]
            gaps.extend(b - a - 1 for a, b in zip(steps, steps[1:]))
            stats['max_extra_gap'] = max(stats['max_extra_gap'], *gaps)
            for event in trace:
                # Joint _score returns expected.new_tensor(...): the controller
                # consumes FP32, while its log replaces score with Python float.
                score = float(np.float32(event['score'])) if label == 'formal_j' else event['score']
                threshold, trigger = step(score, state)
                error = abs(threshold - event['threshold'])
                stats['scored_steps'] += 1
                stats['threshold_mismatches'] += int(error != 0.)
                stats['trigger_mismatches'] += int(trigger != event['trigger'])
                stats['max_threshold_error'] = max(stats['max_threshold_error'], error)
                stats['predictions_scored_across_replans'] += int(bool(event['prediction_crossed_replan']))
    for stats in out.values():
        stats['first_score_steps'] = sorted(stats['first_score_steps'])
        stats['coverage'] = stats['scored_steps'] / stats['eligible_steps']
        if stats['threshold_mismatches'] or stats['trigger_mismatches']:
            raise ValueError('Threshold/state-machine replay disagrees with stored trace')
    formal = d['formal_j']['protocol']
    for suffix, expected in [('scripts/corrector_expanded_eval.py', formal['source_sha256']),
                             ('scripts/corrector_j36m_object_eval.py', formal['object_adapter_sha256']),
                             ('DEPLOYMENT_MANIFEST.json', formal['core_manifest'])]:
        item = next(v for path, v in d['sources'].items() if path.endswith(suffix))
        if item['sha256'] != expected:
            raise ValueError('Frozen source differs from recorded formal protocol: ' + suffix)
    s = d['storage']
    raw_payload = sum(np.prod(r['shape'], dtype=np.int64).item() * 2 for r in s['raw_arrays'])
    if raw_payload != s['frames70'] * s['two_raw_layers_bytes_per_frame']:
        raise ValueError('NPY shapes differ from storage model')
    receipt = {
        'audited_utc': d['audited_utc'],
        'snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'collector_sha256': d['audit_source_sha256'],
        'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cpu_numpy': np.__version__, 'gpu_used': False, 'source_writes': False,
        'threshold_source_sha256': upstream['sha256'],
        'formal_source_sha256': formal['source_sha256'],
        'formal_adapter_sha256': formal['object_adapter_sha256'],
        'core_manifest_sha256': formal['core_manifest'],
        'joint_checkpoint_sha256': formal['joint_checkpoint_sha256'],
        'mlp_checkpoint_sha256': formal['legacy_weights']['mlp']['checkpoint_sha256'],
        'trace_replay': dict(out),
        'score_precision_note': 'J log preserves Python-float fused score; controller consumes its FP32 tensor value. Replaying that actual conversion gives exact thresholds.',
        'storage': {k: s[k] for k in ('episodes70', 'frames70', 'splits70',
            'raw70_file_bytes', 'raw70_allocated_bytes', 'connector70_file_bytes',
            'full_episodes', 'full_frames', 'two_raw_layers_bytes_per_frame',
            'full_raw_payload_bytes', 'full_connector_quantized_payload_bytes')},
        'resources': {k: d['resources'][k] for k in ('memory_limit_bytes', 'memory_current_bytes', 'disks')},
        'historical_ram_peak_measured': False,
        'limitations': ['Header sizes verify shape/size, not the SHA of all 60 GiB tensor contents.',
            'Threshold replay verifies saved scores/state-machine decisions, not model forward or simulator physics.',
            'No per-process historical extraction RSS/peak RAM record exists in the audited reports.',
            'No new model training, GPU inference, simulation, data deletion or experiment modification.']}
    return receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = analyze(a.snapshot)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(a.output), 'replay': result['trace_replay']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

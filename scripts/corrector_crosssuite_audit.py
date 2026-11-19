"""Read-only CPU audit/export of completed task-transfer runs (no Torch/pickle).

Run with system Python on the result machine, stream JSON to the local machine.
The output contains light records and trace scalars; never videos or weights.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def audit_phase(root, expected):
    plan, execution = read(root / 'plan.json'), read(root / 'execution.json')
    require(execution['status'] == 'completed' and execution['planned'] == execution['completed'] == expected
            and execution['errors'] == 0, 'phase not completely finished')
    require(len(plan['cases']) == expected, 'plan count differs')
    expected_cases = {c['id']: c for c in plan['cases']}
    paths = sorted((root / 'trials').glob('*.json'))
    require(len(paths) == expected == len(expected_cases), 'missing/duplicate trial')
    recorder_sha = 'fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873'
    evidence = read(root / 'recording/evidence.json')
    require(evidence['source_sha256'] == recorder_sha and evidence['tracked_file_clean']
            and sha(root / 'recording/lerobot_eval.py') == recorder_sha, 'recording evidence differs')
    result = []
    sealed_files = video_bytes = 0
    signature_groups = {}
    plan_sha = sha(root / 'plan.json')
    for path in paths:
        record = read(path)
        c, metrics = record['case'], record['result']
        require(record['status'] == 'completed' and type(metrics['success']) is bool, 'not a completed bool result')
        require(c == expected_cases[c['id']], 'trial and frozen case differ')
        plain = {k: v for k, v in c.items() if k != 'id'}
        require(hashlib.sha256(json.dumps(plain, sort_keys=True).encode()).hexdigest()[:20] == c['id'], 'case ID hash differs')
        directory = root / 'attempts' / c['id'] / ('attempt_%03d' % record['attempt'])
        seals = read(directory / 'artifact_hashes.json')
        current = {str(p.relative_to(directory)) for p in directory.rglob('*')
                   if p.is_file() and p.name != 'artifact_hashes.json' and p.suffix in ('.json', '.jsonl', '.npz')}
        require(current == set(seals), 'diagnostic artifact set changed')
        for name, digest in seals.items():
            item = (directory / name).resolve()
            item.relative_to(directory.resolve())
            require(sha(item) == digest, 'diagnostic artifact hash differs: ' + str(item))
        sealed_files += len(seals)
        require(len(record['video_artifacts']) == 1, 'video missing/ambiguous')
        for video in record['video_artifacts']:
            item = (root / video['path']).resolve()
            item.relative_to(directory.resolve())
            require(sha(item) == video['sha256'], 'video hash differs')
            video_bytes += item.stat().st_size
        n = metrics['env_steps']
        config, actual = read(directory / 'environment_config.json'), read(directory / 'actual_protocol.json')
        timing = read(directory / 'control_timing.json')
        require(0 < n <= c['config']['episode_length'] == config['episode_length'], 'episode cap differs')
        require(config['suite'] == c['config']['suite'] and config['task_name'] == c['config']['task_name']
                and config['task_id'] == c['config']['task_id'], 'actual task differs')
        # v1 wrote raw.init_state_id AFTER reset. Native reset first selects the
        # requested state then increments that cursor by n_envs=1. Do not relabel
        # the old file or silently treat the post-reset cursor as selected init.
        require(config['init'] == c['config']['init'] + 1
                and config['suite_assets']['init'] == c['config']['init'], 'v1 init cursor contract differs')
        require(actual['case'] == c and actual['protocol_sha256'] == plan_sha, 'actual protocol/case differs')
        require(config['controller'] == 'OSC_POSE' and config['relative'] and config['control_hz'] == 20, 'controller differs')
        require(timing['count'] == n and abs(timing['min_dt'] - .05) < 1e-7
                and abs(timing['max_dt'] - .05) < 1e-7, 'physical timebase differs')
        require(actual['video']['frames'] == n and actual['video']['playback_fps'] == 20, 'native frame count differs')
        require(actual['recording_source_sha256'] == recorder_sha, 'unreviewed recorder')
        frames = [json.loads(l) for l in (directory / 'frame_map.jsonl').read_text().splitlines()]
        require(len(frames) == n and all(f['env_step'] == i and f['frame_index'] == i
                and f['phase'] == 'before_action' and abs(f['pts_seconds'] - i / 20) < 1e-7
                for i, f in enumerate(frames)), 'actual frame mapping differs')
        actions = read(directory / 'executed_actions.json')
        commands = read(directory / 'controller_commands.json')
        queue = read(directory / 'queue_trace.json')
        chunks = read(directory / 'chunk_calls.json')
        require(len(actions) == len(commands) == len(queue) == n and len(chunks) == metrics['chunk_calls'], 'action/queue count differs')
        require(all(q['step'] == i and q['returned'] and q['raw_controller_action'] == actions[i]
                and len(actions[i]) == 7 for i, q in enumerate(commands)), 'actual submitted command differs')
        require(all(ch['shape'] == [1, 50, 7] for ch in chunks), 'generation shape differs')
        require(all(q['step'] == i and q['chunk_calls_after'] - q['chunk_calls_before'] in (0, 1)
                for i, q in enumerate(queue)), 'invalid forward/queue audit')
        method = c['method']
        if method in ('H50', 'H20'):
            horizon = int(method[1:])
            require(metrics['chunk_calls'] == math.ceil(n / horizon), 'baseline actual forward count differs')
            require(all(q['chunk_calls_after'] - q['chunk_calls_before'] == int(i % horizon == 0)
                        and q['queue_after'] == horizon - 1 - i % horizon for i, q in enumerate(queue)), 'baseline used stale/wrong queue')
        signature = {k: actual['first'][k] for k in ('state_sha256', 'input_sha256', 'noise_sha256')}
        pair = (c['condition'], c['seed'])
        require(pair not in signature_groups or signature_groups[pair] == signature, 'paired first state/input/noise differ')
        signature_groups[pair] = signature
        trace = directory / 'monitor_steps.jsonl'
        rows = [json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
        scored = [[x['step'], x['score']] for x in rows if type(x.get('score')) in (int, float)]
        events = [{k: x.get(k) for k in ('step', 'score', 'threshold', 'discarded_actions')}
                  for x in rows if x.get('interrupted')]
        guides = [{k: x.get(k) for k in ('step', 'denoise', 'guidance_loss', 'gradient_l2',
                  'clean_action_update_l2', 'guidance_seconds', 'original_update_formula_exact_equal')}
                  for x in rows if 'gradient_l2' in x]
        require(len(events) == metrics['event_replans'] and len(guides) == metrics['guidance_calls'], 'event/guide counters differ')
        require(all(g['original_update_formula_exact_equal'] and g['gradient_l2'] is not None
                and math.isfinite(g['gradient_l2']) for g in guides), 'OGG actual update audit failed')
        for row in rows:
            if 'per_action_clean_update_l2' in row:
                require(all(abs(v) < 1e-12 for slot in row['per_action_clean_update_l2'] for v in slot[1:]), 'OGG non-first-action update')
        if method in ('H50', 'H20', 'mlp_lvm'):
            require(not guides, 'OGG ran in a non-OGG method')
        if method.startswith('mlp_'):
            require(actual['monitor_horizon_steps'] == 20 and actual['monitor_first_due_step'] == 24, 'monitor schedule differs')
            require(len([x for x in rows if 'score' in x]) == n, 'step trace incomplete')
        result.append({'record': record, 'trial_sha256': sha(path), 'environment': config,
            'selected_init_from_frozen_reset_contract': c['config']['init'],
            'first': signature, 'score_start': actual['monitor_first_due_step'], 'scored': scored,
            'events': events, 'guidance': guides})
    return {'plan': plan, 'execution': execution, 'trials': result,
            'integrity': {'status': 'passed', 'videos_sha256_checked': expected,
                          'sealed_artifacts_sha256_checked': sealed_files,
                          'video_bytes': video_bytes, 'paired_signature_groups': len(signature_groups),
                          'scope': 'native PTS/frame maps and capture assertions plus file hashes, no new full video decode'}}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    queue = read(root / 'status.json')
    require(queue['status'] == 'completed' and queue['returncode'] == 0, 'queue not verified complete')
    plan = read(root / 'launch_plan.json')
    for path, expected in plan['sources'].items():
        require(sha(path) == expected, 'source/input/weight changed: ' + path)
    env_source = Path(plan['runtime']) / 'lerobot/src/lerobot/envs/libero.py'
    source = env_source.read_text()
    require('self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])' in source
            and 'self.init_state_id += self._reset_stride' in source
            and 'self._reset_stride = n_envs' in source, 'native reset cursor semantics changed')
    processes = subprocess.run(['pgrep', '-af', r'[c]orrector_crosssuite_(queue|eval)[.]py'], capture_output=True, text=True)
    require(processes.returncode == 1 and not processes.stdout.strip(), 'evaluation process still alive or process query failed')
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_state = {'returncode': gpu.returncode, 'output': gpu.stdout.strip(), 'error': gpu.stderr.strip()}
    except OSError as exc:
        gpu_state = {'query_available': False, 'error': type(exc).__name__ + ': ' + str(exc)}
    output = {'schema': 'crosssuite_readonly_audit_v1', 'audit_source_sha256': sha(Path(__file__)) if '__file__' in globals() and Path(__file__).is_file() else None,
              'root': str(root), 'queue': queue, 'budget': read(root / 'budget.json'),
              'training_scope': read(root / 'training_scope_audit.json'), 'bridge_equivalence': read(root / 'bridge_equivalence.json'),
              'environment_probe': read(root / 'environment_probe.json'), 'source_files_unchanged': len(plan['sources']),
              'metadata_warning': {'field': 'environment_config.json:init',
                  'meaning': 'post-reset next-init cursor; not selected init', 'offset': 1,
                  'selected_init': 'case.config.init and suite_assets.init, verified against native reset selection/increment order',
                  'native_env_source_sha256': sha(env_source), 'source_results_modified': False},
              'current_process_check': {'evaluators_alive': False, 'gpu_query': gpu_state},
              'disk_free_gib': {str(p): shutil.disk_usage(p).free / 2**30 for p in ('/root', '/root/autodl-tmp')},
              'bridge': audit_phase(root / 'bridge', 4), 'transfer': audit_phase(root / 'transfer', 880),
              'source_writes': False, 'gpu_started': False, 'audited_utc': datetime.now(timezone.utc).isoformat()}
    print(json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


if __name__ == '__main__':
    main()

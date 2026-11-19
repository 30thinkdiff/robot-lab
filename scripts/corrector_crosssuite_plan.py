"""CPU-only, outcome-independent plan and paired analysis for task transfer.

Object is an engineering bridge only. Formal cases never contain Object tasks.
The frozen SmolVLA model's own task-level training coverage is not established.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

METHODS = ('H50', 'H20', 'mlp_lvm', 'mlp_ogg')
CAPS = {'libero_object': 280, 'libero_spatial': 280, 'libero_goal': 300,
        'libero_10': 520, 'libero_90': 400}
PRIMARY = ('libero_spatial', 'libero_goal', 'libero_10')
ROUNDS = ((0, 1000), (1, 1001), (0, 1001), (1, 1000))
NINETY_IDS = (0, 9, 19, 29, 39, 49, 59, 69, 79, 89)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def schedule(inventory, bridge=False):
    """Priority prefix: every suite/task, then repetitions, then fixed 90 subset."""
    object_names = {t['name'] for t in inventory['libero_object']['tasks']}
    groups = []
    if bridge:
        groups = [('bridge', 'libero_object', 1, 0, 1001)]
    else:
        for init, seed in ROUNDS:
            for task in range(10):
                for suite in PRIMARY:
                    groups.append(('primary', suite, task, init, seed))
        for init, seed in ROUNDS:
            for task in NINETY_IDS:
                groups.append(('ninety_subset', 'libero_90', task, init, seed))
        for seed in (1000, 1001):
            for task in range(10):
                for suite in PRIMARY:
                    groups.append(('extra_init', suite, task, 2, seed))
    cases = []
    conditions = {}
    for phase, suite, task, init, seed in groups:
        entry = inventory[suite]['tasks'][task]
        if (entry['task_id'] != task or inventory[suite]['horizon'] != CAPS[suite]
                or not entry['bddl_exists'] or not entry['init_exists']
                or not 0 <= init < entry['init_count']):
            raise ValueError('task index, horizon, assets or init range differs')
        if not bridge and entry['name'] in object_names:
            raise ValueError('formal held-out task overlaps Object')
        condition = f'{suite}/task_{task}/init_{init}'
        conditions[condition] = {**entry, 'suite': suite, 'init': init,
                                 'episode_length': CAPS[suite]}
        cases.extend((condition, seed, method, phase) for method in METHODS)
    if len(cases) != len(set((c, s, m) for c, s, m, _ in cases)):
        raise ValueError('duplicate trial')
    return cases, conditions


def materialize(make_plan, wanted, conditions, methods, protocol):
    """Suite and cap enter case hashes; immutable plans never reuse Object IDs."""
    plan = {'protocol': protocol, 'cases': []}
    for condition, seed, method, phase in wanted:
        c = conditions[condition]
        config = {**methods[method], 'suite': c['suite'], 'task_name': c['name'],
                  'task_id': c['task_id'], 'init': c['init'], 'phase': phase,
                  'episode_length': c['episode_length']}
        plan['cases'].extend(make_plan([condition], [seed], {method: config}, {})['cases'])
    return plan


def configure_instance(cfg, case):
    """Only the new config instance changes; no factory/class/source replacement."""
    c = case['config']
    if (cfg.task != 'libero_object' or cfg.episode_length != 280 or cfg.fps != 20
            or cfg.task_ids != [c['task_id']] or cfg.is_libero_plus
            or cfg.control_mode != 'relative' or cfg.hard_reset or not cfg.init_states):
        raise ValueError('reviewed frozen environment call changed')
    if c['episode_length'] != CAPS.get(c['suite']):
        raise ValueError('suite horizon differs')
    cfg.task = c['suite']
    cfg.episode_length = c['episode_length']
    return cfg


def reset_with_init_receipt(raw, reset, *args, **kwargs):
    """Read the selected index before native reset advances its next-init cursor."""
    cursor = raw.init_state_id
    selected = cursor % len(raw._init_states)
    result = reset(*args, **kwargs)
    if raw.init_state_id != cursor + raw._reset_stride:
        raise ValueError('native reset cursor increment differs')
    return result, {'init': selected, 'init_cursor_after_reset': raw.init_state_id,
                    'reset_stride': raw._reset_stride, 'init_recording_schema': 'selected_index_v2'}


def summarize(root):
    """Raw outcomes plus complete-quartet comparisons; pending is never failure."""
    root = Path(root)
    rows = [read(p) for p in sorted((root / 'trials').glob('*.json'))]
    completed = [r for r in rows if r['status'] == 'completed']
    groups = defaultdict(dict)
    for r in completed:
        c = r['case']
        key = (c['config']['phase'], c['config']['suite'], c['condition'], c['seed'])
        if c['method'] in groups[key]:
            raise ValueError('duplicate matched result')
        groups[key][c['method']] = r
    paired = {k: v for k, v in groups.items() if set(v) == set(METHODS)}
    totals = []
    strata = sorted({(k[0], k[1]) for k in groups})
    for phase, suite in strata:
        matched = [v for k, v in paired.items() if k[:2] == (phase, suite)]
        raw = [r for r in completed if (r['case']['config']['phase'], r['case']['config']['suite']) == (phase, suite)]
        for method in METHODS:
            results = [r['result'] for r in raw if r['case']['method'] == method]
            p = [g[method]['result'] for g in matched]
            row = {'phase': phase, 'suite': suite, 'method': method, 'completed': len(results),
                   'successes': sum(r['success'] for r in results), 'matched_cases': len(p),
                   'paired_successes': sum(r['success'] for r in p),
                   'paired_success_rate': sum(r['success'] for r in p) / len(p) if p else None}
            for name in ('env_steps', 'chunk_calls', 'policy_s', 'monitor_s', 'event_replans',
                         'guidance_calls', 'episode_wall_s', 'peak_torch_allocated_gib'):
                values = [r[name] for r in p if type(r.get(name)) in (int, float)]
                row['paired_mean_' + name] = sum(values) / len(values) if values else None
            totals.append(row)
    comparisons = []
    for phase, suite in strata:
        matched = [v for k, v in paired.items() if k[:2] == (phase, suite)]
        for baseline in ('H50', 'H20'):
            for method in ('mlp_lvm', 'mlp_ogg'):
                comparisons.append({'phase': phase, 'suite': suite, 'baseline': baseline, 'method': method,
                    'matched': len(matched),
                    'method_only_success': sum(g[method]['result']['success'] and not g[baseline]['result']['success'] for g in matched),
                    'baseline_only_success': sum(g[baseline]['result']['success'] and not g[method]['result']['success'] for g in matched)})
    result = {'completed': len(completed), 'complete_quartets': len(paired),
              'partial_quartets': len(groups) - len(paired), 'by_suite_phase': totals, 'paired_outcomes': comparisons,
              'scope': 'Frozen Object-trained Corrector task transfer; base SmolVLA training task coverage unknown. '
                       'Repeated seeds of an init are not distinct physical scenarios. LIBERO-90 is a fixed subset. '
                       'Suite/phase results are separate; no result-based selection or retuning.'}
    write(root / 'analysis/transfer_summary.json', result)
    if totals:
        with (root / 'analysis/transfer_by_suite.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(totals[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(totals)
    return result

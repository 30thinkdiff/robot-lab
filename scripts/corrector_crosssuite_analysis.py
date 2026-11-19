"""CPU analysis of independently audited, fully paired frozen-MLP transfer runs.

Reuse coverage and paired-outcome definitions from the original results audit.
No experiment is started, no weights are loaded and no source result is edited.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile

from corrector_crosssuite_plan import METHODS, PRIMARY
from corrector_results_analysis import coverage_stats, csv_write, discordance

NAMES = {'libero_spatial': 'Spatial', 'libero_goal': 'Goal', 'libero_10': 'Long', 'libero_90': 'LIBERO-90 subset'}
METRICS = ('env_steps', 'chunk_calls', 'event_replans', 'guidance_calls', 'policy_s',
           'monitor_s', 'episode_wall_s', 'peak_torch_allocated_gib')
PAIRS = (('H50', 'H20'), ('H50', 'mlp_lvm'), ('H50', 'mlp_ogg'),
         ('H20', 'mlp_lvm'), ('H20', 'mlp_ogg'), ('mlp_lvm', 'mlp_ogg'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def average(values):
    values = list(values)
    return statistics.mean(values) if values else None


def analyse(data):
    transfer = data['transfer']
    records = transfer['trials']
    require(data['queue']['status'] == 'completed' and data['queue']['returncode'] == 0,
            'queue not completely successful')
    require(transfer['execution']['completed'] == transfer['execution']['planned'] == len(records) == 880
            and transfer['execution']['errors'] == 0 and transfer['integrity']['status'] == 'passed', 'incomplete or unaudited input')
    require(data['source_writes'] is False and data['gpu_started'] is False, 'not a read-only CPU audit')
    conditions = transfer['plan']['protocol']['conditions']
    actual_plan = transfer['plan']['cases']
    expected_ids = {c['id'] for c in actual_plan}
    require(len(actual_plan) == len(expected_ids) == 880, 'duplicate plan ID')
    trials, guide_rows = [], []
    group = defaultdict(dict)
    seen = set()
    for trial in records:
        r = trial['record']
        c, v = r['case'], r['result']
        require(c['id'] in expected_ids and c['id'] not in seen and r['status'] == 'completed'
                and type(v['success']) is bool, 'unknown, duplicate or incomplete trial')
        seen.add(c['id'])
        cfg = c['config']
        require(cfg['suite'] != 'libero_object' and cfg['suite'] in NAMES and c['method'] in METHODS,
                'Object bridge/unknown suite or method entered formal analysis')
        require(c == next(x for x in actual_plan if x['id'] == c['id']), 'record differs from frozen plan')
        if c['method'].startswith('mlp_'):
            require(cfg['corrector_sha256'] == data['training_scope']['checkpoint_sha256'], 'mixed predictor weight')
        key = (c['condition'], c['seed'])
        require(c['method'] not in group[key], 'duplicate paired method')
        group[key][c['method']] = trial
        scored = [{'step': step} for step, score in trial['scored']]
        coverage = coverage_stats(scored, trial['score_start'], v['env_steps'], 1, 20) if c['method'].startswith('mlp_') else None
        require(not coverage or not coverage['duplicate_scores'] and not coverage['out_of_schedule_scores'], 'invalid score schedule')
        require(len(trial['events']) == v['event_replans'] and len(trial['guidance']) == v['guidance_calls'], 'trace counters differ')
        trials.append({'id': c['id'], 'phase': cfg['phase'], 'suite': cfg['suite'],
            'task_id': cfg['task_id'], 'task_name': cfg['task_name'], 'condition': c['condition'],
            'init': cfg['init'], 'seed': c['seed'], 'method': c['method'], 'success': v['success'],
            **{k: v[k] for k in METRICS}, 'callback_wall_s': r['wall_s'],
            'score_samples': len(trial['scored']), 'eligible_score_steps': coverage['eligible_score_steps'] if coverage else None,
            'coverage': coverage['scores'] / coverage['eligible_score_steps'] if coverage and coverage['eligible_score_steps'] else None,
            'max_gap_steps': coverage['max_gap_steps'] if coverage else None,
            'mean_mismatch': average(score for step, score in trial['scored']),
            'events_with_nonempty_queue': sum(e['discarded_actions'] > 0 for e in trial['events']),
            'events_with_empty_queue': sum(e['discarded_actions'] == 0 for e in trial['events']),
            'state_sha256': trial['first']['state_sha256'], 'trial_sha256': trial['trial_sha256'],
            'selected_init_from_reset_contract': trial['selected_init_from_frozen_reset_contract'],
            'logged_post_reset_cursor': trial['environment']['init']})
        for g in trial['guidance']:
            guide_rows.append({'id': c['id'], 'suite': cfg['suite'], 'phase': cfg['phase'],
                'condition': c['condition'], 'seed': c['seed'], 'method': c['method'], **g})
    require(len(group) == 220 and all(set(g) == set(METHODS) for g in group.values()), 'incomplete quartet')
    require(Counter(r['phase'] for r in trials) == {'primary': 480, 'ninety_subset': 160, 'extra_init': 240}, 'phase counts differ')
    for pair in group.values():
        require(len({tuple(sorted(t['first'].items())) for t in pair.values()}) == 1, 'paired signatures differ')
    views = {phase: [r for r in trials if r['phase'] == phase] for phase in ('primary', 'extra_init', 'ninety_subset')}
    views['all_inits'] = trials
    methods, comparisons, task_rows, stability = [], [], [], []
    for cohort, cohort_rows in views.items():
        for suite in sorted({r['suite'] for r in cohort_rows}):
            selected = [r for r in cohort_rows if r['suite'] == suite]
            for method in METHODS:
                subset = [r for r in selected if r['method'] == method]
                subset_ids = {r['id'] for r in subset}
                guides = [g for g in guide_rows if g['id'] in subset_ids]
                scored = sum(r['score_samples'] for r in subset)
                eligible = sum(r['eligible_score_steps'] or 0 for r in subset)
                methods.append({'cohort': cohort, 'suite': suite, 'method': method,
                    'completed': len(subset), 'successes': sum(r['success'] for r in subset),
                    'success_rate': average(r['success'] for r in subset),
                    **{'mean_' + k: average(r[k] for r in subset) for k in METRICS},
                    'monitor_ms_per_env_step': 1000 * sum(r['monitor_s'] for r in subset) / sum(r['env_steps'] for r in subset),
                    'score_samples': scored, 'eligible_score_steps': eligible,
                    'score_coverage': scored / eligible if eligible else None,
                    'max_gap_steps': max((r['max_gap_steps'] for r in subset if r['max_gap_steps'] is not None), default=None),
                    'guidance_calls': len(guides), 'nonzero_gradient_calls': sum(g['gradient_l2'] > 0 for g in guides),
                    'mean_guidance_loss': average(g['guidance_loss'] for g in guides),
                    'mean_guidance_ms_instrumented': average(1000 * g['guidance_seconds'] for g in guides),
                    'median_gradient_l2': statistics.median(g['gradient_l2'] for g in guides) if guides else None,
                    'median_action_update_l2': statistics.median(g['clean_action_update_l2'] for g in guides) if guides else None})
                for task in sorted({r['task_id'] for r in subset}):
                    task_subset = [r for r in subset if r['task_id'] == task]
                    task_rows.append({'cohort': cohort, 'suite': suite, 'task_id': task,
                        'task_name': task_subset[0]['task_name'], 'method': method,
                        'completed': len(task_subset), 'successes': sum(r['success'] for r in task_subset)})
            pairs = defaultdict(dict)
            for r in selected:
                pairs[r['condition'], r['seed']][r['method']] = r['success']
            for baseline, candidate in PAIRS:
                d = discordance(pairs, baseline, candidate)
                # Repeated init/seed observations are correlated. Do not expose
                # a naive case-independent p-value as confirmatory evidence.
                d.pop('exact_mcnemar_two_sided_unadjusted')
                comparisons.append({'cohort': cohort, 'suite': suite, 'baseline': baseline, 'candidate': candidate, **d})
    for suite in sorted(NAMES):
        for method in METHODS:
            states = defaultdict(list)
            for r in trials:
                if r['suite'] == suite and r['method'] == method:
                    states[r['condition']].append(r)
            require(all(len(v) == 2 and {r['seed'] for r in v} == {1000, 1001} for v in states.values()), 'seed/state matrix differs')
            stability.append({'suite': suite, 'method': method, 'task_init_conditions': len(states),
                'both_seeds_success': sum(all(r['success'] for r in v) for v in states.values()),
                'both_seeds_failure': sum(not any(r['success'] for r in v) for v in states.values()),
                'seed_sensitive_conditions': sum(len({r['success'] for r in v}) > 1 for v in states.values())})
    patterns = []
    for (condition, seed), pair in sorted(group.items()):
        first = pair['H50']
        c = first['record']['case']
        cfg = c['config']
        bits = ''.join(str(int(pair[m]['record']['result']['success'])) for m in METHODS)
        patterns.append({'condition': condition, 'seed': seed, 'phase': cfg['phase'], 'suite': cfg['suite'],
            'task_id': cfg['task_id'], 'init': cfg['init'], 'task_name': cfg['task_name'], 'pattern': bits,
            **{m: pair[m]['record']['result']['success'] for m in METHODS},
            **{m + '_id': pair[m]['record']['case']['id'] for m in METHODS}})
    for row in patterns:
        same = [p for p in patterns if p['condition'] == row['condition']]
        row['same_pattern_both_seeds'] = len({p['pattern'] for p in same}) == 1
    predicates = {
        'H20_only': lambda p: p['pattern'] == '0100',
        'correctors_rescue_both_baselines': lambda p: p['pattern'] == '0011',
        'OGG_only': lambda p: p['pattern'] == '0001',
        'OGG_hurts_LVM': lambda p: p['mlp_lvm'] and not p['mlp_ogg'],
        'OGG_rescues_LVM': lambda p: not p['mlp_lvm'] and p['mlp_ogg'],
        'LVM_hurts_H50': lambda p: p['H50'] and not p['mlp_lvm'],
        'all_failure': lambda p: p['pattern'] == '0000',
        'all_success': lambda p: p['pattern'] == '1111'}
    candidates, category_counts = [], []
    for category, predicate in predicates.items():
        matches = [p for p in patterns if predicate(p)]
        category_counts.append({'category': category, 'quartets': len(matches),
                                'task_init_conditions': len({r['condition'] for r in matches})})
        pool = sorted(matches, key=lambda r: (not r['same_pattern_both_seeds'], r['suite'], r['task_id'], r['init'], r['seed']))
        selected = []
        for r in pool:
            if r['suite'] not in {x['suite'] for x in selected}:
                selected.append(r)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            for r in pool:
                if r['condition'] not in {x['condition'] for x in selected}:
                    selected.append(r)
                if len(selected) == 2:
                    break
        candidates.extend({'category': category, **r} for r in selected)
    source = transfer['plan']['protocol']
    overall_main = {}
    for method in METHODS:
        selected = [r for r in trials if r['suite'] in PRIMARY and r['method'] == method]
        primary_rows = [r for r in selected if r['phase'] == 'primary']
        overall_main[method] = {'all_inits_successes': sum(r['success'] for r in selected), 'all_inits_n': len(selected),
            'primary_successes': sum(r['success'] for r in primary_rows), 'primary_n': len(primary_rows)}
    coverage_rows = [r for r in trials if r['method'].startswith('mlp')]
    summary = {'status': 'completed', 'formal_trials': len(trials), 'errors': 0, 'paired_quartets': len(group),
        'primary_trials': 480, 'secondary_ninety_trials': 160, 'extra_init_trials': 240,
        'task_init_conditions': len({r['condition'] for r in trials}), 'eval_seeds': [1000, 1001],
        'tasks_per_suite': {s: len({r['task_id'] for r in trials if r['suite'] == s}) for s in NAMES},
        'initial_state_vector_sha_count': len({r['state_sha256'] for r in trials}),
        'bddl_and_state_fingerprint_count': len({(conditions[r['condition']]['bddl_sha256'], r['state_sha256']) for r in trials}),
        'queue_started_unix': data['budget']['started_unix'], 'queue_finished_unix': data['queue']['updated_unix'],
        'queue_wall_seconds': data['queue']['updated_unix'] - data['budget']['started_unix'],
        'formal_wall_seconds': transfer['execution']['wall_seconds'], 'native_rollout_wall_seconds_sum': sum(r['episode_wall_s'] for r in trials),
        'overall_main_descriptive': overall_main,
        'coverage': {'scores': sum(r['score_samples'] for r in coverage_rows),
            'eligible_steps': sum(r['eligible_score_steps'] for r in coverage_rows),
            'max_gap_steps': max(r['max_gap_steps'] for r in coverage_rows), 'warmup_excluded_steps': 24},
        'ogg': {'events': sum(r['event_replans'] for r in trials if r['method'] == 'mlp_ogg'),
                'guidance_calls': len(guide_rows), 'nonzero_gradient_calls': sum(g['gradient_l2'] > 0 for g in guide_rows)},
        'integrity': {'transfer': transfer['integrity'], 'bridge': data['bridge']['integrity'],
                      'source_files_unchanged': data['source_files_unchanged'], 'current_process_check': data['current_process_check']},
        'metadata_warning': data['metadata_warning'], 'checkpoint_sha256': data['training_scope']['checkpoint_sha256'],
        'policy_revision': source['policy_revision'],
        'frozen_adapter_sha256': source['crosssuite']['adapter_sha256'], 'cpu_audit_source_sha256': data['audit_source_sha256'],
        'audited_utc': data['audited_utc'], 'current_disk_free_gib': data['disk_free_gib'],
        'scope': 'Object-trained frozen MLP, original LVM/cosine OGG. Primary and supplementary cohorts separate. '
                 'Base SmolVLA training task coverage unknown; no zero-shot claim. '
                 'Same task/init evaluated with two seeds is correlated, not independent new scenes. '
                 'No abnormality ground-truth labels; no AUROC/false-alarm claim. '
                 'Videos remain remote; this analysis downloads no full videos or tensors.'}
    return summary, {'methods': methods, 'comparisons': comparisons, 'tasks': task_rows,
                     'seed_stability': stability, 'trials': trials, 'guidance': guide_rows,
                     'patterns': patterns, 'replay_candidates': candidates, 'category_counts': category_counts}


def plots(tables, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'robotlab-crosssuite-mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    colors = ('#94a3b8', '#0f766e', '#3564ac', '#bc6c3c')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, cohort, suites, title in zip(axes, ('primary', 'all_inits'), (list(PRIMARY), list(NAMES)),
            ('Primary: 10 tasks / suite, init 0-1', 'Cumulative: init 0-2; fixed LIBERO-90 subset init 0-1')):
        for j, method in enumerate(METHODS):
            rows = [next(r for r in tables['methods'] if r['cohort'] == cohort and r['suite'] == s and r['method'] == method) for s in suites]
            x = [i + (j - 1.5) * .19 for i in range(len(suites))]
            y = [100 * r['success_rate'] for r in rows]
            bars = ax.bar(x, y, width=.18, label=method.replace('mlp_', 'MLP-').upper(), color=colors[j])
            for b, r in zip(bars, rows):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.4, f"{r['successes']}/{r['completed']}",
                        ha='center', va='bottom', fontsize=7, rotation=90)
        ax.set_xticks(range(len(suites)), [NAMES[s].replace(' subset', '\nsubset') for s in suites])
        ax.set_ylim(0, 101)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Task success (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), fontsize=8, ncol=4)
    fig.suptitle('Frozen Object-trained MLP Corrector: task transfer', fontsize=13)
    fig.text(.5, .015, 'Same-init seeds are correlated; no base-policy zero-shot claim. Native suite horizons, frozen thresholds/weights.',
             ha='center', fontsize=8, color='#475569')
    fig.tight_layout(rect=(0, .045, 1, .86))
    for suffix in ('svg', 'png'):
        fig.savefig(output / ('success_rates.' + suffix), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-plot', action='store_true')
    a = p.parse_args()
    source = a.input.read_bytes()
    summary, tables = analyse(json.loads(source))
    summary['source_audit_sha256'] = hashlib.sha256(source).hexdigest()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for name, rows in tables.items():
        csv_write(a.output / (name + '.csv'), rows)
    if not a.no_plot:
        plots(tables, a.output)
    print(json.dumps({'formal_trials': summary['formal_trials'], 'main': summary['overall_main_descriptive'],
                      'coverage': summary['coverage'], 'ogg': summary['ogg']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""离线重算纯SmolVLA窗口Pilot；核验计划/队列/动作，不加载模型或访问网络。"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

METHODS = ['H50', 'H20', 'H10', 'H5', 'H2', 'H1']


def summarize(z):
    read = lambda name: json.loads(z.read(name))
    execution, plan = read('execution.json'), read('plan.json')
    assert execution['status'] == 'completed' and execution['planned'] == execution['completed'] == 24
    assert execution['errors'] == 0
    protocol = plan['protocol']
    assert protocol['chunk_size'] == 50 and protocol['num_steps'] == 10 and protocol['fps'] == 20
    assert protocol['max_env_steps'] == 280 and protocol['seed'] == 1000
    assert not any(protocol[k] for k in ['corrector', 'lvm', 'ogg', 'automatic_expansion'])
    conditions = {f'libero_object/task_{t}/init_{i}' for t in [1, 9] for i in [0, 2]}
    expected = {(c, m, 1000) for c in conditions for m in METHODS}
    assert {(c['condition'], c['method'], c['seed']) for c in plan['cases']} == expected
    records = [read(n) for n in z.namelist() if n.startswith('trials/') and n.endswith('.json')]
    assert len(records) == 24
    seen, paired, cases = set(), {}, []
    queue_steps = 0
    for record in records:
        case, r = record['case'], record['result']
        key = (case['condition'], case['method'], case['seed'])
        assert key in expected and key not in seen
        seen.add(key)
        assert record['status'] == 'completed' and type(r['success']) is bool and r['queue_verified']
        h = int(case['method'][1:]); steps = r['env_steps']
        assert case['config']['n_action_steps'] == h and not case['config']['uses_corrector']
        assert 1 <= steps <= 280
        assert r['chunk_calls'] == r['sample_actions_calls'] == math.ceil(steps/h)
        assert r['denoise_forward_calls'] == 10*r['chunk_calls']
        assert len(r['chunk_latency_s']) == r['chunk_calls']
        assert math.isclose(sum(r['chunk_latency_s']), r['policy_s'], rel_tol=1e-10)
        assert math.isclose(r['rollout_including_warmup_s']-r['warmup_s'], r['rollout_wall_s'], rel_tol=1e-10)
        assert 0 < r['policy_s'] < r['rollout_wall_s'] < record['wall_s']
        prefix = f"attempts/{case['id']}/attempt_{record['attempt']:03d}/"
        actual = read(prefix+'actual_protocol.json')
        assert actual['horizon'] == h
        assert all(actual[k] == v for k, v in protocol.items())
        trace, actions = read(prefix+'queue_trace.json'), read(prefix+'executed_actions.json')
        assert len(trace) == len(actions) == steps
        for s, row in enumerate(trace):
            assert row['step'] == s and row['fresh_inference'] == (s % h == 0)
            assert row['queue_before'] == (0 if s % h == 0 else h-s % h)
            assert row['queue_after'] == h-1-s % h and row['chunk_index'] == s//h
        queue_steps += steps
        assert all(len(a) == 7 and all(math.isfinite(v) for v in a) for a in actions)
        delta = [[b-a for a, b in zip(actions[s-1], actions[s])] for s in range(1, steps)]
        for label, ids in [('all', list(range(steps-1))),
                           ('boundary', [s-1 for s in range(1, steps) if s % h == 0]),
                           ('within', [s-1 for s in range(1, steps) if s % h != 0])]:
            assert r['continuity'][label+'_n'] == len(ids)
            for name, part in [('translation', slice(0, 3)), ('rotation', slice(3, 6))]:
                values = [math.sqrt(sum(x*x for x in delta[s][part])) for s in ids]
                saved = r['continuity'][label+'_'+name+'_delta_mean']
                assert (saved is None and not values) or math.isclose(saved, statistics.mean(values), rel_tol=1e-10)
        switches = sum((b[6] > 0) != (a[6] > 0) for a, b in zip(actions, actions[1:]))
        assert switches == r['continuity']['gripper_sign_switches']
        assert len(record['video_artifacts']) == 1  # 内容仍在服务器；本地不假称核验视频字节。
        signature = tuple(r[k] for k in ['initial_state_sha256', 'first_input_sha256', 'first_noise_sha256'])
        paired.setdefault((case['condition'], case['seed']), set()).add(signature)
        cases.append({'condition': case['condition'], 'method': case['method'], 'seed': case['seed'],
                      'success': r['success'], 'env_steps': steps, 'chunk_calls': r['chunk_calls'],
                      'policy_s': r['policy_s'], 'rollout_wall_s': r['rollout_wall_s'],
                      'episode_wall_s': record['wall_s'], 'warmup_s': r['warmup_s'],
                      'video_remote_path': r['video_paths'][0]})
    assert seen == expected and all(len(v) == 1 for v in paired.values())
    rows = list(csv.DictReader(io.StringIO(z.read('analysis/pilot_overall.csv').decode())))
    metrics = ['env_steps', 'chunk_calls', 'sample_actions_calls', 'denoise_forward_calls',
               'policy_s', 'rollout_wall_s', 'peak_torch_allocated_gib', 'peak_torch_reserved_gib']
    for row in rows:
        subset = [r for r in records if r['case']['method'] == row['method']]
        assert len(subset) == int(row['completed']) == int(row['planned']) == 4
        assert sum(r['result']['success'] for r in subset) == int(row['successes'])
        assert math.isclose(float(row['success_rate']), int(row['successes'])/4)
        for key in metrics:
            assert math.isclose(float(row['mean_'+key]), statistics.mean(r['result'][key] for r in subset), rel_tol=1e-10)
        assert math.isclose(float(row['mean_episode_wall_s']), statistics.mean(r['wall_s'] for r in subset), rel_tol=1e-10)
        calls = sum(r['result']['chunk_calls'] for r in subset)
        policy_s = sum(r['result']['policy_s'] for r in subset)
        row.update(weighted_chunk_latency_ms=1000*policy_s/calls,
                   mean_rollout_control_hz=sum(r['result']['env_steps'] for r in subset)/sum(r['result']['rollout_wall_s'] for r in subset),
                   mean_denoise_forward_calls=float(row['mean_denoise_forward_calls']))
    findings = {'status': 'verified', 'completed': 24, 'errors': 0, 'paired_conditions': len(paired),
        'audited_queue_steps': queue_steps, 'sample_actions_calls': sum(r['result']['sample_actions_calls'] for r in records),
        'denoise_forward_calls': sum(r['result']['denoise_forward_calls'] for r in records),
        'wall_seconds': execution['wall_seconds'], 'source_sha256': protocol['source_sha256'],
        'protocol': protocol, 'video_bytes_verified_locally': False,
        'limits': ['failure-selected two tasks / two initial states / one seed; no population estimate',
                   'matched initial state/input/noise; different replan schedules use later random draws differently',
                   'timing includes synchronous queue-audit overhead; episode wall includes first policy load',
                   'action-vector differences are not physical jerk; H1 has no within-chunk transitions',
                   '20Hz is simulated control frequency; not a wall-clock inference deadline guarantee']}
    return findings, rows, sorted(cases, key=lambda r:(r['condition'], METHODS.index(r['method'])))


def render(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4), constrained_layout=True)
    names = [r['method'] for r in rows]
    axes[0].bar(names, [int(r['successes']) for r in rows], color='#3476ad')
    axes[0].set(ylim=(0, 4.6), yticks=[0,1,2,3,4], title='Successful trials (4 per method)', ylabel='Count')
    axes[1].plot(names, [float(r['mean_rollout_wall_s']) for r in rows], 'o-', label='Rollout excl. warmup')
    axes[1].plot(names, [float(r['mean_policy_s']) for r in rows], 'o-', label='Full-chunk inference')
    axes[1].set(title='Mean time per episode', ylabel='Seconds'); axes[1].legend(fontsize=8)
    axes[2].plot(names, [float(r['mean_boundary_translation_delta_mean']) for r in rows], 'o-', label='Chunk boundary')
    axes[2].plot(names, [float(r['mean_all_translation_delta_mean']) for r in rows], 'o-', label='All adjacent controls')
    axes[2].set(title='Translation control-vector changes', ylabel='Mean L2 difference'); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output/'success_cost_continuity.png', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    # 审计通过后才创建报告目录；不解压任意路径、不改远程文件。
    with zipfile.ZipFile(args.bundle) as z:
        if z.testzip():
            raise RuntimeError('ZIP损坏')
        findings, rows, cases = summarize(z)
        findings['bundle_sha256'] = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ['execution.json', 'plan.json', 'preflight.json', 'analysis/pilot_overall.csv', 'analysis/pilot_summary.json']:
            (args.output/Path(name).name).write_bytes(z.read(name))
    (args.output/'audit.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2)+'\n')
    for name, data in [('cases.csv', cases), ('cost_analysis.csv', rows)]:
        with (args.output/name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(data)
    if args.plot:
        render(args.output, rows)
    print(json.dumps({k:findings[k] for k in ['status','completed','errors','paired_conditions',
        'audited_queue_steps','sample_actions_calls','denoise_forward_calls','bundle_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

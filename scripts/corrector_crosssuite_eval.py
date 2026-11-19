"""Opt-in task-transfer adapter over the reviewed frozen rollout, no new control loop."""
import argparse
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from corrector_crosssuite_plan import (METHODS, CAPS, read, write, digest, schedule,
                                       materialize, configure_instance, reset_with_init_receipt, summarize)

BASE = Path('/root/autodl-tmp/corrector')


def runtime_setup(runtime):
    os.environ.update(HF_HOME=str(runtime / 'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', LIBERO_CONFIG_PATH=str(runtime / 'libero-config'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(runtime / 'lerobot/src'))


def asset_probe(args, conditions):
    """Actual assets, task order, init and one physics step; not performance trials."""
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.libero import get_task_init_states, TASK_SUITE_MAX_STEPS
    import smolvla_libero_tools as sim
    rows = []
    for suite, task_id in sorted({(c['suite'], c['task_id']) for c in conditions.values()}):
        suite_obj = benchmark.get_benchmark_dict()[suite]()
        task = suite_obj.tasks[task_id]
        expected = next(c for c in conditions.values() if c['suite'] == suite and c['task_id'] == task_id)
        if TASK_SUITE_MAX_STEPS[suite] != CAPS[suite] or task.name != expected['name'] or task.language != expected['language']:
            raise ValueError('installed task definition or horizon differs')
        for field, folder, filename in [('bddl', 'bddl_files', task.bddl_file), ('init', 'init_states', task.init_states_file)]:
            path = Path(get_libero_path(folder)) / task.problem_folder / filename
            if digest(path) != expected[field + '_sha256']:
                raise ValueError('task asset changed')
        if len(get_task_init_states(suite_obj, task_id)) != expected['init_count']:
            raise ValueError('initial state count differs')
        if (args.output / 'STOP').exists() or time.time() >= args.deadline:
            write(args.output / 'environment_probe.json', {'status': 'paused', 'checks': rows})
            return
        cfg = LiberoEnv(task=suite, task_ids=[task_id], fps=20, episode_length=CAPS[suite],
            observation_height=256, observation_width=256, control_mode='relative',
            is_libero_plus=False, hard_reset=False, init_states=True,
            camera_name_mapping={'agentview_image': 'camera1', 'robot0_eye_in_hand_image': 'camera2'})
        env = sim.make_vector(cfg)
        try:
            raw = env.envs[0].unwrapped
            raw.init_state_id = 0
            env.reset(seed=1000)
            if raw.task != expected['name'] or raw.task_description != expected['language']:
                raise ValueError('actual environment task mismatch')
            before = float(raw._env.sim.data.time)
            env.step(np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=np.float32))
            dt = float(raw._env.sim.data.time) - before
            if abs(dt - .05) > 1e-7:
                raise ValueError('actual control dt differs from 20Hz')
            rows.append({'suite': suite, 'task_id': task_id, 'task_name': raw.task, 'dt': dt,
                         'episode_length': CAPS[suite], 'reset_and_step': 'passed'})
            write(args.output / 'environment_probe.json', {'status': 'running', 'checks': rows})
        finally:
            env.close()
    write(args.output / 'environment_probe.json', {'status': 'passed', 'checks': rows,
          'scope': 'environment engineering checks, not robot success-rate evidence'})


def run(args, bridge=False):
    sys.path.insert(0, str(args.frozen_scripts))
    expanded = importlib.import_module('corrector_expanded_eval')
    from motion_pilot import frozen_audit
    frozen = frozen_audit(args.frozen_scripts)
    wanted, conditions = schedule(read(args.inventory), bridge)
    phase = 'bridge' if bridge else 'transfer'
    args.mode = 'ordinary'
    args.j_checkpoint = args.training / 'J_s42/best.pt'
    args.c0_checkpoint = args.training / 'C0_s42/best.pt'
    true_deadline = args.deadline
    # Batch gate owns the true budget. Legacy per-case guard gets one-minute grace
    # to avoid recording a budget boundary as an error; no new episode after STOP.
    args.deadline += 60
    holder = expanded.load(args)
    import smolvla_libero_tools as sim
    from studies.policy_generalization.corrector import batch_evaluation as batch
    old_make, old_run, old_vector = batch.make_plan, batch.run_plan, sim.make_vector
    registry = {k: v for k, v in expanded.registry().items() if k in METHODS}
    active = {}
    receipt = {'schema': 'crosssuite_v1', 'frozen_manifest': frozen,
               'adapter_sha256': digest(__file__), 'planner_sha256': digest(Path(__file__).with_name('corrector_crosssuite_plan.py')),
               'inventory_sha256': digest(args.inventory), 'training_scope_audit': read(args.training_audit),
               'control_implementation': 'frozen original; only suite and native episode cap differ',
               'corrector': 'MLP seed42; original dynamic LVM; cosine OGG eta1/every3; no task calibration',
               'auxiliary_models': 'frozen loader also loads unused J/C0 and encoder; memory is not minimal baseline memory'}

    def make_plan(unused_conditions, unused_seeds, methods, protocol):
        protocol = dict(protocol, study_id='replay_casebank_crosssuite_20261117_v1_' + phase,
            experiment_label='跨任务泛化 · 原MLP四方法' if not bridge else '跨任务适配 · Object一致性检查',
            horizon=None, episode_length_by_suite=CAPS, crosssuite=receipt, conditions=conditions,
            budget_deadline_unix=true_deadline, planned_phases={'primary': 480, 'ninety_subset': 160, 'extra_init': 240})
        return materialize(old_make, wanted, conditions, methods, protocol)

    def vector(cfg):
        case, directory = active['case'], active['directory']
        configure_instance(cfg, case)
        env = old_vector(cfg)
        raw = env.envs[0].unwrapped
        old_reset, old_step = env.reset, env.step

        def reset(*a, **kw):
            result, init_receipt = reset_with_init_receipt(raw, old_reset, *a, **kw)
            expected = conditions[case['condition']]
            controller = raw._env.robots[0].controller
            if (raw.task != expected['name'] or raw.task_description != expected['language']
                    or raw.task_id != expected['task_id'] or raw.episode_length != expected['episode_length']
                    or controller.name != 'OSC_POSE' or not controller.use_delta
                    or init_receipt['init'] != expected['init']):
                raise ValueError('actual task/horizon/OSC controller differs')
            write(directory / 'environment_config.json', {'suite': cfg.task, 'task_name': raw.task,
                'task_id': raw.task_id, 'language': raw.task_description, **init_receipt,
                'episode_length': cfg.episode_length, 'control_hz': 20, 'controller': controller.name,
                'relative': bool(controller.use_delta), 'suite_assets': expected,
                'scope': 'same native rollout/control; no altered observation or controller scaling'})
            return result

        def step(*a, **kw):
            before = float(raw._env.sim.data.time)
            result = old_step(*a, **kw)
            dt = float(raw._env.sim.data.time) - before
            if abs(dt - .05) > 1e-7:
                raise ValueError('simulation control dt is not .05 seconds')
            active.setdefault('control_dt', []).append(dt)
            return result

        env.reset, env.step = reset, step
        return env

    def run_plan(root, plan, callbacks, **kw):
        def wrap(cb):
            def call(case, directory):
                active.clear()
                active.update(case=case, directory=directory)
                result = cb(case, directory)
                steps = result['env_steps']
                dt = active.get('control_dt', [])
                if len(dt) != steps or not 0 < steps <= case['config']['episode_length']:
                    raise ValueError('actual control steps/cap mismatch')
                write(directory / 'control_timing.json', {'count': len(dt), 'min_dt': min(dt), 'max_dt': max(dt), 'hz': 20})
                return result
            return call
        # Frozen runner verifies every resumed video's hash. Also verify all
        # light/native diagnostic artifacts, not only the rendered MP4.
        def hashes(directory):
            return {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*')
                    if p.is_file() and p.name != 'artifact_hashes.json' and p.suffix in ('.json', '.jsonl', '.npz')}
        for f in (Path(root) / 'trials').glob('*.json'):
            r = read(f)
            if r['status'] == 'completed':
                directory = Path(root) / 'attempts' / r['case']['id'] / ('attempt_%03d' % r['attempt'])
                if read(directory / 'artifact_hashes.json') != hashes(directory):
                    raise ValueError('completed diagnostic artifact changed')
        def seal(cb):
            def call(case, directory):
                result = wrap(cb)(case, directory)
                write(directory / 'artifact_hashes.json', hashes(directory))
                return result
            return call
        # STOP file is shared with the supervisor. Real deadline is never reset
        # by worker restart or resuming an incomplete plan.
        remaining = max(.001, true_deadline - time.time())
        try:
            return old_run(root, plan, {m: seal(cb) for m, cb in callbacks.items()},
                           **{**kw, 'max_seconds': remaining})
        finally:
            summarize(root)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(expanded, 'registry', lambda: registry))
            stack.enter_context(patch.object(expanded, 'cases', lambda *_: [(c, s, m) for c, s, m, _ in wanted]))
            stack.enter_context(patch.object(batch, 'make_plan', make_plan))
            stack.enter_context(patch.object(batch, 'run_plan', run_plan))
            stack.enter_context(patch.object(sim, 'make_vector', vector))
            expanded.run(args, phase, holder, conditions)
    except RuntimeError:
        status = read(args.output / phase / 'execution.json')
        if status['status'] != 'paused':
            raise
        write(args.output / (phase + '_worker.json'), {'status': 'paused', 'execution': status})
        return
    if bridge:
        from corrector_loss_queue import verify_cosine_reference
        checks = verify_cosine_reference(args.output / phase, args.reference)
        write(args.output / 'bridge_equivalence.json', {'status': 'passed', 'checks': checks})
    write(args.output / (phase + '_worker.json'), {'status': 'completed'})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['probe', 'bridge', 'transfer'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--frozen-scripts', type=Path, default=BASE / 'dev/expanded-2e6296eff15f/scripts')
    p.add_argument('--runtime', type=Path, default=BASE / 'runtime')
    p.add_argument('--core', type=Path, default=BASE / 'dev/j36m-35ecdd7488bf')
    p.add_argument('--training', type=Path, default=BASE / 'j36m_longrun_20261113_001/training')
    p.add_argument('--calibration', type=Path, default=Path('/root/corrector_j36m_object_20261113_002/j_calibration.json'))
    p.add_argument('--reference', type=Path, default=Path('/root/corrector_gpu_followup_20261116_001/motion_representatives/ordinary'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_setup(args.runtime)
    if args.mode == 'probe':
        sys.path[:0] = [str(args.core / 'scripts'), str(args.core)]
        _, conditions = schedule(read(args.inventory))
        asset_probe(args, conditions)
    else:
        run(args, bridge=args.mode == 'bridge')


if __name__ == '__main__':
    main()

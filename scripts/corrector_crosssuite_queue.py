"""One bounded, resumable GPU process; no training, downloads or deletion.

The four-hour deadline is persisted on first run and is not renewed on resume.
Signals and STOP request an episode-boundary pause, never terminate the worker.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from corrector_crosssuite_plan import read, write, digest, summarize


def resources():
    result = {'system_free_gib': shutil.disk_usage('/root').free / 2**30,
              'data_free_gib': shutil.disk_usage('/root/autodl-tmp').free / 2**30}
    group = Path('/sys/fs/cgroup')
    if (group / 'memory.max').exists():
        limit = (group / 'memory.max').read_text().strip()
        used = int((group / 'memory.current').read_text())
        result['available_cgroup_gib'] = (int(limit) - used) / 2**30 if limit != 'max' else None
    result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], text=True).strip()
    return result


def preflight(plan):
    for path, expected in plan['sources'].items():
        if digest(path) != expected:
            raise ValueError('frozen source/input changed: ' + path)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('GPU occupied; no parallel experiment launched')
    state = resources()
    if min(state['system_free_gib'], state['data_free_gib']) < 5:
        raise RuntimeError('dual-disk 5GiB reserve not available')
    if state.get('available_cgroup_gib') is not None and state['available_cgroup_gib'] < 6:
        raise RuntimeError('available container RAM below startup reserve')
    return state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['run', 'status', 'stop'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='explicitly remove only this queue STOP; keep original deadline')
    a = p.parse_args()
    root = a.root.resolve()
    if a.mode == 'stop':
        (root / 'STOP').touch()
        print('Requested pause at a complete episode boundary; no process killed.')
        return
    if a.mode == 'status':
        print((root / 'status.json').read_text())
        return
    plan = read(root / 'launch_plan.json')
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if plan['max_hours'] != 4:
        raise ValueError('authorized budget is four hours')
    budget_file = root / 'budget.json'
    if not budget_file.exists():
        start = time.time()
        write(budget_file, {'started_unix': start, 'deadline_unix': start + 4 * 3600,
                           'max_seconds': 4 * 3600, 'boundary_overrun': 'at most current episode; no forced kill'})
    budget = read(budget_file)
    stop = root / 'STOP'
    if a.resume:
        if time.time() >= budget['deadline_unix']:
            raise RuntimeError('original budget expired; resume does not authorize additional GPU time')
        stop.unlink(missing_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'child_pid': None, 'deadline_unix': budget['deadline_unix'],
             'phase': 'preflight', 'completed_stages': [], 'resources': preflight(plan)}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.touch())
    child = None
    try:
        for stage in ('probe', 'bridge', 'transfer'):
            if stop.exists() or time.time() >= budget['deadline_unix']:
                state.update(status='paused', reason='operator_stop_or_original_deadline')
                break
            marker = root / ('environment_probe.json' if stage == 'probe' else
                             'bridge_equivalence.json' if stage == 'bridge' else 'transfer_worker.json')
            if marker.exists() and read(marker)['status'] in ('passed', 'completed'):
                state['completed_stages'].append(stage)
                continue
            state['resources'] = preflight(plan)
            command = [plan['python'], '-u', str(Path(plan['code']) / 'corrector_crosssuite_eval.py'), stage,
                       '--output', str(root), '--inventory', plan['inventory'], '--training-audit', plan['training_audit'],
                       '--deadline', str(budget['deadline_unix'])]
            log_path = root / (stage + '.log')
            state.update(phase=stage, log=str(log_path), command=command)
            env = {**os.environ, 'PYTHONPATH': plan['runtime'] + '/lerobot/src', 'PYTHONUNBUFFERED': '1'}
            with log_path.open('ab') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, cwd=plan['code'], start_new_session=True)
            state['child_pid'] = child.pid
            write(root / 'status.json', state)
            last_summary = 0
            while child.poll() is None:
                state['resources'] = resources()
                r = state['resources']
                if (time.time() >= budget['deadline_unix'] or min(r['system_free_gib'], r['data_free_gib']) < 5.15
                        or r.get('available_cgroup_gib') is not None and r['available_cgroup_gib'] < 5):
                    stop.touch()
                execution = root / stage / 'execution.json'
                if execution.exists():
                    state['progress'] = read(execution)
                    if time.time() - last_summary > 60:
                        summarize(root / stage)
                        last_summary = time.time()
                write(root / 'status.json', state)
                time.sleep(3)
            state.update(child_pid=None, returncode=child.returncode)
            if child.returncode:
                raise RuntimeError('worker error; inspect ' + str(log_path) + '; no infinite retry')
            if marker.exists() and read(marker)['status'] in ('passed', 'completed'):
                state['completed_stages'].append(stage)
            elif stop.exists() or time.time() >= budget['deadline_unix']:
                state.update(status='paused', reason='STOP, resource reserve or time budget')
                break
            else:
                raise RuntimeError('stage returned without verified completion marker: ' + stage)
            state.pop('progress', None)
        else:
            state.update(status='completed', phase='completed')
    except BaseException as exc:
        state.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
        if child is not None and child.poll() is None:
            stop.touch()
        raise
    finally:
        for stage in ('bridge', 'transfer'):
            if (root / stage / 'trials').is_dir():
                summarize(root / stage)
        state['updated_unix'] = time.time()
        write(root / 'status.json', state)


if __name__ == '__main__':
    main()

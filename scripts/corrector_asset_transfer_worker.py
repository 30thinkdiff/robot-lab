"""Copy an existing coherent runtime into an empty second-machine project root.

Read-only compressed rsync, inventory SHA validation, then environment/CPU tests.
Never installs packages, extracts dataset features, trains or runs simulation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate_inventory(inv):
    base = '/root/autodl-tmp/corrector'
    roots = {'bootstrap', 'runtime/venv', 'runtime/torch-runtime-2.11-cu128',
             'runtime/lerobot', 'runtime/vla-corrector-reference', 'runtime/hf/hub',
             'night-code-1e80242789', 'dev/j36m-35ecdd7488bf', 'object10/features/episode_813'}
    if inv.get('source_root') != base or inv.get('publish_root') != base or set(inv.get('selected_roots', [])) != roots:
        raise ValueError('Asset roots differ from the explicit minimal preparation recipe')
    files = inv['files']; names = {r['path'] for r in files}
    if not files or len(names) != len(files): raise ValueError('Empty/duplicate inventory')
    dirs = {str(parent) for name in names for parent in PurePosixPath(name).parents}
    for row in files:
        rel = PurePosixPath(row['path'])
        if (rel.is_absolute() or '..' in rel.parts or '\n' in row['path'] or row['kind'] not in ('file', 'symlink')
            or not any(row['path'] == root or row['path'].startswith(root + '/') for root in roots)):
            raise ValueError('Unsafe or unselected asset path')
        if row['kind'] == 'file':
            if row['bytes'] < 0 or len(row['sha256']) != 64 or row['mode'] & 0o6000:
                raise ValueError('Invalid file metadata or privileged executable')
        else:
            target = posixpath.normpath(posixpath.join(base, posixpath.dirname(row['path']), row['target']))
            if not target.startswith(base + '/') or target[len(base) + 1:] not in names | dirs:
                raise ValueError('External or missing symlink dependency')
    total = sum(r['bytes'] for r in files)
    if total != inv['total_bytes'] or total > 16 * 2**30:
        raise ValueError('Wrong byte total or asset budget exceeded')
    return files


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    private = root / '.transfer'
    with (private / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inv_file = private / 'asset_inventory.json'
        inv = json.loads(inv_file.read_text())
        publish = Path(inv['publish_root'])
        payload = root / 'payload'
        if publish != Path('/root/autodl-tmp/corrector') or publish.exists():
            raise FileExistsError('Only a new, empty destination project root is allowed')
        if (root / 'READY.json').exists():
            raise FileExistsError('Already verified')
        files = validate_inventory(inv)
        payload.mkdir(exist_ok=True)
        total = sum(r['bytes'] for r in files)
        children, logs = {}, []
        state = dict(status='running', phase='preflight', pid=os.getpid(),
            started_utc=datetime.now(timezone.utc).isoformat(), expected_bytes=total,
            files=len(files), source_read_only=True, feature_extraction_started=False,
            model_training_started=False, simulation_started=False)

        def update(**values):
            state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
            tmp = private / 'status.tmp'
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
            tmp.replace(root / 'status.json')

        def received():
            n = 0
            for row in files:
                if row['kind'] == 'file':
                    f = payload / row['path']
                    if f.is_file() and not f.is_symlink():
                        n += min(row['bytes'], f.stat().st_size)
            return n

        try:
            update()
            if shutil.disk_usage(root).free < total - received() + 5 * 2**30:
                raise RuntimeError('Insufficient space including 5 GiB reserve')
            parts, sizes = [[] for _ in range(4)], [0] * 4
            for row in sorted(files, key=lambda r: -r['bytes']):
                i = sizes.index(min(sizes)); parts[i].append(row); sizes[i] += row['bytes']
            ssh = ['ssh', '-p', str(a.port), '-i', str(private / 'id_ed25519'),
                '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'UserKnownHostsFile=' + str(private / 'known_hosts'),
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4']
            commands, attempts, finished = {}, {}, set()
            for i, rows in enumerate(parts):
                listing = private / ('files_%d.txt' % i)
                listing.write_text(''.join(r['path'] + '\n' for r in rows))
                commands[i] = ['rsync', '-rltpz', '--partial', '--append-verify', '--stats',
                    '--timeout=180', '--files-from=' + str(listing), '-e', shlex.join(ssh),
                    a.host + ':./', str(payload) + '/']

            def launch(i):
                attempts[i] = attempts.get(i, 0) + 1
                log = (private / ('rsync_%d.log' % i)).open('a'); logs.append(log)
                children[i] = subprocess.Popen(commands[i], stdout=log, stderr=subprocess.STDOUT)

            initial, start = received(), time.monotonic()
            for i in commands: launch(i)
            while len(finished) < 4:
                for i, child in list(children.items()):
                    if i in finished or child.poll() is None: continue
                    if child.returncode == 0: finished.add(i)
                    elif child.returncode in (10, 12, 30, 35, 255) and attempts[i] < 3:
                        time.sleep(15); launch(i)
                    else: raise RuntimeError('rsync part%d failed: %d' % (i, child.returncode))
                update(phase='compressed_transfer', child_pids=[c.pid for i, c in children.items() if i not in finished and c.poll() is None],
                    received_bytes=received(), preexisting_bytes=initial, transfer_seconds=time.monotonic() - start,
                    completed_partitions=len(finished), attempts=attempts,
                    rate_note='Uncompressed files materialized / wall time; not compressed network bytes')
                if len(finished) < 4: time.sleep(15)
            update(phase='verifying_files', verified_files=0)
            for i, row in enumerate(files, 1):
                f = payload / row['path']
                if row['kind'] == 'symlink':
                    if not f.is_symlink() or os.readlink(f) != row['target']:
                        raise ValueError('Symlink mismatch: ' + row['path'])
                elif f.is_symlink() or not f.is_file() or f.stat().st_size != row['bytes'] or digest(f) != row['sha256']:
                    raise ValueError('File SHA/size mismatch: ' + row['path'])
                if i % 250 == 0: update(verified_files=i)
            if publish.exists(): raise FileExistsError('Publish root appeared; not replacing')
            payload.rename(publish)  # own new tree only, same filesystem
            update(phase='environment_validation', verified_files=len(files), published_root=str(publish))
            python = publish / 'runtime/venv/bin/python'
            env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            probe = subprocess.run([str(python), '-c',
                'import json,sys,torch,numpy,transformers; print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,numpy=numpy.__version__,transformers=transformers.__version__,cuda_available=torch.cuda.is_available(),gpu_model=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))'],
                env=env, text=True, capture_output=True, timeout=180)
            (private / 'environment_probe.log').write_text(probe.stdout + probe.stderr)
            if probe.returncode: raise RuntimeError('Existing runtime incompatible; see environment_probe.log')
            versions = json.loads(probe.stdout.strip().splitlines()[-1])
            if not (versions['python'].startswith('3.12.') and versions['torch'] == '2.11.0+cu128' and versions['numpy'] == '2.1.3' and versions['transformers'] == '5.5.4' and versions['cuda_available']):
                raise ValueError('Runtime version/GPU availability differs from baseline: ' + str(versions))
            bundle = private / 'foundation.zip'
            expected_sha = (private / 'foundation.sha256').read_text().strip()
            if digest(bundle) != expected_sha: raise ValueError('Foundation ZIP SHA mismatch')
            code = publish / 'dev' / ('slc-foundation-' + expected_sha[:12])
            code.mkdir()
            with zipfile.ZipFile(bundle) as z:
                for name in z.namelist():
                    rel = PurePosixPath(name)
                    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe ZIP')
                z.extractall(code)
            env.update(PYTHONPATH=str(code / 'tools'),
                ROBOTLAB_SLC_BASELINE_DIR=str(publish / 'night-code-1e80242789'),
                ROBOTLAB_SLC_SOURCE_ROOT=str(publish / 'runtime/vla-corrector-reference'),
                ROBOTLAB_SLC_EXPORTER=str(publish / 'dev/j36m-35ecdd7488bf/scripts/corrector_multilayer_pilot.py'))
            update(phase='destination_cpu_tests', versions=versions)
            tests = subprocess.run([str(python), '-m', 'unittest', 'robotlab_single_layer.test_core', '-v'],
                cwd=code, env=env, text=True, capture_output=True, timeout=600)
            (private / 'destination_cpu_tests.log').write_text(tests.stdout + tests.stderr)
            if tests.returncode: raise RuntimeError('Destination CPU tests failed; no extraction/training started')
            ready = dict(status='completed', files=len(files), bytes=total, all_inventory_verified=True,
                inventory_sha256=digest(inv_file), versions=versions, code_root=str(code),
                foundation_sha256=expected_sha, destination_cpu_tests='passed',
                real_corrector_gpu_smoke_completed=False, feature_extraction_started=False,
                formal_training_started=False, simulation_started=False,
                completed_utc=datetime.now(timezone.utc).isoformat())
            (root / 'READY.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2) + '\n')
            update(**ready, phase='ready_for_real_gpu_probe')
            # The source's bounded public-key entry is removed by the next
            # authenticated coordinator check; stop retaining its private key now.
            (private / 'id_ed25519').unlink(missing_ok=True)
        except BaseException as error:
            for child in children.values():
                if child.poll() is None: child.send_signal(signal.SIGINT)
            for child in children.values(): child.wait()
            update(status='error', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            for log in logs: log.close()


if __name__ == '__main__': main()

"""Resumable, CPU-only SSH/rsync pull of an explicit raw-data inventory.

Run on the destination machine with its own short-lived, read-only source key.
READY is published only after every destination file passes size and SHA256.
No source deletion, training, dependency installation or GPU imports.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--host', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--parallel', type=int, choices=range(1, 5), default=4)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    private = root / '.transfer'
    with (private / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'READY.json').exists():
            raise FileExistsError('Already verified: refusing to run transfer again')
        inventory_file = private / 'source_inventory.json'
        inventory = json.loads(inventory_file.read_text())
        files = inventory['files']
        if not files or len({r['path'] for r in files}) != len(files):
            raise ValueError('Empty or duplicate inventory')
        destination = root / 'source'
        destination.mkdir(exist_ok=True)
        for row in files:
            rel = PurePosixPath(row['path'])
            if rel.is_absolute() or '..' in rel.parts or '\n' in row['path'] or not row['path'].endswith('.parquet'):
                raise ValueError('Unsafe raw-data path')
            path = destination / row['path']
            if any(x.is_symlink() for x in [path, *path.parents] if x != Path('/')):
                raise ValueError('Destination symlink is not permitted')
        total = sum(r['bytes'] for r in files)
        state = {'status': 'running', 'phase': 'preflight', 'started_utc': now(),
                 'pid': os.getpid(), 'files': len(files), 'expected_bytes': total,
                 'source_read_only': True, 'gpu_used': False}

        def update(**values):
            state.update(values, updated_utc=now())
            tmp = private / 'status.tmp'
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
            tmp.replace(root / 'status.json')

        def received():
            return sum(min((destination / r['path']).stat().st_size, r['bytes'])
                       for r in files if (destination / r['path']).exists())

        children = {}
        logs = []
        try:
            if shutil.disk_usage(root).free < total - received() + 5 * 2**30:
                raise RuntimeError('Insufficient destination space including 5 GiB reserve')
            ssh = ['ssh', '-p', str(a.port), '-i', str(private / 'id_ed25519'),
                   '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                   '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'UserKnownHostsFile=' + str(private / 'known_hosts'),
                   '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
                   '-o', 'ServerAliveCountMax=4']
            # Disjoint partitions avoid simultaneous writes to the same file.
            parts = [[] for _ in range(min(a.parallel, len(files)))]
            sizes = [0] * len(parts)
            for row in sorted(files, key=lambda r: -r['bytes']):
                i = sizes.index(min(sizes))
                parts[i].append(row)
                sizes[i] += row['bytes']
            commands, attempts, finished = {}, {}, set()
            for i, rows in enumerate(parts):
                filelist = private / ('files_%d.txt' % i)
                filelist.write_text(''.join(r['path'] + '\n' for r in rows))
                commands[i] = ['rsync', '-rt', '--partial', '--append-verify', '--stats',
                    '--timeout=180', '--files-from=' + str(filelist),
                    '-e', shlex.join(ssh), a.host + ':./', str(destination) + '/']

            def launch(i):
                attempts[i] = attempts.get(i, 0) + 1
                log = (private / ('rsync_%d.log' % i)).open('a')
                logs.append(log)
                log.write('\n' + now() + ' attempt ' + str(attempts[i]) + '\n'); log.flush()
                children[i] = subprocess.Popen(commands[i], stdout=log, stderr=subprocess.STDOUT)

            initial_received = received()
            transfer_start = time.monotonic()
            for i in commands:
                launch(i)
            while len(finished) < len(commands):
                for i, child in list(children.items()):
                    if i in finished or child.poll() is None:
                        continue
                    if child.returncode == 0:
                        finished.add(i)
                    elif child.returncode in (10, 12, 30, 35, 255) and attempts[i] < 3:
                        time.sleep(15)
                        launch(i)
                    else:
                        raise RuntimeError('rsync partition %d failed; code=%d; see .transfer/rsync_%d.log' % (i, child.returncode, i))
                update(phase='transferring', parallel=len(commands), attempts=attempts,
                       child_pids=[c.pid for i, c in children.items() if i not in finished and c.poll() is None],
                       completed_partitions=len(finished), received_bytes=received(),
                       preexisting_bytes=initial_received,
                       transfer_seconds=time.monotonic() - transfer_start)
                if len(finished) < len(commands):
                    time.sleep(10)
            transfer_seconds = time.monotonic() - transfer_start
            update(phase='verifying_sha256', received_bytes=received(), verified_files=0,
                   transfer_seconds=transfer_seconds)
            for i, row in enumerate(files, 1):
                path = destination / row['path']
                if not path.is_file() or path.stat().st_size != row['bytes']:
                    raise ValueError('Size mismatch: ' + row['path'])
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 2**20), b''):
                        digest.update(block)
                if digest.hexdigest() != row['sha256']:
                    raise ValueError('SHA256 mismatch: ' + row['path'])
                update(verified_files=i)
            metadata = root / 'metadata'
            metadata.mkdir(exist_ok=True)
            manifest = metadata / 'object10_manifest.json'
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != inventory['object10_manifest_sha256']:
                raise ValueError('Episode manifest SHA256 mismatch')
            receipt = {'status': 'completed', 'completed_utc': now(), 'files': len(files),
                       'bytes': total, 'all_sha256_verified': True,
                       'inventory_sha256': hashlib.sha256(inventory_file.read_bytes()).hexdigest(),
                       'object10_manifest_sha256': inventory['object10_manifest_sha256'],
                       'transfer_seconds': transfer_seconds,
                       'preexisting_bytes': initial_received,
                       'mean_new_file_bytes_mib_s': (total - initial_received) / 2**20 / transfer_seconds,
                       'rate_note': 'New completed file bytes / this transfer wall time; not a network byte counter.',
                       'source_read_only': True, 'gpu_used': False}
            with (root / 'READY.json').open('x') as f:
                json.dump(receipt, f, ensure_ascii=False, indent=2); f.write('\n')
            update(status='completed', phase='verified', **{k: v for k, v in receipt.items() if k != 'status'})
        except BaseException as error:
            for child in children.values():
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
            for child in children.values():
                child.wait()
            update(status='error', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            for log in logs:
                log.close()


if __name__ == '__main__':
    main()

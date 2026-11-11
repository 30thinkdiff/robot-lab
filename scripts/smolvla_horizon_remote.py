"""Pilot专用一次SSH部署、状态取回与结果取回；不改冻结源码或旧输出。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

BASE = Path('/root/autodl-tmp/corrector')
OUTPUT = BASE/'object10/horizon_pilot_20261111_001'
FILES = ['smolvla_horizon_pilot.py', 'smolvla_baseline_benchmark.py',
         'smolvla_libero_tools.py', 'smolvla_training_workbench.py']


def build_payload():
    here = Path(__file__).resolve().parent
    files = {name: (here/name).read_bytes() for name in FILES}
    files['batch_evaluation.py'] = (here.parent/'studies/policy_generalization/corrector/batch_evaluation.py').read_bytes()
    digest = hashlib.sha256(b''.join(name.encode()+files[name] for name in sorted(files))).hexdigest()[:12]
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
        archive.writestr('pilot_manifest.json', json.dumps({name: hashlib.sha256(raw).hexdigest()
            for name, raw in files.items()}, indent=2)+'\n')
    return output.getvalue(), digest


def apply(args):
    # 只允许本Pilot固定根目录；不把用户给出的路径当作任意远程执行/覆盖目标。
    import importlib.util
    raw = sys.stdin.buffer.read(2**20)
    if hashlib.sha256(raw).hexdigest() != args.sha:
        raise RuntimeError('上传校验失败，不启动')
    target = BASE/'dev'/('horizon-pilot-'+args.code)
    if target.exists() or OUTPUT.exists():
        raise RuntimeError('源码目录或输出目录已存在；拒绝覆盖或重复启动')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    allowed = set(FILES+['batch_evaluation.py', 'pilot_manifest.json'])
    if set(archive.namelist()) != allowed or len(archive.namelist()) != len(allowed):
        raise RuntimeError('上传文件集合异常')
    target.mkdir(parents=True, exist_ok=False)
    for name in archive.namelist():
        (target/name).write_bytes(archive.read(name))
    spec = importlib.util.spec_from_file_location('pilot', target/'smolvla_horizon_pilot.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(target))
    spec.loader.exec_module(module)
    # 检查失败仅留下新源码，不创建实验输出、不启动模型。
    audit = module.preflight(BASE)
    command = [str(BASE/'runtime/sim-venv/bin/python'), '-u', str(target/'smolvla_horizon_pilot.py'), 'run']
    logfile = BASE/'horizon-pilot-20261111-001.log'
    if logfile.exists():
        raise RuntimeError('日志文件已存在；拒绝覆盖历史日志')
    env = os.environ.copy()
    env.update(HF_HOME=str(BASE/'runtime/hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', MPLBACKEND='Agg')
    env.pop('PYTHONPATH', None)
    with logfile.open('x') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    receipt = {'pid': process.pid, 'code': str(target), 'output': str(OUTPUT), 'log': str(logfile),
               'command': command, 'pilot_only': True, 'preflight': audit}
    (target/'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def remote_status():
    root = OUTPUT
    result = {'output_exists': root.exists()}
    for name in ['execution.json', 'analysis/pilot_summary.json']:
        if (root/name).is_file():
            result[name] = json.loads((root/name).read_text())
    log = BASE/'horizon-pilot-20261111-001.log'
    if log.is_file():
        result['log_tail'] = log.read_text(errors='replace').splitlines()[-20:]
    result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                           '--format=csv,noheader,nounits'], text=True).strip()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def remote_export():
    if not (OUTPUT/'execution.json').is_file():
        raise RuntimeError('尚无Pilot执行记录')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.csv'}:
                if path.stat().st_size > 2**20:
                    raise RuntimeError('结果文件过大，停止导出')
                archive.write(path, path.relative_to(OUTPUT))
    sys.stdout.buffer.write(buffer.getvalue())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['install', 'status', 'fetch', 'apply', 'remote-status', 'remote-export'])
    p.add_argument('--ssh-host', default='connect.nmb1.seetacloud.com')
    p.add_argument('--sha'); p.add_argument('--code')
    p.add_argument('--output', type=Path, default=Path('outputs/smolvla_horizon_pilot_20261111/results.zip'))
    args = p.parse_args()
    if args.mode == 'apply':
        apply(args); return
    if args.mode == 'remote-status':
        remote_status(); return
    if args.mode == 'remote-export':
        remote_export(); return
    payload = b''
    mode = {'install': 'apply', 'status': 'remote-status', 'fetch': 'remote-export'}[args.mode]
    remote = [str(BASE/'bootstrap/bin/python'), '-c', Path(__file__).read_text(), mode]
    if args.mode == 'install':
        payload, digest = build_payload()
        remote += ['--sha', hashlib.sha256(payload).hexdigest(), '--code', digest]
    command = ['ssh', '-T', '-o', 'ConnectTimeout=15', args.ssh_host, shlex.join(remote)]
    if args.mode == 'fetch':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise RuntimeError('本地结果包已存在，不覆盖；请另设--output')
        with args.output.open('xb') as file:
            result = subprocess.run(command, input=payload, stdout=file)
        if result.returncode:
            raise RuntimeError('取回失败，保留文件用于检查')
        with zipfile.ZipFile(args.output) as archive:
            if archive.testzip():
                raise RuntimeError('ZIP校验失败')
        print(json.dumps({'bundle': str(args.output.resolve()), 'bytes': args.output.stat().st_size}))
    else:
        result = subprocess.run(command, input=payload)
        if result.returncode:
            raise RuntimeError('远程检查/部署失败；没有自动重试或扩大实验')


if __name__ == '__main__':
    main()

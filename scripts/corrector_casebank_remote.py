"""交互SSH：检查、部署、状态、暂停、同计划续跑与取回Replay完整样本库。"""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

BASE=Path('/root/autodl-tmp/corrector')
ROOT=Path('/root/robotlab_replay_casebank_20261112_003')
BRANCH='experiment/replay-casebank-20261112-v3'
HERE=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
FILES=['corrector_casebank_plan.py','corrector_casebank_eval.py','corrector_casebank_capture.py',
       'corrector_online_runtime.py','corrector_hybrid.py','corrector_models.py','corrector_train.py',
       'corrector_data_workbench.py','corrector_executed_monitor.py','corrector_consistency_control.py',
       'corrector_overnight_eval.py',
       'smolvla_horizon_pilot.py','smolvla_baseline_benchmark.py','smolvla_libero_tools.py','smolvla_training_workbench.py']


def read(path):return json.loads(Path(path).read_text())
def alive(pid):return bool(pid) and Path(f'/proc/{pid}').exists()


def inspect_remote():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    old=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(p.parent.name)==os.getpid():continue
            argv=p.read_bytes().decode(errors='replace').split('\0')
            names=[Path(a).name for a in argv if a.endswith('.py')]
            if any(n.startswith(('corrector_','smolvla_','lerobot_')) for n in names):old.append({'pid':int(p.parent.name),'scripts':names})
        except (OSError,ValueError):pass
    return {'utc':datetime.now(timezone.utc).isoformat(),'gpu':gpu,'compute_processes':processes,'experiment_processes':old,
        'system_free_gib':shutil.disk_usage('/root').free/2**30,'data_free_gib':shutil.disk_usage(BASE).free/2**30,
        'night_status':read('/root/corrector_overnight_20261111_v2/results/execution.json') if Path('/root/corrector_overnight_20261111_v2/results/execution.json').exists() else None,
        'casebank_exists':ROOT.exists()}


def payload():
    files={n:(HERE/n).read_bytes() for n in FILES}
    practice=HERE.parent/'studies/policy_generalization/corrector'
    for name,path in [('batch_evaluation.py',practice/'batch_evaluation.py'),('corrector_online_practice.py',practice/'online_practice.py'),
       ('selection.json',practice/'manifests/replay_casebank_20261112.json'),('models.json',practice/'manifests/overnight_models_20261111.json'),
       ('calibration_rule.json',practice/'manifests/closed_loop_calibration_20261111.json')]:files[name]=path.read_bytes()
    code=hashlib.sha256(b''.join(n.encode()+files[n] for n in sorted(files))).hexdigest()[:12];buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for n,raw in files.items():z.writestr(n,raw)
    return buf.getvalue(),code


def remote(args):
    if args.mode=='remote-inspect':print(json.dumps(inspect_remote(),ensure_ascii=False,indent=2));return
    if args.mode=='remote-status':
        result=inspect_remote()
        for name in ['launch.json','results/pipeline_status.json','results/execution.json','results/analysis/casebank_summary.json']:
            if (ROOT/name).is_file():result[name]=read(ROOT/name)
        if (ROOT/'launch.log').exists():
            with (ROOT/'launch.log').open('rb') as f:f.seek(max(0,(ROOT/'launch.log').stat().st_size-9000));result['log_tail']=f.read().decode(errors='replace').splitlines()[-25:]
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.mode=='remote-pause':
        if not (ROOT/'results').is_dir():raise RuntimeError('没有已启动的样本库队列')
        (ROOT/'results/STOP').touch();print('已请求回合边界暂停；不杀进程、不删除数据。');return
    if args.mode in ['remote-install','remote-resume']:
        audit=inspect_remote()
        fields=[x.strip() for x in audit['gpu'].split(',')]
        if audit['compute_processes'] or audit['experiment_processes'] or len(fields)!=4 or '4090' not in fields[0] or float(fields[1])>128 or float(fields[3])>5:
            raise RuntimeError('GPU或旧实验进程冲突，停止：'+json.dumps(audit,ensure_ascii=False))
        if audit['system_free_gib']<8 or audit['data_free_gib']<5:raise RuntimeError('空间不足，不启动或清理旧数据')
        if subprocess.check_output(['git','-C',str(BASE),'diff','HEAD','--name-only'],text=True).strip():raise RuntimeError('远程跟踪源码有修改，不切分支')
        if args.mode=='remote-install':
            if ROOT.exists():raise RuntimeError('新输出目录已存在，不覆盖；请status或resume')
            raw=sys.stdin.buffer.read(2**20)
            if hashlib.sha256(raw).hexdigest()!=args.sha:raise RuntimeError('部署包SHA不符')
            code=BASE/'dev'/('replay-casebank-'+args.code)
            if code.exists():raise RuntimeError('代码目录冲突，不覆盖')
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                expected=set(FILES+['batch_evaluation.py','corrector_online_practice.py','selection.json','models.json','calibration_rule.json'])
                if set(z.namelist())!=expected or len(z.namelist())!=len(expected):raise RuntimeError('上传文件集合不符')
                code.mkdir(parents=True)
                for n in z.namelist():(code/n).write_bytes(z.read(n))
            subprocess.run(['git','-C',str(BASE),'switch','-c',BRANCH],check=True)
            ROOT.mkdir();(ROOT/'results').mkdir()
        else:
            previous=read(ROOT/'launch.json');code=Path(previous['code'])
            if alive(previous.get('pid')):raise RuntimeError('原进程仍在，不重复启动')
            if subprocess.check_output(['git','-C',str(BASE),'branch','--show-current'],text=True).strip()!=BRANCH:raise RuntimeError('不在原分支，不自动切换')
            if (ROOT/'results/STOP').exists():raise RuntimeError('存在操作员STOP；先核对后在远程显式移除再resume')
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(HF_HOME=str(BASE/'runtime/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        cmd=[str(BASE/'runtime/sim-venv/bin/python'),'-u',str(code/'corrector_casebank_eval.py'),'--output',str(ROOT/'results'),'--hours',str(args.hours)]
        if args.mode=='remote-resume':cmd.append('--resume')
        with (ROOT/'launch.log').open('a') as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        receipt={'pid':child.pid,'code':str(code),'output':str(ROOT/'results'),'log':str(ROOT/'launch.log'),'branch':BRANCH,'preflight':audit,'no_training':True}
        (ROOT/'launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(receipt,ensure_ascii=False,indent=2),file=sys.stderr,flush=True)
        if not args.wait_and_fetch:return
    if args.mode in ['remote-fetch','remote-install','remote-resume']:
        if args.wait_and_fetch:
            until=time.monotonic()+args.wait_minutes*60;last=-1
            while time.monotonic()<until:
                state=read(ROOT/'results/pipeline_status.json') if (ROOT/'results/pipeline_status.json').exists() else {}
                execution=read(ROOT/'results/execution.json') if (ROOT/'results/execution.json').exists() else {}
                if execution.get('completed',0)!=last:
                    print(json.dumps({'phase':state.get('phase'),'completed':execution.get('completed',0),'planned':execution.get('planned'),'status':state.get('status')},ensure_ascii=False),file=sys.stderr,flush=True);last=execution.get('completed',0)
                if state.get('status') in ['completed','paused','error']:break
                if not alive(read(ROOT/'launch.json').get('pid')):raise RuntimeError('后台进程退出，先status检查，不盲目重启')
                time.sleep(15)
            else:raise RuntimeError('等待预算到达；后台任务保持运行，稍后fetch')
        state=read(ROOT/'results/pipeline_status.json')
        if not args.snapshot and state.get('status') not in ['completed','paused']:raise RuntimeError('尚未完成或存在程序错误；只读已完成回合可显式使用fetch --snapshot')
        root=ROOT/'results';records=[]
        for p in sorted((root/'trials').glob('*.json')):
            raw=p.read_bytes();record=json.loads(raw)
            if record.get('status')=='completed':records.append((p,raw,record))
        if not records:raise RuntimeError('尚无正式完成回合，技术探针不冒充样本库')
        execution=read(root/'execution.json')
        # zipfile支持不可seek的stdout；仅导出本次新目录内文件，不复制权重/旧数据。
        with zipfile.ZipFile(sys.stdout.buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
            def put(p):
                if not p.is_file() or p.is_symlink() or p.suffix not in {'.json','.jsonl','.csv','.md','.mp4','.npz','.py'}:return
                p.resolve().relative_to(root.resolve())
                if p.stat().st_size>40*2**20:raise RuntimeError('单文件超过预算，停止取回')
                z.write(p,'results/'+p.relative_to(root).as_posix())
            for p in root.glob('*.json'):
                if p.name not in {'execution.json','pipeline_status.json'}:put(p)
            for p in (root/'recording').glob('*'):put(p)
            for p,raw,record in records:
                z.writestr('results/'+p.relative_to(root).as_posix(),raw)
                directory=root/'attempts'/record['case']['id']/f"attempt_{record['attempt']:03d}"
                directory.resolve().relative_to(root.resolve())
                for child in sorted(directory.rglob('*')):put(child)
            if not args.snapshot:
                for p in (root/'analysis').glob('*'):put(p)
            z.writestr('results/source_execution.json',json.dumps(execution,ensure_ascii=False))
            z.writestr('results/execution.json',json.dumps({**execution,'status':'snapshot' if args.snapshot else execution['status'],
                'source_status':execution['status'],'completed':len(records),'snapshot_completed_only':True},ensure_ascii=False))
            z.writestr('results/pipeline_status.json',json.dumps({**state,'status':'snapshot' if args.snapshot else state['status'],
                'source_status':state['status'],'completed':len(records)},ensure_ascii=False))
        return
    raise ValueError(args.mode)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inspect','install','status','pause','resume','fetch','remote-inspect','remote-install','remote-status','remote-pause','remote-resume','remote-fetch'])
    p.add_argument('--ssh-host',default=os.environ.get('ROBOTLAB_SSH_HOST','autodl-corrector'))
    p.add_argument('--control-path',type=Path,help='复用本机已认证的SSH ControlMaster；不读取或接收密码')
    p.add_argument('--hours',type=float,default=8);p.add_argument('--wait-and-fetch',action='store_true');p.add_argument('--wait-minutes',type=int,default=540)
    p.add_argument('--snapshot',action='store_true',help='只读取运行中已完成回合，不包含正在写入的attempt')
    p.add_argument('--sha');p.add_argument('--code');args=p.parse_args()
    if args.mode.startswith('remote-'):remote(args);return
    cmd=['python3','-c',Path(__file__).read_text(),'remote-'+args.mode,'--hours',str(args.hours),'--wait-minutes',str(args.wait_minutes)];raw=b''
    if args.wait_and_fetch:cmd.append('--wait-and-fetch')
    if args.snapshot:cmd.append('--snapshot')
    if args.mode=='install':raw,code=payload();cmd+=['--sha',hashlib.sha256(raw).hexdigest(),'--code',code]
    ssh=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    if args.control_path:ssh+=['-S',str(args.control_path),'-o','BatchMode=yes']
    ssh += [args.ssh_host,shlex.join(cmd)]
    if args.mode=='fetch' or args.wait_and_fetch:
        folder=HERE.parent/'tools/robotlab_replay/.local/casebank_downloads';folder.mkdir(parents=True,exist_ok=True)
        dest=folder/('casebank_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip');partial=dest.with_suffix('.partial')
        with partial.open('xb') as f:r=subprocess.run(ssh,input=raw,stdout=f)
        if r.returncode:raise RuntimeError('SSH操作/等待失败；后台已启动任务不会因此终止，稍后status/fetch，不重复install')
        partial.replace(dest)
        from corrector_casebank_import import install_bundle
        result=install_bundle(dest,HERE.parent/'tools/robotlab_replay/.local')
        print(json.dumps(result,ensure_ascii=False));return
    if subprocess.run(ssh,input=raw).returncode:raise RuntimeError('远程检查或操作停止，保留现场，不自动重试')


if __name__=='__main__':main()

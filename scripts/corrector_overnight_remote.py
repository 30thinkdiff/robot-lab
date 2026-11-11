"""一次交互SSH部署夜间冻结实验；状态、暂停、恢复、取回不依赖代理隧道。"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
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

BASE=Path('/root/autodl-tmp/corrector');ROOT=Path('/root/corrector_overnight_20261111_v2')
BRANCH='experiment/overnight-generalization-20261111-v2'
FILES=['corrector_overnight_eval.py','corrector_overnight_metrics.py','corrector_overnight_queue.py',
       'analyse_corrector_overnight.py','corrector_consistency_control.py','corrector_consistency_eval.py',
       'corrector_executed_monitor.py','corrector_hybrid.py','corrector_hybrid_pilot.py','corrector_online_runtime.py',
       'corrector_models.py','corrector_train.py','corrector_data_workbench.py','smolvla_horizon_pilot.py',
       'smolvla_baseline_benchmark.py','smolvla_libero_tools.py','smolvla_training_workbench.py']
EXTRA=['batch_evaluation.py','corrector_online_practice.py','calibration_rule.json','overnight_models.json']


def payload():
    here=Path(__file__).resolve().parent;files={n:(here/n).read_bytes() for n in FILES}
    practice=here.parent/'studies/policy_generalization/corrector'
    for target,source in [('batch_evaluation.py','batch_evaluation.py'),('corrector_online_practice.py','online_practice.py'),
                          ('calibration_rule.json','manifests/closed_loop_calibration_20261111.json'),
                          ('overnight_models.json','manifests/overnight_models_20261111.json')]:
        files[target]=(practice/source).read_bytes()
    code=hashlib.sha256(b''.join(n.encode()+files[n] for n in sorted(files))).hexdigest()[:12];buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in files.items():z.writestr(name,value)
    return buf.getvalue(),code


def read(path):return json.loads(Path(path).read_text())


def alive(pid):return bool(pid) and Path(f'/proc/{pid}').exists()


def preflight(code):
    spec=importlib.util.spec_from_file_location('overnight_preflight',code/'smolvla_horizon_pilot.py')
    module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(code));spec.loader.exec_module(module)
    audit=module.preflight(BASE)
    if read(BASE/'object10/consistency_closed_loop_20261111_001/pipeline_status.json')['status']!='completed':
        raise RuntimeError('上一轮48回合未完成，不启动')
    free=shutil.disk_usage('/root').free/2**30
    if free<8:raise RuntimeError(f'系统盘仅{free:.2f}GiB，至少需8GiB才能开始本次有限计划')
    audit['system_free_gib']=free
    if subprocess.check_output(['git','-C',str(BASE),'diff','HEAD','--name-only'],text=True).strip():
        raise RuntimeError('远程已跟踪源码有修改，不切换分支')
    return audit


def launch(code,deadline,audit):
    dt=datetime.fromisoformat(deadline)
    if dt.tzinfo is None or dt<=datetime.now(timezone.utc):raise RuntimeError('截止时间无效或已过')
    if (ROOT/'results/STOP').exists():raise RuntimeError('存在STOP，确认无活动任务后显式移除再resume')
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(HF_HOME=str(BASE/'runtime/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    cmd=[str(BASE/'bootstrap/bin/python'),'-u',str(code/'corrector_overnight_queue.py'),'--deadline',deadline]
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'launch.log').open('a') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    result={'pid':process.pid,'branch':BRANCH,'code':str(code),'output':str(ROOT/'results'),'deadline':deadline,
        'log':str(ROOT/'worker.log'),'planned_upper_bound':3400,'first_stage':600,'no_training':True,
        'models':5,'raw_features':'per-episode RAM only; small audit windows retained',
        'new_results_on_system_disk':True,'video_budget_gib':1.5,'preflight':audit}
    (ROOT/'launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'launch_history.jsonl').open('a') as history:history.write(json.dumps(result,ensure_ascii=False)+'\n')
    # 一次登录内等候实际首组，避免仅凭PID就宣称训练/评测已正常开始。
    until=time.monotonic()+180;last=-1
    while time.monotonic()<until:
        state=read(ROOT/'queue_status.json') if (ROOT/'queue_status.json').exists() else {}
        pipeline=read(ROOT/'results/pipeline_status.json') if (ROOT/'results/pipeline_status.json').exists() else {}
        execution=read(ROOT/'results/execution.json') if (ROOT/'results/execution.json').exists() else {}
        completed=execution.get('completed',0)
        if completed!=last:
            print(json.dumps({'startup_phase':pipeline.get('phase','starting'),'completed':completed,
                              'errors':execution.get('errors',0)},ensure_ascii=False),file=sys.stderr,flush=True);last=completed
        if state.get('status') in ['error','paused'] or pipeline.get('status') in ['error','paused']:
            raise RuntimeError('启动检查停止：'+json.dumps({'queue':state,'pipeline':pipeline},ensure_ascii=False))
        if completed>=6:
            result['startup_verified']=True;result['execution']=execution
            result['baseline_action_parity']=read(ROOT/'results/interface_smoke.json');break
        if not alive(process.pid):raise RuntimeError(f'监督器已退出，查看{ROOT}/launch.log')
        time.sleep(3)
    else:result['startup_verified']=False;result['note']='仍在后台运行，首组尚未确认；使用status查看，不重复install'
    print(json.dumps(result,ensure_ascii=False,indent=2))


def install(args):
    raw=sys.stdin.buffer.read(2**20)
    if hashlib.sha256(raw).hexdigest()!=args.sha:raise RuntimeError('上传SHA错误')
    code=BASE/'dev'/('overnight-'+args.code)
    if ROOT.exists() or code.exists():raise RuntimeError('新目录冲突，不覆盖；已有计划请用resume')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if set(z.namelist())!=set(FILES+EXTRA) or len(z.namelist())!=len(FILES+EXTRA):raise RuntimeError('上传文件集合异常')
        code.mkdir(parents=True,exist_ok=False)
        for name in z.namelist():(code/name).write_bytes(z.read(name))
    audit=preflight(code)
    if subprocess.run(['git','-C',str(BASE),'show-ref','--verify','--quiet','refs/heads/'+BRANCH]).returncode==0:
        raise RuntimeError('独立分支已存在，停止，不重复启动')
    subprocess.run(['git','-C',str(BASE),'switch','-c',BRANCH],check=True)
    launch(code,args.until,audit)


def remote(args):
    if args.mode=='remote-repair':
        old=Path('/root/corrector_overnight_20261111');trace=(old/'worker.log').read_text(errors='replace')
        state=read(old/'results/pipeline_status.json')
        if state.get('status')!='error' or 'UnpicklingError' not in state.get('error',''):
            raise RuntimeError('不是已确认的旧NumPy反序列化错误，不自动恢复：'+json.dumps(state,ensure_ascii=False))
        if 'in get_task_init_states' not in trace or 'libero/benchmark/__init__.py' not in trace:
            raise RuntimeError('完整栈未指向旧LIBERO初始状态入口，停止，不猜测或改检查点：'+trace[-9000:])
        if any(read(p).get('status')=='completed' for p in (old/'results/trials').glob('*.json')):
            raise RuntimeError('旧目录已有正式完成回合，停止自动迁移，需先核对')
        until=time.monotonic()+40
        while True:
            previous=read(old/'queue_status.json')
            if not alive(previous.get('pid')) and not alive(previous.get('child_pid')):break
            if time.monotonic()>=until:raise RuntimeError('旧父/子进程尚未退出，不并行启动')
            time.sleep(2)
        diagnosis={'cause':'legacy LIBERO get_task_init_states calls default torch.load; existing LeRobot compatible getter reused',
            'old_root_preserved':str(old),'old_error':state,'no_checkpoint_change':True}
        print(json.dumps({'diagnosis':diagnosis},ensure_ascii=False),file=sys.stderr,flush=True)
        install(args)
        (ROOT/'recovery_diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n')
        return
    if args.mode=='remote-install':install(args);return
    if args.mode=='remote-status':
        result={}
        for name in ['launch.json','queue_status.json','results/pipeline_status.json','results/execution.json','results/analysis/overnight_summary.json']:
            if (ROOT/name).exists():result[name]=read(ROOT/name)
        state=result.get('queue_status.json',{});result['supervisor_alive']=alive(state.get('pid'));result['child_alive']=alive(state.get('child_pid'))
        if (ROOT/'worker.log').exists():
            with (ROOT/'worker.log').open('rb') as file:
                file.seek(max(0,(ROOT/'worker.log').stat().st_size-6000));result['log_tail']=file.read().decode(errors='replace').splitlines()[-15:]
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.mode=='remote-pause':
        if not (ROOT/'results').is_dir():raise RuntimeError('没有已启动计划')
        (ROOT/'results/STOP').touch();print('已请求回合边界暂停；不杀进程、不删结果。');return
    if args.mode=='remote-resume':
        previous=read(ROOT/'launch.json');state=read(ROOT/'queue_status.json') if (ROOT/'queue_status.json').exists() else {}
        if alive(state.get('pid')) or alive(state.get('child_pid')):raise RuntimeError('仍有原进程，拒绝重复启动')
        code=Path(previous['code']);audit=preflight(code)
        if subprocess.check_output(['git','-C',str(BASE),'branch','--show-current'],text=True).strip()!=BRANCH:
            raise RuntimeError('远程不在原实验分支，不自动切换')
        if args.clear_stop and (ROOT/'results/STOP').exists():(ROOT/'results/STOP').unlink()
        launch(code,args.until or previous['deadline'],audit);return
    if args.mode=='remote-fetch':
        if not (ROOT/'results').exists():raise RuntimeError('尚无结果')
        buf=io.BytesIO()
        # Snapshot only complete per-case records; exclude large windows, raw commands and videos.
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            candidates=list(ROOT.glob('*.json'))+list((ROOT/'results').glob('*.json'))+list((ROOT/'results/analysis').glob('*'))
            candidates+=list((ROOT/'results/trials').glob('*.json'))
            for path in candidates:
                if path.is_file() and path.suffix in {'.json','.csv','.md'} and path.stat().st_size<=64*2**20:
                    data=path.read_bytes()
                    if path.parent.name=='trials' and json.loads(data)['status']!='completed':continue
                    z.writestr(str(path.relative_to(ROOT)),data)
        sys.stdout.buffer.write(buf.getvalue());return
    raise ValueError(args.mode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['install','repair','status','pause','resume','fetch','remote-install','remote-repair','remote-status','remote-pause','remote-resume','remote-fetch'])
    p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com');p.add_argument('--until')
    p.add_argument('--clear-stop',action='store_true');p.add_argument('--sha');p.add_argument('--code')
    p.add_argument('--output',type=Path,default=Path('/home/hhd/robot-lab/outputs/corrector_overnight_20261111'))
    args=p.parse_args()
    if args.mode.startswith('remote-'):remote(args);return
    if args.mode in ['install','repair'] and args.until is None:args.until='2026-11-12T10:00:00+08:00'
    cmd=[str(BASE/'bootstrap/bin/python'),'-c',Path(__file__).read_text(),'remote-'+args.mode];raw=b''
    if args.until:cmd+=['--until',args.until]
    if args.clear_stop:cmd.append('--clear-stop')
    if args.mode in ['install','repair']:raw,code=payload();cmd+=['--sha',hashlib.sha256(raw).hexdigest(),'--code',code]
    ssh=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',args.ssh_host,shlex.join(cmd)]
    if args.mode=='fetch':
        args.output.mkdir(parents=True,exist_ok=True)
        from datetime import datetime
        target=args.output/('snapshot_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip');partial=target.with_suffix('.partial')
        with partial.open('xb') as file:r=subprocess.run(ssh,input=raw,stdout=file)
        if r.returncode:raise RuntimeError('SSH取回失败，不发布半成品')
        with zipfile.ZipFile(partial) as z:
            if z.testzip():raise RuntimeError('快照ZIP校验失败')
        partial.replace(target);print(json.dumps({'bundle':str(target),'bytes':target.stat().st_size}));return
    if subprocess.run(ssh,input=raw).returncode:raise RuntimeError('远程检查或操作停止，未自动重试；保留错误现场')


if __name__=='__main__':main()

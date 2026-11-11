"""部署有限完整任务闭环对照；预冻结阈值、真实动作历史、普通重规划，无OGG。"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile

BASE=Path('/root/autodl-tmp/corrector')
OUTPUT=BASE/'object10/consistency_closed_loop_20261111_001'
BRANCH='experiment/executed-closed-loop-pilot-20261111'
FILES=['corrector_consistency_control.py','corrector_consistency_eval.py','corrector_executed_monitor.py',
       'corrector_hybrid.py','corrector_hybrid_pilot.py','corrector_online_runtime.py','corrector_models.py',
       'corrector_train.py','corrector_data_workbench.py','smolvla_horizon_pilot.py',
       'smolvla_baseline_benchmark.py','smolvla_libero_tools.py','smolvla_training_workbench.py']


def payload():
    here=Path(__file__).resolve().parent;files={n:(here/n).read_bytes() for n in FILES}
    root=here.parent/'studies/policy_generalization/corrector'
    for n,s in [('batch_evaluation.py','batch_evaluation.py'),('corrector_online_practice.py','online_practice.py')]:
        files[n]=(root/s).read_bytes()
    files['calibration_rule.json']=(root/'manifests/closed_loop_calibration_20261111.json').read_bytes()
    code=hashlib.sha256(b''.join(n.encode()+files[n] for n in sorted(files))).hexdigest()[:12]
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    return buf.getvalue(),code


def apply(args):
    raw=sys.stdin.buffer.read(2**20)
    if hashlib.sha256(raw).hexdigest()!=args.sha:raise RuntimeError('上传SHA错误')
    target=BASE/'dev'/('consistency-control-'+args.code);log=BASE/'consistency-closed-loop-20261111-001.log'
    if target.exists() or OUTPUT.exists() or log.exists():raise RuntimeError('目录/日志冲突，停止，不覆盖')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        allowed=set(FILES+['batch_evaluation.py','corrector_online_practice.py','calibration_rule.json'])
        if set(z.namelist())!=allowed or len(z.namelist())!=len(allowed):raise RuntimeError('上传文件集合异常')
        target.mkdir(parents=True,exist_ok=False)
        for n in z.namelist():(target/n).write_bytes(z.read(n))
    sys.path.insert(0,str(target))
    spec=importlib.util.spec_from_file_location('closed_loop_preflight',target/'smolvla_horizon_pilot.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);audit=m.preflight(BASE)
    prior=json.loads((BASE/'object10/executed_consistency_pilot_20261111_002/pipeline_status.json').read_text())
    if prior['status']!='completed' or prior['trials']!=4:raise RuntimeError('前序被动Pilot未完成')
    if subprocess.check_output(['git','-C',str(BASE),'diff','HEAD','--name-only'],text=True).strip():
        raise RuntimeError('已跟踪源码有修改，不切分支')
    if subprocess.run(['git','-C',str(BASE),'show-ref','--verify','--quiet','refs/heads/'+BRANCH]).returncode==0:
        raise RuntimeError('独立分支已存在，不重复部署')
    subprocess.run(['git','-C',str(BASE),'switch','-c',BRANCH],check=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(HF_HOME=str(BASE/'runtime/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    cmd=[str(BASE/'runtime/sim-venv/bin/python'),'-u',str(target/'corrector_consistency_eval.py')]
    with log.open('x') as file:
        process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    result={'pid':process.pid,'branch':BRANCH,'code':str(target),'output':str(OUTPUT),'log':str(log),
        'methods':['H50','H10','planned_lvm','executed_lvm'],'trials':48,
        'initial_states':[3,4,5],'seeds':[1002,1003],'tasks':[1,9],
        'ogg':False,'threshold':json.loads((target/'calibration_rule.json').read_text())['rule'],
        'preflight':audit,'automatic_expansion':False}
    (target/'launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def status():
    result={'output_exists':OUTPUT.exists()}
    for n in ['pipeline_status.json','execution.json','analysis/closed_loop_summary.json']:
        if (OUTPUT/n).is_file():result[n]=json.loads((OUTPUT/n).read_text())
    log=BASE/'consistency-closed-loop-20261111-001.log'
    if log.is_file():result['log_tail']=log.read_text(errors='replace').splitlines()[-20:]
    print(json.dumps(result,ensure_ascii=False,indent=2))


def export(wait_minutes=0):
    deadline=time.monotonic()+wait_minutes*60
    while True:
        state=OUTPUT/'pipeline_status.json';v=json.loads(state.read_text()) if state.is_file() else {}
        if v.get('status')=='completed':break
        if v.get('status') in {'error','paused','interrupted_or_error'} or time.monotonic()>=deadline:
            print('尚未导出：'+json.dumps(v,ensure_ascii=False)+'；请查看status，不改阈值。',file=sys.stderr)
            raise SystemExit(2)
        progress=OUTPUT/'execution.json'
        execution=json.loads(progress.read_text()) if progress.is_file() else {}
        print(json.dumps({'waiting_for_completion':True,'completed':execution.get('completed'),
                          'planned':48,'errors':execution.get('errors')},ensure_ascii=False),file=sys.stderr,flush=True)
        time.sleep(min(15,max(0,deadline-time.monotonic())))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.csv'}:
                if p.stat().st_size>16*2**20:raise RuntimeError('结果文件超出轻量导出预算')
                z.write(p,p.relative_to(OUTPUT))
    sys.stdout.buffer.write(buf.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['install','status','fetch','apply','remote-status','remote-export'])
    p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com');p.add_argument('--sha');p.add_argument('--code')
    p.add_argument('--wait-minutes',type=float,default=0,help='仅fetch：只读等待完成，单次SSH；0立即检查')
    p.add_argument('--output',type=Path,default=Path('/home/hhd/robot-lab/outputs/consistency_closed_loop_20261111/results.zip'))
    args=p.parse_args()
    if args.mode=='apply':apply(args);return
    if args.mode=='remote-status':status();return
    if args.mode=='remote-export':export(args.wait_minutes);return
    cmd=[str(BASE/'bootstrap/bin/python'),'-c',Path(__file__).read_text(),
         {'install':'apply','status':'remote-status','fetch':'remote-export'}[args.mode]]
    raw=b''
    if args.mode=='install':
        raw,code=payload();cmd+=['--sha',hashlib.sha256(raw).hexdigest(),'--code',code]
    if args.mode=='fetch':
        if not 0<=args.wait_minutes<=120:raise ValueError('等待分钟应在0～120之间')
        cmd+=['--wait-minutes',str(args.wait_minutes)]
    ssh=['ssh','-T','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',args.ssh_host,shlex.join(cmd)]
    if args.mode=='fetch':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise RuntimeError('本地完整结果已存在，不覆盖')
        partial=args.output.with_suffix('.zip.partial')
        with partial.open('wb') as file:r=subprocess.run(ssh,input=raw,stdout=file)
        if r.returncode:
            if partial.stat().st_size==0:partial.unlink()
            if r.returncode==2:print('尚未完成，可以稍后重试。');return
            raise RuntimeError('SSH取回失败，未发布半成品')
        with zipfile.ZipFile(partial) as z:
            if z.testzip() or json.loads(z.read('pipeline_status.json'))['status']!='completed':raise RuntimeError('结果校验失败')
        partial.replace(args.output);print(json.dumps({'bundle':str(args.output),'bytes':args.output.stat().st_size}))
    elif subprocess.run(ssh,input=raw).returncode:raise RuntimeError('远程检查/部署停止，不自动重试')


if __name__=='__main__':main()

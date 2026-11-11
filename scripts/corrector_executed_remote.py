"""SSH部署被动实际动作到期监测；只采集/评分，不中断、不OGG、不训练。"""
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
import zipfile

BASE=Path('/root/autodl-tmp/corrector')
OUTPUT=BASE/'object10/executed_consistency_pilot_20261111_002'
BRANCH='experiment/executed-consistency-pilot-20261111-v2'
FILES=['corrector_executed_monitor.py','corrector_executed_pilot.py','corrector_hybrid.py',
       'corrector_hybrid_pilot.py','corrector_online_runtime.py','corrector_models.py',
       'corrector_train.py','corrector_data_workbench.py','smolvla_horizon_pilot.py',
       'smolvla_baseline_benchmark.py','smolvla_libero_tools.py','smolvla_training_workbench.py']


def payload():
    here=Path(__file__).resolve().parent;files={n:(here/n).read_bytes() for n in FILES}
    for n,s in [('batch_evaluation.py','batch_evaluation.py'),('corrector_online_practice.py','online_practice.py')]:
        files[n]=(here.parent/'studies/policy_generalization/corrector'/s).read_bytes()
    code=hashlib.sha256(b''.join(n.encode()+files[n] for n in sorted(files))).hexdigest()[:12]
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    return b.getvalue(),code


def apply(args):
    raw=sys.stdin.buffer.read(2**20)
    if hashlib.sha256(raw).hexdigest()!=args.sha:raise RuntimeError('上传SHA错误')
    target=BASE/'dev'/('executed-monitor-'+args.code);log=BASE/'executed-consistency-20261111-002.log'
    if target.exists() or OUTPUT.exists() or log.exists():raise RuntimeError('目录/日志冲突，不覆盖、不重复启动')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        allowed=set(FILES+['batch_evaluation.py','corrector_online_practice.py'])
        if set(z.namelist())!=allowed or len(z.namelist())!=len(allowed):raise RuntimeError('文件集合异常')
        target.mkdir(parents=True,exist_ok=False)
        for n in z.namelist():(target/n).write_bytes(z.read(n))
    sys.path.insert(0,str(target))
    spec=importlib.util.spec_from_file_location('executed_preflight',target/'smolvla_horizon_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=module.preflight(BASE)
    if json.loads((BASE/'object10/hybrid_pilot_20261111_001/pipeline_status.json').read_text()).get('status')!='completed':
        raise RuntimeError('混合Pilot未完成')
    if subprocess.check_output(['git','-C',str(BASE),'diff','HEAD','--name-only'],text=True).strip():
        raise RuntimeError('已跟踪源码有修改，停止，不切分支')
    if subprocess.run(['git','-C',str(BASE),'show-ref','--verify','--quiet','refs/heads/'+BRANCH]).returncode==0:
        raise RuntimeError('实验分支已存在，停止，不覆盖')
    subprocess.run(['git','-C',str(BASE),'switch','-c',BRANCH],check=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(HF_HOME=str(BASE/'runtime/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    command=[str(BASE/'runtime/sim-venv/bin/python'),'-u',str(target/'corrector_executed_pilot.py')]
    with log.open('x') as file:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    receipt={'pid':child.pid,'branch':BRANCH,'code':str(target),'output':str(OUTPUT),'log':str(log),
        'trials':4,'monitor_only':True,'interruptions_allowed':False,'ogg_allowed':False,'preflight':audit,
        'feature_capture_max_gib':4*280*128*960*4/2**30,'automatic_expansion':False}
    (target/'launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


def status():
    result={'output_exists':OUTPUT.exists()}
    for n in ['pipeline_status.json','execution.json']:
        if (OUTPUT/n).is_file():result[n]=json.loads((OUTPUT/n).read_text())
    log=BASE/'executed-consistency-20261111-002.log'
    if log.is_file():result['log_tail']=log.read_text(errors='replace').splitlines()[-25:]
    print(json.dumps(result,ensure_ascii=False,indent=2))


def export():
    state=OUTPUT/'pipeline_status.json';value=json.loads(state.read_text()) if state.is_file() else {}
    if value.get('status')!='completed':
        print('被动Pilot尚未全部完成，先查看status，不导出半成品。',file=sys.stderr);raise SystemExit(2)
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.csv'}:
                if p.stat().st_size>16*2**20:raise RuntimeError('结果文件过大')
                z.write(p,p.relative_to(OUTPUT))
    sys.stdout.buffer.write(b.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['install','status','fetch','apply','remote-status','remote-export'])
    p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com');p.add_argument('--sha');p.add_argument('--code')
    p.add_argument('--output',type=Path,default=Path('/home/hhd/robot-lab/outputs/executed_consistency_pilot_20261111_v2/results.zip'))
    args=p.parse_args()
    if args.mode=='apply':apply(args);return
    if args.mode=='remote-status':status();return
    if args.mode=='remote-export':export();return
    remote=[str(BASE/'bootstrap/bin/python'),'-c',Path(__file__).read_text(),
        {'install':'apply','status':'remote-status','fetch':'remote-export'}[args.mode]]
    raw=b''
    if args.mode=='install':
        raw,code=payload();remote+=['--sha',hashlib.sha256(raw).hexdigest(),'--code',code]
    cmd=['ssh','-T','-o','ConnectTimeout=15',args.ssh_host,shlex.join(remote)]
    if args.mode=='fetch':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise RuntimeError('完整结果包已存在，不覆盖')
        partial=args.output.with_suffix('.zip.partial')
        with partial.open('wb') as file:result=subprocess.run(cmd,input=raw,stdout=file)
        if result.returncode:
            if partial.stat().st_size==0:partial.unlink()
            if result.returncode==2:print('尚未完成，可稍后重试；没有生成完整结果包。');return
            raise RuntimeError('SSH取回失败，未覆盖完整结果包')
        with zipfile.ZipFile(partial) as z:
            if z.testzip() or json.loads(z.read('pipeline_status.json')).get('status')!='completed':
                raise RuntimeError('结果包不完整')
        partial.replace(args.output);print(json.dumps({'bundle':str(args.output),'bytes':args.output.stat().st_size}))
    else:
        if subprocess.run(cmd,input=raw).returncode:raise RuntimeError('部署/状态检查停止，不自动重试')


if __name__=='__main__':main()

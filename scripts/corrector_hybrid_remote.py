"""交互SSH部署独立混合Corrector Pilot；冻结权重只读、兼容性不符即停止。"""
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
OUTPUT=BASE/'object10/hybrid_pilot_20261111_001'
BRANCH='experiment/hybrid-corrector-pilot-20261111'
FILES=['corrector_hybrid.py','corrector_hybrid_pilot.py','corrector_online_runtime.py',
       'corrector_models.py','corrector_train.py','corrector_data_workbench.py',
       'smolvla_horizon_pilot.py','smolvla_baseline_benchmark.py',
       'smolvla_libero_tools.py','smolvla_training_workbench.py']


def payload():
    here=Path(__file__).resolve().parent
    files={name:(here/name).read_bytes() for name in FILES}
    for name,source in [('batch_evaluation.py','batch_evaluation.py'),('corrector_online_practice.py','online_practice.py')]:
        files[name]=(here.parent/'studies/policy_generalization/corrector'/source).read_bytes()
    sha=hashlib.sha256(b''.join(name.encode()+files[name] for name in sorted(files))).hexdigest()[:12]
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
    return buf.getvalue(),sha


def apply(args):
    raw=sys.stdin.buffer.read(2**20)
    if hashlib.sha256(raw).hexdigest()!=args.sha:raise RuntimeError('上传SHA校验失败')
    target=BASE/'dev'/('hybrid-pilot-'+args.code)
    log=BASE/'hybrid-pilot-20261111-001.log'
    if target.exists() or OUTPUT.exists() or log.exists():
        raise RuntimeError('新代码/输出/日志目录冲突，不覆盖，不重复启动')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        allowed=set(FILES+['batch_evaluation.py','corrector_online_practice.py'])
        if set(z.namelist())!=allowed or len(z.namelist())!=len(allowed):
            raise RuntimeError('上传文件集合异常')
        target.mkdir(parents=True,exist_ok=False)
        for name in z.namelist():(target/name).write_bytes(z.read(name))
    sys.path.insert(0,str(target))
    spec=importlib.util.spec_from_file_location('hybrid_preflight',target/'smolvla_horizon_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=module.preflight(BASE)
    horizon=json.loads((BASE/'object10/horizon_pilot_20261111_001/execution.json').read_text())
    if horizon['status']!='completed' or horizon['completed']!=24 or horizon['errors']!=0:
        raise RuntimeError('前一Pilot没有正常完成')
    origin=subprocess.check_output(['git','-C',str(BASE),'remote','get-url','origin'],text=True).strip()
    if origin not in {'https://github.com/30thinkdiff/corrector.git','git@github.com:30thinkdiff/corrector.git'}:
        raise RuntimeError('远程工程origin异常，不创建实验分支')
    if subprocess.check_output(['git','-C',str(BASE),'diff','HEAD','--name-only'],text=True).strip():
        raise RuntimeError('远程存在已跟踪源码修改；先报告，不切分支')
    exists=subprocess.run(['git','-C',str(BASE),'show-ref','--verify','--quiet','refs/heads/'+BRANCH])
    if exists.returncode==0:raise RuntimeError('实验分支已存在，拒绝重复部署')
    subprocess.run(['git','-C',str(BASE),'switch','-c',BRANCH],check=True)
    command=[str(BASE/'runtime/sim-venv/bin/python'),'-u',str(target/'corrector_hybrid_pilot.py')]
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(HF_HOME=str(BASE/'runtime/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with log.open('x') as f:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                               env=env,start_new_session=True)
    receipt={'pid':child.pid,'branch':BRANCH,'code':str(target),'output':str(OUTPUT),'log':str(log),
             'stages':['compatibility','shape_gradient_probe','3 forced interface checks','12 natural paired trials'],
             'preflight':audit,'retraining':False,'automatic_expansion':False}
    (target/'launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


def status():
    result={'output_exists':OUTPUT.exists()}
    for name in ['pipeline_status.json','compatibility.json','shape_gradient_probe.json',
                 'forced_check/execution.json','paired/execution.json','paired/analysis/hybrid_summary.json']:
        if (OUTPUT/name).is_file():result[name]=json.loads((OUTPUT/name).read_text())
    log=BASE/'hybrid-pilot-20261111-001.log'
    if log.is_file():result['log_tail']=log.read_text(errors='replace').splitlines()[-25:]
    print(json.dumps(result,ensure_ascii=False,indent=2))


def export():
    state=OUTPUT/'pipeline_status.json'
    value=json.loads(state.read_text()) if state.is_file() else {}
    if value.get('status')!='completed':
        print('Pilot尚未全部完成，不导出半成品；请稍后再运行fetch。',file=sys.stderr)
        raise SystemExit(2)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.csv'}:
                if p.stat().st_size>16*2**20:raise RuntimeError('单文件结果过大，停止导出')
                z.write(p,p.relative_to(OUTPUT))
    sys.stdout.buffer.write(buf.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['install','status','fetch','apply','remote-status','remote-export'])
    p.add_argument('--ssh-host',default='connect.nmb1.seetacloud.com')
    p.add_argument('--sha');p.add_argument('--code')
    p.add_argument('--output',type=Path,default=Path('/home/hhd/robot-lab/outputs/hybrid_pilot_20261111/results.zip'))
    args=p.parse_args()
    if args.mode=='apply':apply(args);return
    if args.mode=='remote-status':status();return
    if args.mode=='remote-export':export();return
    remote=[str(BASE/'bootstrap/bin/python'),'-c',Path(__file__).read_text(),
            {'install':'apply','status':'remote-status','fetch':'remote-export'}[args.mode]]
    raw=b''
    if args.mode=='install':
        raw,code=payload();remote+=['--sha',hashlib.sha256(raw).hexdigest(),'--code',code]
    command=['ssh','-T','-o','ConnectTimeout=15',args.ssh_host,shlex.join(remote)]
    if args.mode=='fetch':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise RuntimeError('本地完整结果包已存在，不覆盖')
        partial=args.output.with_suffix(args.output.suffix+'.partial')
        with partial.open('wb') as out:
            result=subprocess.run(command,input=raw,stdout=out)
        if result.returncode:
            if partial.stat().st_size==0:partial.unlink()
            if result.returncode==2:
                print('尚未全部完成；未生成本地结果包，可稍后重试。');return
            raise RuntimeError('取回失败；临时文件不冒充完整结果包，可重试')
        with zipfile.ZipFile(partial) as z:
            if z.testzip():raise RuntimeError('ZIP校验失败')
            if json.loads(z.read('pipeline_status.json')).get('status')!='completed':
                raise RuntimeError('结果包没有完整完成标记')
        partial.replace(args.output)
        print(json.dumps({'bundle':str(args.output),'bytes':args.output.stat().st_size}))
    else:
        result=subprocess.run(command,input=raw)
        if result.returncode:raise RuntimeError('远程检查/部署停止，请报告错误，不自动重试')


if __name__=='__main__':main()

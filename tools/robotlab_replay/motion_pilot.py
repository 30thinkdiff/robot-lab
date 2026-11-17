"""New opt-in diagnostic adapter over frozen expanded evaluation; default inspect.

No installed third-party or frozen deployment files are edited. `run` requires a
GPU; do not invoke it on the current CPU-only instance. Completed trials resume
through the original batch_evaluation runner; interrupted attempts are retained.
"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

SCHEMA='robotlab_motion_v1'
BASE=Path('/root/autodl-tmp/corrector')
METHODS=['H50','H20','mlp_lvm','mlp_ogg','j_lvm','j_ogg']


def frozen_audit(scripts):
    scripts=Path(scripts).resolve();root=scripts.parent
    manifest=json.loads((root/'EXPANDED_MANIFEST.json').read_text())
    for name,sha in manifest.items():
        p=(root/name).resolve();p.relative_to(root)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('frozen source changed: '+name)
    if manifest.get('scripts/corrector_expanded_eval.py')!='86860f248b46fc3917fd3ee07aa4bd0be5bbb0c0e0701ea88a2ea9d482064d70':
        raise ValueError('adapter supports only reviewed final expanded deployment')
    return manifest


def selection(cases,methods):
    result=[]
    if len(set(methods))!=len(methods) or not set(methods)<=set(METHODS):raise ValueError('unknown/duplicate method')
    for item in cases:
        match=re.fullmatch(r'(libero_object/task_([0-9])/init_([0-9]+)):([0-9]+)',item)
        if not match:raise ValueError('case must be libero_object/task_T/init_I:SEED')
        condition,task,init,seed=match.groups()
        if int(init)>=50:raise ValueError('initial state index outside supported range')
        for method in methods:result.append((condition,int(seed),method))
    if len(result)!=len(set(result)) or not 1<=len(result)<=24:raise ValueError('duplicate/oversized pilot; maximum 24 trials')
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['inspect','run'],nargs='?',default='inspect')
    p.add_argument('--frozen-scripts',type=Path,default=BASE/'dev/expanded-2e6296eff15f/scripts')
    p.add_argument('--runtime',type=Path,default=BASE/'runtime');p.add_argument('--core',type=Path,default=BASE/'dev/j36m-35ecdd7488bf')
    p.add_argument('--training',type=Path,default=BASE/'j36m_longrun_20261113_001/training')
    p.add_argument('--calibration',type=Path,default=Path('/root/corrector_j36m_object_20261113_002/j_calibration.json'))
    p.add_argument('--output',type=Path,default=Path('/root/corrector_motion_pilot_001'))
    p.add_argument('--case',action='append',dest='cases');p.add_argument('--methods',nargs='+',default=METHODS)
    p.add_argument('--hours',type=float,default=2);p.add_argument('--substep-stride',type=int,default=0)
    p.add_argument('--capture',choices=['on','off'],default='on',help='off is a separate same-condition control, not reused output')
    a=p.parse_args();manifest=frozen_audit(a.frozen_scripts)
    wanted=selection(a.cases or ['libero_object/task_1/init_0:1001','libero_object/task_9/init_2:1000'],a.methods)
    if not 0<a.hours<=4 or not 0<=a.substep_stride<=100:raise ValueError('pilot hours/stride out of range')
    receipt={'schema':SCHEMA,'frozen_manifest':manifest,'cases':wanted,'substep_stride':a.substep_stride,'capture':a.capture,
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'capture_sha256':hashlib.sha256(Path(__file__).with_name('motion_capture.py').read_bytes()).hexdigest(),
        'scope':'diagnostic rollout with added CPU/IO; not a zero-overhead performance benchmark'}
    if a.mode=='inspect':
        print(json.dumps({'status':'cpu_source_check_passed','trials':len(wanted),'gpu_started':False,
            'cases':wanted,'frozen_files_verified':len(manifest)},ensure_ascii=False,indent=2));return
    # Everything below is explicit future opt-in, never called by inspect.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU occupied; no experiment launched')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[v.strip() for v in gpu.split(',')]
    if len(fields)!=3 or '4090' not in fields[0] or float(fields[1])>128 or float(fields[2])>5:
        raise RuntimeError('single idle 4090 not verified; no experiment launched')
    output=a.output.resolve()
    if any(output.is_relative_to(d.resolve()) for d in [a.frozen_scripts.parent,a.core,a.runtime,a.training]):
        raise ValueError('output must be outside frozen source, environment and training roots')
    if output.exists():
        old=output/'motion_adapter.json'
        if not old.exists() or json.loads(old.read_text())!=json.loads(json.dumps(receipt)):
            raise FileExistsError('output exists with a different/unverified protocol')
    else:output.mkdir(parents=True)
    lock=(output/'motion_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not(output/'motion_adapter.json').exists():(output/'motion_adapter.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:(output/'STOP').touch())
    sys.path.insert(0,str(a.frozen_scripts.resolve()))
    expanded=importlib.import_module('corrector_expanded_eval')
    args=SimpleNamespace(**vars(a));args.mode='ordinary';args.output=output;args.deadline=time.time()+a.hours*3600
    args.j_checkpoint=a.training/'J_s42/best.pt';args.c0_checkpoint=a.training/'C0_s42/best.pt'
    expanded.guard(args)
    from motion_capture import MotionCapture,ActionDecoder
    snapshot=a.runtime/'hf/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d'
    # Validate the small numeric decoder before loading any GPU model.
    decoder=ActionDecoder.from_snapshot(snapshot)
    h=expanded.load(args)
    from lerobot.scripts import lerobot_eval
    from studies.policy_generalization.corrector import batch_evaluation
    old_eval=lerobot_eval.eval_policy;old_run=batch_evaluation.run_plan
    active={}
    def evaluate(*aa,**kw):
        if aa or not active or len(kw['env'].envs)!=1:raise ValueError('unsupported evaluation call layout')
        if a.capture=='off':return old_eval(**kw)
        with MotionCapture(kw['env'].envs[0].unwrapped,kw['policy'],active['directory']/'motion',decoder,
            method=active['case']['method'],provenance={'case':active['case'],'adapter':receipt},
            substep_stride=a.substep_stride):return old_eval(**kw)
    def run_plan(root,plan,callbacks,**kw):
        plan=copy.deepcopy(plan);plan['protocol'].update(motion_capture=receipt,
            study_id='replay_casebank_motion_diagnostic_v1',experiment_label='动作跟踪诊断 · 新采集')
        def wrap(callback):
            def call(case,directory):
                active.update(case=case,directory=directory)
                try:
                    result=callback(case,directory)
                    if a.capture=='off':
                        return {**result,'motion_capture':{'schema':SCHEMA,'capture':'disabled_control','files_sha256':{}}}
                    motion=json.loads((directory/'motion/status.json').read_text())
                    if motion['status']!='completed' or motion['steps']!=result['env_steps']:
                        raise RuntimeError('motion and rollout completion differ')
                    files={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((directory/'motion').rglob('*.json*')) if p.is_file()}
                    return {**result,'motion_capture':{'schema':SCHEMA,'files_sha256':files}}
                finally:active.clear()
            return call
        # Verify motion artifacts as well as the original runner's video checks on resume.
        for p in (Path(root)/'trials').glob('*.json'):
            r=json.loads(p.read_text())
            if r['status']=='completed':
                directory=Path(root)/'attempts'/r['case']['id']/('attempt_%03d'%r['attempt'])
                files=r['result']['motion_capture']['files_sha256']
                for name,sha in files.items():
                    f=(directory/name).resolve();f.relative_to(directory.resolve())
                    if hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise ValueError('completed motion artifact changed')
        return old_run(root,plan,{m:wrap(cb) for m,cb in callbacks.items()},**kw)
    state={'status':'running','trials':len(wanted),'scope':'motion diagnostics; physics not yet validated before first real rollout'}
    try:
        all_conditions=expanded.ordinary_conditions();conditions={c:all_conditions[c] for c,_,_ in wanted}
        with ExitStack() as stack:
            stack.enter_context(patch.object(expanded,'cases',lambda mode,conditions:wanted))
            stack.enter_context(patch.object(lerobot_eval,'eval_policy',evaluate))
            stack.enter_context(patch.object(batch_evaluation,'run_plan',run_plan))
            expanded.run(args,'ordinary',h,conditions)
        state['status']='completed'
    except BaseException as e:
        state.update(status='paused' if (output/'STOP').exists() else 'error',error=type(e).__name__+': '+str(e));raise
    finally:(output/'motion_pilot_status.json').write_text(json.dumps(state,ensure_ascii=False)+'\n')


if __name__=='__main__':main()

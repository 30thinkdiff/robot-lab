"""H10/H5/H1 on the original frozen 32 Plus conditions; no control Corrector.

Reuses the immutable expanded recorder, environment and queue evaluator, plus
the previously validated Horizon QueueAudit via AST extraction. No source edit,
training, download, extra evaluation seeds, or automatic expansion.
"""
import argparse
import ast
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import importlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

BASE=Path('/root/autodl-tmp/corrector')
HORIZONS=(10,5,1)
AUDIT_SHA='6f03efa535aab4cd36a2f45368683956ff9ddd075a698896b6713c4ffcf682eb'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_plan(path):
    plan=json.loads(Path(path).read_text());p=plan['protocol'];conditions=p['conditions']
    expected={'fps':20,'horizon':280,'chunk_size':50,'denoise':10,'vla_dtype':'float32',
        'policy_revision':'a24f1f380461b665158d52acd140152c79e8b36d',
        'plus_revision':'4976dc30028e805ff8094b55501d532c48fec182'}
    if len(conditions)!=32 or any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('original 32-condition Plus protocol differs')
    for method in ['H50','H20']:
        old=[c for c in plan['cases'] if c['method']==method]
        if {(c['condition'],c['seed']) for c in old}!={(c,1000) for c in conditions} or len(old)!=32:
            raise ValueError('original baseline condition/seed set differs')
    return plan


def audit_api(path):
    if sha(path)!=AUDIT_SHA:raise ValueError('frozen Horizon audit source SHA differs')
    tree=ast.parse(Path(path).read_text());body=[n for n in tree.body if
        isinstance(n,ast.ClassDef) and n.name=='QueueAudit' or isinstance(n,ast.FunctionDef) and n.name=='continuity']
    if len(body)!=2:raise ValueError('audit class/continuity function missing')
    namespace={'time':time,'math':math,'hashlib':hashlib}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),namespace)
    return namespace['QueueAudit'],namespace['continuity']


def registry():
    return {f'H{h}':{'steps':h,'action_steps':h,'family':'baseline','model_label':'SmolVLA'} for h in HORIZONS}


def corrected_plan(make_plan,conditions,seeds,methods,protocol,receipt):
    if set(methods)!=set(registry()) or list(seeds)!=[1000]:raise ValueError('unexpected methods/seeds')
    methods={m:{**c,'uses_corrector':False,'corrector_sha256':None} for m,c in methods.items()}
    protocol={**protocol,'study_id':'replay_casebank_plus_horizon_20261116',
        'experiment_label':'Plus原32条件 · H10/H5/H1','horizon_followup':receipt,
        'corrector_enabled':False,'lvm_enabled':False,'ogg_enabled':False,
        'warmup_chunks_per_episode':0,
        'loader_limit':'legacy loader instantiates auxiliary frozen models; none used in baseline control'}
    return make_plan(conditions,seeds,methods,protocol)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['inspect','run'],nargs='?',default='inspect')
    p.add_argument('--output',type=Path,default=Path('/root/corrector_plus_horizon_20261116_001'))
    p.add_argument('--frozen-scripts',type=Path,default=BASE/'dev/expanded-2e6296eff15f/scripts')
    p.add_argument('--audit-source',type=Path,default=BASE/'dev/horizon-pilot-f13415bb39d0/smolvla_horizon_pilot.py')
    p.add_argument('--old-plan',type=Path,default=Path('/root/corrector_plus_expanded_20261113_001/results/plus/plan.json'))
    p.add_argument('--runtime',type=Path,default=BASE/'runtime');p.add_argument('--core',type=Path,default=BASE/'dev/j36m-35ecdd7488bf')
    p.add_argument('--training',type=Path,default=BASE/'j36m_longrun_20261113_001/training')
    p.add_argument('--calibration',type=Path,default=Path('/root/corrector_j36m_object_20261113_002/j_calibration.json'))
    p.add_argument('--plus-source',type=Path,default=BASE/'runtime/libero-plus-4976dc3')
    p.add_argument('--plus-config',type=Path,default=BASE/'plus_resources_20261113_001/config')
    p.add_argument('--hours',type=float,default=6);a=p.parse_args()
    if not 0<a.hours<=12:raise ValueError('hours must be 0..12')
    from motion_pilot import frozen_audit
    frozen=frozen_audit(a.frozen_scripts);old=source_plan(a.old_plan);conditions=old['protocol']['conditions']
    QueueAudit,continuity=audit_api(a.audit_source)
    receipt={'schema':'plus_horizon_followup_v1','source_sha256':sha(__file__),'old_plan_sha256':sha(a.old_plan),
        'audit_source_sha256':sha(a.audit_source),'frozen_manifest':frozen,'horizons':HORIZONS,
        'evaluation_seeds':[1000],'planned':96,'control_fps':20,'chunk_size':50,'denoise_steps':10,
        'warmup':'disabled to match original Plus; original audit reused without its two warmup calls'}
    if a.mode=='inspect':
        print(json.dumps({'status':'cpu_protocol_passed','planned':96,'conditions':32,'horizons':HORIZONS,
            'gpu_started':False,'old_plan_sha256':receipt['old_plan_sha256']},indent=2));return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; not starting')
    if a.output.exists():
        if not(a.output/'horizon_adapter.json').exists() or json.loads((a.output/'horizon_adapter.json').read_text())!=json.loads(json.dumps(receipt)):
            raise FileExistsError('output/protocol conflict')
    else:a.output.mkdir(parents=True)
    lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not(a.output/'horizon_adapter.json').exists():(a.output/'horizon_adapter.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:(a.output/'STOP').touch())
    state={'status':'running','phase':'loading_models','planned':96,'pid':__import__('os').getpid()}
    def status():(a.output/'status.json').write_text(json.dumps(state,ensure_ascii=False)+'\n')
    status()
    try:
        sys.path.insert(0,str(a.frozen_scripts));expanded=importlib.import_module('corrector_expanded_eval')
        args=SimpleNamespace(**vars(a));args.mode='plus';args.deadline=time.time()+a.hours*3600
        args.j_checkpoint=a.training/'J_s42/best.pt';args.c0_checkpoint=a.training/'C0_s42/best.pt'
        expanded.guard(args);h=expanded.load(args)
        from lerobot.scripts import lerobot_eval
        from studies.policy_generalization.corrector import batch_evaluation
        original_eval=lerobot_eval.eval_policy;original_make=batch_evaluation.make_plan;original_run=batch_evaluation.run_plan
        active={}
        def evaluate(*aa,**kw):
            if aa or not active:raise ValueError('evaluation call outside declared case')
            policy=kw['policy'];window=int(active['case']['method'][1:]);audit=QueueAudit(policy,window,active['directory'],warmup=0)
            audit.warm=True  # original Plus has no extra warmup inference/reset
            with ExitStack() as stack:
                for obj,name,value in [(policy,'_get_action_chunk',audit.chunk),(policy,'select_action',audit.select),
                    (policy.model,'sample_actions',audit.sample),(policy.model,'denoise_step',audit.denoise_step)]:
                    stack.enter_context(patch.object(obj,name,value))
                result=original_eval(**kw)
            audit.check();active['audit']={'steps':audit.steps,'chunk_calls':audit.calls,'sample_actions_calls':audit.samples,
                'denoise_forward_calls':audit.denoise,'window':window,'queue_verified':True,'warmup_calls':0}
            (active['directory']/'horizon_queue_audit.json').write_text(json.dumps({'summary':active['audit'],'steps':audit.trace})+'\n')
            return result
        def run_plan(root,plan,callbacks,**kw):
            if len(plan['cases'])!=96 or any(c['config']['uses_corrector'] for c in plan['cases']):
                raise ValueError('expected exactly 96 pure-policy cases')
            def wrap(callback):
                def call(case,directory):
                    active.update(case=case,directory=directory)
                    try:
                        result=callback(case,directory);audit=active['audit']
                        if result['env_steps']!=audit['steps'] or result['chunk_calls']!=audit['chunk_calls']:
                            raise ValueError('recorder/audit call count differs')
                        commands=json.loads((directory/'executed_actions.json').read_text())
                        return {**result,'horizon_audit':audit,'action_continuity':continuity(commands,audit['window'])}
                    finally:active.clear()
                return call
            return original_run(root,plan,{m:wrap(cb) for m,cb in callbacks.items()},**kw)
        wanted=[(c,1000,m) for c in conditions for m in registry()]
        with ExitStack() as stack:
            stack.enter_context(patch.object(expanded,'registry',registry))
            stack.enter_context(patch.object(expanded,'cases',lambda mode,conditions:wanted))
            stack.enter_context(patch.object(batch_evaluation,'make_plan',lambda c,s,m,p:corrected_plan(original_make,c,s,m,p,receipt)))
            stack.enter_context(patch.object(batch_evaluation,'run_plan',run_plan))
            stack.enter_context(patch.object(lerobot_eval,'eval_policy',evaluate))
            state.update(phase='plus_horizons');status();expanded.run(args,'plus',h,conditions)
        state.update(status='completed',phase='completed')
    except BaseException as e:
        state.update(status='paused' if (a.output/'STOP').exists() or time.time()>=getattr(locals().get('args'),'deadline',float('inf')) else 'error',
            error=type(e).__name__+': '+str(e));raise
    finally:status()


if __name__=='__main__':main()

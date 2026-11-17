"""Frozen-weight bounded OGG loss comparisons over the audited expanded recorder."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

BASE=Path('/root/autodl-tmp/corrector')
ORDINARY=[('libero_object/task_1/init_0',1001),('libero_object/task_9/init_2',1000)]
PLUS=[('plus/cheese/sensor_noise/1',1000),('plus/soup/language_instructions/1',1000)]
METHODS=('mlp_ogg','j_ogg')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_pairs(path,domain):
    """Explicit bounded extension; original Pilot defaults remain available."""
    if path is None:return ORDINARY if domain=='ordinary' else PLUS
    import re
    selection=json.loads(Path(path).read_text())
    if selection.get('schema')!='ogg_loss_extension_selection_v1':raise ValueError('selection schema differs')
    pairs=selection['new_pairs'][domain]
    if not 1<=len(pairs)<=8:raise ValueError('extension must be a bounded selection')
    result=[]
    for condition,seed in pairs:
        if condition not in selection['conditions'][domain] or type(seed)is not int or seed not in selection['seeds']:
            raise ValueError('condition/seed outside frozen extension')
        if domain=='ordinary' and not re.fullmatch(r'libero_object/task_[0-9]/init_[0-2]',condition):
            raise ValueError('ordinary physical condition differs')
        if domain=='plus' and not condition.startswith('plus/'):raise ValueError('Plus condition differs')
        result.append((condition,seed))
    if len(result)!=len(set(result)):raise ValueError('duplicate requested condition/seed')
    reused={tuple(v) for v in selection['reused_pairs'][domain]}
    expected={(c,s) for c in selection['conditions'][domain] for s in selection['seeds']}
    if reused&set(result) or reused|set(result)!=expected:raise ValueError('new/reused condition matrix differs')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['inspect','run'],nargs='?',default='inspect')
    p.add_argument('--domain',choices=['ordinary','plus'],required=True)
    p.add_argument('--loss',choices=['cosine','mse','mse_cosine'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen-scripts',type=Path,default=BASE/'dev/expanded-2e6296eff15f/scripts')
    p.add_argument('--runtime',type=Path,default=BASE/'runtime')
    p.add_argument('--core',type=Path,default=BASE/'dev/j36m-35ecdd7488bf')
    p.add_argument('--training',type=Path,default=BASE/'j36m_longrun_20261113_001/training')
    p.add_argument('--calibration',type=Path,default=Path('/root/corrector_j36m_object_20261113_002/j_calibration.json'))
    p.add_argument('--plus-source',type=Path,default=BASE/'runtime/libero-plus-4976dc3')
    p.add_argument('--plus-config',type=Path,default=BASE/'plus_resources_20261113_001/config')
    p.add_argument('--old-plan',type=Path,default=Path('/root/corrector_plus_expanded_20261113_001/results/plus/plan.json'))
    p.add_argument('--hours',type=float,default=1)
    p.add_argument('--selection',type=Path,help='new immutable extension selection; does not edit original Pilot')
    a=p.parse_args()
    if not 0<a.hours<=2:raise ValueError('bounded pilot hours')
    from motion_pilot import frozen_audit
    pairs=selected_pairs(a.selection,a.domain);planned=len(pairs)*len(METHODS)
    receipt={'schema':'ogg_loss_pilot_v1','domain':a.domain,'loss_mode':a.loss,
        'adapter_sha256':sha(__file__),'loss_source_sha256':sha(Path(__file__).with_name('corrector_ogg_losses.py')),
        'frozen_manifest':frozen_audit(a.frozen_scripts),'cases':pairs,
        'methods':METHODS,'eta':1.,'combination':'raw feature mean MSE + 1*cosine; no rescaling',
        'joint_weights':'same frozen calibration; reliable layers renormalized identically',
        'target':'detector expected delta minus actual observed delta',
        'support':'first clean action, first seven normalized dimensions only',
        'lvm_unchanged':True,'checkpoint_read_only':True,'planned':planned}
    if a.selection:receipt['extension_selection_sha256']=sha(a.selection)
    if a.mode=='inspect':print(json.dumps(receipt,ensure_ascii=False));return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; not starting')
    if a.output.exists():
        if not(a.output/'loss_adapter.json').exists() or json.loads((a.output/'loss_adapter.json').read_text())!=json.loads(json.dumps(receipt)):
            raise FileExistsError('output/protocol conflict')
    else:a.output.mkdir(parents=True)
    lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (a.output/'loss_adapter.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:(a.output/'STOP').touch())
    sys.path.insert(0,str(a.frozen_scripts));expanded=importlib.import_module('corrector_expanded_eval')
    args=SimpleNamespace(**vars(a));args.mode=a.domain;args.deadline=time.time()+a.hours*3600
    args.j_checkpoint=a.training/'J_s42/best.pt';args.c0_checkpoint=a.training/'C0_s42/best.pt'
    state={'status':'running','phase':'loading','pid':os.getpid()}
    def status():(a.output/'status.json').write_text(json.dumps(state)+'\n')
    status()
    try:
        expanded.guard(args);h=expanded.load(args)
        import corrector_online_runtime as legacy
        from j36m import online as joint
        from corrector_ogg_losses import mlp_factory,joint_factory
        from studies.policy_generalization.corrector import batch_evaluation as batch
        original_make,original_run=batch.make_plan,batch.run_plan
        spec=expanded.registry();chosen={m:spec[m] for m in METHODS}
        old=json.loads(a.old_plan.read_text())
        all_conditions=expanded.ordinary_conditions() if a.domain=='ordinary' else old['protocol']['conditions']
        conditions={c:all_conditions[c] for c,_ in pairs}
        wanted=[(c,s,m) for c,s in pairs for m in METHODS]
        active={}
        def report(row):
            if not active:raise RuntimeError('guidance outside declared trial')
            with (active['directory']/'guidance_loss_details.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        def make_plan(cs,seeds,methods,protocol):
            methods={m:{**v,'ogg_loss_mode':a.loss} for m,v in methods.items()}
            tag='ogg_loss_extension' if a.selection else 'ogg_loss'
            protocol={**protocol,'study_id':f'replay_casebank_{tag}_{a.domain}_{a.loss}_20261116',
                'experiment_label':f'OGG Loss'+('扩展' if a.selection else '')+f' · {a.domain} · {a.loss}', 'ogg_loss':receipt}
            return original_make(cs,seeds,methods,protocol)
        def run_plan(root,plan,callbacks,**kw):
            if len(plan['cases'])!=planned:raise ValueError('pilot case count differs')
            def wrap(callback):
                def call(case,directory):
                    active.update(directory=directory,case=case)
                    try:
                        result=callback(case,directory)
                        result['ogg_loss_mode']=a.loss
                        return result
                    finally:active.clear()
                return call
            return original_run(root,plan,{m:wrap(cb) for m,cb in callbacks.items()},**kw)
        with ExitStack() as stack:
            for obj,name,value in [(legacy,'guided_velocity',mlp_factory(legacy.guided_velocity,a.loss,report)),
                (joint,'guided_velocity',joint_factory(joint.guided_velocity,a.loss,report)),
                (expanded,'registry',lambda:chosen),(expanded,'cases',lambda *_:wanted),
                (batch,'make_plan',make_plan),(batch,'run_plan',run_plan)]:
                stack.enter_context(patch.object(obj,name,value))
            state.update(phase='paired_rollouts');status();expanded.run(args,a.domain,h,conditions)
        state.update(status='completed',phase='completed')
    except BaseException as exc:
        state.update(status='paused' if (a.output/'STOP').exists() else 'error',error=type(exc).__name__+': '+str(exc));raise
    finally:status()


if __name__=='__main__':main()

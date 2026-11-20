"""Full Object10 seed42 evaluation through the immutable historical benchmark."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from .baseline import load_reference,digest
from .train_monitor import read,write


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('base','code','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--layer',choices=['block_02','block_08'],required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();base=a.base;runtime=base/'runtime'
    os.environ.update(HF_HOME=str(runtime/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(runtime/'libero-config'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    sys.path.insert(0,str(runtime/'lerobot/src'))
    ref=load_reference(base/'night-code-1e80242789',runtime/'vla-corrector-reference')
    sys.path.insert(0,str(base/'night-code-1e80242789'))
    import torch
    torch.set_num_threads(4)
    from .online import load_predictor,corrector_class
    import smolvla_baseline_benchmark as bench
    from lerobot.scripts import lerobot_eval
    from robotlab_replay.motion_capture import MotionCapture,ActionDecoder
    import smolvla_libero_tools as sim
    predictor,protocol=load_predictor(ref,a.checkpoint,runtime/'vla-corrector-reference','cuda')
    if protocol['visual_layer']!=a.layer:raise ValueError('Wrong feature layer')
    extended=corrector_class(ref,a.code/'scripts/corrector_multilayer.py')
    source=a.checkpoint.parent
    if read(source/'training_status.json')['status']!='completed' or not read(source/'result.json')['checkpoint_reload_verified']:
        raise ValueError('Only a completed, reload-verified formal model is allowed')
    history=read(Path(__file__).with_name('historical_object60.json'))
    if digest(bench.__file__)!=history['protocol']['evaluation_entry_sha256']:raise ValueError('Historical benchmark source changed')
    make=bench.make_plan;run=bench.run_plan;evaluate=lerobot_eval.eval_policy
    completed_before=sum(read(x).get('status')=='completed' for x in (a.output/'trials').glob('*.json'))
    label='S2' if a.layer=='block_02' else 'S8'
    def plan(conditions,seeds,methods,pr):
        if len(conditions)!=30 or seeds!=[1000,1001]:raise ValueError('Expected fixed sixty historical conditions')
        if pr['conditions']!=history['protocol']['conditions']:raise ValueError('Historical tasks, initial states or instructions changed')
        for key in ('policy_revision','lerobot_revision','libero_revision','fps','horizon','control_mode','corrector_settings'):
            if pr[key]!=history['protocol'][key]:raise ValueError('Historical deployment protocol mismatch: '+key)
        for mode,cfg in methods.items():
            cfg.update(model_label=label+' / 原Residual MLP',feature_layer=a.layer,model_type='mlp',training_seed=42)
        pr.update(study_id='single_layer_object60_'+label+'_20261119',experiment_label=label+'完整Object10 · 原MLP控制',
            control_fps=20,score_start=24,horizon_steps=20,feature_layer=a.layer,
            adapter_sha256=digest(__file__),online_adapter_sha256=digest(Path(__file__).with_name('online.py')),
            checkpoint_sha256=digest(a.checkpoint),training_protocol_sha256=digest(source/'protocol.json'),
            limitations='Training uses stored FP16 raw layers; original MLP uses quantized Connector. Online raw encoder FP32; predictor AMP BF16.',
            prediction_retention='original pending single-action predictions retained; cooldown0; dynamic threshold')
        return make(conditions,seeds,methods,pr)
    active={};count=0
    def run_plan(root,pl,callbacks,**kw):
        def wrap(callback):
            def call(case,directory):
                nonlocal count
                active.update(case=case,directory=directory)
                result=callback(case,directory);count+=1
                if a.pilot and count>=max(0,2-completed_before):(a.output/'STOP').touch()
                return result
            return call
        return run(root,pl,{k:wrap(v) for k,v in callbacks.items()},**kw)
    def controller(policy,pred,pr,method,src,directory,**kwargs):
        return extended(policy,pred,pr,method,src,directory,layer=a.layer,diagnostics=True,**kwargs)
    def eval_policy(**kw):
        case,directory=active['case'],active['directory'];env=kw['env'];policy=kw['policy']
        snapshot=runtime/'hf/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d'
        if policy.config.num_steps!=10 or policy.config.chunk_size!=50:raise ValueError('Frozen action generation changed')
        commands=[];queue=[];step=env.step;select=policy.select_action
        def submit(action,*args,**kwargs):
            row=dict(step=len(commands),returned=False,action=action.tolist())
            commands.append(row);out=step(action,*args,**kwargs);row['returned']=True;return out
        def choose(*args,**kwargs):
            before=len(policy._queues['action']);out=select(*args,**kwargs)
            queue.append(dict(step=len(queue),before=before,after=len(policy._queues['action']),returned=True));return out
        with ExitStack() as stack:
            stack.enter_context(patch.object(env,'step',submit));stack.enter_context(patch.object(policy,'select_action',choose))
            stack.enter_context(MotionCapture(env.envs[0].unwrapped,policy,directory/'motion',ActionDecoder.from_snapshot(snapshot),
                method=label.lower()+'_'+case['method'],provenance=dict(checkpoint_sha256=protocol['checkpoint_sha256'],feature_layer=a.layer)))
            info=evaluate(**kw)
        n=len(commands)
        if n!=len(queue) or not all(x['returned'] for x in commands):raise ValueError('Incomplete control recording')
        write(directory/'controller_commands.json',commands);write(directory/'queue_trace.json',queue)
        (directory/'frame_map.jsonl').write_text(''.join(json.dumps(dict(frame=i,step=i,phase='before_action'))+'\n' for i in range(n)))
        recorder=runtime/'lerobot/src/lerobot/scripts/lerobot_eval.py'
        write(directory/'recording_evidence.json',dict(source_sha256=digest(recorder),frames=n,video_phase='before_action',
            note='Binding to this actual invocation; video/control count verified by frozen benchmark'))
        return info
    if a.pilot and completed_before>=2:return
    args=['single-layer-formal','--workdir',str(runtime),'--output',str(a.output),'--suite','libero_object',
        '--task-ids',*[str(i) for i in range(10)],'--initial-state-ids','0','1','2','--seeds','1000','1001',
        '--methods','lvm','ogg','--corrector-checkpoint',str(a.checkpoint),'--source-root',str(runtime/'vla-corrector-reference'),
        '--control-implementation','reference','--video-policy','all','--audit-mode','all','--trace-mode','all',
        '--min-free-gib','5','--stop-file',str(a.output/'STOP')]
    with ExitStack() as stack:
        stack.enter_context(patch.object(ref['runtime'],'load_predictor',lambda *x,**k:(predictor,protocol)))
        stack.enter_context(patch.object(ref['runtime'],'OnlineCorrector',controller))
        stack.enter_context(patch.object(bench,'make_plan',plan));stack.enter_context(patch.object(bench,'run_plan',run_plan))
        stack.enter_context(patch.object(lerobot_eval,'eval_policy',eval_policy));stack.enter_context(patch.object(sys,'argv',args))
        bench.main()


if __name__=='__main__':main()

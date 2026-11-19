"""Real LIBERO short rollouts for logging/motion parity, never a success benchmark."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from .baseline import digest, load_reference, model_config
from .extraction import atomic_status
from .online import corrector_class


def array_hash(value):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    import hashlib
    value=np.asarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def run(a):
    if a.output.exists():raise FileExistsError('Use a fresh acceptance directory')
    a.output.mkdir(parents=True)
    state=dict(status='running',phase='loading',planned=12,completed=0,
               formal_training_started=False,formal_success_benchmark=False)
    atomic_status(a.output/'status.json',state)
    try:
        os.environ.update(HF_HOME=str(a.base/'runtime/hf'),HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
            LIBERO_CONFIG_PATH=str(a.base/'runtime/libero-config'),
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        import subprocess
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU is occupied; no acceptance started')
        sys.path.insert(0,str(a.base/'runtime/lerobot/src'))
        reference=load_reference(a.base/'night-code-1e80242789',a.base/'runtime/vla-corrector-reference')
        import smolvla_libero_tools as sim
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.configs import LiberoEnv
        from lerobot.scripts.lerobot_eval import eval_policy
        from lerobot.utils.random_utils import set_seed
        from robotlab_replay.motion_capture import MotionCapture,ActionDecoder
        snapshot=a.base/'runtime/hf/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d'
        torch.set_num_threads(4)
        cfg=SmolVLAConfig.from_pretrained(snapshot,local_files_only=True)
        cfg.device,cfg.n_action_steps,cfg.num_steps,cfg.pretrained_path='cuda',50,10,snapshot
        cfg.load_vlm_weights=False
        policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg,local_files_only=True,strict=True).to('cuda').eval().requires_grad_(False)
        torch.set_num_threads(4)
        from collections import Counter
        wrapper=policy.model.vlm_with_expert
        dtype_audit={name:dict(Counter(str(p.dtype) for p in module.parameters() if p.is_floating_point()))
            for name,module in [('wrapper',wrapper),('vision',wrapper.get_vlm_model().vision_model),
                                ('connector',wrapper.get_vlm_model().connector),('expert',wrapper.lm_expert)]}
        atomic_status(a.output/'policy_dtype_audit.json',dtype_audit)
        pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=snapshot,
                      preprocessor_overrides={'device_processor':{'device':'cuda'}})
        decoder=ActionDecoder.from_snapshot(snapshot)
        extended=corrector_class(reference,a.code/'scripts/corrector_multilayer.py')
        results=[]
        for layer in ('block_02','block_08'):
            checkpoint=a.probe/(layer+'_discarded_probe_weights.pt')
            gpu_record=json.loads((a.probe/'result.json').read_text())
            matched=next(r for r in gpu_record['reports'] if r['layer']==layer)
            if digest(checkpoint)!=matched['checkpoint_sha256']:raise ValueError('Probe weights changed')
            cls=reference['models'].precision_class(reference['model_cls'],'amp-bfloat16')
            predictor=cls(model_config(reference)).cuda().eval().requires_grad_(False)
            predictor.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
            protocol=dict(h_window=1,anchor_history=3,k_step=10,fps=10.,prediction_seconds=1.,visual_layer=layer)
            for mode in ('lvm','ogg'):
                paired=[]
                for variant in ('reference_raw','diagnostics_off','motion_on'):
                    directory=a.output/(layer+'_'+mode+'_'+variant);directory.mkdir()
                    state.update(phase='real_control_parity',layer=layer,method=mode,variant=variant)
                    atomic_status(a.output/'status.json',state)
                    policy.config.n_action_steps=50;policy.reset()
                    env_cfg=LiberoEnv(task='libero_object',task_ids=[1],fps=20,episode_length=80,
                        observation_height=256,observation_width=256,control_mode='relative',
                        is_libero_plus=False,hard_reset=False,init_states=True,
                        camera_name_mapping={'agentview_image':'camera1','robot0_eye_in_hand_image':'camera2'})
                    env=sim.make_vector(env_cfg);env_pre,env_post=env_cfg.get_env_processors()
                    raw=env.envs[0].unwrapped;raw.init_state_id=0
                    set_seed(1000)
                    if variant=='reference_raw':
                        # Baseline state/queue/guidance methods remain inherited verbatim.
                        # The only substitution is the necessary raw-layer encode.
                        baseline_cls=type('OriginalControllerRawObservation',(reference['runtime'].OnlineCorrector,),
                                          {'encode':extended.encode,'layer':layer})
                        controller=baseline_cls(policy,predictor,protocol,mode,a.base/'runtime/vla-corrector-reference',directory)
                    else:
                        controller=extended(policy,predictor,protocol,mode,a.base/'runtime/vla-corrector-reference',directory,
                                             layer=layer,diagnostics=variant=='motion_on')
                    trajectory=[];noises=[];chunks=[];first=[]
                    original_step,original_noise,original_chunk=env.step,policy.model.sample_noise,policy._get_action_chunk
                    def step(action,*args,**kwargs):
                        before=array_hash(raw._env.get_sim_state())
                        result=original_step(action,*args,**kwargs)
                        trajectory.append(dict(before=before,command=array_hash(action),after=array_hash(raw._env.get_sim_state())))
                        return result
                    def noise(*args,**kwargs):
                        value=original_noise(*args,**kwargs);noises.append(array_hash(value));return value
                    def chunk(*args,**kwargs):
                        value=original_chunk(*args,**kwargs);chunks.append(array_hash(value));return value
                    def observe(batch):
                        value=pre(batch)
                        if not first:first.append({k:array_hash(v) for k,v in value.items() if isinstance(v,torch.Tensor)})
                        return value
                    begin=time.perf_counter()
                    try:
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(env,'step',step))
                            stack.enter_context(patch.object(policy.model,'sample_noise',noise))
                            stack.enter_context(patch.object(policy,'_get_action_chunk',chunk))
                            stack.enter_context(controller.installed())
                            if variant=='motion_on':
                                stack.enter_context(MotionCapture(raw,policy,directory/'motion',decoder,method=mode,
                                    provenance=dict(scope='engineering parity only',layer=layer,checkpoint_sha256=digest(checkpoint))))
                            info=eval_policy(env=env,policy=policy,env_preprocessor=env_pre,env_postprocessor=env_post,
                                preprocessor=observe,postprocessor=post,n_episodes=1,max_episodes_rendered=1,
                                videos_dir=directory/'videos',start_seed=1000)
                        rows=[json.loads(s) for s in (directory/'monitor_steps.jsonl').read_text().splitlines()]
                        common_keys=('step','score','threshold','trigger','debug_forced','observed_norm',
                                     'expected_norm','interrupted','discarded_actions','denoise','guidance_loss')
                        common=[{k:r[k] for k in common_keys if k in r} for r in rows if 'score' in r or 'guidance_loss' in r]
                        if sum(r.get('score') is not None for r in rows)<15:raise ValueError('Too few real due predictions for LVM parity')
                        videos=[sim.verify_video(p,20) for p in info['video_paths']]
                        if len(videos)!=1 or videos[0]['frames']!=len(trajectory):raise ValueError('Video/control frame mismatch')
                        item=dict(layer=layer,mode=mode,variant=variant,steps=len(trajectory),
                            first_input=first,trajectory=trajectory,noises=noises,chunks=chunks,trace=common,
                            threshold_state=plain(asdict(controller.state)),events=controller.events,
                            guidance_calls=controller.guidance_calls,pending=[(p[0],array_hash(p[1]),array_hash(p[2])) for p in controller.pending],
                            target=array_hash(controller.target_now) if controller.target_now is not None else None,
                            video_sha256=digest(info['video_paths'][0]),seconds=time.perf_counter()-begin,
                            engineering_weights_sha256=digest(checkpoint))
                        if variant=='motion_on':
                            motion=json.loads((directory/'motion/status.json').read_text())
                            if motion['status']!='completed' or motion['steps']!=len(trajectory):raise ValueError('Motion/control count mismatch')
                        atomic_status(directory/'parity_trace.json',item)
                        if paired:
                            for key in ('steps','first_input','trajectory','noises','chunks','trace','threshold_state','events',
                                        'guidance_calls','pending','target','video_sha256'):
                                if item[key]!=paired[0][key]:raise ValueError('Real GPU parity mismatch: '+layer+'/'+mode+'/'+variant+'/'+key)
                        paired.append(item)
                        state['completed']+=1;atomic_status(a.output/'status.json',state)
                    finally:env.close()
                results.append(dict(layer=layer,method=mode,variants=3,steps=paired[0]['steps'],
                    events=paired[0]['events'],guidance_calls=paired[0]['guidance_calls'],
                    exact_controls_states_inputs_noise_chunks_scores_thresholds_targets_queues_and_video=True,
                    durations_s=[p['seconds'] for p in paired]))
            del predictor
        result=dict(status='completed',scope='real LIBERO short engineering parity, not a robot performance benchmark',
            task='libero_object/task_1/init_0',seed=1000,cap_steps=80,results=results,
            weights='discarded one-step engineering probe models; not formally trained S2/S8',
            lvm_state_parity=True,capture_on_off_parity=True,extra_physics_steps=False,
            adapter_sha256=digest(__file__),online_adapter_sha256=digest(Path(__file__).with_name('online.py')),
            motion_capture_sha256=digest(a.code/'tools/robotlab_replay/motion_capture.py'))
        atomic_status(a.output/'result.json',result)
        state.update(status='completed',phase='parity_verified');atomic_status(a.output/'status.json',state)
    except BaseException as error:
        state.update(status='error',error=type(error).__name__+': '+str(error));atomic_status(a.output/'status.json',state);raise


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('base','code','probe','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a)


if __name__=='__main__':main()

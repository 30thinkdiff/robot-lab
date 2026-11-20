"""Six fixed S2/S8 seed runs; frozen trainer, epoch-boundary STOP and explicit resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .train_monitor import read,write,status,alive


def jobs():
    return [dict(name='%s_s%d'%(label,seed),layer=layer,seed=seed)
            for seed in (42,43,44) for label,layer in [('S2','block_02'),('S8','block_08')]]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def child_environment():
    env=dict(os.environ);tools=str(Path(__file__).resolve().parents[1])
    env.update(PYTHONPATH=tools+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
        GIT_OPTIONAL_LOCKS='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def validate_config(cfg,allow_pending=False):
    expected=jobs()[:2] if cfg.get('training_scope')=='seed42_only' else jobs()
    if cfg.get('training_scope') not in (None,'seed42_only') or cfg['jobs']!=expected:
        raise ValueError('Only the original six runs or the explicit seed42 pair are authorized')
    for path,sha in cfg['pinned_files'].items():
        if digest(path)!=sha:raise ValueError('Training queue source/data receipt changed: '+path)
    validation=read(cfg['validation_receipt'])
    if validation is None and allow_pending:return False
    if not validation or validation.get('status')!='completed' or (validation['episodes'],validation['frames'],validation['train_pairs'],validation['validation_pairs'])!=(454,66984,49447,12089):
        raise ValueError('Full454 input validation has not completed successfully')
    if validation['manifest_sha256']!=digest(cfg['manifest']):raise ValueError('Validation/manifest mismatch')
    for key,name in [('data_adapter_sha256','data.py'),('training_adapter_sha256','training.py')]:
        if validation[key]!=digest(Path(__file__).with_name(name)):raise ValueError('Validation adapter differs: '+name)
    binding=read(Path(cfg['output'])/'VALIDATION_BINDING.json')
    if binding and binding['sha256']!=digest(cfg['validation_receipt']):raise ValueError('Validated data receipt changed')
    return True


def attempt(root,job):
    directory=root/'runs'/job['name'];directory.mkdir(parents=True,exist_ok=True)
    old=read(directory/'index.json')
    if old:
        output=Path(old['output']);saved=read(output/'training_status.json');result=read(output/'result.json')
        if saved and saved.get('status')=='completed' and result and result.get('checkpoint_reload_verified'):
            receipt=read(output/'QUEUE_COMPLETION.json')
            if receipt:
                for name,sha in receipt['files_sha256'].items():
                    if digest(output/name)!=sha:raise ValueError('Completed training artifact changed: '+name)
            return output,False,True
        if (output/'latest.pt').is_file() and (output/'protocol.json').is_file():return output,True,False
        # Explicit queue resume may retry a startup which has no recoverable
        # epoch. Old partial files remain in the previous numbered attempt.
        number=old['attempt']+1
    else:number=1
    output=directory/('attempt_%03d'%number)
    if output.exists():raise FileExistsError('Do not overwrite an old training attempt')
    write(directory/'index.json',dict(job=job,attempt=number,output=str(output),
        startup_retry_reason='no complete latest checkpoint; previous attempt preserved' if old else None))
    return output,False,False


def fit_one(cfg,name,output,resume):
    validate_config(cfg);job=next(j for j in cfg['jobs'] if j['name']==name)
    from .baseline import load_reference
    from .training import fit
    from .train_monitor import observe
    from . import data
    from unittest.mock import patch
    reference=load_reference(Path(cfg['base'])/'night-code-1e80242789',Path(cfg['base'])/'runtime/vla-corrector-reference')
    # Observer delegates original tensor operations and retains scalar counters
    # only. A separate regression checks parameters/gradients/RNG equality.
    verify=data.verify_episode;began=time.perf_counter();count=0;frames=0
    def counted(spec,layer):
        nonlocal count,frames
        result=verify(spec,layer);count+=1;frames+=spec['frames']
        elapsed=time.perf_counter()-began
        write(output.parent/(output.name+'_input_check.json'),dict(status='completed' if count==454 else 'running',
            episodes_checked=count,planned=454,frames_checked=frames,frames_total=66984,
            eta_seconds=max(0,66984-frames)/(frames/elapsed),layer=layer))
        return result
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(data,'verify_episode',counted))
        stack.enter_context(observe(reference,output))
        if cfg.get('pipeline'):
            if cfg['pipeline'].get('kind')=='packed_fp16_transport':
                from .packed_pipeline import require_acceptance,loader_override
            else:
                from .fast_pipeline import require_acceptance,loader_override
            from . import training
            pipeline=cfg['pipeline'];require_acceptance(pipeline,cfg['manifest'])
            original_arguments=training.arguments
            def arguments(*args,**kwargs):
                value=original_arguments(*args,**kwargs);value.loader_workers=pipeline['workers'];return value
            stack.enter_context(patch.object(training,'arguments',arguments))
            stack.enter_context(loader_override(reference,pipeline['workers']))
        if cfg.get('execution'):
            if (cfg.get('pipeline') or {}).get('kind')!='packed_fp16_transport':
                raise ValueError('Execution optimization requires the accepted lossless packed pipeline')
            from .execution_gate import require_acceptance as execution_acceptance
            from .execution_pipeline import execution_override
            execution_acceptance(cfg['execution'],cfg['manifest'])
            stack.enter_context(execution_override(reference,**cfg['execution']['selected']))
        fit(reference,cfg['manifest'],job['layer'],output,gates=cfg['gates'][job['layer']],
            source_root=Path(cfg['base'])/'runtime/vla-corrector-reference',seed=job['seed'],
            microbatch=32,device='cuda',resume=resume)


def supervise(cfg,root):
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='running',phase='checking',pid=os.getpid(),jobs=cfg['jobs'],
                   runs_planned=len(cfg['jobs']),runs_completed=0,child_pid=None)
        if cfg.get('execution'):state['execution_policy']=cfg['execution']['selected']
        def save():
            state.update(updated_unix=time.time(),data_free_gib=shutil.disk_usage('/root/autodl-tmp').free/2**30,
                         system_free_gib=shutil.disk_usage('/').free/2**30)
            write(root/'queue_status.json',state)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(root/'STOP').touch())
        save()
        try:
            while not validate_config(cfg,allow_pending=True):
                if (root/'STOP').exists():state.update(status='paused',reason='operator_stop');save();return
                validation=read(Path(cfg['validation_receipt']).with_name('status.json'))
                if validation and validation.get('status')=='error':raise RuntimeError('Input validation failed: '+str(validation))
                if not alive(cfg.get('validation_pid')):raise RuntimeError('Input validation process exited without a complete receipt')
                state.update(phase='waiting_full_input_verification',input_validation=validation);save();time.sleep(5)
            binding=root/'VALIDATION_BINDING.json'
            if not binding.exists():write(binding,dict(sha256=digest(cfg['validation_receipt']),path=cfg['validation_receipt']))
            state.pop('input_validation',None)
            for job in cfg['jobs']:
                if (root/'STOP').exists():
                    state.update(status='paused',reason='operator_stop');save();return
                if min(shutil.disk_usage('/root/autodl-tmp').free,shutil.disk_usage('/').free)<7*2**30:
                    state.update(status='paused',reason='disk_reserve_plus_checkpoint_room');save();return
                output,resume,done=attempt(root,job)
                if done:state['runs_completed']+=1;save();continue
                busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if busy:raise RuntimeError('GPU occupied by another process; no training launched: '+busy)
                if resume and (output/'STOP').exists():
                    signals=root/'signals';signals.mkdir(exist_ok=True)
                    (output/'STOP').rename(signals/(job['name']+'_STOP_'+str(time.time_ns())))
                logs=root/'logs';logs.mkdir(exist_ok=True)
                logpath=logs/(job['name']+'_'+str(time.time_ns())+'.log')
                cmd=[sys.executable,'-u','-m','robotlab_single_layer.train_queue','fit','--config',str(root/'queue_config.json'),
                     '--job',job['name'],'--output',str(output)]
                if resume:cmd.append('--resume-fit')
                with logpath.open('x') as log:
                    process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                             env=child_environment())
                    state.update(phase='input_verification_or_model_initialization',current_job=job,current_output=str(output),child_pid=process.pid,
                                 current_log=str(logpath),resume=resume)
                    save()
                    while process.poll() is None:
                        if min(shutil.disk_usage('/root/autodl-tmp').free,shutil.disk_usage('/').free)<5*2**30:
                            (root/'STOP').touch();state['reason']='disk_below_5gib'
                        if (root/'STOP').exists() and output.is_dir():
                            (output/'STOP').touch();state['phase']='pausing_at_epoch_boundary'
                        elif (output/'live_progress.json').exists():
                            live=read(output/'live_progress.json')
                            if live:state['phase']=live['phase']
                        save();time.sleep(5)
                state['child_pid']=None
                if process.returncode:raise RuntimeError('Training exited %s; see %s'%(process.returncode,logpath))
                saved=read(output/'training_status.json');result=read(output/'result.json')
                if saved and saved.get('status')=='paused':
                    state.update(status='paused',reason=saved.get('reason'));save();return
                if not result or not result.get('checkpoint_reload_verified') or not saved or saved.get('status')!='completed':
                    raise RuntimeError('Training child ended without verified best-checkpoint result')
                if result['parameters']!=37266176 or result['split']['samples']!={'train':49447,'validation':12089}:
                    raise ValueError('Formal parameter/sample counts differ from plan')
                files=('latest.pt','best_model.pt','protocol.json','result.json','history.json','split_indices.npz')
                write(output/'QUEUE_COMPLETION.json',dict(status='completed',job=job,
                    files_sha256={name:digest(output/name) for name in files},checkpoint_reload_verified=True))
                state['runs_completed']+=1;save()
            state.update(status='completed',phase='all_planned_trainings_verified',current_output=None);save()
        except BaseException as e:
            state.update(status='error',error=type(e).__name__+': '+str(e));save();raise


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['start','resume','status','pause','worker','fit'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--job');p.add_argument('--output',type=Path)
    p.add_argument('--resume-fit',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['output'])
    if a.mode=='status':print(json.dumps(status(root),ensure_ascii=False,indent=2));return
    if a.mode=='pause':
        root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();print('已请求在完整epoch边界暂停队列');return
    if a.mode=='fit':fit_one(cfg,a.job,a.output,a.resume_fit);return
    if a.mode=='worker':supervise(cfg,root);return
    validate_config(cfg,allow_pending=True)
    if a.mode=='start' and root.exists():raise FileExistsError('Use a new training queue directory or explicit resume')
    root.mkdir(parents=True,exist_ok=True)
    old=read(root/'queue_config.json')
    if old and old!=cfg:raise ValueError('Cannot resume with a changed queue configuration')
    for file in ('launcher.json','queue_status.json'):
        record=read(root/file)
        if record and alive(record.get('pid')):raise RuntimeError('Existing queue process alive')
    if old:
        state=read(root/'queue_status.json') or {}
        if alive(state.get('child_pid')):raise RuntimeError('Existing training child still alive')
    if (root/'STOP').exists():
        signals=root/'signals';signals.mkdir(exist_ok=True)
        (root/'STOP').rename(signals/('queue_STOP_'+str(time.time_ns())))
    if not old:write(root/'queue_config.json',cfg)
    logpath=root/('queue_'+str(time.time_ns())+'.log')
    cmd=[sys.executable,'-u','-m','robotlab_single_layer.train_queue','worker','--config',str(root/'queue_config.json')]
    with logpath.open('x') as log:process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        env=child_environment(),start_new_session=True)
    receipt=dict(pid=process.pid,log=str(logpath),started_unix=time.time(),config=str(root/'queue_config.json'))
    write(root/'launcher.json',receipt);print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()

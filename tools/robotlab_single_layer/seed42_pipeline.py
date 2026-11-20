"""Explicit two-model training completion -> frozen Object60 evaluation handoff."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .baseline import digest
from .train_monitor import read,write,alive


def trained(root,name):
    index=read(root/'runs'/name/'index.json')
    if not index:return None
    path=Path(index['output']);status=read(path/'training_status.json');result=read(path/'result.json')
    if not status or status['status']!='completed':return None
    if not result or not result.get('checkpoint_reload_verified'):raise RuntimeError('Completed model failed reload validation')
    return path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['run','status','pause','resume'])
    p.add_argument('--config',required=True,type=Path);a=p.parse_args();cfg=read(a.config);root=Path(cfg['output'])
    root.mkdir(exist_ok=True);training=Path(cfg['training_output']);manager=cfg['training_manager']
    if a.mode=='status':
        value=read(root/'status.json') or {};value['process_alive']=alive(value.get('pid'))
        value['training']=read(training/'queue_status.json')
        value['evaluations']={k:read(root/k/'execution.json') for k in ('S2','S8')}
        print(json.dumps(value,ensure_ascii=False,indent=2));return
    if a.mode=='pause':
        (root/'STOP').touch();print('Requested epoch/episode boundary pause');return
    if a.mode=='resume':
        old=read(root/'status.json') or {}
        if alive(old.get('pid')):raise RuntimeError('Pipeline already alive')
        if (root/'STOP').exists():(root/'STOP').rename(root/('STOP_'+str(time.time_ns())))
        env=dict(os.environ,PYTHONPATH=cfg['code']+'/tools',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        log=root/('pipeline_'+str(time.time_ns())+'.log')
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-m','robotlab_single_layer.seed42_pipeline','run','--config',str(a.config)],
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(log))));return
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='running',phase='training_seed42_pair',pid=os.getpid(),restarts=0)
        def save():state['updated_unix']=time.time();write(root/'status.json',state)
        def child(cmd,log):
            with log.open('a') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;save()
            while proc.poll() is None:
                if (root/'STOP').exists():
                    for label in ('S2','S8'):
                        d=root/label
                        if d.exists():(d/'STOP').touch()
                time.sleep(5)
            state['child_pid']=None;save()
            if proc.returncode:raise RuntimeError('Child failed; preserve '+str(log))
        try:
            for name,sha in cfg['pinned_files'].items():
                if digest(name)!=sha:raise ValueError('Source/config changed: '+name)
            while not all(trained(training,n) for n in ('S2_s42','S8_s42')):
                q=read(training/'queue_status.json') or {}
                if (root/'STOP').exists():
                    subprocess.run([sys.executable,manager,'pause'],check=True)
                    state.update(status='paused',phase='training_pause_requested');save();return
                if q.get('status')=='error':raise RuntimeError('Training error; '+str(q))
                if not alive(q.get('pid')) and not alive(q.get('child_pid')):
                    if state['restarts']>=2:raise RuntimeError('Automatic restart budget exhausted; explicit resume required')
                    subprocess.run([sys.executable,manager,'resume'],check=True);state['restarts']+=1
                save();time.sleep(30)
            q=read(training/'queue_status.json') or {}
            while alive(q.get('pid')) or alive(q.get('child_pid')):
                time.sleep(5);q=read(training/'queue_status.json') or {}
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Unexpected GPU process')
            for label,layer in [('S2','block_02'),('S8','block_08')]:
                if (root/'STOP').exists():state.update(status='paused');save();return
                src=trained(training,label+'_s42');checkpoint=src/'best_model.pt';sha=digest(checkpoint)
                old=read(root/(label+'_checkpoint.json'))
                if old and old['sha256']!=sha:raise ValueError('Evaluation checkpoint changed')
                write(root/(label+'_checkpoint.json'),dict(path=str(checkpoint),sha256=sha))
                stop=root/label/'STOP'
                if stop.exists():stop.rename(root/(label+'_STOP_'+str(time.time_ns())))
                state.update(phase=label+'_formal_object60_lvm_ogg');save()
                child([cfg['sim_python'],'-u','-m','robotlab_single_layer.formal_eval','--base',cfg['base'],'--code',cfg['code'],
                    '--checkpoint',str(checkpoint),'--layer',layer,'--output',str(root/label)],root/(label+'_evaluation.log'))
                result=read(root/label/'execution.json')
                if result['status']=='paused':state.update(status='paused');save();return
                if result['status']!='completed' or result['completed']!=120 or result['errors']!=0:raise ValueError('Evaluation incomplete')
                if digest(checkpoint)!=sha:raise ValueError('Frozen model changed during evaluation')
            history=read(Path(__file__).with_name('historical_object60.json'))['trials']
            rows=list(history)
            for label in ('S2','S8'):
                for trial in (root/label/'trials').glob('*.json'):
                    r=read(trial)
                    if r['status']=='completed':
                        c=r['case'];v=r['result'];rows.append(dict(condition=c['condition'],seed=c['seed'],method=label.lower()+'_'+c['method'],success=v['success'],env_steps=v['env_steps']))
            if len(rows)!=480:raise ValueError('Expected four historical plus four new methods, sixty each')
            write(root/'comparison.json',dict(rows=rows,summary={m:dict(completed=sum(r['method']==m for r in rows),successes=sum(r['success'] for r in rows if r['method']==m)) for m in sorted({r['method'] for r in rows})},historical_rows_reused=240,new_rows=240))
            state.update(status='completed',phase='all_240_evaluations_complete');save()
        except BaseException as e:
            state.update(status='error',error=type(e).__name__+': '+str(e));save();raise


if __name__=='__main__':main()

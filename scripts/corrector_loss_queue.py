"""Bounded sequential loss pilot, waiting for the authorized Horizon queue."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from corrector_gpu_followup_queue import read,write,sha,verify_complete


def verify_cosine_reference(phase, reference):
    """Require identical actual controls/queues under an unchanged cosine path."""
    def records(root):
        rows=[read(p) for p in (Path(root)/'trials').glob('*.json')]
        return {(r['case']['condition'],r['case']['seed'],r['case']['method']):r for r in rows if r['status']=='completed'}
    old=records(reference);new=records(phase);checks=[]
    for key,r in new.items():
        other=old[key]
        left=Path(phase)/'attempts'/r['case']['id']/('attempt_%03d'%r['attempt'])
        right=Path(reference)/'attempts'/other['case']['id']/('attempt_%03d'%other['attempt'])
        if any(r['result'][k]!=other['result'][k] for k in ['success','env_steps','chunk_calls','event_replans','guidance_calls']):
            raise ValueError('cosine baseline result changed')
        for name in ['executed_actions.json','queue_trace.json']:
            if read(left/name)!=read(right/name):raise ValueError('cosine baseline actual command/queue changed')
        checks.append({'condition':key[0],'seed':key[1],'method':key[2],'controls_and_queue_exact':True})
    return checks


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;plan=read(root/'plan.json');lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in plan['sources'].items():
        if sha(name)!=h:raise ValueError('source changed: '+name)
    state={'status':'running','phase':'waiting_horizon_queue','pid':os.getpid(),'child_pid':None,'completed_jobs':{}}
    stop=root/'STOP';deadline=plan['deadline_unix'];child=None;job=None
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.touch())
    try:
        previous=Path(plan['wait_for'])
        while True:
            old=read(previous/'queue_status.json')
            if old['status']=='completed':
                for sub,expected in [('motion_representatives/ordinary',12),('plus_horizons/plus',96)]:
                    execution=read(previous/sub/'execution.json')
                    if execution['status']!='completed' or execution['completed']!=expected or execution['errors']:
                        raise RuntimeError('preceding completion gate differs')
                break
            if old['status']!='running':raise RuntimeError('preceding queue paused/errored; no launch')
            if stop.exists() or time.time()>=deadline:state.update(status='paused',reason='STOP/budget');return
            write(root/'status.json',state);time.sleep(10)
        for job in plan['jobs']:
            state.update(phase=job['name'],execution=str(Path(job['phase_root'])/'execution.json'),log=job['log'])
            path=Path(job['phase_root'])/'execution.json'
            if path.exists() and read(path)['status']=='completed':
                state['completed_jobs'][job['name']]=verify_complete(job);continue
            if stop.exists() or time.time()>=deadline:state.update(status='paused',reason='STOP/budget');break
            import shutil
            if any(shutil.disk_usage(p).free<5*2**30 for p in ['/root','/root/autodl-tmp']):
                raise RuntimeError('disk reserve below 5GiB')
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU occupied; no concurrent launch')
            env={**os.environ,'PYTHONPATH':'/root/autodl-tmp/corrector/runtime/lerobot/src','PYTHONUNBUFFERED':'1'}
            with Path(job['log']).open('ab') as log:
                child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=plan['code'])
            state['child_pid']=child.pid;write(root/'status.json',state)
            while child.poll() is None:
                if stop.exists() or time.time()>=deadline:
                    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
                if path.exists():state['progress']=read(path)
                write(root/'status.json',state);time.sleep(3)
            state['child_pid']=None
            if stop.exists() or time.time()>=deadline:state.update(status='paused',reason='STOP/budget');break
            if child.returncode:raise RuntimeError('child failed; inspect '+job['log'])
            state['completed_jobs'][job['name']]=verify_complete(job);state.pop('progress',None)
            if job.get('cosine_reference'):
                checks=verify_cosine_reference(job['phase_root'],job['cosine_reference'])
                write(root/(job['name']+'_equivalence.json'),{'status':'passed','checks':checks})
        else:state.update(status='completed',phase='completed')
    except BaseException as exc:
        state.update(status='error',error=type(exc).__name__+': '+str(exc))
        if child is not None and child.poll() is None and job:
            output=Path(job['output']);output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
        raise
    finally:write(root/'status.json',state)


if __name__=='__main__':main()

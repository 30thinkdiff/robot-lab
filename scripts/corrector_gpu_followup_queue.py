"""Sequential bounded GPU follow-up; immutable job JSON, STOP at episode boundary."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def verify_complete(job):
    phase=Path(job['phase_root']);execution=read(phase/'execution.json')
    if execution['status']!='completed' or execution['completed']!=job['planned'] or execution.get('errors',0):
        raise RuntimeError('phase not fully completed: '+job['name'])
    records=[read(p) for p in (phase/'trials').glob('*.json')]
    if len(records)!=job['planned'] or any(r['status']!='completed' for r in records):
        raise RuntimeError('trial count/status differs')
    for r in records:
        for v in r['video_artifacts']:
            p=(phase/v['path']).resolve();p.relative_to(phase.resolve())
            if sha(p)!=v['sha256']:raise ValueError('video SHA differs')
        directory=phase/'attempts'/r['case']['id']/('attempt_%03d'%r['attempt'])
        for name,h in r['result'].get('motion_capture',{}).get('files_sha256',{}).items():
            p=(directory/name).resolve();p.relative_to(directory.resolve())
            if sha(p)!=h:raise ValueError('motion SHA differs')
    return {'completed':len(records),'errors':0,'video_and_motion_hashes_verified':True}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['run','status']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;plan=read(root/'followup_plan.json')
    if a.mode=='status':
        print(json.dumps(read(root/'queue_status.json'),ensure_ascii=False,indent=2));return
    lock=(root/'followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(root/'motion_equivalence.json')['status']!='passed':raise RuntimeError('motion equivalence gate has not passed')
    for name,h in plan['sources'].items():
        if sha(name)!=h:raise ValueError('source changed: '+name)
    state={'status':'running','phase':'preflight','pid':os.getpid(),'child_pid':None,'completed_jobs':{},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    stop=root/'STOP';deadline=time.time()+plan['max_hours']*3600
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.touch())
    child=None;job=None
    try:
        for job in plan['jobs']:
            if stop.exists() or time.time()>=deadline:state.update(status='paused',reason='STOP or budget');break
            if Path(job['phase_root'],'execution.json').exists():
                execution=read(Path(job['phase_root'])/'execution.json')
                if execution['status']=='completed':
                    state['completed_jobs'][job['name']]=verify_complete(job);continue
            for disk in ['/root','/root/autodl-tmp']:
                import shutil
                if shutil.disk_usage(disk).free<5*2**30:raise RuntimeError('disk reserve below 5GiB: '+disk)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if apps:raise RuntimeError('GPU occupied before next stage; no duplicate launch')
            state.update(phase=job['name'],log=job['log'],execution=str(Path(job['phase_root'])/'execution.json'))
            env=os.environ.copy();env['PYTHONPATH']='/root/autodl-tmp/corrector/runtime/lerobot/src';env['PYTHONUNBUFFERED']='1'
            with Path(job['log']).open('ab') as log:child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=plan['code'])
            state['child_pid']=child.pid;write(root/'queue_status.json',state)
            while child.poll() is None:
                if stop.exists() or time.time()>=deadline:
                    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
                path=Path(job['phase_root'])/'execution.json'
                if path.exists():state['progress']=read(path)
                write(root/'queue_status.json',state);time.sleep(5)
            state.update(child_pid=None,returncode=child.returncode)
            if child.returncode:
                if stop.exists():state.update(status='paused',reason='STOP');break
                raise RuntimeError(job['name']+' exited '+str(child.returncode)+'; inspect '+job['log'])
            state['completed_jobs'][job['name']]=verify_complete(job)
            state.pop('progress',None);write(root/'queue_status.json',state)
        else:state.update(status='completed',phase='completed')
    except BaseException as e:
        state.update(status='error',error=type(e).__name__+': '+str(e))
        if child is not None and child.poll() is None and job:
            out=Path(job['output']);out.mkdir(parents=True,exist_ok=True);(out/'STOP').touch()
        raise
    finally:write(root/'queue_status.json',state)


if __name__=='__main__':main()

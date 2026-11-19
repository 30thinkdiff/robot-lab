"""Explicit extraction job operations: start/resume/status/pause, no training auto-start."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .progress import status, process_alive


def worker_environment():
    """A fresh child Python does not inherit the launcher's sys.path edits."""
    env=dict(os.environ)
    tools=str(Path(__file__).resolve().parents[1])
    previous=[p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and p!=tools]
    env['PYTHONPATH']=os.pathsep.join([tools,*previous])
    env.setdefault('OMP_NUM_THREADS','4')
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    return env


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['start','resume','worker','status','pause'])
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['output'])
    if a.mode=='status':print(json.dumps(status(root),ensure_ascii=False,indent=2));return
    if a.mode=='pause':
        root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch()
        print(json.dumps(dict(pause_requested=True,boundary='complete verified episode',root=str(root))));return
    if a.mode in ('start','resume'):
        saved=status(root)
        if saved.get('process_alive') and saved.get('status') not in ('completed','error','paused'):
            raise RuntimeError('Existing worker is still alive; do not launch a duplicate')
        launch_path=root/'launcher.json'
        if launch_path.exists() and process_alive(json.loads(launch_path.read_text()).get('pid')):
            raise RuntimeError('An existing launch is still alive')
        root.mkdir(parents=True,exist_ok=True)
        if a.mode=='start' and (root/'JOB_IDENTITY.json').exists():raise FileExistsError('Existing job; use explicit resume')
        if (root/'STOP').exists():
            if a.mode!='resume':raise RuntimeError('STOP exists; use explicit resume to archive it')
            signals=root/'signals';signals.mkdir(exist_ok=True)
            (root/'STOP').rename(signals/('STOP_resumed_'+str(time.time_ns())))
        logpath=root/('worker_'+str(time.time_ns())+'.log')
        cmd=[sys.executable,'-u','-m','robotlab_single_layer.job','worker','--config',str(a.config.resolve())]
        with logpath.open('x') as log:
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=worker_environment())
        receipt=dict(pid=child.pid,log=str(logpath),command=cmd,started_unix=time.time(),config=str(a.config.resolve()))
        launch_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,ensure_ascii=False));return
    root.mkdir(parents=True,exist_ok=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(root/'STOP').touch())
    from .extraction import run
    run(cfg['manifest'],base=cfg['base'],data_root=cfg['data_root'],gates=cfg['gates'],
        output=cfg['output'],microbatch=cfg['microbatch'])


if __name__=='__main__':main()

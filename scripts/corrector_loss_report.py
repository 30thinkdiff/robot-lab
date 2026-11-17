"""Fetch a completed loss pilot through existing Replay-v1 readonly exporter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import datetime,hashlib,json,pathlib,subprocess
root=pathlib.Path(REMOTE_ROOT)
read=lambda p:json.loads(p.read_text())
status=read(root/'status.json')
if status['status']!='completed':raise RuntimeError('loss pilot not completed: '+json.dumps(status))
plan=read(root/'plan.json');folder=root/'exports';folder.mkdir(exist_ok=True)
namespace={'__name__':'readonly_export'}
exec(EXPORT_SOURCE,namespace)
data={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'plan':plan,'stages':[],'equivalence':{},'exports':[]}
for job in plan['jobs']:
 phase=pathlib.Path(job['phase_root']);out=folder/(job['name']+'.zip')
 if not out.exists():namespace['export'](phase,out)
 data['exports'].append({'stage':job['name'],'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size})
 entry={'name':job['name'],'execution':read(phase/'execution.json'),'protocol':read(phase/'plan.json')['protocol'],'trials':[]}
 for p in sorted((phase/'trials').glob('*.json')):
  r=read(p);c=r['case'];d=phase/'attempts'/c['id']/('attempt_%03d'%r['attempt'])
  monitor=[json.loads(line) for line in (d/'monitor_steps.jsonl').read_text().splitlines() if line.strip()]
  attempts=[row for row in monitor if 'guidance_loss' in row or 'ogg_loss' in row or row.get('event')=='joint_ogg']
  guide=[row for row in attempts if row.get('updated') is not False]
  details=d/'guidance_loss_details.jsonl'
  entry['trials'].append({'record':r,'actual_protocol':read(d/'actual_protocol.json'),
   'guidance':guide,'guidance_skips':[row for row in attempts if row.get('updated') is False],
   'guidance_details':[v for line in details.read_text().splitlines() if (v:=json.loads(line)).get('updated') is not False] if details.exists() else [],
   'commands':read(d/'executed_actions.json')})
 data['stages'].append(entry)
for p in root.glob('*_equivalence.json'):data['equivalence'][p.stem]=read(p)
print(json.dumps(data,ensure_ascii=False,allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--ssh-host',default='autodl');p.add_argument('--ssh-port',type=int)
    p.add_argument('--remote-root',default='/root/corrector_ogg_loss_20261116_001')
    p.add_argument('--output',type=Path,default=Path('outputs/ogg_loss_20261116'));a=p.parse_args()
    if not a.remote_root.startswith('/root/corrector_ogg_loss') or '..' in Path(a.remote_root).parts:raise ValueError('unexpected experiment root')
    remote='REMOTE_ROOT='+repr(a.remote_root)+'\nEXPORT_SOURCE='+repr(Path(__file__).with_name('corrector_replay_export.py').read_text())+'\n'+REMOTE
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if a.ssh_port:ssh+=['-p',str(a.ssh_port)]
    result=subprocess.run(ssh+[a.ssh_host,'/root/autodl-tmp/corrector/bootstrap/bin/python','-'],input=remote.encode(),capture_output=True,check=True)
    # The existing exporter prints small receipts before the final JSON line.
    data=json.loads(result.stdout.decode().splitlines()[-1]);a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/'snapshot.json'
    if manifest.exists():raise FileExistsError('use new output directory; preserve previous snapshot')
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for artifact in data['exports']:
        destination=a.output/(artifact['stage']+'.zip')
        scp=['scp','-o','BatchMode=yes']
        if a.ssh_port:scp+=['-P',str(a.ssh_port)]
        subprocess.run(scp+[a.ssh_host+':'+artifact['path'],str(destination)],check=True)
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=artifact['sha256']:raise ValueError('download SHA differs')
    print(json.dumps({'snapshot':str(manifest),'trials':sum(len(s['trials']) for s in data['stages']),
        'bytes':sum(v['bytes'] for v in data['exports']),'status':data['status']['status']}))


if __name__=='__main__':main()

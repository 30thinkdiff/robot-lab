"""Read-only metadata snapshot for Nov16 GPU follow-up; no video/model transfer."""
import argparse,datetime,json,shlex,subprocess
from pathlib import Path

REMOTE=r'''
import json,pathlib,statistics,hashlib,datetime,shutil
root=pathlib.Path('/root/corrector_gpu_followup_20261116_001')
read=lambda p:json.loads(p.read_text())
report={'queue':read(root/'queue_status.json'),'motion_equivalence':read(root/'motion_equivalence.json'),'phases':{}}
def process_matches(pid,script,output):
 try:argv=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
 except OSError:return False
 return any(pathlib.Path(a).name==script for a in argv) and str(output) in argv
report['captured_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
report['queue_process_matches']=process_matches(report['queue']['pid'],'corrector_gpu_followup_queue.py',root)
report['disk_free_gib']={p:shutil.disk_usage(p).free/2**30 for p in ['/root','/root/autodl-tmp']}
old=pathlib.Path('/root/corrector_plus_expanded_20261113_001/results/plus');refs={}
report['original_baselines']=[]
report['plus_conditions']=read(old/'plan.json')['protocol']['conditions']
for p in (old/'trials').glob('*.json'):
 r=read(p);c=r['case']
 if c['method'] in ['H50','H20'] and r['status']=='completed':
  report['original_baselines'].append({'case':c,'result':{k:v for k,v in r['result'].items() if k!='video_paths'}})
 if c['method']=='H20' and r['status']=='completed':
  d=old/'attempts'/c['id']/('attempt_%03d'%r['attempt']);refs[c['condition'],c['seed']]=read(d/'actual_protocol.json')['first']
for name,folder in [('motion_representatives',root/'motion_representatives/ordinary'),('plus_horizons',root/'plus_horizons/plus')]:
 if not(folder/'plan.json').exists():continue
 phase={'execution':read(folder/'execution.json'),'plan_sha256':hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),'cases':[]}
 for p in sorted((folder/'trials').glob('*.json')):
  r=read(p);c=r['case'];row={'case':c,'status':r['status'],'attempt':r['attempt'],'error':r.get('error')}
  if r['status']=='completed':
   result=r['result'];row['result']={k:v for k,v in result.items() if k not in ['video_paths','motion_capture']};row['video_artifacts']=r['video_artifacts']
   d=folder/'attempts'/c['id']/('attempt_%03d'%r['attempt']);actual=read(d/'actual_protocol.json');row['first_signatures']={k:actual['first'][k] for k in ['state_sha256','input_sha256','noise_sha256']}
   if name=='plus_horizons':
    expected=refs[c['condition'],c['seed']];row['matches_original_H20_signatures']=all(expected[k]==v for k,v in row['first_signatures'].items())
   if (d/'motion/steps.jsonl').exists():
    sums={'tracking_position_m':0.,'tracking_orientation_rad':0.,'nominal_position_m':0.,'nominal_orientation_rad':0.};maxima=dict(sums);n=0;updates=clipped=0
    with (d/'motion/steps.jsonl').open() as f:
     for line in f:
      s=json.loads(line);n+=1;updates+=s['controller_updates'];clipped+=s['clipped_updates']
      values={'tracking_position_m':s['tracking_error']['position_m'],'tracking_orientation_rad':s['tracking_error']['orientation_rad'],'nominal_position_m':s['nominal_block_error']['position_m'],'nominal_orientation_rad':s['nominal_block_error']['orientation_rad']}
      for k,v in values.items():sums[k]+=v;maxima[k]=max(maxima[k],v)
    row['motion_summary']={'steps':n,'controller_updates':updates,'clipped_updates':clipped,'mean':{k:v/n for k,v in sums.items()},'max':maxima}
  phase['cases'].append(row)
 methods={}
 for m in sorted({c['method'] for c in read(folder/'plan.json')['cases']}):
  completed=[r for r in phase['cases'] if r['case']['method']==m and r['status']=='completed'];metrics={}
  for key in ['env_steps','chunk_calls','policy_s','monitor_s','episode_wall_s','peak_torch_allocated_gib']:
   values=[r['result'][key] for r in completed if type(r['result'].get(key)) in [float,int]];metrics['mean_'+key]=statistics.mean(values) if values else None
  methods[m]={'completed':len(completed),'successes':sum(r['result']['success'] for r in completed),**metrics}
 phase['methods']=methods;report['phases'][name]=phase
print(json.dumps(report,ensure_ascii=False,allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--ssh-host',default='autodl');p.add_argument('--ssh-port',type=int)
    p.add_argument('--output',type=Path,default=Path('outputs/gpu_followup_20261116'));a=p.parse_args()
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15']
    if a.ssh_port:cmd+=['-p',str(a.ssh_port)]
    cmd +=[a.ssh_host,shlex.join(['/root/autodl-tmp/corrector/bootstrap/bin/python','-c',REMOTE])]
    r=subprocess.run(cmd,capture_output=True,text=True,check=True);data=json.loads(r.stdout)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/(stamp+'.json')
    with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'snapshot':str(path),'queue_status':data['queue']['status'],'phase':data['queue']['phase'],
        'phases':{k:v['execution'] for k,v in data['phases'].items()}},ensure_ascii=False))


if __name__=='__main__':main()

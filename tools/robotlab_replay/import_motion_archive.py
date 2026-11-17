"""Read-only retrieval of six existing paired action-chunk traces; no GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,io,json,pathlib,sys,zipfile
request=json.load(sys.stdin)
root=pathlib.Path('/root/corrector_plus_expanded_20261113_001/results/plus').resolve()
snapshot=pathlib.Path('/root/autodl-tmp/corrector/runtime/hf/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d')
members={};total=0;audits=[]
def include(name,p):
 global total
 if p.stat().st_size>2*2**20:raise ValueError('individual file above 2MiB')
 raw=p.read_bytes();total+=len(raw)
 if total>8*2**20:raise ValueError('8MiB total budget exceeded')
 members[name]=raw
if len(request)!=6:raise ValueError('requires the six declared paired trials')
for item in request:
 cid=item['case_id']
 if len(cid)!=20 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('invalid case id')
 trial=root/'trials'/(cid+'.json');raw=trial.read_bytes();r=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!=item['trial_sha256'] or r['status']!='completed':raise ValueError('source trial changed')
 c=r['case']
 if (c['condition'],c['seed'],c['method'])!=(item['condition'],item['seed'],item['method']):raise ValueError('case differs')
 base='attempts/'+cid+'/attempt_%03d/'%r['attempt'];d=root/base
 files=[]
 for p in sorted(d.iterdir()):
  if not p.is_file():continue
  entry={'name':p.name,'bytes':p.stat().st_size}
  if p.suffix=='.json' and p.stat().st_size<200000:
   v=json.loads(p.read_bytes());entry['keys']=list(v) if isinstance(v,dict) else list(v[0]) if v and isinstance(v[0],dict) else 'array'
  files.append(entry)
 for n in ['actual_protocol.json','queue_trace.json','chunk_calls.json','actions.json','controller_commands.json']:
  include(base+n,d/n)
 chunks=sorted((d/'chunks').glob('chunk_*.npz'))
 if not chunks or len(chunks)>40:raise ValueError('unexpected chunk count')
 for p in chunks:
  p.resolve().relative_to(root);include(base+'chunks/'+p.name,p)
 members['trials/'+cid+'.json']=raw
 audits.append({'case_id':cid,'method':c['method'],'condition':c['condition'],'seed':c['seed'],'files':files,'chunks':len(chunks),
  'motion_states_present':(d/'motion/steps.jsonl').exists(),
  'limit':'file/schema audit only; initial_sim_state.pt is a single initial snapshot, not a motion time series'})
for n in ['config.json','policy_postprocessor.json','policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
 include('processor/'+n,snapshot/n)
controllers=list(pathlib.Path('/root/autodl-tmp/corrector/runtime/sim-venv/lib').glob('python*/site-packages/robosuite/controllers/config/osc_pose.json'))
if len(controllers)!=1:raise ValueError('controller config path ambiguous')
include('controller/osc_pose.json',controllers[0])
manifest={'schema':'historical_motion_evidence_v1','remote_read_only':True,'gpu_work':False,'audit':audits,
 'members':{n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for n,v in members.items()},
 'omitted':['video already imported separately','PT/pickle, large models and visual tensors'],
 'action_semantics':'returned final policy chunk; includes guidance when method is OGG; not an unguided counterfactual'}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for n,v in members.items():z.writestr(n,v)
 z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''


def validate(archive, requested):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>9*2**20:
            raise ValueError('duplicate members or archive budget exceeded')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('unsafe member path')
            if p.suffix not in {'.json','.npz','.safetensors'} or stat.S_ISLNK(z.getinfo(name).external_attr>>16):
                raise ValueError('unknown member format/symlink')
        m=json.loads(z.read('manifest.json'))
        if set(names)!=set(m['members'])|{'manifest.json'}:raise ValueError('unknown archive member')
        for n,v in m['members'].items():
            raw=z.read(n)
            if len(raw)!=v['bytes'] or hashlib.sha256(raw).hexdigest()!=v['sha256']:
                raise ValueError('member SHA mismatch')
        for item in requested:
            raw=z.read('trials/'+item['case_id']+'.json')
            if hashlib.sha256(raw).hexdigest()!=item['trial_sha256']:raise ValueError('unrequested trial')
        if {a['case_id'] for a in m['audit']}!={i['case_id'] for i in requested}:
            raise ValueError('audit case set differs')
        return m


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--ssh-host',default='autodl');p.add_argument('--control-path',type=Path)
    a=p.parse_args()
    from data import Catalog
    c=Catalog();request=[]
    try:
        for r in c.runs.values():
            if (r['_protocol'].get('study_id')=='replay_casebank_representative_plus_20261114'
                    and r['condition']=='plus/cheese/sensor_noise/1' and r['seed']==1000):
                request.append({'case_id':r['_record']['case']['id'],'method':r['method'],'condition':r['condition'],
                    'seed':r['seed'],'trial_sha256':hashlib.sha256(r['_source'].read(r['member'])).hexdigest()})
    finally:c.close()
    if len(request)!=6:raise ValueError('requires previously verified six-method local selection')
    key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:16]
    output=HERE/'.local/motion_archive'/('historical_'+key);output.parent.mkdir(exist_ok=True)
    archive=output.with_suffix('.zip')
    if not archive.exists():
        cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30']
        if a.control_path:cmd+=['-S',str(a.control_path)]
        cmd+=[a.ssh_host,shlex.join(['/root/autodl-tmp/corrector/bootstrap/bin/python','-c',REMOTE])]
        partial=archive.with_suffix('.zip.partial')
        with partial.open('xb') as f:r=subprocess.run(cmd,input=json.dumps(request).encode(),stdout=f)
        if r.returncode:raise RuntimeError('read-only export failed; partial retained, no import')
        validate(partial,request);partial.rename(archive)
    m=validate(archive,request)
    if not output.exists():
        output.mkdir()
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                target=output/n;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(z.read(n))
    for name,entry in m['members'].items():
        target=output/name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('partial or changed local import; preserve existing files and use a new directory')
    print(json.dumps({'cases':len(m['audit']),'chunks':sum(r['chunks'] for r in m['audit']),
        'output':str(output),'archive_bytes':archive.stat().st_size,
        'motion_state_time_series':sum(r['motion_states_present'] for r in m['audit']),
        'remote_read_only':True,'gpu_work':False},ensure_ascii=False))


if __name__=='__main__':main()

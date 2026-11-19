"""Package/deploy the explicitly authorized four-hour, four-method task transfer.

Use an SSH alias or pass --host/--port. No credential or server address is stored.
Results are read-only; fetch takes a small text snapshot, not weights or videos.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
FILES = {name: REPO / 'scripts' / name for name in (
    'corrector_crosssuite_plan.py', 'corrector_crosssuite_eval.py', 'corrector_crosssuite_queue.py',
    'corrector_loss_queue.py', 'corrector_gpu_followup_queue.py')}
FILES['motion_pilot.py'] = REPO / 'tools/robotlab_replay/motion_pilot.py'
DEFAULT_ROOT = '/root/corrector_crosssuite_20261117_001'

INSTALL = r'''
import base64,hashlib,io,json,os,re,shutil,subprocess,sys,time,zipfile
from pathlib import Path
cfg=json.loads(sys.stdin.read());root=Path(cfg['root']);blob=base64.b64decode(cfg['zip'])
if hashlib.sha256(blob).hexdigest()!=cfg['zip_sha256']:raise ValueError('ZIP hash')
if root.exists():raise FileExistsError('Existing output preserved; use status or explicit resume, never reinstall')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if apps:raise RuntimeError('GPU occupied')
old=subprocess.run(['pgrep','-af',r'[c]orrector_.*[.]py|[s]molvla_.*[.]py'],text=True,capture_output=True)
if old.stdout.strip():raise RuntimeError('Old experiment/queue process exists: '+old.stdout)
for disk in ['/root','/root/autodl-tmp']:
 if shutil.disk_usage(disk).free<5*2**30:raise RuntimeError('disk reserve below 5GiB')
b=Path('/root/autodl-tmp/corrector');inventory=json.loads((Path(cfg['inventory'])).read_text())
pfile=b/'object10/night_20261109_001/runs/mlp_s42/protocol.json'
mfile=b/'object10/manifest.json';p=json.loads(pfile.read_text());m=json.loads(mfile.read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
weight=b/'object10/night_20261109_001/runs/mlp_s42/best_model.pt'
if sha(weight)!='05d55ba9452142be66840b3bef6fad10226480b6115be4bbbcf0842116732fad':raise ValueError('MLP checkpoint differs')
if (p['model_type'],p['seed'],p['fps'],p['k_step'],p['prediction_seconds'],p['anchor_history'])!=('mlp',42,10.0,10,1.0,3):raise ValueError('MLP timing/type changed')
by_id={r['episode']:r for r in m['records']}
records=[dict(episode=r['episode'],split=r['split'],task=by_id[r['episode']]['task']) for r in p['records']]
canon=lambda x:' '.join(re.sub('[^a-z0-9]+',' ',x.lower()).split())
object_tasks={canon(t['language']) for t in inventory['libero_object']['tasks']}
if {canon(r['task']) for r in records}!=object_tasks:raise ValueError('Corrector records not exactly Object10')
if any(canon(t['language']) in object_tasks for suite in ['libero_spatial','libero_goal','libero_10','libero_90'] for t in inventory[suite]['tasks']):raise ValueError('held-out task language overlaps Object training')
if len({r['episode'] for r in records})!=len(records):raise ValueError('duplicate fit records')
if set(r['episode'] for r in records if r['split']=='train') & set(r['episode'] for r in records if r['split']=='validation'):raise ValueError('train/validation episode leakage')
root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir()
with zipfile.ZipFile(io.BytesIO(blob)) as z:
 for info in z.infolist():
  if info.filename not in cfg['files'] or '/' in info.filename:raise ValueError('unexpected source member')
  (code/info.filename).write_bytes(z.read(info))
(root/'deployment.zip').write_bytes(blob)
(root/'suite_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
audit={'status':'passed','records_checked':len(records),'split_counts':{s:sum(r['split']==s for r in records) for s in ['train','validation']},
 'tasks':sorted(object_tasks),'checkpoint_sha256':sha(weight),'protocol_sha256':sha(pfile),'dataset_manifest_sha256':sha(mfile),
 'base_policy_revision':p['encoder_policy_revision'],'base_policy_task_training_coverage':'not documented in cached model card; no base-policy zero-shot claim',
 'held_out_suites_language_overlap':False,'protocol_records':records}
(root/'training_scope_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
sources={str(code/name):sha(code/name) for name in cfg['files']}
sources.update({str(root/'suite_inventory.json'):sha(root/'suite_inventory.json'),str(root/'training_scope_audit.json'):sha(root/'training_scope_audit.json'),str(weight):sha(weight),str(pfile):sha(pfile),str(mfile):sha(mfile)})
factory=b/'night-code-1e80242789/corrector_models.py'
if sha(factory)!='811386798aa491d8c0b43c08cd400bf00f397ad380484693689a1912976c4201':raise ValueError('MLP factory changed')
sources[str(factory)]=sha(factory)
plan={'max_hours':4,'code':str(code),'runtime':str(b/'runtime'),'python':str(b/'runtime/sim-venv/bin/python'),
 'inventory':str(root/'suite_inventory.json'),'training_audit':str(root/'training_scope_audit.json'),'sources':sources,
 'methods':['H50','H20','mlp_lvm','mlp_ogg'],'fixed_rollout_ceiling':880,'priority':[480,160,240],
 'checkpoint_read_only':True,'training':False,'downloads':False,'zip_sha256':cfg['zip_sha256']}
(root/'launch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
command=[str(b/'bootstrap/bin/python'),'-u',str(code/'corrector_crosssuite_queue.py'),'run','--root',str(root)]
with (root/'launch.log').open('xb') as log:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=code)
print(json.dumps({'pid':child.pid,'root':str(root),'log':str(root/'launch.log'),'code_zip_sha256':cfg['zip_sha256'],
 'formal_rollout_ceiling':880,'primary':480,'methods':plan['methods'],'max_hours':4,'training_scope':{k:v for k,v in audit.items() if k!='protocol_records'}},ensure_ascii=False))
'''


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['pack', 'install', 'status', 'stop', 'resume', 'fetch'])
    p.add_argument('--host', default='autodl')
    p.add_argument('--port', type=int)
    p.add_argument('--root', default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, default=REPO / 'outputs/autodl_setup/crosssuite_20261117')
    a = p.parse_args()
    if not a.root.startswith('/root/corrector_crosssuite_') or any(x in a.root for x in ('..', "'", '\n', ' ')):
        raise ValueError('scoped absolute output root required')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    if a.port:
        ssh += ['-p', str(a.port)]
    ssh.append(a.host)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode in ('pack', 'install'):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, path in FILES.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, path.read_bytes())
        blob = buffer.getvalue()
        sha = hashlib.sha256(blob).hexdigest()
        path = a.output / ('crosssuite_' + sha[:12] + '.zip')
        path.write_bytes(blob)
        path.with_suffix('.zip.sha256').write_text(sha + '  ' + path.name + '\n')
        print(json.dumps({'zip': str(path), 'sha256': sha}), flush=True)
        if a.mode == 'pack':
            return
        cfg = {'zip': base64.b64encode(blob).decode(), 'zip_sha256': sha, 'files': list(FILES),
               'root': a.root, 'inventory': '/root/corrector_crosssuite_preflight_20261117_001/suite_inventory.json'}
        encoded = base64.b64encode(INSTALL.encode()).decode()
        command = 'python3 -c "import base64;exec(base64.b64decode(\'' + encoded + '\'))"'
        subprocess.run(ssh + [command], input=json.dumps(cfg).encode(), check=True)
        return
    if a.mode in ('stop', 'resume'):
        # JSON literal, not shell interpolation of arbitrary paths.
        script = 'from pathlib import Path\nimport subprocess\nr=Path(' + repr(a.root) + ')\n'
        if a.mode == 'stop':
            script += "(r/'STOP').touch()\nprint('Episode-boundary STOP requested')\n"
        else:
            script += "with (r/'launch.log').open('ab') as log:\n subprocess.Popen(['/root/autodl-tmp/corrector/bootstrap/bin/python','-u',str(r/'code/corrector_crosssuite_queue.py'),'run','--root',str(r),'--resume'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint('Resume requested; original four-hour deadline retained')\n"
        subprocess.run(ssh + ['python3 -'], input=script.encode(), check=True)
        return
    script = "import json,os,time\nfrom pathlib import Path\nr=Path(" + repr(a.root) + ")\ns={}\n"
    script += "for name in ['status.json','budget.json','environment_probe.json','bridge_equivalence.json','transfer/execution.json','transfer/analysis/transfer_summary.json']:\n p=r/name\n if p.exists():s[name]=json.loads(p.read_text())\n"
    if a.mode == 'status':
        script += "s.pop('environment_probe.json',None)\ns.pop('transfer/analysis/transfer_summary.json',None)\n"
    script += "q=s.get('status.json',{})\npid=q.get('pid')\ncmd=Path('/proc')/str(pid)/'cmdline'\ns['live_check']={'checked_unix':time.time(),'queue_process_alive':bool(pid and cmd.exists() and b'corrector_crosssuite_queue.py' in cmd.read_bytes())}\n"
    script += "print(json.dumps(s,ensure_ascii=False,indent=2))\n"
    result = subprocess.run(ssh + ['python3 -'], input=script.encode(), capture_output=True)
    if result.returncode:
        raise RuntimeError('Read-only SSH query failed; remote queue was not stopped or restarted: '
                           + result.stderr.decode(errors='replace').strip())
    if a.mode == 'fetch':
        import datetime
        dest = a.output / ('snapshot_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
        dest.write_bytes(result.stdout)
        print(json.dumps({'snapshot': str(dest), 'bytes': len(result.stdout)}))
    else:
        print(result.stdout.decode())


if __name__ == '__main__':
    main()

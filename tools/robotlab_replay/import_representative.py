"""Read-only, bounded import of outcome-diverse completed Plus replays.

Uses the existing metrics snapshot to select cases before contacting SSH. Writes
only Replay's ignored local directory. Does not launch simulations or models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from corrector_casebank_import import install_bundle
from representative_cases import FOUR, SIX, select_groups

REMOTE = r'''
import hashlib,io,json,pathlib,sys,zipfile
request=json.load(sys.stdin);phase=request['phase']
if phase not in ['plus','plus_replicates']:raise ValueError('unknown phase')
root=(pathlib.Path('/root/corrector_plus_expanded_20261113_001/results')/phase).resolve()
read=lambda n:json.loads((root/n).read_bytes())
plan_raw=(root/'plan.json').read_bytes();plan=json.loads(plan_raw);execution=read('execution.json')
if execution['status']!='completed':raise ValueError('source phase not complete')
if hashlib.sha256(plan_raw).hexdigest()!=request['plan_sha256']:raise ValueError('plan changed since snapshot')
if not 1<=len(request['trials'])<=80:raise ValueError('trial count exceeds budget')
files={};records=[];total=0
def add(name):
 global total
 p=(root/name).resolve();p.relative_to(root)
 if not p.is_file() or p.stat().st_size>24*2**20:raise ValueError('missing/oversized artifact: '+name)
 total+=p.stat().st_size
 if total>100*2**20:raise ValueError('100MiB import budget exceeded; choose fewer cases')
 files[name]=p
for item in request['trials']:
 cid=item['id']
 if len(cid)!=20 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('invalid trial id')
 raw=(root/'trials'/(cid+'.json')).read_bytes();r=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('trial changed since snapshot')
 if r['status']!='completed' or r['result'].get('capture_schema')!='replay_casebank_v1':raise ValueError('requires native completed capture')
 if r['case'] not in plan['cases']:raise ValueError('trial is not in frozen plan')
 records.append(r);add('trials/'+cid+'.json')
 base='attempts/'+cid+'/attempt_%03d/'%r['attempt']
 for n in ['actual_protocol.json','frame_map.jsonl','executed_actions.json','controller_commands.json','queue_trace.json','actions.json','chunk_calls.json','monitor_steps.jsonl']:
  if (root/(base+n)).exists():add(base+n)
 for v in r['video_artifacts']:
  if not v['path'].startswith(base+'videos/') or not v['path'].endswith('.mp4'):raise ValueError('video not bound to trial')
  add(v['path'])
  if hashlib.sha256(files[v['path']].read_bytes()).hexdigest()!=v['sha256']:raise ValueError('video SHA differs')
for n in ['recording/lerobot_eval.py','recording/evidence.json']:add(n)
protocol=dict(plan['protocol']);protocol.update(study_id='replay_casebank_representative_'+phase+'_20261114',
 experiment_label='Plus'+('首轮' if phase=='plus' else '选例复测')+' · 成败组合精选',
 selection=request['selection'],selection_scope='outcome-selected diagnostic cases; not an unbiased success-rate sample')
manifest={'source_plan_sha256':request['plan_sha256'],'source_execution':execution,'remote_read_only':True,
 'omitted':['full tensors/NPZ, noise and state PT, intermediate chunks; retained at source'],
 'retained':'original trial bytes, video, frame map, commands, queue, score and OGG summaries'}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,'results/'+name)
 for name,value in [('plan.json',{'protocol':protocol,'cases':[r['case'] for r in records]}),
   ('execution.json',{'status':'completed','completed':len(records),'planned':len(records),'errors':0,'source_execution':execution}),
   ('export_manifest.json',manifest),('source_plan.json',plan)]:
  z.writestr('results/'+name,json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
sys.stdout.buffer.write(buf.getvalue())
'''


def snapshot_selection(snapshot):
    with zipfile.ZipFile(snapshot) as z:
        records = {}
        plans = {}
        for phase in ['plus', 'plus_replicates']:
            prefix = 'plus/' + phase + '/'
            plans[phase] = json.loads(z.read(prefix + 'plan.json'))
            records[phase] = [(n, z.read(n)) for n in sorted(z.namelist())
                if n.startswith(prefix + 'trials/') and n.endswith('.json')]
        selected = select_groups([json.loads(raw) for _, raw in records['plus']],
            plans['plus']['protocol']['conditions'])
        # One pre-existing replication of a diagnostic disagreement, not a new seed search.
        contrast = next((g for g in selected if g['pattern'] == '0011'), None)
        repeats = []
        if contrast:
            candidates = select_groups([json.loads(raw) for _, raw in records['plus_replicates']
                if json.loads(raw)['case']['condition'] == contrast['condition']],
                plans['plus_replicates']['protocol']['conditions'])
            same = next((g for g in candidates if g['pattern'] == contrast['pattern']), None)
            if same:
                repeats = [{**same, 'reason': 'existing independent evaluation seed for selected disagreement'}]
        output = []
        for phase, groups in [('plus', selected), ('plus_replicates', repeats)]:
            if not groups:
                continue
            keys = {(g['condition'], g['seed']) for g in groups}
            trials = []
            for _, raw in records[phase]:
                r = json.loads(raw); c = r['case']
                if (c['condition'], c['seed']) in keys and c['method'] in SIX:
                    if r['status'] != 'completed':
                        raise ValueError('selected method not completed')
                    trials.append({'id': c['id'], 'sha256': hashlib.sha256(raw).hexdigest()})
            if len(trials) != len(groups) * len(SIX):
                raise ValueError('selected six-method group is incomplete')
            output.append({'phase': phase, 'plan_sha256': hashlib.sha256(z.read('plus/'+phase+'/plan.json')).hexdigest(),
                'trials': trials, 'selection': {'methods_order': list(FOUR), 'groups': groups,
                    'retained_methods': list(SIX), 'rule': 'one per observed pattern, then category coverage'}})
        return output


def ordinary_manifest(catalog, snapshot):
    """Reference existing ordinary trials; retain the original per-run protocols."""
    from casebank import inventory
    groups = inventory(catalog)['groups']
    joint = {(g['condition'], g['seed']): g for g in groups
        if g['study'] == 'replay_casebank_expanded_ordinary_20261113_derived'}
    candidates = []
    for g in groups:
        if g['study'] not in ['legacy_five_models', 'replay_casebank_20261112_001']:
            continue
        j = joint.get((g['condition'], g['seed']))
        if not j:
            continue
        members = {m: j['methods'][m] if m.startswith('j_') else g['methods'][m] for m in FOUR}
        pattern = ''.join(str(int(members[m]['success'])) for m in FOUR)
        candidates.append((not all(v['replay_complete'] for v in members.values()),
            g['condition'], g['seed'], pattern, members))
    chosen = {}; result = []
    for incomplete, condition, seed, pattern, members in sorted(candidates, key=lambda x: x[:4]):
        if pattern in chosen:
            continue
        chosen[pattern] = True; refs = {}
        for method, item in members.items():
            r = catalog.runs[item['id']]
            expected = joint[condition, seed]['methods']['j_lvm']
            jp = catalog.runs[expected['id']]['_protocol']
            actual_sha = r['_record']['case']['config'].get('corrector_sha256')
            if actual_sha is None and method.startswith('mlp_'):
                actual_sha = r['_protocol'].get('all_weights', {}).get('mlp', {}).get('checkpoint_sha256')
            checkpoint = jp['joint_checkpoint_sha256'] if method.startswith('j_') else jp['legacy_weights']['mlp']['checkpoint_sha256']
            if actual_sha != checkpoint:
                raise ValueError('ordinary checkpoint mismatch')
            if r['_protocol'].get('policy_revision') != jp['policy_revision']:
                raise ValueError('ordinary policy revision mismatch')
            refs[method] = {'id': r['id'], 'method': method, 'recorded_method': r['method'],
                'source_study': r['_protocol'].get('study_id', r['experiment']),
                'trial_sha256': hashlib.sha256(r['_source'].read(r['member'])).hexdigest()}
        result.append({'condition': condition, 'seed': seed, 'pattern': pattern,
            'replay_complete': not incomplete, 'methods': refs})
    with zipfile.ZipFile(snapshot) as z:
        evidence = z.read('ordinary/bridge_receipt.json')
    if json.loads(evidence)['status'] != 'passed':
        raise ValueError('ordinary bridge not passed')
    return {'schema': 'reviewed_replay_comparisons_v1',
        'study': 'reviewed_ordinary_j_mlp_20261114', 'evidence_file': 'ordinary_bridge.evidence',
        'evidence_sha256': hashlib.sha256(evidence).hexdigest(),
        'pairing_note': '同task/init/eval seed、冻结SmolVLA及对应checkpoint；旧MLP/Case Bank与新J跨批次，4回合桥接已通过；训练数据量不同。逐回合协议保持原样，缺视频/动作明确标缺。',
        'methods_order': list(FOUR), 'groups': result}, evidence


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['plan', 'fetch'])
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--ssh-host', default='autodl')
    p.add_argument('--control-path', type=Path)
    a = p.parse_args(); requests = snapshot_selection(a.snapshot)
    if a.mode == 'plan':
        print(json.dumps(requests, ensure_ascii=False, indent=2)); return
    state = HERE / '.local'; destination = state / 'casebank_downloads'
    destination.mkdir(parents=True, exist_ok=True)
    for request in requests:
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
        output = destination / ('representative_' + request['phase'] + '_' + key + '.zip')
        if not output.exists():
            if shutil.disk_usage(destination).free < 750 * 2**20:
                raise ValueError('require 750MiB local free space before bounded import; no cleanup performed')
            cmd = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
            if a.control_path:
                cmd += ['-S', str(a.control_path)]
            cmd += [a.ssh_host, shlex.join(['/root/autodl-tmp/corrector/bootstrap/bin/python', '-c', REMOTE])]
            partial = output.with_suffix('.zip.partial')
            with partial.open('xb') as f:
                r = subprocess.run(cmd, input=json.dumps(request).encode(), stdout=f)
            if r.returncode:
                raise RuntimeError('read-only SSH export failed; partial retained, not installed')
            partial.rename(output)
        print(json.dumps(install_bundle(output, state), ensure_ascii=False), flush=True)
    from data import Catalog
    c = Catalog()
    try:
        manifest, evidence = ordinary_manifest(c, a.snapshot)
        folder = state / 'comparison_sets'; folder.mkdir(exist_ok=True)
        for name, raw in [('ordinary_bridge.evidence', evidence),
            ('ordinary_j_mlp.json', (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode())]:
            target = folder / name
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError('comparison manifest conflict; preserve existing: '+name)
            else:
                with target.open('xb') as f:
                    f.write(raw)
        print(json.dumps({'ordinary_comparison_groups': len(manifest['groups']),
            'patterns': {g['pattern']: {'condition':g['condition'], 'seed':g['seed'], 'complete':g['replay_complete']} for g in manifest['groups']},
            'remote_read_only': True, 'gpu_work': False}, ensure_ascii=False))
    finally:
        c.close()


if __name__ == '__main__':
    main()

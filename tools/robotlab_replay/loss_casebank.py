"""Reviewed cross-loss views reference existing immutable trials, never copy them."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODES=('cosine','mse','mse_cosine')
METHODS=('mlp_ogg','j_ogg')


def protocol_signature(run):
    p=run['_protocol'];loss=p['ogg_loss']
    fields=['policy_revision','vla_dtype','joint_visual_encoder_dtype','joint_checkpoint_sha256','legacy_weights',
        'fps','control_fps','data_fps','due_steps','first_due_step','horizon','chunk_size','denoise','threshold',
        'joint_layer_calibration_sha256','prediction_retention','plus_revision']
    if any(k not in p for k in fields):raise ValueError('loss comparison protocol incomplete')
    frozen={k:p[k] for k in fields}
    frozen['condition']=p['conditions'][run['condition']]
    config=run['_record']['case']['config']
    if any(config.get(k)!=v for k,v in {'steps':50,'action_steps':50,'mode':'ogg','uses_corrector':True}.items()):
        raise ValueError('loss action queue/control configuration differs')
    frozen['guidance']={k:loss[k] for k in ['eta','combination','joint_weights','target','support','lvm_unchanged','checkpoint_read_only','loss_source_sha256','frozen_manifest']}
    return hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()


def validate_reference(alias,ref,run):
    method,mode=alias.split('@',1)
    if method not in METHODS or mode not in MODES or ref.get('loss_mode')!=mode:
        raise ValueError('unknown loss alias')
    p=run['_protocol'];r=run['_record']
    if (run['method']!=method or run['model']!=('MLP' if method=='mlp_ogg' else 'J-36M')
        or p.get('ogg_loss',{}).get('loss_mode')!=mode or r['case']['config'].get('ogg_loss_mode')!=mode
        or r.get('result',{}).get('ogg_loss_mode')!=mode):raise ValueError('loss model/method alias mismatch')


def validate_group(methods,group,evidence):
    if evidence.get('kind')!='ogg_loss_modes' or set(methods)!={m+'@'+v for m in METHODS for v in MODES}:
        raise ValueError('loss comparison matrix differs')
    checks=[r for r in evidence['first_guide_checks'] if r['condition']==group['condition'] and r['seed']==group['seed']]
    if len(checks)!=2 or {r['method'] for r in checks}!=set(METHODS) or any(
        not (r.get('first_candidate_exact') is True or r.get('no_guidance_all_modes') is True and r.get('all_controls_exact') is True)
        or r.get('initial_signatures_and_pre_guide_commands_exact') is not True for r in checks):
        raise ValueError('loss numerical pairing evidence missing')
    signatures={protocol_signature(r) for r in methods.values()}
    if len(signatures)!=1 or next(iter(signatures))!=group.get('protocol_signature'):
        raise ValueError('loss frozen protocols differ')
    states=set()
    for r in methods.values():
        record=r['_record'];base=r['_prefix']+f"attempts/{record['case']['id']}/attempt_{record.get('attempt',1):03d}/"
        actual=r['_source'].json(base+'actual_protocol.json') or {}
        state=actual.get('first',{}).get('state_sha256')
        if not isinstance(state,str) or len(state)!=64 or any(c not in '0123456789abcdef' for c in state):
            raise ValueError('loss initial state fingerprint missing')
        states.add(state)
    if len(states)!=1:raise ValueError('loss initial physical state differs across methods')
    return next(iter(states))


def build(catalog,input_dir,tag='pilot'):
    sys.path.insert(0,str(ROOT/'scripts'))
    from corrector_loss_analysis import analyse
    input_dir=Path(input_dir);raw=(input_dir/'snapshot.json').read_bytes();data=json.loads(raw)
    summary,_,_=analyse(data,input_dir,expected_trials=data['plan']['planned'])
    evidence={'status':'passed','kind':'ogg_loss_modes','source_snapshot_sha256':hashlib.sha256(raw).hexdigest(),
              'first_guide_checks':summary['first_guide_checks']}
    evidence_raw=(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n').encode()
    source_records={t['record']['case']['id']:t['record'] for stage in data['stages'] for t in stage['trials']}
    selected={}
    for r in catalog.runs.values():
        record=source_records.get(r['_record']['case']['id'])
        if record is None:continue
        # Import/export preserves original trial bytes; require the same parsed record.
        if r['_record']!=record:continue
        c=record['case'];mode=c['config']['ogg_loss_mode'];alias=c['method']+'@'+mode
        key=(r['condition'],r['seed'])
        if alias in selected.setdefault(key,{}):raise ValueError('duplicate loss trial binding')
        selected[key][alias]=r
    manifests=[];file_name=f'ogg_loss_{tag}.evidence'
    for domain in ['ordinary','plus']:
        groups=[]
        for (condition,seed),runs in sorted(selected.items()):
            if condition.startswith('plus/')!=(domain=='plus'):continue
            group={'condition':condition,'seed':seed,'protocol_signature':protocol_signature(next(iter(runs.values()))),'methods':{}}
            for alias,r in runs.items():
                group['methods'][alias]={'id':r['id'],'method':alias,'recorded_method':r['method'],
                    'loss_mode':alias.split('@')[1],'source_study':r['_protocol']['study_id'],
                    'trial_sha256':hashlib.sha256(r['_source'].read(r['member'])).hexdigest()}
            validate_group(runs,group,evidence);groups.append(group)
        if not groups:raise ValueError('no complete recorded loss groups for '+domain)
        manifests.append({'schema':'reviewed_replay_comparisons_v1','comparison_kind':'ogg_loss_modes',
            'study':f'reviewed_ogg_losses_{tag}_{domain}_20261116','evidence_file':file_name,
            'evidence_sha256':hashlib.sha256(evidence_raw).hexdigest(),'groups':groups,
            'pairing_note':'同condition/seed；同预测器三损失冻结协议与首候选已核验。MLP/J训练数据不同，跨模型仅诊断对照。'})
    if sum(len(g['methods']) for m in manifests for g in m['groups'])!=data['plan']['planned']:
        raise ValueError('not all completed loss trials have local bindings')
    return manifests,evidence_raw


def install(catalog,input_dir,tag='pilot'):
    if not tag.replace('_','').isalnum() or len(tag)>32:raise ValueError('invalid local view tag')
    manifests,evidence=build(catalog,input_dir,tag);folder=catalog.state/'comparison_sets';folder.mkdir(parents=True,exist_ok=True)
    files={f'ogg_loss_{tag}.evidence':evidence}
    for domain,m in zip(['ordinary','plus'],manifests):files[f'ogg_loss_{tag}_{domain}.json']=(json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode()
    for name,raw in files.items():
        p=folder/name
        if p.exists() and p.read_bytes()!=raw:raise FileExistsError('different existing view; use a new tag')
    for name,raw in files.items():
        p=folder/name
        if not p.exists():
            with p.open('xb') as f:f.write(raw)
    return {'views':len(manifests),'groups':sum(len(m['groups']) for m in manifests),'referenced_trials':sum(len(g['methods']) for m in manifests for g in m['groups']),
        'copied_trials':0,'overwritten':False}


if __name__=='__main__':
    from data import Catalog
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',default='pilot');a=p.parse_args()
    c=Catalog()
    try:print(json.dumps(install(c,a.input,a.tag),ensure_ascii=False))
    finally:c.close()

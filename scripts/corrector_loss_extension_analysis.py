"""Explicit derived merge: old24 + disjoint new72, never relabel as one run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from corrector_loss_analysis import analyse,MODES


def merge(pilot,extension,selection):
    target=json.loads(Path(selection).read_text());batches=[];rows=[];checks=[];seen=set();protocols=[];states={}
    for name,folder,n in [('pilot',Path(pilot),24),('extension',Path(extension),72)]:
        raw=(folder/'snapshot.json').read_bytes();data=json.loads(raw);summary,items,_=analyse(data,folder,n)
        batches.append({'name':name,'trials':n,'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'source_zip_sha256':data['plan']['zip_sha256']})
        for stage in data['stages']:
            for t in stage['trials']:
                c=t['record']['case']['condition'];state=t['actual_protocol']['first']['state_sha256']
                if c in states and states[c]!=state:raise ValueError('initial state differs within one named test condition')
                states[c]=state
        checks.extend(summary['first_guide_checks']);protocols.extend(s['protocol'] for s in data['stages'])
        for row in items:
            key=(row['condition'],row['seed'],row['method'],row['loss_mode'])
            if key in seen:raise ValueError('old/new overlap; no duplicate counting')
            seen.add(key);rows.append({**row,'source_batch':name,'source_snapshot_sha256':batches[-1]['snapshot_sha256']})
    expected={(c,s,m,l) for cs in target['conditions'].values() for c in cs for s in target['seeds'] for m in ['mlp_ogg','j_ogg'] for l in MODES}
    if len(expected)!=96 or seen!=expected:raise ValueError('combined physical condition/seed/method/loss matrix differs')
    frozen_fields=['policy_revision','vla_dtype','joint_visual_encoder_dtype','joint_checkpoint_sha256','legacy_weights',
        'fps','control_fps','data_fps','due_steps','first_due_step','horizon','chunk_size','denoise','threshold',
        'joint_layer_calibration_sha256','prediction_retention']
    # Plus instantiates additional unused legacy predictors; compare the actual MLP checkpoint/factory only.
    def frozen(p):
        values={k:p[k] for k in frozen_fields if k!='legacy_weights'}
        values['mlp_weights']=p['legacy_weights']['mlp']
        values['guidance']={k:p['ogg_loss'][k] for k in ['eta','combination','joint_weights','target','support','loss_source_sha256','frozen_manifest']}
        return json.dumps(values,sort_keys=True)
    if len({frozen(p) for p in protocols})!=1:raise ValueError('frozen control protocol differs across batches')
    for domain,conditions in target['conditions'].items():
        cohort=[p for p in protocols if p['mode']==domain]
        if not cohort or len({p['plus_revision'] for p in cohort})!=1:raise ValueError('domain source revision differs')
        for condition in conditions:
            definitions={json.dumps(p['conditions'][condition],sort_keys=True) for p in cohort if condition in p['conditions']}
            if len(definitions)!=1:raise ValueError('condition definition differs across batches')
    index={(r['condition'],r['seed'],r['method'],r['loss_mode']):r for r in rows}
    aggregate={};paired={};repeat={};patterns=[]
    for method in ['mlp_ogg','j_ogg']:
        aggregate[method]={};paired[method]={};repeat[method]={}
        for mode in MODES:
            subset=[r for r in rows if r['method']==method and r['loss_mode']==mode]
            aggregate[method][mode]={'completed':len(subset),'successes':sum(r['success'] for r in subset),
                **{'mean_'+k:statistics.mean(r[k] for r in subset) for k in ['env_steps','event_replans','guidance_calls','episode_wall_s']}}
            stable_s=stable_f=mixed=0
            for cs in target['conditions'].values():
                for c in cs:
                    successes=sum(index[c,s,method,mode]['success'] for s in target['seeds'])
                    stable_s+=successes==2;stable_f+=successes==0;mixed+=successes==1
            repeat[method][mode]={'two_seed_success_conditions':stable_s,'two_seed_failure_conditions':stable_f,'seed_sensitive_conditions':mixed}
            if mode!='cosine':
                pairs=[(index[c,s,method,'cosine']['success'],index[c,s,method,mode]['success']) for cs in target['conditions'].values() for c in cs for s in target['seeds']]
                paired[method][mode]={'new_only_success':sum(not a and b for a,b in pairs),'cosine_only_success':sum(a and not b for a,b in pairs),
                    'both_success':sum(a and b for a,b in pairs),'both_failure':sum(not a and not b for a,b in pairs)}
    for domain,conditions in target['conditions'].items():
        for c in conditions:
            for s in target['seeds']:
                patterns.append({'domain':domain,'condition':c,'seed':s,
                    **{m+'@'+l:index[c,s,m,l]['success'] for m in ['mlp_ogg','j_ogg'] for l in MODES}})
    return {'status':'completed','combined_trials':96,'new_trials':72,'reused_trials':24,'errors':0,
        'test_conditions':8,'condition_seed_groups':16,'eval_seeds':target['seeds'],
        'initial_state_signatures':len(set(states.values())),'initial_state_sha256_by_condition':states,
        'state_fingerprint_limit':'MuJoCo initial state vector only; does not encode camera, language or all static model/asset configuration',
        'source_batches':batches,'frozen_control_signature_sha256':hashlib.sha256(frozen(protocols[0]).encode()).hexdigest(),
        'methods':aggregate,'paired_vs_cosine':paired,'seed_repeat':repeat,
        'by_domain':{domain:{method:{mode:{'completed':len([r for r in rows if r['domain']==domain and r['method']==method and r['loss_mode']==mode]),
            'successes':sum(r['success'] for r in rows if r['domain']==domain and r['method']==method and r['loss_mode']==mode)} for mode in MODES}
            for method in ['mlp_ogg','j_ogg']} for domain in ['ordinary','plus']},
        'active_first_candidate_groups':sum(c.get('first_candidate_exact') is True for c in checks),
        'no_guidance_matched_groups':sum(c.get('no_guidance_all_modes') is True for c in checks),
        'scope':'8 deliberately selected test conditions; 2 eval seeds; diagnostic expansion, not full Plus/Object10 benchmark'},rows,patterns


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--extension',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary,rows,patterns=merge(a.pilot,a.extension,a.selection);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'combined_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,items in [('combined_trials.csv',rows),('combined_patterns.csv',patterns)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator="\n");w.writeheader();w.writerows(items)
    print(json.dumps({'trials':96,'methods':summary['methods'],'paired':summary['paired_vs_cosine']},ensure_ascii=False))


if __name__=='__main__':main()

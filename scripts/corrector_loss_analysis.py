"""Validate/aggregate actual paired OGG loss trials and numerical guide traces."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np

MODES=('cosine','mse','mse_cosine')


def analyse(data, archives,expected_trials=24):
    if data['status']['status']!='completed':raise ValueError('pilot not complete')
    index={};rows=[];guide_rows=[]
    for stage in data['stages']:
        if stage['execution']['completed']!=stage['execution']['planned'] or stage['execution']['errors'] or stage['execution']['status']!='completed':raise ValueError('stage incomplete')
        for trial in stage['trials']:
            r=trial['record'];c=r['case'];v=r['result'];mode=v['ogg_loss_mode'];key=(c['condition'],c['seed'],c['method'],mode)
            if key in index or r['status']!='completed' or type(v['success']) is not bool:raise ValueError('duplicate/incomplete trial')
            if len(trial['guidance'])!=v['guidance_calls']:raise ValueError('actual guidance count differs')
            details=trial['guidance_details']
            if mode!='cosine' and len(details)!=len(trial['guidance']):raise ValueError('loss details count differs')
            for i,g in enumerate(trial['guidance']):
                if not g.get('original_update_formula_exact_equal'):raise ValueError('velocity update audit failed')
                slots=g['per_action_clean_update_l2'][0]
                if any(value!=0 for value in slots[1:]):raise ValueError('unrequested multi-action update')
                extra=details[i] if details else {}
                if extra and not np.isclose(g.get('guidance_loss',g.get('ogg_loss')),extra['loss'],rtol=1e-6):raise ValueError('logged loss differs')
                guide_rows.append({'condition':c['condition'],'seed':c['seed'],'method':c['method'],'loss_mode':mode,
                    'step':g['step'],'denoise':g.get('denoise'),'loss':g.get('guidance_loss',g.get('ogg_loss')),
                    'gradient_l2':g['gradient_l2'],'clean_action_update_l2':g['clean_action_update_l2'],
                    'guidance_seconds':g['guidance_seconds'],'raw_mse':extra.get('raw_mse'),
                    'cosine_loss':extra.get('cosine_loss'),'per_layer_components':json.dumps(extra.get('per_layer_components'),sort_keys=True)})
            index[key]=(stage['name'],trial)
            rows.append({'domain':stage['name'].split('_')[-1],'condition':c['condition'],'seed':c['seed'],'method':c['method'],'loss_mode':mode,
                **{k:v[k] for k in ['success','env_steps','chunk_calls','event_replans','guidance_calls','policy_s','monitor_s','episode_wall_s','peak_torch_allocated_gib']}})
    if expected_trials not in [24,72] or data['plan']['planned']!=expected_trials or len(index)!=expected_trials or len({k[:3] for k in index})!=expected_trials//3:raise ValueError('pilot matrix differs')
    checks=[]
    def candidate(stage,trial):
        r=trial['record'];g=trial['guidance'][0];base=f"results/attempts/{r['case']['id']}/attempt_{r['attempt']:03d}/"
        with zipfile.ZipFile(archives/(stage+'.zip')) as z:
            raw=z.read(base+g['candidate_arrays']['path'])
        if hashlib.sha256(raw).hexdigest()!=g['candidate_arrays']['sha256']:raise ValueError('candidate SHA differs')
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:return arrays['candidate_clean_before']
    for key in sorted({k[:3] for k in index}):
        name,base=index[key+('cosine',)];initial=base['actual_protocol']['first']
        if not base['guidance']:
            for mode in MODES[1:]:
                _,trial=index[key+(mode,)]
                if trial['guidance'] or base['commands']!=trial['commands']:raise ValueError('unguided controls differ across loss modes')
                if any(initial[k]!=trial['actual_protocol']['first'][k] for k in ['state_sha256','input_sha256','noise_sha256']):raise ValueError('initial signatures differ')
            checks.append({'condition':key[0],'seed':key[1],'method':key[2],'first_guide_step':None,
                'initial_signatures_and_pre_guide_commands_exact':True,'first_candidate_exact':None,
                'no_guidance_all_modes':True,'all_controls_exact':True})
            continue
        first=base['guidance'][0];before=candidate(name,base)
        for mode in MODES[1:]:
            name,trial=index[key+(mode,)]
            if not trial['guidance']:raise ValueError('first guidance missing in alternate mode')
            other=trial['guidance'][0]
            if any(initial[k]!=trial['actual_protocol']['first'][k] for k in ['state_sha256','input_sha256','noise_sha256']):raise ValueError('initial signatures differ')
            if first['step']!=other['step'] or base['commands'][:first['step']]!=trial['commands'][:first['step']]:raise ValueError('pre-guidance controls differ')
            if not np.array_equal(before,candidate(name,trial)):raise ValueError('first pre-guidance candidate differs')
        checks.append({'condition':key[0],'seed':key[1],'method':key[2],'first_guide_step':first['step'],
            'initial_signatures_and_pre_guide_commands_exact':True,'first_candidate_exact':True})
    methods={}
    for method in ['mlp_ogg','j_ogg']:
        methods[method]={}
        for mode in MODES:
            group=[r for r in rows if r['method']==method and r['loss_mode']==mode]
            guides=[r for r in guide_rows if r['method']==method and r['loss_mode']==mode]
            methods[method][mode]={'completed':len(group),'successes':sum(r['success'] for r in group),
                **{'mean_'+k:statistics.mean(r[k] for r in group) for k in ['env_steps','event_replans','guidance_calls','episode_wall_s']},
                'guide_calls':len(guides),'mean_guidance_ms':statistics.mean(r['guidance_seconds'] for r in guides)*1000 if guides else None,
                'median_gradient_l2':statistics.median(r['gradient_l2'] for r in guides) if guides else None}
        for mode in MODES[1:]:
            baseline={k[:3]:v[1]['record']['result']['success'] for k,v in index.items() if k[2]==method and k[3]=='cosine'}
            changes=[(baseline[k[:3]],v[1]['record']['result']['success']) for k,v in index.items() if k[2]==method and k[3]==mode]
            methods[method][mode]['gained_successes']=sum(not a and b for a,b in changes)
            methods[method][mode]['lost_successes']=sum(a and not b for a,b in changes)
            ratios=[v[1]['guidance'][0]['gradient_l2']/index[k[:3]+('cosine',)][1]['guidance'][0]['gradient_l2']
                for k,v in index.items() if k[2]==method and k[3]==mode and v[1]['guidance'] and index[k[:3]+('cosine',)][1]['guidance'] and index[k[:3]+('cosine',)][1]['guidance'][0]['gradient_l2']>0]
            methods[method][mode]['median_same_first_candidate_gradient_ratio_vs_cosine']=statistics.median(ratios) if ratios else None
    return {'status':'completed','trials':expected_trials,'errors':0,'methods':methods,'first_guide_checks':checks,
        'cosine_reproduction':data['equivalence'],'source_zip_sha256':data['plan']['zip_sha256'],
        'scope':{'test_conditions':len({k[0] for k in index}),'eval_seeds':sorted({k[1] for k in index}),
                 'initial_state_signatures':len({t['actual_protocol']['first']['state_sha256'] for s in data['stages'] for t in s['trials']}),
                 'limit':'selected diagnostic conditions; not population success-rate estimation'}},rows,guide_rows


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-trials',type=int,choices=[24,72],default=24);a=p.parse_args()
    raw=(a.input/'snapshot.json').read_bytes();data=json.loads(raw);summary,rows,guides=analyse(data,a.input,a.expected_trials)
    summary.update(snapshot_sha256=hashlib.sha256(raw).hexdigest(),captured_utc=data['captured_utc'])
    calibration_path=a.input/'j_calibration.json'
    if calibration_path.exists():
        calibration_raw=calibration_path.read_bytes();expected=data['stages'][0]['protocol']['joint_layer_calibration_sha256']
        if hashlib.sha256(calibration_raw).hexdigest()!=expected:raise ValueError('J calibration SHA differs')
        cal=json.loads(calibration_raw)
        summary['joint_calibration']={k:cal[k] for k in ['weights','norm_floor','checkpoint_sha256']}
    p0=data['stages'][0]['protocol']
    summary['frozen_models']={'joint':p0['joint_checkpoint_sha256'],'mlp':p0['legacy_weights']['mlp']}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,items in [('trials.csv',rows),('guidance.csv',guides)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator="\n");w.writeheader();w.writerows(items)
    print(json.dumps(summary['methods'],ensure_ascii=False))


if __name__=='__main__':main()

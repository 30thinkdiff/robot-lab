"""混合Corrector Pilot离线审计：分开强制检查/自然结果，核验触发与引导证据。"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


def audit(archive, reference=None):
    read=lambda n:json.loads(archive.read(n))
    pipeline=read('pipeline_status.json');protocol=read('protocol.json')
    assert pipeline['status']=='completed' and pipeline['checkpoint_sha_unchanged']
    assert pipeline['forced_checks']==3 and pipeline['paired_trials']==12 and not pipeline['automatic_expansion']
    assert protocol['eta']==1 and protocol['guidance_every']==3 and protocol['guidance_indices']==[0,3,6,9]
    assert protocol['fps']==20 and protocol['data_fps']==10 and protocol['due_steps']==20
    assert protocol['chunk_size']==50 and protocol['action_steps']==50 and protocol['num_steps']==10
    assert protocol['experiment_branch']=='experiment/hybrid-corrector-pilot-20261111'
    assert protocol['configuration']=={'A':{'detector':'single','guide':'single','video_policy':'all'},
        'B':{'detector':'single','guide':'a10','video_policy':'all'},'C':{'detector':'a10','guide':'a10','video_policy':'all'}}
    compatibility=read('compatibility.json')
    assert compatibility['compatible'] and compatibility['episodes_checked']==454
    a,b=read('frozen_single_protocol.json'),read('frozen_a10_protocol.json')
    for key in ['records','data_manifest_sha256','fps','k_step','prediction_seconds','encoder_policy_revision']:
        assert a[key]==b[key]
    records_by_stage={}
    for stage,n in [('forced_check',3),('paired',12)]:
        execution=read(stage+'/execution.json')
        assert execution['status']=='completed' and execution['completed']==execution['planned']==n and execution['errors']==0
        records_by_stage[stage]=[read(p) for p in archive.namelist() if p.startswith(stage+'/trials/') and p.endswith('.json')]
        assert len(records_by_stage[stage])==n
    cases=[];calls=[];events=[];data={};matched={}
    for stage,records in records_by_stage.items():
        plan=read(stage+'/plan.json');seen=set()
        for record in records:
            assert record['status']=='completed' and type(record['result']['success']) is bool
            c,r=record['case'],record['result'];cfg=c['method']
            assert c in plan['cases'] and c['id'] not in seen and c['seed']==1000
            seen.add(c['id'])
            prefix=f"{stage}/attempts/{c['id']}/attempt_{record['attempt']:03d}/"
            actual=read(prefix+'actual_protocol.json')
            assert actual['protocol']==protocol and actual['forced_check']==(stage=='forced_check')
            assert actual['k']==20 and actual['warmup_steps']==4 and actual['source_history_stride']==2
            ts=read(prefix+'targets.json');trace=[json.loads(l) for l in archive.read(prefix+'monitor_steps.jsonl').decode().splitlines()]
            actions=read(prefix+'executed_actions.json')
            steps=[t for t in trace if 'score' in t]
            gs=[t for t in trace if t.get('type')=='hybrid_guidance']
            triggers=[t for t in trace if t.get('type')=='hybrid_trigger']
            assert len(steps)==len(actions)==r['env_steps']
            assert [t['step'] for t in steps]==list(range(r['env_steps']))
            assert all(len(v)==7 and all(math.isfinite(x) for x in v) for v in actions)
            scored=[t for t in steps if t['score'] is not None]
            assert len(scored)==r['scored_steps'] and r['score_eligible_steps']==max(0,r['env_steps']-24)
            assert len(ts)==len(triggers)==r['event_replans']
            assert len(gs)==r['guidance_calls']==4*len(ts)
            assert len(r['chunk_latency_s'])==r['chunk_calls'] and math.isclose(sum(r['chunk_latency_s']),r['policy_s'])
            assert math.isclose(sum(g['guidance_seconds'] for g in gs),r['guidance_s'])
            for target in ts:
                assert target['target_shape']==[1,128,960] and target['prediction_origin_step']==target['step']-20
                assert target['target_source']=='detector_expected_minus_observed'
                assert any(t['step']==target['step'] and t.get('interrupted') for t in steps)
                event_calls=[g for g in gs if g['event']==target['event']]
                assert [g['denoise'] for g in event_calls]==[0,3,6,9]
                assert all(g['step']==target['step'] for g in event_calls)
                if stage=='paired':events.append({'configuration':cfg,'condition':c['condition'],**target})
            for g in gs:
                assert g['event'] in {t['event'] for t in ts}
                offsets=[0] if cfg=='A' else list(range(0,20,2))
                assert g['condition_offsets']==offsets
                assert all(math.isfinite(g[k]) and g[k]>=0 for k in ['gradient_l2','velocity_update_l2','clean_action_update_l2','guidance_seconds'])
                assert math.isclose(g['velocity_update_l2'],g['gradient_l2'],rel_tol=1e-6)
                assert math.isclose(g['clean_action_update_l2'],g['gradient_l2']*g['timestep'][0],rel_tol=2e-6,abs_tol=1e-10)
                position=g['per_action_clean_update_l2'][0]
                assert len(position)==50 and all(v==0 for i,v in enumerate(position) if i not in offsets)
                assert math.isclose(math.sqrt(sum(v*v for v in position)),g['clean_action_update_l2'],rel_tol=2e-6,abs_tol=1e-10)
                if stage=='paired':calls.append({'configuration':cfg,'condition':c['condition'],**g,
                    'first_action_clean_update_l2':position[0]})
            if stage=='paired':
                assert not any(t.get('debug_forced') for t in steps)
                case={'configuration':cfg,'condition':c['condition'],'success':r['success'],
                      **{k:r[k] for k in ['env_steps','chunk_calls','policy_s','monitor_s','guidance_s',
                       'event_replans','guidance_calls','scored_steps','score_eligible_steps','sequence_prediction_skips',
                       'cancelled_condition_predictions','peak_torch_allocated_gib']},'episode_wall_s':record['wall_s'],
                       'video_remote_path':r['video_paths'][0]}
                case['score_coverage']=r['scored_steps']/r['score_eligible_steps']
                cases.append(case);data[(c['condition'],cfg)]=(record,ts,gs,actions,steps)
                signature=tuple(r[k] for k in ['initial_state_sha256','first_input_sha256','first_noise_sha256'])
                matched.setdefault(c['condition'],set()).add(signature)
            else:
                assert any(t.get('debug_forced') and t['step']==24 for t in steps)
        assert seen=={c['id'] for c in plan['cases']}
    assert len(matched)==4 and all(len(v)==1 for v in matched.values())
    ab=[]
    for cond in sorted(matched):
        ar,at,ag,aa,ascores=data[(cond,'A')];br,bt,bg,ba,bscores=data[(cond,'B')]
        assert at and bt
        step=at[0]['step'];assert step==bt[0]['step']
        assert at[0]['target_sha256']==bt[0]['target_sha256']
        assert aa[:step]==ba[:step]
        assert [(s['score'],s['threshold']) for s in ascores[:step+1]]==[(s['score'],s['threshold']) for s in bscores[:step+1]]
        first_difference=next((i for i,(x,y) in enumerate(zip(aa,ba)) if x!=y),None)
        assert first_difference==step
        row={'condition':cond,'A_success':ar['result']['success'],'B_success':br['result']['success'],
             'first_event_step':step,'target_sha256':at[0]['target_sha256'],'prefix_actions_exactly_equal':True,
             'scores_equal_through_first_event':True,'first_action_divergence_step':first_difference}
        for label,guide in [('A',ag[0]),('B',bg[0])]:
            for key in ['ogg_loss','gradient_l2','clean_action_update_l2']:
                row[label+'_'+key]=guide[key]
            row[label+'_first_action_clean_update_l2']=guide['per_action_clean_update_l2'][0][0]
        ab.append(row)
    # 重算跨配置表，不只相信服务器已有汇总。
    overall=list(csv.DictReader(io.StringIO(archive.read('paired/analysis/hybrid_overall.csv').decode())))
    for row in overall:
        subset=[c for c in cases if c['configuration']==row['configuration']]
        assert len(subset)==int(row['completed'])==int(row['planned'])==4
        assert int(row['successes'])==sum(c['success'] for c in subset)
        assert float(row['success_rate'])==int(row['successes'])/4
        for key in subset[0]:
            if 'mean_'+key in row:
                assert math.isclose(float(row['mean_'+key]),statistics.mean(c[key] for c in subset),rel_tol=1e-9)
        assert math.isclose(float(row['score_coverage']),sum(c['scored_steps'] for c in subset)/sum(c['score_eligible_steps'] for c in subset))
    history=[]
    if reference:
        for cfg,prefix in [('A','night/evaluation/transformer-h1_s42/'),('C','actions/experiments/evaluation/transformer-a10-h1_s42/')]:
            old={}
            for name in reference.namelist():
                if name.startswith(prefix) and '/paired/trials/' in name:
                    r=json.loads(reference.read(name));c=r['case']
                    if c['seed']==1000 and c['method']=='ogg':old[c['condition']]=r['result']
            for case in [c for c in cases if c['configuration']==cfg]:
                metrics=['success','env_steps','chunk_calls','event_replans','guidance_calls']
                assert all(old[case['condition']][key]==case[key] for key in metrics)
                history.append({'configuration':cfg,'condition':case['condition'],'matched_metrics':metrics})
    findings={'status':'verified','natural_trials':12,'forced_checks':3,'errors':0,
        'checkpoint_sha_unchanged_reported':True,'paired_initial_conditions_verified':True,
        'AB_prefix_actions_scores_and_first_target_equal_cases':4,'historical_reproduced_cases':history,
        'formal_guidance_calls':len(calls),'formal_trigger_events':len(events),
        'video_and_target_tensor_bytes_verified_locally':False,
        'causal_limits':['four conditions, one extra failure, no population-level superiority claim',
                        'B-C changes scores and detector-generated target as well as coverage',
                        'loss values of distinct learned predictors are not interchangeable quality scores',
                        'post-first-event trajectories differ; aggregate gradient means are not matched counterfactuals']}
    return findings,overall,cases,events,calls,ab


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-bundle',type=Path)
    p.add_argument('--plot',action='store_true');args=p.parse_args()
    with zipfile.ZipFile(args.bundle) as z:
        assert z.testzip() is None
        reference=zipfile.ZipFile(args.reference_bundle) if args.reference_bundle else None
        try:findings,overall,cases,events,calls,ab=audit(z,reference)
        finally:
            if reference:reference.close()
        args.output.mkdir(parents=True,exist_ok=True)
        for name in ['pipeline_status.json','protocol.json','compatibility.json','shape_gradient_probe.json',
                     'paired/analysis/hybrid_overall.csv','paired/execution.json']:
            (args.output/Path(name).name).write_bytes(z.read(name))
        (args.output/'forced_execution.json').write_bytes(z.read('forced_check/execution.json'))
    findings['bundle_sha256']=hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('cases.csv',cases),('events.csv',events),('guidance_calls.csv',calls),('AB_first_event.csv',ab)]:
        with (args.output/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12.2,3.8),constrained_layout=True)
        names=['A','B','C']
        axes[0].bar(names,[int(r['successes']) for r in overall]);axes[0].set(ylim=(0,4.5),yticks=range(5),title='Successes (4 trials each)')
        axes[1].bar(names,[100*float(r['score_coverage']) for r in overall]);axes[1].set(ylim=(0,110),title='Scoring coverage (%)')
        case=next(r for r in ab if r['condition']=='libero_object/task_1/init_0')
        axes[2].bar(['A: single','B: A10'],[case['A_first_action_clean_update_l2'],case['B_first_action_clean_update_l2']])
        axes[2].set(title='Same first target, step 61',ylabel='First-action clean update L2')
        for ax in axes:ax.spines[['right','top']].set_visible(False)
        fig.savefig(args.output/'hybrid_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({k:findings[k] for k in ['status','natural_trials','forced_checks','errors',
        'AB_prefix_actions_scores_and_first_target_equal_cases','formal_guidance_calls','formal_trigger_events','bundle_sha256']}))


if __name__=='__main__':main()

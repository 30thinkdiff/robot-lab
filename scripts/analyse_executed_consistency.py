"""已执行动作一致性Pilot离线核验；无独立标签时不伪造异常检测指标。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def percentile(values,p):
    a=sorted(values)
    if not a:return None
    x=(len(a)-1)*p/100;lo=int(x);hi=min(lo+1,len(a)-1)
    return a[lo]*(hi-x)+a[hi]*(x-lo) if hi!=lo else a[lo]


def stats(values):
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'median':percentile(values,50),'p05':percentile(values,5),'p95':percentile(values,95),
            'min':min(values) if values else None,'max':max(values) if values else None}


def longest_missing(rows,key):
    longest=run=0
    for r in rows:
        run=run+1 if r[key] is None else 0;longest=max(longest,run)
    return longest


def audit(z,baseline):
    read=lambda n:json.loads(z.read(n))
    state,execution,p=read('pipeline_status.json'),read('execution.json'),read('protocol.json')
    assert state['status']=='completed' and state['checkpoint_sha_unchanged'] and state['controller_actions_equal_to_baseline']
    assert state['interruptions']==state['ogg_calls']==0 and not state['automatic_expansion']
    assert execution['completed']==execution['planned']==4 and execution['errors']==0 and execution['status']=='completed'
    assert p['control_fps']==20 and p['data_fps']==10 and p['horizon_steps']==20
    assert p['action_offsets']==list(range(0,20,2)) and p['executed_first_due']==20 and p['common_evaluation_start']==24
    assert not any(p[k] for k in ['thresholds_applied','interruptions_allowed','ogg_allowed','retraining','automatic_expansion'])
    references={}
    for n in baseline.namelist():
        if n.startswith('trials/'):
            old=json.loads(baseline.read(n));case=old['case']
            if case['method']=='H50':
                path=f"attempts/{case['id']}/attempt_{old['attempt']:03d}/executed_actions.json"
                references[case['condition']]=(old,json.loads(baseline.read(path)))
    records=[read(n) for n in z.namelist() if n.startswith('trials/') and n.endswith('.json')]
    assert len(records)==4
    windows=[];cases=[];latencies=[];feature_captures=[];condition_differences=[]
    for record in records:
        c,r=record['case'],record['result'];condition=c['condition']
        assert record['status']=='completed' and r['baseline_unchanged'] and c['seed']==1000
        assert r['events']==r['guidance_calls']==0
        prefix=f"attempts/{c['id']}/attempt_{record['attempt']:03d}/"
        capture=read(prefix+'feature_capture.json');commands=read(prefix+'controller_commands.json')
        rows=[json.loads(l) for l in z.read(prefix+'monitor_steps.jsonl').decode().splitlines()]
        plans=read(prefix+'planned_action_conditions.json');times=read(prefix+'predictor_latencies.json')
        assert len(rows)==len(commands)==r['env_steps']==capture['shape'][0]
        assert capture['shape'][1:]==[128,960] and capture['dtype']=='float32' and capture['reset_count']==1
        assert capture['baseline_actions_exactly_equal']
        old,old_actions=references[condition]
        assert all(r[k]==old['result'][k] for k in ['success','env_steps','chunk_calls'])
        assert [cmd['raw_controller_action'] for cmd in commands]==old_actions
        for t,(cmd,row) in enumerate(zip(commands,rows)):
            assert cmd['step']==row['step']==t and cmd['simulation_seconds']==t/20 and cmd['returned']
            assert all(len(cmd[k])==7 and all(math.isfinite(v) for v in cmd[k])
                       for k in ['raw_controller_action','processor_normalized_action','policy_normalized_action'])
            assert not row['trigger'] and row['threshold'] is None and not row['threshold_applied']
            assert not row['interruptions_allowed'] and not row['ogg_allowed'] and not row.get('interrupted')
            assert row['eligible_executed']==(t>=20) and row['eligible_common']==(t>=24)
            if t<20:
                assert row['executed_score'] is None;continue
            origin=t-20;assert row['origin_step']==origin and row['action_window']==[origin,t-1]
            assert row['action_indices']==list(range(origin,t,2)) and row['prediction_due_step']==t
            assert math.isfinite(row['executed_score'])
            reconstructed=1-row['dot']/(max(row['expected_norm'],1e-8)*max(row['observed_norm'],1e-8))
            assert math.isclose(row['executed_score'],reconstructed,abs_tol=3e-6)
            if origin>=20:assert row['origin_z_sha256']==rows[origin]['observed_z_sha256']
            if t>=24:
                windows.append({'condition':condition,'due_step':t,'origin_step':origin,
                    'due_seconds':t/20,'episode_success':r['success'],
                    'planned_score':row['planned_score'],'executed_score':row['executed_score'],
                    'window_type':'overlap' if row['planned_score'] is not None else 'executed_only',
                    'planned_missing_reason':None})
        assert sum(row['executed_score'] is not None for row in rows)==len(rows)-20
        common=rows[24:];scored=sum(v['planned_score'] is not None for v in common)
        assert scored==r['planned_scored_steps'] and len(common)==r['executed_scored_steps']==r['common_eligible_steps']
        assert sum(v.get('prediction_skipped')=='insufficient_planned_actions' for v in rows)==r['planned_skips']
        assert sum(v.get('cancelled_on_new_plan',0) for v in rows)==r['planned_cancellations']
        assert len(plans)==sum(t['role']=='planned' for t in times)==len(rows)-4-r['planned_skips']
        assert sum(t['role']=='executed' for t in times)==len(rows)-20
        # 重放原计划到期/取消调度，不能把取消的预测当成负样本或评分。
        pending=set();cancelled_due=set();by_origin={v['origin_step']:v for v in plans}
        for t,row in enumerate(rows):
            assert (t in pending)==(row['planned_score'] is not None)
            pending.discard(t)
            if 'cancelled_on_new_plan' in row:
                assert row['cancelled_on_new_plan']==len(pending)
                cancelled_due.update(pending);pending.clear()
            if t in by_origin:
                entry=by_origin[t];assert entry['due_step']==t+20 and entry['offsets']==list(range(0,20,2))
                pending.add(t+20)
        for w in [v for v in windows if v['condition']==condition and v['window_type']=='executed_only']:
            if w['due_step'] in cancelled_due:w['planned_missing_reason']='cancelled_on_new_plan'
            else:
                assert rows[w['origin_step']].get('prediction_skipped')=='insufficient_planned_actions'
                w['planned_missing_reason']='insufficient_planned_actions'
        for row in common:
            if row['planned_score'] is not None:
                planned=by_origin[row['origin_step']]['normalized_actions']
                actual=[commands[i]['processor_normalized_action'] for i in row['action_indices']]
                difference=max(abs(a-b) for ap,bp in zip(planned,actual) for a,b in zip(ap,bp))
                condition_differences.append(difference)
        source_groups=[('predictor',times),('shared_visual',read(prefix+'shared_visual_latencies.json')),
                       ('normalization',read(prefix+'normalization_latencies.json'))]
        for category,source in source_groups:
            for item in source:
                assert math.isfinite(item['seconds']) and item['seconds']>0
                latencies.append({'condition':condition,'step':item['step'],
                    'role':item['role'] if category=='predictor' else category,'milliseconds':1000*item['seconds']})
        actual=read(prefix+'actual_protocol.json');assert actual['protocol']==p
        cases.append({'condition':condition,'success':r['success'],'env_steps':r['env_steps'],
            'common_eligible':len(common),'planned_scored':scored,'executed_scored':len(common),
            'planned_coverage':scored/len(common),'executed_coverage':1.,
            'planned_max_gap_steps':longest_missing(common,'planned_score'),
            'executed_max_gap_steps':longest_missing(common,'executed_score'),
            'planned_skips':r['planned_skips'],'planned_cancellations':r['planned_cancellations'],
            'normalization_max_roundtrip_error':actual['normalizer_max_roundtrip_error'],
            'baseline_exact':True,'episode_wall_s':record['wall_s'],
            'peak_torch_allocated_gib':r['peak_torch_allocated_gib'],'video_remote_path':r['video_paths'][0]})
        feature_captures.append({'condition':condition,**capture})
    paired=[w for w in windows if w['planned_score'] is not None]
    score_groups={}
    for outcome in [True,False]:
        for kind in ['all','overlap','executed_only']:
            subset=[w for w in windows if w['episode_success']==outcome and (kind=='all' or w['window_type']==kind)]
            label=('successful_episode' if outcome else 'failed_episode')+'/'+kind
            score_groups[label]={'executed':stats([w['executed_score'] for w in subset]),
                                 'planned':stats([w['planned_score'] for w in subset if w['planned_score'] is not None])}
    latency_groups={}
    for role in ['planned','executed','shared_visual','normalization']:
        values=[v for v in latencies if v['role']==role]
        latency_groups[role]={'all':stats([v['milliseconds'] for v in values])}
        steady=[]
        for condition in sorted({v['condition'] for v in values}):
            steady.extend(v['milliseconds'] for v in [w for w in values if w['condition']==condition][5:])
        latency_groups[role]['excluding_first_5_per_episode']=stats(steady)
    summary={'status':'verified','trials':4,'errors':0,'wall_seconds':execution['wall_seconds'],
        'successes':sum(c['success'] for c in cases),'control_unchanged':True,
        'controller_commands':sum(c['env_steps'] for c in cases),'interruptions':0,'ogg_calls':0,
        'common_eligible_windows':len(windows),'planned_scored':len(paired),'executed_scored':len(windows),
        'planned_coverage':len(paired)/len(windows),'executed_coverage':1.,
        'newly_scored_windows':len(windows)-len(paired),
        'new_windows_from_planned_skips':sum(w['planned_missing_reason']=='insufficient_planned_actions' for w in windows),
        'new_windows_from_planned_cancellations':sum(w['planned_missing_reason']=='cancelled_on_new_plan' for w in windows),
        'planned_max_gap_steps':max(c['planned_max_gap_steps'] for c in cases),'executed_max_gap_steps':0,
        'overlap_absolute_score_difference':stats([abs(w['executed_score']-w['planned_score']) for w in paired]),
        'overlap_action_condition_max_abs_difference':stats(condition_differences),
        'score_distributions_by_episode_outcome_not_anomaly_labels':score_groups,
        'latency_milliseconds':latency_groups,
        'feature_bytes_recorded_remotely':sum(c['shape'][0]*128*960*4 for c in feature_captures),
        'raw_feature_and_video_bytes_verified_locally':False,
        'anomaly_detection_metrics':{'status':'not_identifiable_without_independent_window_labels',
                                    'AUROC':None,'recall':None,'false_positive_rate':None,'threshold':None},
        'interpretation':'retrospective due-consistency; coverage is scheduling availability, not detection recall'}
    return summary,cases,windows,latencies,feature_captures


def annotated_auc(windows,path):
    rows=list(csv.DictReader(path.open()))
    labels={}
    for row in rows:
        if not row.get('label'):continue
        if row['label'] not in {'normal','anomaly'} or row.get('annotation_source') not in {'manual_video','independent_sim_event'}:
            raise ValueError('标签须来自独立人工视频/仿真事件，不能把结局或模型分数冒充标签')
        key=(row['condition'],int(row['due_step']))
        if key in labels:raise ValueError('标签重复')
        labels[key]=row['label']=='anomaly'
    selected=[w for w in windows if (w['condition'],w['due_step']) in labels]
    if len(selected)!=len(labels):raise ValueError('标签时刻不属于本次共同评测窗口')
    overlap=[w for w in selected if w['planned_score'] is not None]
    def ranking(valid,method):
        pos=[w[method+'_score'] for w in valid if labels[(w['condition'],w['due_step'])]]
        neg=[w[method+'_score'] for w in valid if not labels[(w['condition'],w['due_step'])]]
        return {'labelled_valid_windows':len(valid),'positive':len(pos),'negative':len(neg),
            'AUROC':sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None,
            'warning':'descriptive ranking of correlated windows; no calibrated threshold'}
    return {'same_labelled_overlap':{m:ranking(overlap,m) for m in ['executed','planned']},
            'executed_all_labelled':ranking(selected,'executed'),
            'labelled_windows':len(selected),'overlap_windows':len(overlap),
            'threshold':None,'note':'compare methods only on same_labelled_overlap; full executed coverage is a separate scope'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--baseline-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path);p.add_argument('--plot',action='store_true');args=p.parse_args()
    with zipfile.ZipFile(args.bundle) as z,zipfile.ZipFile(args.baseline_bundle) as baseline:
        assert z.testzip() is None
        summary,cases,windows,latencies,captures=audit(z,baseline)
        args.output.mkdir(parents=True,exist_ok=True)
        for n in ['pipeline_status.json','execution.json','protocol.json','compatibility.json']:
            (args.output/n).write_bytes(z.read(n))
    summary['bundle_sha256']=hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    if args.labels:summary['optional_independent_labels']=annotated_auc(windows,args.labels)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.output/'feature_inventory.json').write_text(json.dumps(captures,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('cases.csv',cases),('windows.csv',windows),('latencies.csv',latencies)]:
        with (args.output/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    template=args.output/'anomaly_annotation_template.csv'
    if not template.exists():
        with template.open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['condition','due_step','origin_step','label','annotation_source','notes'],lineterminator='\n')
            writer.writeheader();writer.writerows({k:w[k] for k in ['condition','due_step','origin_step']} for w in windows)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11.5,7),constrained_layout=True)
        axes[0,0].bar(['Planned','Executed'],[100*summary['planned_coverage'],100]);axes[0,0].set(ylim=(0,110),title='Common-window scoring coverage (%)')
        subset=[w for w in windows if w['condition']=='libero_object/task_1/init_0']
        for name,color,style in [('executed','#e08025','-'),('planned','#2277aa','--')]:
            axes[0,1].plot([w['due_step'] for w in subset],[w[name+'_score'] for w in subset],label=name,color=color,linestyle=style)
        axes[0,1].set(title='Task1/init0: raw consistency scores',xlabel='Control step',ylabel='1 - cosine');axes[0,1].legend()
        for outcome in [True,False]:
            values=[w['executed_score'] for w in windows if w['episode_success']==outcome]
            axes[1,0].hist(values,bins=25,alpha=.5,density=True,label='Episode success' if outcome else 'Episode failure')
        axes[1,0].set(title='Executed scores by episode outcome (not anomaly truth)',xlabel='1 - cosine');axes[1,0].legend(fontsize=8)
        roles=['planned','executed','shared_visual','normalization']
        med=[summary['latency_milliseconds'][r]['excluding_first_5_per_episode']['median'] for r in roles]
        axes[1,1].bar(roles,med);axes[1,1].set(title='Median component latency, excluding first 5/case',ylabel='ms')
        for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
        fig.savefig(args.output/'executed_consistency.png',dpi=170);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['status','trials','errors','controller_commands','planned_scored',
        'executed_scored','common_eligible_windows','newly_scored_windows','planned_max_gap_steps','executed_max_gap_steps','bundle_sha256']}))


if __name__=='__main__':main()

"""完整任务对照离线审计：冻结规则、配对成败、队列消费、真实历史与成本。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

METHODS=['H50','H10','planned_lvm','executed_lvm']


def audit(z):
    read=lambda n:json.loads(z.read(n))
    execution=read('execution.json');state=read('pipeline_status.json');plan=read('plan.json');p=read('protocol.json')
    assert execution['status']==state['status']=='completed' and execution['completed']==execution['planned']==48
    assert execution['errors']==0 and state['checkpoint_sha_unchanged'] and not state['automatic_expansion'] and state['ogg_calls']==0
    assert plan['protocol']==p and p['ogg'] is False and p['test_result_tuning'] is False
    assert p['seeds']==[1002,1003] and p['max_steps']==280 and p['chunk_size']==50 and p['num_steps']==10
    rule=p['calibration']['rule'];values=sorted(p['calibration']['source']['scores'])
    i=(len(values)-1)*.95;lo=int(i);hi=min(lo+1,len(values)-1)
    assert rule['threshold']==values[lo]*(hi-i)+values[hi]*(i-lo)
    records=[read(n) for n in z.namelist() if n.startswith('trials/') and n.endswith('.json')]
    assert len(records)==48
    cases=[];signatures={};seen=set();triggers=[];trajectories={}
    for r in records:
        c,result=r['case'],r['result'];method=c['method'];n=result['env_steps']
        assert r['status']=='completed' and type(result['success']) is bool and c in plan['cases'] and c['id'] not in seen
        seen.add(c['id']);assert result['guidance_calls']==0 and 1<=n<=280
        assert c['seed'] in [1002,1003] and c['condition'] in p['conditions']
        assert p['conditions'][c['condition']]['initial_state_id'] in [3,4,5]
        signature=tuple(result['initial_signature']);key=(c['condition'],c['seed'])
        signatures.setdefault(key,set()).add(signature)
        prefix=f"attempts/{c['id']}/attempt_{r['attempt']:03d}/"
        assert read(prefix+'actual_protocol.json')['protocol']==p
        actions=read(prefix+'executed_actions.json');assert len(actions)==n
        assert all(len(a)==7 and all(math.isfinite(v) for v in a) for a in actions)
        effective=0;trace=[];max_gap=gap=0
        if method in ['H50','H10']:
            h=50 if method=='H50' else 10
            assert result['chunk_calls']==math.ceil(n/h) and result['monitor_s']==0 and result['event_replans']==0
        else:
            trace=[json.loads(l) for l in z.read(prefix+'monitor_steps.jsonl').decode().splitlines()]
            commands=read(prefix+'controller_commands.json')
            assert len(trace)==len(commands)==n
            assert [t['step'] for t in trace]==list(range(n)) and all(t['mode']==method for t in trace)
            assert [a['raw_controller_action'] for a in commands]==actions
            assert all(a['returned'] and a['step']==i for i,a in enumerate(commands))
            count=0;ewma=None;last=None;next_allowed=0;events=0;q=0;calls=0;scored=0
            pending=set()
            for t,row in enumerate(trace):
                score=row['score'];assert row['threshold']==rule['threshold']
                if method=='executed_lvm':
                    assert (score is not None)==(t>=24)
                    if t>=24:
                        assert row['origin_step']==t-20 and row['action_indices']==list(range(t-20,t,2))
                else:
                    assert (score is not None)==(t in pending);pending.discard(t)
                should=False
                if score is not None:
                    assert math.isfinite(score);scored+=1
                    if last is not None and t!=last+1:count=0
                    last=t;ewma=score if ewma is None else rule['alpha']*score+(1-rule['alpha'])*ewma
                    count=count+1 if ewma>rule['threshold'] and t>=next_allowed else 0
                    should=count>=rule['consecutive']
                    if should:count=0;next_allowed=t+rule['cooldown_steps']
                    assert math.isclose(row['ewma'],ewma,rel_tol=1e-12)
                    assert row['next_allowed']==next_allowed
                assert row['trigger']==should
                if should:
                    assert row['discarded_actions']==q
                    events+=1;effective+=int(q>0)
                    triggers.append({'condition':c['condition'],'seed':c['seed'],'method':method,
                        'step':t,'score':score,'ewma':ewma,'threshold':rule['threshold'],'discarded_actions':q})
                    q=0
                    if method=='planned_lvm':
                        assert row['cancelled_on_interrupt']==len(pending);pending.clear()
                fresh=q==0
                if fresh:q=50;calls+=1
                q-=1
                if method=='planned_lvm':
                    if fresh:
                        assert row['cancelled_on_new_plan']==len(pending);pending.clear()
                    if t>=4 and row.get('prediction_skipped') is None:pending.add(t+20)
            assert events==result['event_replans'] and calls==result['chunk_calls']
            assert scored==result['scored_steps'] and result['eligible_steps']==max(0,n-24)
            for row in trace[24:]:
                gap=gap+1 if row['score'] is None else 0
                max_gap=max(max_gap,gap)
        trajectories[(key,method)]={'actions':actions,'trace':trace}
        cases.append({'condition':c['condition'],'seed':c['seed'],'method':method,'success':result['success'],
            **{k:result[k] for k in ['env_steps','chunk_calls','policy_s','monitor_s','event_replans','scored_steps','eligible_steps',
                                     'planned_skips','planned_cancellations','peak_torch_allocated_gib']},
            'effective_interruptions':effective,'longest_unscored_gap_steps':max_gap if trace else None,
            'episode_wall_s':r['wall_s'],'video_remote_path':result['video_paths'][0]})
    assert len(signatures)==12 and all(len(s)==1 for s in signatures.values()) and seen=={c['id'] for c in plan['cases']}
    rows=[]
    for method in METHODS:
        sub=[r for r in cases if r['method']==method];assert len(sub)==12
        row={'method':method,'completed':len(sub),'successes':sum(r['success'] for r in sub),'success_rate':sum(r['success'] for r in sub)/12}
        for k in ['env_steps','chunk_calls','policy_s','monitor_s','event_replans','effective_interruptions','episode_wall_s','peak_torch_allocated_gib']:
            row['mean_'+k]=statistics.mean(c[k] for c in sub)
        eligible=sum(c['eligible_steps'] for c in sub);row['score_coverage']=sum(c['scored_steps'] for c in sub)/eligible if eligible else None
        row['scored_steps']=sum(c['scored_steps'] for c in sub);row['eligible_steps']=eligible
        row['longest_unscored_gap_steps']=max(c['longest_unscored_gap_steps'] for c in sub) if eligible else None
        rows.append(row)
    mechanisms=[]
    for key in sorted(signatures):
        base=trajectories[(key,'H50')]['actions'];entry={'condition':key[0],'seed':key[1]}
        for method in ['planned_lvm','executed_lvm']:
            data=trajectories[(key,method)];events=[r for r in data['trace'] if r['trigger']]
            first=events[0]['step'] if events else None
            divergence=next((i for i,(a,b) in enumerate(zip(base,data['actions'])) if a!=b),None)
            # The first action difference must occur at the recorded intervention.
            assert divergence==first
            entry[method+'_first_trigger']=first
            entry[method+'_first_action_divergence_from_H50']=divergence
        first=entry['executed_lvm_first_trigger'];planned=trajectories[(key,'planned_lvm')]
        entry['planned_score_at_executed_first_trigger']=planned['trace'][first]['score'] if first is not None else None
        entry['shared_actions_before_executed_first_trigger']=(
            planned['actions'][:first]==trajectories[(key,'executed_lvm')]['actions'][:first] if first is not None else None)
        mechanisms.append(entry)
    paired=[]
    for reference in ['H50','H10','planned_lvm']:
        b=c=both_ok=both_fail=0
        for key in signatures:
            base=next(r for r in cases if (r['condition'],r['seed'])==key and r['method']==reference)
            new=next(r for r in cases if (r['condition'],r['seed'])==key and r['method']=='executed_lvm')
            b+=int(new['success'] and not base['success']);c+=int(base['success'] and not new['success'])
            both_ok+=int(new['success'] and base['success']);both_fail+=int(not new['success'] and not base['success'])
        discordant=b+c
        paired.append({'reference':reference,'candidate':'executed_lvm','matched_cases':12,
            'candidate_only_success':b,'reference_only_success':c,'both_success':both_ok,'both_failure':both_fail,
            'net_successes':b-c,'paired_success_rate_difference':(b-c)/12,
            'descriptive_exact_McNemar_p':min(1.,2*sum(math.comb(discordant,j) for j in range(min(b,c)+1))/2**discordant) if discordant else 1.,
            'limit':'exploratory, fixed two tasks, 6 init states x 2 noise seeds; no multiplicity correction or population claim'})
    findings={'status':'verified','completed':48,'errors':0,'matched_condition_seed_groups':12,
        'rule_replayed':True,'queue_calls_replayed':True,'ogg_calls':0,'checkpoint_sha_unchanged_reported':True,
        'wall_seconds':execution['wall_seconds'],'calibration':p['calibration'],
        'limits':['single successful reference trajectory, not independently normal-labelled calibration',
                  'only 2 known tasks, 6 physical initial states and 2 noise seeds',
                  'same fixed rule, different score availability changes EWMA/persistence history',
                  'raw videos not exported or locally reviewed','no OGG or retraining']}
    return findings,rows,cases,paired,triggers,mechanisms


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    with zipfile.ZipFile(args.bundle) as z:
        assert z.testzip() is None
        findings,rows,cases,paired,triggers,mechanisms=audit(z)
        args.output.mkdir(parents=True,exist_ok=True)
        for n in ['pipeline_status.json','execution.json','protocol.json']:(args.output/n).write_bytes(z.read(n))
    findings['bundle_sha256']=hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    for name,data in [('overall.csv',rows),('cases.csv',cases),('paired.csv',paired),('triggers.csv',triggers),
                      ('first_interventions.csv',mechanisms)]:
        if data:
            with (args.output/name).open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12.5,4),constrained_layout=True)
        names=[r['method'] for r in rows]
        axes[0].bar(names,[r['successes'] for r in rows]);axes[0].set(ylim=(0,13),title='Complete-task successes (12 each)')
        axes[1].bar(names,[r['mean_chunk_calls'] for r in rows]);axes[1].set(title='Mean full VLA chunk calls')
        axes[2].bar(names,[r['mean_episode_wall_s'] for r in rows]);axes[2].set(title='Mean episode wall time (s)')
        for ax in axes:
            ax.tick_params(axis='x',rotation=25);ax.spines[['top','right']].set_visible(False)
        fig.savefig(args.output/'closed_loop_performance.png',dpi=170);plt.close(fig)
    print(json.dumps({'audit':findings['status'],'methods':rows,'paired':paired},ensure_ascii=False))


if __name__=='__main__':main()

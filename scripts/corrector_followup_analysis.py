"""Deterministic final aggregation of Nov16 read-only metadata snapshots."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import statistics

METHODS=['H50','H20','H10','H5','H1']


def analyse(data):
    phase=data['phases']['plus_horizons'];execution=phase['execution']
    if execution['status']!='completed' or execution['completed']!=96 or execution['errors']:
        raise ValueError('final analysis requires all 96 error-free trials')
    rows=data['original_baselines']+phase['cases'];index={}
    for r in rows:
        c=r['case'];key=(c['condition'],c['seed'],c['method'])
        if key in index or type(r['result']['success']) is not bool:raise ValueError('duplicate/unknown outcome')
        index[key]=r
    conditions=data['plus_conditions']
    if len(conditions)!=32 or set(index)!={(c,1000,m) for c in conditions for m in METHODS}:
        raise ValueError('condition/seed/method matrix differs')
    if not all(r.get('matches_original_H20_signatures') is True and r['result']['horizon_audit']['queue_verified'] is True for r in phase['cases']):
        raise ValueError('initial-signature or queue audit failed')
    methods={};pairs={};categories={};flat=[]
    for method in METHODS:
        group=[r for key,r in index.items() if key[2]==method]
        metrics={k:statistics.mean(r['result'][k] for r in group) for k in ['env_steps','chunk_calls','policy_s','episode_wall_s','peak_torch_allocated_gib']}
        successes=sum(r['result']['success'] for r in group)
        methods[method]={'completed':len(group),'successes':successes,'success_rate':successes/len(group),**{'mean_'+k:v for k,v in metrics.items()}}
        if method in ['H10','H5','H1']:
            methods[method]['mean_denoise_forward_calls']=statistics.mean(r['result']['horizon_audit']['denoise_forward_calls'] for r in group)
            methods[method]['command_translation_delta_mean']=statistics.mean(r['result']['action_continuity']['all_translation_delta_mean'] for r in group)
            b=[(index[(c,1000,'H20')]['result']['success'],index[(c,1000,method)]['result']['success']) for c in conditions]
            pairs[method]={'both_success':sum(a and b for a,b in b),'both_failure':sum(not a and not b for a,b in b),
                'new_only_success':sum(not a and b for a,b in b),'H20_only_success':sum(a and not b for a,b in b)}
        for r in group:
            c=r['case'];v=r['result'];flat.append({'condition':c['condition'],'category':conditions[c['condition']]['category'],
                'seed':c['seed'],'method':method,**{k:v[k] for k in ['success','env_steps','chunk_calls','policy_s','episode_wall_s']},
                'same_initial_signatures_as_H20':r.get('matches_original_H20_signatures','historical baseline')})
    for category in sorted({v['category'] for v in conditions.values()}):
        names=[c for c,v in conditions.items() if v['category']==category]
        categories[category]={'n':len(names),**{m:sum(index[(c,1000,m)]['result']['success'] for c in names) for m in METHODS}}
    patterns=[{'condition':c,'seed':1000,'category':conditions[c]['category'],
        **{m:index[(c,1000,m)]['result']['success'] for m in METHODS}} for c in sorted(conditions)]
    return {'execution':execution,'methods':methods,'paired_vs_H20':pairs,'categories':categories,
        'initial_signatures_verified':96,'queue_audits_passed':96,'scope':'32 previously selected Plus conditions, eval seed1000; not full benchmark or independent seed replication'},flat,patterns


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.snapshot.read_bytes();data=json.loads(raw);summary,flat,patterns=analyse(data)
    summary.update(captured_utc=data['captured_utc'],source_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        queue_status=data['queue']['status'],disk_free_gib=data['disk_free_gib'])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('plus_horizon_results.csv',flat),('plus_horizon_pairs.csv',patterns)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    print(json.dumps(summary['methods'],ensure_ascii=False))


if __name__=='__main__':main()

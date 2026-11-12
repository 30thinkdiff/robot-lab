"""只读Case Bank研究覆盖审计；分开协议队列与物理状态，不启动模型或实验。"""
import argparse
from collections import defaultdict, Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools/robotlab_replay'))
from data import Catalog
from casebank import MODELS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def signature(result):
    value=result.get('initial_signature')
    if isinstance(value,list) and len(value)==3:return tuple(value)
    keys=['initial_state_sha256','first_input_sha256','first_noise_sha256']
    return tuple(result[k] for k in keys) if all(result.get(k) for k in keys) else None


def summarize_instances(instances, rows):
    used={i for ids in instances for i in ids};selected=[rows[i] for i in sorted(used)]
    def complete(ids):return all(rows[i]['replay_complete'] for i in ids)
    return {'cases':len(instances),'physical_states':len({r['condition'] for r in selected}),
            'evaluation_seed_values':sorted({r['seed'] for r in selected}),
            'evaluation_seed_value_count':len({r['seed'] for r in selected}),
            'local_complete_cases':sum(complete(ids) for ids in instances),
            'cases_with_all_traces':sum(all(rows[i]['trace'] or rows[i]['baseline'] for i in ids) for ids in instances),
            'cases_with_all_commands':sum(all(rows[i]['commands'] for i in ids) for ids in instances),
            'cases_with_numeric_ogg':sum(any(rows[i]['numeric_ogg'] for i in ids) for ids in instances),
            'all_initial_hashes_verified':sum(all(rows[i]['signature'] for i in ids) and len({tuple(rows[i]['signature']) for i in ids})==1 for ids in instances),
            'examples':[{'cohort':rows[ids[0]]['cohort'],'condition':rows[ids[0]]['condition'],
                         'seed':rows[ids[0]]['seed'],'trials':[rows[i]['id'] for i in ids]} for ids in instances[:8]]}


def put_group(groups,row,index):
    key=(row['cohort'],row['phase'],row['condition'],row['seed']);method=row['method']
    if method in groups[key]:raise ValueError('协议组内出现重复方法，不按结局择优合并')
    groups[key][method]=index


def analyze(progress, output, overnight=None):
    if output.exists():raise FileExistsError('派生输出已存在；用新目录，不覆盖旧审计')
    catalog=Catalog();rows=[];groups=defaultdict(dict);excluded=[];protocol_checks={};baseline_check={}
    try:
        for r in catalog.runs.values():
            if r['experiment']!='Object10 五模型' or r['forced'] or r['method'] not in ['H50','H20']:continue
            result=r['_record'].get('result',{});key=(r['condition'],r['seed'],r['method'])
            value=tuple(result.get(k) for k in ['success','env_steps','chunk_calls'])
            if key in baseline_check and baseline_check[key]!=value:raise ValueError('五模型重复基线结局/步数/调用不同，禁止当作相同基线去重')
            baseline_check[key]=value
        for r in catalog.runs.values():
            if r['forced'] or r['status']!='completed':continue
            record=r['_record'];result=record.get('result',{});p=r['_protocol'];config=record['case'].get('config',{})
            if type(result.get('success')) is not bool:
                excluded.append({'id':r['id'],'reason':'缺少布尔结局，不按失败处理'});continue
            casebank=p.get('study_id','').startswith('replay_casebank_')
            if casebank:cohort=p['study_id'];method=r['method']
            elif r['experiment']=='Object10 五模型':
                cohort='legacy_five_models'
                if r['method'] in ['H50','H20']:
                    if r['model']!='MLP':continue  # 同一基线五份归档，只取固定MLP对照。
                    method=r['method']
                elif r['model'] in MODELS and r['method'] in ['lvm','ogg']:method=MODELS[r['model']]+'_'+r['method']
                else:continue
            elif r['experiment']=='混合 Corrector':
                cohort='hybrid_pilot';method={'A':'transformer-h1_ogg','B':'hybrid_t1_a10_ogg','C':'transformer-a10-h1_ogg'}.get(r['method'])
                if method is None:continue
            elif 'consistency_closed_loop' in r['experiment']:
                cohort='consistency_closed_loop';method=r['method']
            else:continue
            fps=p.get('control_fps',p.get('fps'));policy=p.get('policy_revision')
            if fps!=20 or policy!='a24f1f380461b665158d52acd140152c79e8b36d':
                raise ValueError('核心控制频率/策略revision不一致，拒绝配对')
            protocol_checks[cohort]={'control_fps':fps,'policy_revision':policy,
                'pairing':'same frozen cohort, task/init/eval seed; initial hashes additionally verified where recorded',
                'cross_cohort_pooling':False}
            d=catalog.details(r['id']);base=r['_prefix']+f"attempts/{record['case']['id']}/attempt_{record.get('attempt',1):03d}/";s=r['_source']
            baseline=method in ['H50','H20','H10'];trace=s.exists(base+'monitor_steps.jsonl');commands=s.exists(base+'controller_commands.json') or s.exists(base+'executed_actions.json')
            guide_rows=[e['data'] for e in d['events'] if e['kind']=='ogg'];coverage=d['coverage']
            row={'id':r['id'],'trial_id':record['case']['id'],'cohort':cohort,'phase':config.get('phase','formal'),
                'condition':r['condition'],'seed':r['seed'],'method':method,'model':r['model'],'success':r['success'],
                'baseline':baseline,'steps':r['steps'],'events':result.get('event_replans'),
                'guide_calls':result.get('guidance_calls'),'trace':trace,'commands':commands,
                'video':d['video']['available'],'verified_video':d['video'].get('alignment')=='recording_verified',
                'frame_map':s.exists(base+'frame_map.jsonl'),
                'ogg_summary':bool(guide_rows),'numeric_ogg':any('candidate_arrays' in x for x in guide_rows),
                'coverage':coverage.get('fraction') if coverage else None,'max_gap_steps':coverage.get('max_gap_steps') if coverage else None,
                'score_count':coverage.get('scores') if coverage else None,'signature':signature(result),
                'wall_s':record.get('wall_s'),'source_member':r['member'],
                'trial_sha256':hashlib.sha256(s.read(r['member'])).hexdigest()}
            row['replay_complete']=row['verified_video'] and commands and (trace or baseline)
            put_group(groups,row,len(rows));rows.append(row)
        if overnight and overnight.exists():
            with zipfile.ZipFile(overnight) as z:
                p=json.loads(z.read('results/plan.json'))['protocol']
                if p['control_fps']!=20 or p['policy_revision']!='a24f1f380461b665158d52acd140152c79e8b36d':raise ValueError('夜间策略协议不符')
                for name in sorted(z.namelist()):
                    if not name.startswith('results/trials/') or not name.endswith('.json'):continue
                    record=json.loads(z.read(name));c=record['case'];result=record.get('result',{})
                    if record['status']!='completed' or type(result.get('success')) is not bool:continue
                    method={'single_lvm':'transformer-h1_lvm','single_ogg':'transformer-h1_ogg'}.get(c['method'],c['method'])
                    row={'id':'overnight:'+c['id'],'trial_id':c['id'],'cohort':'overnight_snapshot','phase':c['config']['phase'],
                        'condition':c['condition'],'seed':c['seed'],'method':method,'model':'recorded protocol',
                        'success':result['success'],'baseline':method in ['H50','H10'],'steps':result['env_steps'],
                        'events':result['event_replans'],'guide_calls':result['guidance_calls'],
                        'trace':False,'commands':False,'video':False,'verified_video':False,'frame_map':False,
                        'ogg_summary':False,'numeric_ogg':False,'coverage':result['scored_steps']/result['eligible_steps'] if result['eligible_steps'] else None,
                        'max_gap_steps':None,'score_count':result['scored_steps'],'signature':signature(result),'wall_s':record['wall_s'],
                        'replay_complete':False,'source_member':name,'trial_sha256':hashlib.sha256(z.read(name)).hexdigest()}
                    put_group(groups,row,len(rows));rows.append(row)
        patterns=defaultdict(list);pairs=defaultdict(list)
        configs=['mlp','transformer-h1','transformer-h3','transformer-a10-h1','transformer-a10-h3']
        comparisons=[(m+'_lvm',m+'_ogg','LVM vs OGG') for m in configs]
        comparisons += [('mlp_ogg',m+'_ogg','预测器 vs MLP OGG') for m in configs if m!='mlp']
        comparisons += [('transformer-h1_ogg','hybrid_t1_a10_ogg','同T-H1检测器，不同引导器'),('planned_lvm','executed_lvm','规划 vs 已执行监测')]
        for key,g in groups.items():
            cohort,phase,condition,seed=key
            # 有哈希的组必须完全一致；无哈希的历史组仅声称冻结协议匹配。
            known=[tuple(rows[i]['signature']) for i in g.values() if rows[i]['signature']]
            if known and len(set(known))!=1:raise ValueError('同条件初始状态/输入/噪声SHA不一致，不允许严格配对')
            if all(m in g for m in ['H50','H20','mlp_ogg']):
                ids=[g[m] for m in ['H50','H20','mlp_ogg']];pattern=''.join('1' if rows[i]['success'] else '0' for i in ids)
                patterns[cohort,pattern].append(ids)
            for a,b,family in comparisons:
                if a not in g or b not in g:continue
                ids=[g[a],g[b]];outcome=''.join('1' if rows[i]['success'] else '0' for i in ids)
                pairs[cohort,family,a,b,outcome].append(ids)
        phenomena=defaultdict(list)
        for i,r in enumerate(rows):
            if r['baseline'] or not r['trace']:continue
            if r['max_gap_steps'] is not None and r['max_gap_steps']>=10:phenomena[r['cohort'],'至少10步无评分'].append([i])
            if r['success'] is False and r['events']==0 and (r['score_count'] or 0)>0:phenomena[r['cohort'],'失败且有评分但未中断（非漏检真值）'].append([i])
            if r['events'] is not None and r['events']*100/max(1,r['steps'])>=5:phenomena[r['cohort'],'每100步至少5次中断（非过触发真值）'].append([i])
            if r['success'] is False and (r['events'] or 0)>0:phenomena[r['cohort'],'触发重规划后最终仍失败'].append([i])
        p=json.loads(progress.read_text()) if progress else None
        runtime=runtime_summary(p) if p else {'state':'remote progress unavailable'}
        summary={'created_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime,'records':len(rows),
            'legacy_baseline_groups_verified_equal':len(baseline_check),
            'casebank_plan_sha256':next((hashlib.sha256(r['_source'].read(r['_prefix']+'plan.json')).hexdigest() for r in catalog.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_20261112_001'),None),
            'cohorts':dict(Counter(r['cohort'] for r in rows)),'protocol_checks':protocol_checks,'excluded':excluded,
            'patterns':[{'cohort':c,'pattern':v,**summarize_instances(ids,rows)} for (c,v),ids in sorted(patterns.items())],
            'pair_matrix':[{'cohort':c,'comparison':f,'a':a,'b':b,'outcome_a_b':v,**summarize_instances(ids,rows)} for (c,f,a,b,v),ids in sorted(pairs.items())],
            'phenomena':[{'cohort':c,'phenomenon':name,**summarize_instances(ids,rows)} for (c,name),ids in sorted(phenomena.items())],
            'trace_proxy_definitions':{'long_gap_steps':10,'control_fps':20,'dense_interruptions_per_100_steps':5,'abnormality_ground_truth_available':False},
            'limits':['同一物理task/init的不同seed不算不同场景','不同cohort不合并为一次严格配对',
                '评测seed唯一值不等于独立物理场景数','旧结果缺首输入/噪声SHA时只报告冻结协议匹配',
                '历史归档中的重复基线只取固定MLP组；跨队列重复不汇总成功率',
                '缺少异常/接触等逐步真值，不能把未触发称漏检或频繁触发称误报',
                '夜间快照没有逐步trace/动作/本地视频，不能冒充完整回放',
                'Case Bank没有多层视觉特征；事件NPZ是现有单表示，不能替代多层缓存']}
        output.mkdir(parents=True);(output/'coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        write_csv(output/'trials.csv',rows)
        if p:
            write_csv(output/'runtime_costs.csv',[{'trial_id':r['case']['id'],'condition':r['case']['condition'],'seed':r['case']['seed'],
                'method':r['case']['method'],'started_utc':r['started_utc'],'wall_s':r['wall_s'],'artifact_bytes':r['bytes']} for r in p['last_trials']])
        for key in ['patterns','pair_matrix','phenomena']:
            write_csv(output/(key+'.csv'),[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items() if k!='examples'} for r in summary[key]])
        write_research_plan(summary,rows,output)
        return summary
    finally:catalog.close()


def runtime_summary(progress):
    rows=sorted(progress['last_trials'],key=lambda r:r['started_utc']);n=len(rows);recent=rows[-min(30,n):]
    seconds=statistics.mean(r['wall_s'] for r in recent);size=statistics.mean(r['bytes'] for r in recent)
    remaining=progress['execution']['planned']-n
    return {'execution':progress['execution'],'compute_processes':progress['gpu'],'completed':n,
        'all_mean_episode_seconds':statistics.mean(r['wall_s'] for r in rows),'recent_n':len(recent),
        'recent_mean_seconds':seconds,'recent_median_seconds':statistics.median(r['wall_s'] for r in recent),
        'recent_mib_per_episode':size/2**20,'root_gib':progress['root_bytes']/2**30,
        'remaining_trials':remaining,'estimated_remaining_hours':remaining*seconds/3600,
        'estimated_remaining_gib':remaining*size/2**30,'estimated_total_gib':(progress['root_bytes']+remaining*size)/2**30,
        'completed_physical_states':len({r['case']['condition'] for r in rows}),
        'completed_seed_values':sorted({r['case']['seed'] for r in rows}),
        'phase_counts':dict(Counter(r['case']['config']['phase'] for r in rows)),
        'method_counts':dict(Counter(r['case']['method'] for r in rows)),
        'projection_limit':'最近30条的方法/难度与剩余不完全相同；字节为已写文件大小，非完整磁盘分配量；原4GiB保护可能先暂停'}


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def write_research_plan(summary,rows,output):
    cohort='replay_casebank_20261112_001';current=[r for r in rows if r['cohort']==cohort]
    chosen=set();evidence=[]
    for family in ['patterns','pair_matrix','phenomena']:
        for entry in summary[family]:
            if entry['cohort']!=cohort:continue
            if family=='pair_matrix' and entry['outcome_a_b'] not in ['01','10']:continue
            physical=set()
            for example in entry['examples']:
                if example['condition'] in physical:continue
                physical.add(example['condition']);chosen.update(example['trials'])
                evidence.append({'category':entry.get('pattern',entry.get('phenomenon',entry.get('comparison'))),**example})
                if len(physical)>=2:break
    # 新清单是研究决策，不改旧run_plan；候选补测仅说明缺口，绝不交给启动器。
    optional=[]
    for condition in ['libero_object/task_2/init_1','libero_object/task_1/init_1']:
        optional.extend({'condition':condition,'seed':1000,'method':m} for m in ['H50','H20','mlp_ogg'])
    plan={'review_id':'casebank_research_20261112_v2','status':'review_complete_no_new_gpu_queue',
        'goal':'representative replay mechanisms, not unbiased population success rates',
        'parent_study_id':cohort,'preserve_original_plan_and_stop':True,
        'original_660_plan_sha256':summary.get('casebank_plan_sha256'),
        'coverage_file_sha256':sha(output/'coverage.json'),
        'original_selection_sha256':sha(REPO/'studies/policy_generalization/corrector/manifests/replay_casebank_20261112.json'),
        'completed_retained_trials':len(current),'selected_review_trials':len(chosen),
        'selected_review_trial_ids':sorted(r['trial_id'] for r in current if r['id'] in chosen),
        'selected_local_replay_ids':sorted(chosen),'all_completed_trial_ids':sorted(r['trial_id'] for r in current),
        'new_gpu_cases':[],'new_eval_seeds':[],'new_training':False,'automatic_launch':False,
        'reason':'119完整回合已覆盖全部四类基本模式、预测器/OGG/混合引导翻转和已执行监测差异；先消费现有证据',
        'gaps':[{'category':'011与111各补一个不同物理状态','currently_complete_physical_states_each':1,
            'prospective_cases_only':optional,'prospective_new_trials':6,'would_raise_total_to':len(current)+6,
            'expected_information_gain':'各增加第二个物理案例，增加机制展示代表性，不重新估计总体成功率',
            'decision':'超过120目标且现有基本模式已满足；本次不自动运行'},
            {'category':'同检测器不同OGG引导翻转','currently_complete_physical_states':1,
             'decision':'现有旧Pilot与新复现指向同一状态，不能当作两个独立场景；暂不搜索seed'},
            {'category':'真实漏检/误报真值','decision':'缺独立异常/接触标签；继续同类评测不能补足真值'}],
        'evidence':evidence,'limits':summary['limits']}
    (output/'reduced_plan_v2.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    write_csv(output/'review_cases.csv',[r for r in current if r['id'] in chosen])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--progress',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--overnight',type=Path,default=REPO/'outputs/corrector_overnight_20261111/snapshot_20261112_105732.zip');a=p.parse_args()
    s=analyze(a.progress,a.output,a.overnight);print(json.dumps({'records':s['records'],'cohorts':s['cohorts'],'runtime':s['runtime']},ensure_ascii=False,indent=2))

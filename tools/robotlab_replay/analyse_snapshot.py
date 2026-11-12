"""只读夜间快照，输出匹配条件统计与五模型预测误差诊断；不运行模型。"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from analyse_corrector_overnight import auc, correlation

METHODS = ['H50','H10','planned_lvm','executed_lvm','single_lvm','single_ogg']
MODELS = ['mlp','transformer-h1','transformer-h3','transformer-a10-h1','transformer-a10-h3']


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def analyse(snapshot, output):
    with zipfile.ZipFile(snapshot) as z:
        if sum(i.file_size for i in z.infolist()) > 512*2**20:raise ValueError('快照超过512MiB预算')
        execution=json.loads(z.read('results/execution.json'))
        queue=json.loads(z.read('queue_status.json'))
        protocol=json.loads(z.read('results/plan.json'))['protocol']
        records=[json.loads(z.read(n)) for n in z.namelist() if n.startswith('results/trials/') and n.endswith('.json')]
        if any(r['status']!='completed' for r in records):raise ValueError('快照包含非完成记录')
        if len(records)!=execution['completed']:raise ValueError('执行汇总与原trial数量不一致，停止，避免混用快照')
    groups=defaultdict(dict)
    for r in records:
        c=r['case'];key=(c['condition'],c['seed'])
        if c['method'] in groups[key]:raise ValueError('重复case条件，不合并统计')
        groups[key][c['method']]=r
    tables=[];pairs=[];task_rows=[]
    for scope,methods in [('core_six',METHODS),('matched_four',METHODS[:4])]:
        common={key:g for key,g in groups.items() if all(m in g for m in methods)}
        for method in methods:
            rows=[g[method]['result'] for g in common.values()]
            eligible=sum(r['eligible_steps'] for r in rows)
            tables.append({'scope':scope,'method':method,'completed':len(rows),
                'physical_initial_states':len({k[0] for k in common}),
                'successes':sum(r['success'] for r in rows),'success_rate':statistics.mean(r['success'] for r in rows),
                'mean_env_steps':statistics.mean(r['env_steps'] for r in rows),
                'mean_chunk_calls':statistics.mean(r['chunk_calls'] for r in rows),
                'mean_event_replans':statistics.mean(r['event_replans'] for r in rows),
                'mean_policy_s':statistics.mean(r['policy_s'] for r in rows),
                'mean_monitor_s':statistics.mean(r['monitor_s'] for r in rows),
                'mean_rollout_wall_s':statistics.mean(r['rollout_wall_s'] for r in rows),
                'score_coverage':sum(r['scored_steps'] for r in rows)/eligible if eligible else None})
        for a,b in [('H50','H10'),('H50','executed_lvm'),('planned_lvm','executed_lvm'),('executed_lvm','H10'),('single_lvm','single_ogg')]:
            if a not in methods or b not in methods:continue
            gain=sum(g[b]['result']['success'] and not g[a]['result']['success'] for g in common.values())
            loss=sum(g[a]['result']['success'] and not g[b]['result']['success'] for g in common.values())
            pairs.append({'scope':scope,'reference':a,'candidate':b,'matched':len(common),
                'candidate_only_success':gain,'reference_only_success':loss,'net_successes':gain-loss})
        if scope=='matched_four':
            for task in range(10):
                sub=[g for k,g in common.items() if f'/task_{task}/' in k[0]]
                task_rows.append({'task':task,'matched':len(sub),**{m:sum(g[m]['result']['success'] for g in sub) for m in methods}})
    # 所有五模型在同一批H50实际轨迹、同一executed窗口上比较，避免把不同控制器的轨迹差别归给预测模型。
    predictions=[]
    for model in MODELS:
        for scope in ['full','early']:
            rows=[]
            for r in records:
                if r['case']['method']!='H50':continue
                values=[m for m in r['result']['prediction_summary'] if m['model']==model and m['condition_role']=='executed' and m['scope']==scope]
                if len(values)!=1:raise ValueError('五模型共同轨迹摘要缺失或重复')
                m=values[0]
                if scope=='early' and not m['early_prefix_complete']:continue
                rows.append({**m,'success':r['result']['success']})
            errors=[r['mse'] for r in rows];failures=[int(not r['success']) for r in rows]
            predictions.append({'controller':'H50','condition_role':'executed','model':model,'scope':scope,'episodes':len(rows),
                'episode_macro_mse':statistics.mean(errors),
                'episode_macro_relative_mse':statistics.mean(r['relative_mse'] for r in rows),
                'success_mse':statistics.mean(r['mse'] for r in rows if r['success']),
                'failure_mse':statistics.mean(r['mse'] for r in rows if not r['success']),
                'error_vs_failure_correlation':correlation(errors,failures),
                'episode_failure_auc':auc(errors,failures)})
    summary={'source_archive':snapshot.name,'source_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'execution':execution,'queue_status':queue['status'],'checkpoint_sha_unchanged':queue.get('result',{}).get('checkpoint_sha_unchanged'),
        'weights':{m:{k:v for k,v in protocol['all_weights'][m].items() if k.endswith('sha256')} for m in MODELS},
        'paired_tables':tables,'paired_outcomes':pairs,'by_task':task_rows,'prediction_on_same_H50_trajectories':predictions,
        'limits':['H20 absent from frozen plan','same physical initial states repeated across evaluation seeds',
            'single methods only in core; compare six methods on common core conditions',
            'full-episode MSE may follow failure; early-prefix association is not causation',
            'five predictors scored common trajectories; they did not each control those trajectories',
            'timing includes passive shared capture; no native throughput claim']}
    core_n=next(r['completed'] for r in tables if r['scope']=='core_six')
    four_n=next(r['completed'] for r in tables if r['scope']=='matched_four')
    h50_n=sum(r['case']['method']=='H50' for r in records)
    pair=lambda scope,a,b:next(r for r in pairs if r['scope']==scope and r['reference']==a and r['candidate']==b)
    ogg_pair=pair('core_six','single_lvm','single_ogg')
    executed_pair=pair('matched_four','planned_lvm','executed_lvm')
    base_pair=pair('matched_four','H50','executed_lvm')
    h10_pair=pair('matched_four','executed_lvm','H10')
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'matched_methods.csv',tables);write_csv(output/'paired_outcomes.csv',pairs)
    write_csv(output/'task_successes.csv',task_rows);write_csv(output/'prediction_accuracy.csv',predictions)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    report=['# 夜间冻结扩展实验分析 · 2026-11-12','',
        f"快照 `{snapshot.name}`，SHA256 `{summary['source_sha256']}`。只读取原记录，不加载权重或运行仿真。",'',
        f"队列状态 **{queue['status']}**；执行状态 **{execution['status']} / {execution.get('reason')}**；完成 **{execution['completed']}/{execution['planned']}**，错误 **{execution['errors']}**。",'',
        '北京时间10:00截止，在回合边界停止启动新任务，保存结果后暂停；不是全部3400回合完成，也不是异常崩溃。saved child_pid为空，权重SHA未变；本快照不代替实时GPU进程查询。',
        f"实际执行耗时 {execution['wall_seconds']/3600:.2f}小时。末次资源记录：系统盘剩余{queue['resources']['system_free_gib']:.2f}GiB、数据盘{queue['resources']['data_free_gib']:.2f}GiB；不是空间不足退出。",'',
        '## 方法、目标与公平比较','',
        f'核心计划600回合=Object10全部10任务×init10～14×评测seed1004/1005×6方法；当前共同完成{core_n}组（{core_n*6}回合）。扩展部分init15～49只跑前4方法。总体3400是截止前尽力执行的有限上界，不承诺全部跑完。', '',
        '| 方法 | 控制含义 |','| --- | --- |',
        '| H50 / H10 | 纯SmolVLA，生成50步，只执行50/10步后清空缓存并重新推理 |',
        '| planned_lvm | A10-H1以未来规划动作预测；保留计划有效性检查；新固定规则触发普通重规划，无OGG |',
        '| executed_lvm | A10-H1以实际已提交历史动作到期评分；同一新固定规则触发普通重规划，无OGG |',
        '| single_lvm | 单动作Transformer-H1，用原监测规则触发普通重规划；不是MLP |',
        '| single_ogg | 同一单动作Transformer-H1与原规则，再加原单动作OGG |','',
        'H20没有纳入本轮：当时选H10作为高频强基线以限制组合，牺牲了与此前H20的直接可比性，应作为设计不足标注。旧H20的任务/init/seed不同，不能补入本轮做公平配对。当前不修改冻结计划、不启动补测。', '',
        '主要研究问题：到期时使用真实执行动作，能否减少规划窗口失效并改善闭环结果；预测误差与回合成败是否有关。五个seed42冻结预测器还对同一实际轨迹诊断，不等于五模型各自控制全部回合。', '',
        'single方法与A10方法的阈值/检测规则不同，不能把两者差异解释为纯模型架构效应。新规则阈值0.3976114362478256、EWMA alpha0.35、连续5步、cooldown20；旧规则见原冻结协议。', '',
        f'## 六方法共同的{core_n}组条件','',
        f"{tables[0]['physical_initial_states']}个物理(task,init)，同一状态可重复两个评测seed；每方法{core_n}回合。这张表避免把只跑核心的single方法与全部扩展总体直接排名。",'',
        '| 方法 | 成功/完成 | 成功率 | 平均VLA块调用 | 平均重规划 | 评分覆盖率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in tables:
        if r['scope']=='core_six':
            cov=f"{r['score_coverage']:.2%}" if r['score_coverage'] is not None else '不适用'
            report.append(f"| {r['method']} | {r['successes']}/{r['completed']} | {r['success_rate']:.2%} | {r['mean_chunk_calls']:.2f} | {r['mean_event_replans']:.2f} | {cov} |")
    report+=['',f"单动作OGG相对single_lvm有{ogg_pair['candidate_only_success']}个独有成功、{ogg_pair['reference_only_success']}个反向失败；没有证据说OGG在本批带来稳定收益。", '',
        f'## 四方法共同的{four_n}组条件','',
        '| 方法 | 成功/完成 | 成功率 | 平均VLA块调用 | 平均监测秒 | 评分覆盖率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in tables:
        if r['scope']=='matched_four':
            cov=f"{r['score_coverage']:.2%}" if r['score_coverage'] is not None else '不适用'
            report.append(f"| {r['method']} | {r['successes']}/{r['completed']} | {r['success_rate']:.2%} | {r['mean_chunk_calls']:.2f} | {r['mean_monitor_s']:.3f} | {cov} |")
    report+=['',f"executed对planned：{executed_pair['candidate_only_success']}个独有成功、{executed_pair['reference_only_success']}个反向失败，净{executed_pair['net_successes']:+d}/{four_n}（{100*executed_pair['net_successes']/four_n:+.2f}个百分点）；对H50为{base_pair['candidate_only_success']}/{base_pair['reference_only_success']}，净{base_pair['net_successes']:+d}/{four_n}。H10相对executed有{h10_pair['candidate_only_success']}个独有成功、{h10_pair['reference_only_success']}个反向失败。", '',
        '这支持“已执行动作到期监测”有用，但不是覆盖率单变量的因果证明：条件动作、评分和触发时刻同时改变。H10平均块调用更多；是否选择它要结合推理成本，不能只看成功率。逐任务成功数保存在task_successes.csv。', '',
        f'## 五预测器在同一批{h50_n}条H50真实轨迹上的准度','',
        f'逐回合宏平均；都使用实际提交动作、相同视觉目标/窗口。这里H50有{h50_n}条，控制器公平比较仍只用共同{four_n}条。完整回合与固定早期前缀分开；缺完整早期前缀的回合只从早期统计排除，详见CSV各分母。', '',
        '| 预测器 | 全回合MSE | 早期MSE | 全回合误差→失败AUC | 早期误差→失败AUC |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        full=next(r for r in predictions if r['model']==model and r['scope']=='full')
        early=next(r for r in predictions if r['model']==model and r['scope']=='early')
        report.append(f"| {model} | {full['episode_macro_mse']:.4f} | {early['episode_macro_mse']:.4f} | {full['episode_failure_auc']:.3f} | {early['episode_failure_auc']:.3f} |")
    report+=['','A10-H1在这批相同轨迹上的MSE更低，但不能据此直接推断其OGG更好。全回合误差与失败强相关；固定早期窗口（origin4～44、due24～64）下关联明显减弱。可能存在失败过程使特征预测变差的反向关系；不能把全回合AUC当提前异常检测能力。', '',
        'AUC分类对象是回合最终失败，不是逐步异常；窗口和重复init/seed相关，未作独立样本显著性声明。没有独立事件标签，不输出“异常检测准确率”。按任务/控制方法分层的原始聚合仍在快照的prediction_success_relationship.csv。', '',
        '## 下一步复盘建议','',
        '先看H50/H20/MLP+OGG的已保留失败视频，再对planned/executed配对中的新增成功与反向失败按任务取少量trace核查。H20补测需要新独立配对计划；本轮不自动恢复或追加。', '',
        '所有数据由snapshot原JSON复算；配对表、逐任务表、预测表与权重SHA见同目录CSV/summary.json。报告不包含私人SSH地址、原始视频或checkpoint。']
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n')
    return summary


def plot(summary, output):
    """可选标准科研图；matplotlib不属于Replay运行依赖。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'H50':'H50','H10':'H10','planned_lvm':'Planned LVM','executed_lvm':'Executed LVM',
            'single_lvm':'T-H1 LVM','single_ogg':'T-H1 OGG'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,scope,label in zip(axes,['core_six','matched_four'],['Six methods','Four methods']):
        rows=[r for r in summary['paired_tables'] if r['scope']==scope]
        title=f"{label}: {rows[0]['completed']} matched cases each"
        ax.barh([labels[r['method']] for r in rows],[100*r['success_rate'] for r in rows],color=['#8aa5bd','#219a90','#c5a36c','#447fbc','#9e92bb','#7764a5'][:len(rows)])
        for i,r in enumerate(rows):ax.text(100*r['success_rate']+.8,i,f"{r['successes']}/{r['completed']}",va='center',fontsize=9)
        ax.set(xlim=(0,106),xlabel='Success rate (%)',title=title);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen weights / Object10 / new initial states / paired evaluation seeds',fontsize=11)
    fig.savefig(output/'matched_success.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    rows=summary['prediction_on_same_H50_trajectories'];names=['MLP','T-H1','T-H3','A10-H1','A10-H3']
    for scope,color in [('full','#407daf'),('early','#25a092')]:
        selected=[next(r for r in rows if r['model']==m and r['scope']==scope) for m in MODELS]
        axes[0].plot(names,[r['episode_macro_mse'] for r in selected],'o-',label=scope,color=color)
        axes[1].plot(names,[r['episode_failure_auc'] for r in selected],'o-',label=scope,color=color)
    axes[0].set(ylabel='Episode macro MSE',title='Same H50 trajectories / executed conditions')
    axes[1].set(ylabel='Episode failure AUC (descriptive)',ylim=(.45,1.02),title='Outcome association, not anomaly detection')
    axes[1].axhline(.5,color='#abb5bf',linestyle='--',linewidth=1)
    for ax in axes:ax.legend();ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Early: origins 4..44 / due 24..64. Full-episode errors may follow failure.',fontsize=11)
    fig.savefig(output/'prediction_accuracy.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true',help='使用已有matplotlib另存两张PNG；不安装依赖')
    args=p.parse_args();s=analyse(args.snapshot,args.output)
    if args.plot:plot(s,args.output)
    print(json.dumps({'completed':s['execution']['completed'],'status':s['execution']['status'],'output':str(args.output)},ensure_ascii=False))

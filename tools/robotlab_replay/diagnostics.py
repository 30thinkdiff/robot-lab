"""从已记录的JSON生成可追溯事实与小型热图；不推断接触、失败因果或缺失Tensor。"""
import math


def numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def vector(value, limit=256):
    if not isinstance(value,list) or not 0<len(value)<=limit or not all(numeric(v) for v in value):return None
    return value


def summarize(detail):
    cards=[];run=detail['run'];steps=detail['steps'];cov=detail.get('coverage');events=detail['events']
    def add(kind,title,text,step=None):cards.append({'kind':kind,'title':title,'text':text,'step':step})
    if run['status']!='completed':add('warning','程序记录未完成','先核对执行状态和错误；不把程序中断当作机器人任务失败。')
    elif run['success'] is False:add('fact','任务结果：失败',f"已记录{len(steps)}个环境步。日志没有独立失败原因标签，不能自动认定抓取失败或OGG致因。",len(steps)-1 if steps else None)
    elif run['success'] is True:add('fact','任务结果：成功','任务结局来自原trial；不把成功结局当作每个窗口都正常。')
    else:add('missing','任务结局未记录','原trial没有明确的success布尔值，不按成功或失败统计。')
    if detail['video']['alignment']!='recording_verified':
        add('missing','视频证据受限',detail['video']['message'])
    if cov and cov.get('fraction') is not None:
        if cov['fraction']<1:
            first=detail.get('first_due');gap=None
            for row in steps:
                if first is not None and row['step']>=first and row.get('score') is None and row.get('monitor_state')=='无到期评分':gap=row['step'];break
            add('warning','存在监测空窗',f"评分覆盖{cov['scores']}/{cov['eligible_score_steps']}，最长空窗{cov['max_gap_steps']}步。这是评分可用性，不是异常召回率。",gap)
        else:add('fact','可评分窗口覆盖完整',f"记录{cov['scores']}/{cov['eligible_score_steps']}个可评分窗口；完整覆盖不能证明检测准确。")
    elif not (run['method'].startswith('H') and run['method'][1:].isdigit()):
        add('missing','评分覆盖无法核验','缺少逐步评分或首到期步/频率协议，不从首个已观测分数反推100%覆盖。')
    triggers=sorted({e['step'] for e in events if e['kind']=='replan'})
    guides=[e for e in events if e['kind']=='ogg']
    if triggers:add('fact','已记录重规划',f'共{len(triggers)}个触发步，首次step{triggers[0]}。可跳到该步核查分数、阈值与队列。',triggers[0])
    if guides:
        measured=sum(numeric(e['data'].get('gradient_l2')) for e in guides)
        add('fact','OGG确有调用记录',f"共{len(guides)}条引导记录，其中{measured}条有梯度范数；调用或Loss下降本身不能证明纠错成功。",guides[0]['step'])
        if not measured:add('missing','旧日志缺少OGG梯度','当前实验没有记录梯度范数/逐动作更新量；不会从其他模型的日志补入。')
    has_commands=any(vector(s.get('submitted_action') or (s.get('controller_command') or {}).get('raw_controller_action')) for s in steps)
    if not has_commands:add('missing','逐步动作未记录或未取回','无法从视频恢复真实控制命令；动作热图保持缺失。')
    return {'cards':cards,'basis':'recorded facts and missing-evidence checks; no automated causal diagnosis',
        'source':detail['source'],'full_visual_tensor_available':False,
        'tensor_limit':'未加载完整视觉/候选动作Tensor；JSON动作与逐动作更新范数不等同完整Tensor。'}


def heatmaps(detail, step, radius=12):
    if type(step) is not int or step<0 or step>=len(detail['steps']):raise ValueError('热图step超出本回合')
    if type(radius) is not int or not 1<=radius<=40:raise ValueError('窗口半径须为1～40')
    commands=[];width=None;notes=[]
    for s in detail['steps'][max(0,step-radius):step+radius+1]:
        a=s.get('submitted_action')
        if a is None:a=(s.get('controller_command') or {}).get('raw_controller_action')
        a=vector(a,64)
        if a is None:continue
        if width is None:width=len(a)
        if len(a)!=width:notes.append(f"step{s['step']}动作维度不同，未拼接");continue
        commands.append({'step':s['step'],'values':a})
    updates=[]
    for e in detail['steps'][step]['events']:
        if e['kind']!='ogg':continue
        values=e['data'].get('per_action_clean_update_l2')
        if isinstance(values,list) and len(values)==1 and isinstance(values[0],list):values=values[0]
        values=vector(values)
        if values is None or any(v<0 for v in values):continue
        updates.append({'denoise':e['data'].get('denoise'),'values':values})
    if updates and len({len(r['values']) for r in updates})!=1:
        notes.append('同一步的OGG更新维度不一致，未拼接');updates=[]
    return {'step':step,'commands':commands,'updates':updates,
        'command_units':'实际提交控制命令；分量语义与单位以原协议为准，不视作机器人测量状态',
        'update_units':'每个候选归一化动作的更新L2范数；不是实际执行位移，不包含7个分量的完整梯度',
        'missing_commands':not commands,'missing_updates':not updates,'notes':notes,'source':detail['source']}


def alignment_steps(details, primary_step, alignment=None):
    if type(primary_step) is not int or primary_step<0:
        raise ValueError('step越界')
    if alignment is None:
        if primary_step>=max(len(d['steps']) for d in details):raise ValueError('step越界')
        return [primary_step]*len(details),None
    if len(details)!=2 or not isinstance(alignment,dict) or set(alignment)!={'a','b','label'}:
        raise ValueError('只接受两个配对回合的人工锚点')
    clocks=[d.get('control_fps') for d in details]
    if not all(numeric(v) and v>0 for v in clocks) or not math.isclose(*clocks,rel_tol=1e-8):
        raise ValueError('控制频率未知或不一致，不用固定步差强行对齐')
    a,b=alignment['a'],alignment['b'];label=alignment['label']
    if any(type(v) is not int or v<0 or v>=len(d['steps']) for v,d in zip([a,b],details)):
        raise ValueError('人工锚点越界')
    if not isinstance(label,str) or not label.strip() or len(label)>120:raise ValueError('需填写1～120字人工锚点说明')
    if primary_step>=max(len(details[0]['steps']),len(details[1]['steps'])-(b-a)):raise ValueError('step越界')
    return [primary_step,primary_step+b-a],{'mode':'manual_event_offset','a':a,'b':b,'label':label,
        'limit':'manual constant offset; not automatic manipulation-phase alignment or time warping'}

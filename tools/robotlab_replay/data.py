"""只读实验适配层。JSON/CSV/ZIP，不反序列化checkpoint或Tensor。"""
from collections import defaultdict
from functools import lru_cache
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.dont_write_bytecode=True  # 复用旧分析函数时不向原脚本目录写入字节缓存。
sys.path.insert(0,str(REPO/'scripts'))
from corrector_results_analysis import coverage_stats,residual_mse_from_norms


def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


class Source:
    def __init__(self,path):
        self.path=Path(path).resolve();self.zip=zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names=set(self.zip.namelist()) if self.zip else set()
    def exists(self,name):return name in self.names if self.zip else self.file(name).is_file()
    def file(self,name):
        p=(self.path/name).resolve();p.relative_to(self.path);return p
    def read(self,name):
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('不接受越界成员')
        if self.zip:
            item=self.zip.getinfo(name)
            if item.file_size>24*2**20:raise ValueError('单个结构化文件超过24MiB，未读取')
            return self.zip.read(name)
        p=self.file(name)
        if p.stat().st_size>24*2**20:raise ValueError('结构化文件超出预算')
        return p.read_bytes()
    def json(self,name,default=None):return clean(json.loads(self.read(name))) if self.exists(name) else default
    def close(self):
        if self.zip:self.zip.close()


def verify_recording(root):
    meta=root/'evidence.json';source=root/'lerobot_eval.py'
    if not meta.exists() or not source.exists():return {'verified':False,'reason':'尚无录制实现证据'}
    evidence=json.loads(meta.read_text())
    if (evidence.get('git_revision')!='7e241bd630a3719a56157a497ce5d08f244784f1'
        or not evidence.get('tracked_file_clean') or digest(source)!=evidence.get('source_sha256')):
        return {'verified':False,'reason':'录制源码版本/SHA核对未通过'}
    tree=ast.parse(source.read_text());rollout=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
    loops=[n for n in ast.walk(rollout) if isinstance(n,(ast.For,ast.While))]
    env_steps=[n.lineno for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' and isinstance(n.func.value,ast.Name) and n.func.value.id=='env']
    renders=sorted(n.lineno for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='render_callback')
    control_loop=next((n for n in loops if any(n.lineno<line<=n.end_lineno for line in env_steps)),None)
    slicing=any(isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='stacked_frames'
        and isinstance(n.slice,ast.Slice) and isinstance(n.slice.upper,ast.BinOp)
        and isinstance(n.slice.upper.left,ast.Name) and n.slice.upper.left.id=='done_index'
        and isinstance(n.slice.upper.op,ast.Add) and isinstance(n.slice.upper.right,ast.Constant) and n.slice.upper.right.value==1 for n in ast.walk(tree))
    ok=bool(control_loop and env_steps and renders and renders[0]<control_loop.lineno and any(r>env_steps[0] for r in renders) and slicing)
    return {'verified':ok,**evidence,'phase':'frame n = Z_n, before submitted u_n; terminal Z_N is not retained',
        'render_lines':renders,'env_step_lines':env_steps,'reason':None if ok else '未识别到预期录制顺序'}


def recording_evidence(state):return verify_recording(state/'imported_video/recording')


@lru_cache(maxsize=24)
def probe_video(path,mtime):
    try:
        cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
             'stream=width,height,avg_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time','-show_frames','-of','json',path]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=True);info=json.loads(p.stdout)
        pts=[float(f['best_effort_timestamp_time']) for f in info.get('frames',[]) if 'best_effort_timestamp_time' in f]
        return {'stream':info.get('streams',[{}])[0],'pts':pts,'decoded_frames':len(pts)}
    except (OSError,subprocess.SubprocessError,ValueError,KeyError) as e:return {'error':type(e).__name__+': 无法核验视频PTS/帧数','pts':[]}


class Catalog:
    def __init__(self,sources=None,state=None):
        self.state=Path(state or HERE/'.local').resolve();self.sources=[];self.runs={};self.errors=[]
        defaults=[REPO/'outputs'/p for p in ['hybrid_pilot_20261111/results.zip','corrector_audit_20261111/results_bundle.zip',
            'executed_consistency_pilot_20261111_v2/results.zip','consistency_closed_loop_20261111/results.zip','smolvla_horizon_pilot_20261111/results.zip']]
        # 多次取回快照只索引每个study最完整的新批次，避免重复计数/旧快照覆盖新回合。
        banks={}
        for batch in sorted((self.state/'import_batches').glob('casebank_*')):
            if not (batch/'manifest.json').is_file():continue
            try:
                if (batch/'archive.json').is_file():
                    entry=json.loads((batch/'archive.json').read_text());folder=(self.state/entry['archive']).resolve();folder.relative_to(self.state)
                    if folder.suffix!='.zip':raise ValueError('索引归档类型异常')
                    with zipfile.ZipFile(folder) as z:
                        plan=json.loads(z.read('results/plan.json'));execution=json.loads(z.read('results/execution.json'))
                else:
                    folder=batch/'results';plan=json.loads((folder/'plan.json').read_text());execution=json.loads((folder/'execution.json').read_text())
                study=plan['protocol']['study_id'];rank=(int(execution['completed']),batch.stat().st_mtime_ns)
                if study not in banks or rank>banks[study][0]:banks[study]=(rank,folder)
            except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
        defaults += [entry[1] for entry in banks.values()]
        for path in sources if sources is not None else defaults:
            if not Path(path).exists():continue
            try:self.add(Source(path))
            except (OSError,ValueError,zipfile.BadZipFile,KeyError) as e:self.errors.append(Path(path).name+': '+str(e))
        self.video_index={};self.video_trial_hash=defaultdict(set);manifest=self.state/'imported_video/manifest.json'
        if manifest.exists():
            imports=json.loads(manifest.read_text()).get('videos',[])
            for v in imports:
                p=(manifest.parent/v['file']).resolve()
                try:p.relative_to(manifest.parent.resolve())
                except ValueError:continue
                if p.is_file() and p.suffix.lower() in {'.mp4','.webm'}:
                    self.video_index[v['sha256']]=p
                    if v.get('trial_sha256'):self.video_trial_hash[v['sha256']].add(v['trial_sha256'])
            # 首次导入清单没有trial哈希时，只从当时读取的真实Hybrid归档补全绑定；不把SHA相同的视频套到其他回合。
            canonical=REPO/'outputs/hybrid_pilot_20261111/results.zip'
            if canonical.exists():
                with zipfile.ZipFile(canonical) as z:
                    for name in z.namelist():
                        if not name.startswith('paired/trials/') or not name.endswith('.json'):continue
                        raw=z.read(name);record=json.loads(raw)
                        for v in imports:
                            if v.get('trial_sha256'):continue
                            if (v['path'] in record.get('result',{}).get('video_paths',[]) and
                                any(a.get('sha256')==v['sha256'] for a in record.get('video_artifacts',[]))):
                                self.video_trial_hash[v['sha256']].add(hashlib.sha256(raw).hexdigest())
        self.recorder=recording_evidence(self.state)
        # 新批次只追加。每个视频分别绑定原trial哈希；相同视频内容可以有多个经核验的来源。
        for manifest in sorted((self.state/'import_batches').glob('*/manifest.json')):
            try:
                proof=verify_recording(manifest.parent/'recording')
                if proof.get('verified') and not self.recorder.get('verified'):self.recorder=proof
                same_source=proof.get('verified') and proof.get('source_sha256')==self.recorder.get('source_sha256')
                for v in json.loads(manifest.read_text())['videos']:
                    p=(manifest.parent/v['file']).resolve();p.relative_to(manifest.parent.resolve())
                    if p.is_file() and p.suffix.lower() in {'.mp4','.webm'}:
                        self.video_index.setdefault(v['sha256'],p)
                        if same_source and v.get('trial_sha256'):self.video_trial_hash[v['sha256']].add(v['trial_sha256'])
            except (OSError,ValueError,KeyError,SyntaxError,StopIteration) as e:
                self.errors.append(manifest.parent.name+': 导入证据未通过 '+type(e).__name__)
        self.details=lru_cache(maxsize=20)(self._detail)

    def close(self):
        for s in self.sources:s.close()

    def add(self,source):
        self.sources.append(source)
        names=source.names if source.zip else {str(p.relative_to(source.path).as_posix()) for p in source.path.rglob('plan.json')}
        plans=sorted(n for n in names if n=='plan.json' or n.endswith('/plan.json'))
        for plan_name in plans:
            prefix=plan_name[:-len('plan.json')];plan=source.json(plan_name)
            if not isinstance(plan,dict) or not isinstance(plan.get('cases'),list):continue
            protocol=plan.get('protocol',{})
            if not isinstance(protocol,dict):protocol={};self.errors.append(plan_name+': protocol不是对象，时间基准未知')
            trial_names=(sorted(n for n in source.names if n.startswith(prefix+'trials/') and n.endswith('.json')) if source.zip
                else [str(p.relative_to(source.path).as_posix()) for p in source.file(prefix+'trials').glob('*.json')])
            for member in trial_names:
                try:
                    record=source.json(member);case=record['case'];result=record.get('result') or {};cid=case['id']
                    if not isinstance(result,dict):raise ValueError('result不是对象')
                    uid=hashlib.sha256((str(source.path)+'::'+member).encode()).hexdigest()[:20]
                    match=re.search(r'task_(\d+)/init_(\d+)',case.get('condition',''))
                    model_match=re.search(r'evaluation/([^/]+)/',prefix)
                    model=model_match[1].split('_s')[0] if model_match else ('SmolVLA' if 'horizon' in source.path.parent.name else 'A10-H1')
                    if 'hybrid' in source.path.parent.name:model={'A':'T-H1 → T-H1','B':'T-H1 → A10-H1','C':'A10-H1 → A10-H1'}.get(case['method'],'Hybrid')
                    variant={'mlp':'MLP','transformer-h1':'T-H1','transformer-h3':'T-H3','transformer-a10-h1':'A10-H1','transformer-a10-h3':'A10-H3'}.get(model,model)
                    variant=case.get('config',{}).get('model_label',variant)
                    if case['method'] in ['single_lvm','single_ogg']:variant='T-H1'
                    forced='forced_check' in prefix or bool(protocol.get('debug_forced_event'))
                    group='混合 Corrector' if 'hybrid' in source.path.parent.name else ('Object10 五模型' if 'corrector_audit' in str(source.path) else source.path.parent.name.replace('_20261111',''))
                    group=protocol.get('experiment_label',group)
                    self.runs[uid]={'id':uid,'model':variant,'method':case['method'],'task':int(match[1]) if match else None,
                        'init':int(match[2]) if match else None,'seed':case.get('seed'),'condition':case.get('condition'),
                        'success':result.get('success'),'status':record['status'],'steps':result.get('env_steps'),
                        'experiment':group,'forced':forced,'source_name':source.path.name,'member':member,
                        '_source':source,'_prefix':prefix,'_record':record,'_protocol':protocol}
                except (ValueError,KeyError,TypeError) as e:self.errors.append(member+': '+str(e))

    def public(self):
        rows=[]
        for r in self.runs.values():
            row={k:v for k,v in r.items() if not k.startswith('_')}
            row['video_available']=bool(self.video_path(r));rows.append(row)
        return {'runs':rows,'warnings':self.errors,'read_only':True,'sources':len(self.sources),
                'recorder_verified':self.recorder['verified']}

    def video_path(self,run):
        for v in run['_record'].get('video_artifacts',[]):
            if v.get('sha256') in self.video_index:return self.video_index[v['sha256']]
            if not run['_source'].zip:
                try:
                    p=run['_source'].file(run['_prefix']+v['path'])
                    if p.is_file() and p.suffix.lower() in {'.mp4','.webm'}:return p
                except ValueError:continue
        return None

    def _detail(self,uid):
        r=self.runs[uid];source=r['_source'];record=r['_record'];c=record['case'];result=record.get('result') or {};p=r['_protocol']
        base=r['_prefix']+f"attempts/{c['id']}/attempt_{record.get('attempt',1):03d}/"
        actual=source.json(base+'actual_protocol.json') or source.json(base+'actual_protocol.export.json') or {}
        result={**result,'episode_wall_s':record.get('wall_s')}
        n=int(result.get('env_steps') or 0);fps=p.get('control_fps',actual.get('fps',p.get('fps')));fps=float(fps) if finite(fps) and fps>0 else None
        if not 0<=n<=100000:raise ValueError('回合步数异常，拒绝分配无界内存')
        warnings=[];rows=[];baseline=bool(re.fullmatch(r'H[0-9]+',c['method']))
        if source.exists(base+'monitor_steps.jsonl'):
            for i,line in enumerate(source.read(base+'monitor_steps.jsonl').decode().splitlines()):
                if not line.strip():continue
                try:
                    row=json.loads(line)
                    if not isinstance(row,dict):
                        warnings.append(f'trace第{i+1}行不是对象，已跳过');continue
                    if any(key in row and row[key] is not None and not finite(row[key]) for key in ['score','threshold','executed_score','planned_score']):
                        warnings.append(f'trace第{i+1}行存在非法数值，按缺失显示')
                    row=clean(row);step=row.get('step')
                    if isinstance(step,int) and not isinstance(step,bool) and 0<=step<n:rows.append(row)
                    else:warnings.append(f'trace第{i+1}行step越界/缺失，未加入时间轴')
                except (ValueError,TypeError):warnings.append(f'trace第{i+1}行无法解析')
        elif not baseline:warnings.append('当前实验未记录或未取回逐步monitor trace')
        pp=actual.get('predictor_protocol',{});horizon=actual.get('monitor_horizon_steps',p.get('horizon_steps',p.get('due_steps')))
        if horizon is None and finite(p.get('prediction_seconds')) and fps:horizon=round(p['prediction_seconds']*fps)
        first=actual.get('monitor_first_due_step',p.get('score_start',p.get('common_evaluation_start')))
        if first is None and horizon is not None and p.get('warmup_steps') is not None:first=int(horizon+p['warmup_steps'])
        stride=int(p.get('corrector_settings',{}).get('stride',1));scored=[a for a in rows if finite(a.get('score'))]
        if first is None and pp and horizon is not None:
            first=int(horizon+(max(pp.get('h_window',1),pp.get('anchor_history',1))-1)*(fps/pp.get('fps',fps)))
        if baseline and not rows:first=None
        coverage=coverage_stats(scored,int(first),n,stride,fps) if first is not None and fps else None
        if coverage:coverage['fraction']=coverage['scores']/coverage['eligible_score_steps'] if coverage['eligible_score_steps'] else None
        elif rows:warnings.append('缺少明确的首个可评分步协议，不以首个观测评分反推分母')
        steps=[{'step':i,'score':None,'threshold':None,'events':[],'raw':[]} for i in range(n)];events=[]
        shapes=[p.get('visual_shape'),actual.get('visual_shape')]+[row.get('target_shape') for row in rows]
        dimensions=[int(shape[-2])*int(shape[-1]) for shape in shapes if isinstance(shape,list) and len(shape)>=2
                    and all(isinstance(v,int) and v>0 for v in shape[-2:])]
        feature_dimension=dimensions[0] if dimensions and len(set(dimensions))==1 else None
        def event(step,kind,label,data):
            item={'step':step,'kind':kind,'label':label,'data':data};events.append(item);steps[step]['events'].append(item)
        for row in rows:
            step=row['step'];s=steps[step];s['raw'].append(row)
            for key in ['score','threshold','ewma','expected_norm','observed_norm','executed_score','planned_score']:
                if key in row and (finite(row[key]) or row[key] is None):s[key]=row[key]
            if row.get('type')=='hybrid_trigger':event(step,'target','纠偏目标生成',row)
            if row.get('interrupted') is True or row.get('trigger') is True and row.get('type')!='hybrid_trigger':event(step,'replan','异常中断 / 重规划',row)
            if 'ogg_loss' in row or 'guidance_loss' in row:event(step,'ogg','OGG引导 · 去噪'+str(row.get('denoise','?')),row)
            if row.get('prediction_skipped'):event(step,'skip','规划动作不足 · 跳过预测',row)
            cancelled=sum(row[k] for k in ['cancelled_predictions','cancelled_on_interrupt','cancelled_on_new_plan'] if finite(row.get(k)))
            if cancelled:event(step,'cancel',f'取消 {cancelled} 条待到期预测',row)
            if feature_dimension and finite(row.get('score')) and finite(row.get('expected_norm')) and finite(row.get('observed_norm')):
                s['derived_mse']=residual_mse_from_norms(row['expected_norm'],row['observed_norm'],row['score'],feature_dimension)
                s['derived_mse_basis']=f'由范数/cosine及记录的{feature_dimension}维推导，非保存的预测Tensor'
        h=c.get('config',{}).get('n_action_steps',c.get('config',{}).get('action_steps'))
        if h is None:h=int(c['method'][1:]) if re.fullmatch('H[0-9]+',c['method']) else p.get('action_steps',50)
        queue=0;blocks=0
        for s in steps:
            if any(e['kind']=='replan' for e in s['events']):queue=0
            if queue==0:queue=int(h);blocks+=1;s['block_start_derived']=True
            s['block_derived']=blocks;s['queue_remaining_derived']=queue-1;queue-=1
            s['monitor_state']=('未启用Corrector' if baseline else '未记录') if not rows else ('预热' if first is not None and s['step']<first else ('到期评分可用' if finite(s['score']) else '无到期评分'))
        blocks_valid=bool(n and result.get('chunk_calls')==blocks)
        if not blocks_valid:
            for s in steps:
                for k in ['block_derived','block_start_derived','queue_remaining_derived']:s.pop(k,None)
            if n:warnings.append('动作块生成时刻未完整记录；队列重建与总调用次数不一致，未显示推导块号')
        actions=source.json(base+'executed_actions.json');commands=source.json(base+'controller_commands.json')
        if actions is not None and len(actions)==n:
            for s,a in zip(steps,actions):s['submitted_action']=a
        elif actions is not None:warnings.append('动作数量与环境步数不符，未强行对齐')
        if commands and len(commands)==n:
            for s,a in zip(steps,commands):s['controller_command']=a
        queue_trace=source.json(base+'queue_trace.json')
        if (isinstance(queue_trace,list) and len(queue_trace)==n and all(isinstance(q,dict) and q.get('step')==i
                and type(q.get('plan_id')) is int and type(q.get('queue_after')) is int
                and type(q.get('chunk_calls_before')) is int and type(q.get('chunk_calls_after')) is int for i,q in enumerate(queue_trace))):
            for s,q in zip(steps,queue_trace):
                s['block_actual']=q['plan_id'];s['queue_remaining_actual']=q['queue_after'];s['block_start_actual']=q['chunk_calls_after']>q['chunk_calls_before']
        video_path=self.video_path(r);video={'available':False,'alignment':'unavailable','message':'当前实验视频未取回；trace和事件仍可分析','pts':[]}
        if video_path:
            expected=next((v.get('sha256') for v in record.get('video_artifacts',[]) if v.get('sha256') and (self.video_index.get(v['sha256'])==video_path or not source.zip)),None)
            proof=probe_video(str(video_path),video_path.stat().st_mtime_ns)
            matched=proof.get('decoded_frames')==n;verified=bool(expected and digest(video_path)==expected)
            pts=proof.get('pts',[])
            ordered=bool(pts and abs(pts[0])<1e-6 and all(finite(v) for v in pts) and all(a<b for a,b in zip(pts,pts[1:])))
            bound=bool(expected and self.video_index.get(expected)==video_path and
                hashlib.sha256(source.read(r['member'])).hexdigest() in self.video_trial_hash.get(expected,set()))
            exact=bool(matched and ordered and bound and verified and self.recorder['verified'] and fps)
            message=('录制顺序、SHA、逐帧PTS与帧数已核验：帧n=动作u_n之前的Z_n；末次动作后的Z_N不在视频中。' if exact
                else 'ffprobe不可用或无法读取PTS，没有建立逐步同步。' if proof.get('error')
                else '视频帧数与环境步数不一致，没有建立逐步同步。' if not matched
                else '视频SHA未核验，未建立精确同步。' if not verified
                else 'PTS顺序/起点未通过校验，未建立精确同步。' if not ordered
                else '帧数和SHA匹配，但没有绑定到本视频的录制相位证据；仅为顺序对应。')
            video={'available':True,**proof,'url':'/api/video/'+uid,'sha256_verified':verified,'frame_count_matches_steps':matched,
                'pts_monotonic_and_zero_origin':ordered,'recording_evidence_bound_to_video_sha':bound,
                'alignment':'recording_verified' if exact else 'ordinal_only' if matched and verified else 'unverified',
                'message':message,
                'recorder':self.recorder}
            if not matched:video['pts']=[]
        targets=source.json(base+'targets.json')
        public={k:v for k,v in r.items() if not k.startswith('_')}
        return clean({'run':public,'metrics':result,'coverage':coverage,'first_due':first,'horizon_steps':horizon,
            'control_fps':fps,'steps':steps,'events':events,'video':video,'warnings':list(dict.fromkeys(warnings)),
            'targets':targets,'queue_trace':queue_trace,'actual_protocol':actual,'protocol':p,
            'source':{'archive':source.path.name,'member':r['member'],'trial_sha256':hashlib.sha256(source.read(r['member'])).hexdigest()},
            'limits':['环境步对齐不等于操作阶段对齐','仿真控制时间、视频PTS与推理墙钟分别呈现',
                '未保存的预测Tensor/机器人测量状态不恢复或伪造','推导动作块仅在总调用次数核对一致时显示，区别于逐次调用实测']})

    def pairs(self,uid):
        a=self.runs[uid]
        return [{k:v for k,v in r.items() if not k.startswith('_')} for r in self.runs.values()
            if r['id']!=uid and r['condition']==a['condition'] and r['seed']==a['seed'] and not r['forced']]

"""Read-only training counters and explicitly conditional early-stop ETAs."""
from contextlib import contextmanager
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo


def read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def alive(pid):
    if not pid:return False
    try:
        text=Path('/proc/%s/stat'%pid).read_text()
        return text[text.rfind(')')+2:].split()[0]!='Z'
    except FileNotFoundError:return False
    except OSError:
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False


def history_summary(history,cap=150,patience=5,timing_after_epoch=0):
    if not history:return dict(epochs_completed=0,epoch_cap=cap,patience=patience,
        seconds_per_epoch=None,eta_to_epoch_cap_hours=None,eta_if_no_further_improvement_hours=None,
        eta_note='等待首轮训练与验证完成；早停时刻不能事先确定')
    best=min(history,key=lambda r:r['validation']['loss'])
    completed=history[-1]['epoch'];stale=completed-best['epoch']
    recent=[r['seconds'] for r in history if r['epoch']>timing_after_epoch and r.get('seconds',0)>0][-3:]
    speed=sum(recent)/len(recent) if recent else None
    remaining=max(0,cap-completed);earliest=min(remaining,max(0,patience-stale))
    return dict(epochs_completed=completed,epoch_cap=cap,patience=patience,best_epoch=best['epoch'],
        best_validation=best['validation'],last_validation=history[-1]['validation'],
        stale_epochs=stale,optimizer_steps=history[-1]['step'],seconds_per_epoch=speed,
        eta_to_epoch_cap_hours=remaining*speed/3600 if speed else None,
        eta_if_no_further_improvement_hours=earliest*speed/3600 if speed else None,
        eta_note='最近3轮训练+验证平均耗时；不含checkpoint写入。上限轮数情景/此后无改善早停情景，均非保证完成时间')


@contextmanager
def observe(reference,output):
    """Delegate identical forwards/backwards, retain no tensors or extra scalars."""
    train=reference['train'];output=Path(output)
    history=read(output/'history.json') or []
    state=dict(epoch_in_progress=history[-1]['epoch'] if history else 0,phase='initializing',
               train_samples_after_backward=0,validation_samples_forwarded=0,epoch_optimizer_updates=0)
    last_write=0.;last_model=None;n=0;phase_started=time.time();epoch_started=None
    forward,update=train.reference_forward,train.reference_update
    def emit(force=False):
        nonlocal last_write
        now=time.time()
        if force or now-last_write>=10:
            state['phase_elapsed_seconds']=now-phase_started
            if epoch_started is not None:state['training_elapsed_seconds']=now-epoch_started
            output.mkdir(parents=True,exist_ok=True)
            write(output/'live_progress.json',dict(state,updated_unix=now,pid=os.getpid(),
                train_samples_total=49447,validation_samples_total=12089,
                scope='training counts only; exact frozen tensor operations unchanged'))
            last_write=now
    def observed_forward(model,batch,*args,**kwargs):
        nonlocal n,last_model,phase_started,epoch_started
        out=forward(model,batch,*args,**kwargs)
        phase='training' if model.training else 'validation'
        if phase!=state['phase'] or id(model)!=last_model:
            if phase=='training':
                if state['phase']=='validation' and state['validation_samples_forwarded']==12089:
                    state['last_validation_seconds']=time.time()-phase_started
                epoch_started=time.time()
                state.update(epoch_in_progress=state['epoch_in_progress']+1,
                             train_samples_after_backward=0,epoch_optimizer_updates=0)
            else:
                if state['phase']=='training' and epoch_started is not None:
                    state['last_train_epoch_seconds']=time.time()-epoch_started
                state['validation_samples_forwarded']=0
            state['phase']=phase;last_model=id(model);phase_started=time.time();emit(True)
        n=len(out[0])
        if phase=='validation':state['validation_samples_forwarded']+=n;emit()
        return out
    def observed_update(model,optimizer,loss,should_step):
        result=update(model,optimizer,loss,should_step)
        state['train_samples_after_backward']+=n
        state['epoch_optimizer_updates']+=int(should_step)
        emit()
        return result
    with patch.object(train,'reference_forward',observed_forward),patch.object(train,'reference_update',observed_update):
        try:yield
        finally:emit(True)


def timing(history,live,timing_after_epoch=0):
    measured=history_summary(history,timing_after_epoch=timing_after_epoch)['seconds_per_epoch']
    if measured:return measured,'最近已完成epoch实测'
    if not live:return None,'等待训练样本速度'
    n=live.get('train_samples_after_backward',0);elapsed=live.get('training_elapsed_seconds',0)
    val=live.get('last_validation_seconds')
    if n<512 or elapsed<=0 or not val:return None,'等待初始验证与至少512个训练样本'
    train=live.get('last_train_epoch_seconds') if live.get('phase')=='validation' else None
    return (train or elapsed*49447/n)+val,'临时按当前轮样本速度＋初始验证估计；首个完整epoch后替换'


def status(root):
    root=Path(root);value=read(root/'queue_status.json') or dict(status='not_started')
    config=read(root/'queue_config.json')
    pipeline=(config or {}).get('pipeline',{})
    if pipeline:
        value['data_pipeline']=dict(kind=pipeline.get('kind','mlp_no_history'),workers=pipeline.get('workers'),
            stored_feature_dtype='float16',model_input_and_target_dtype='float32')
    receipt=read(pipeline['receipt']) if pipeline.get('receipt') else None
    pilot_speed=receipt.get('estimated_epoch_seconds') if receipt and receipt.get('status')=='passed' else None
    launch=read(root/'launcher.json') or {}
    if launch.get('started_unix',0)>value.get('updated_unix',0) and alive(launch.get('pid')):
        value.update(status='starting',pid=launch['pid'])
    value['process_alive']=alive(value.get('pid'))
    if value.get('status')=='running' and not value['process_alive']:
        value['status']='process_missing_requires_resume'
    output=value.get('current_output')
    if output:
        path=Path(output);history=read(path/'history.json') or []
        lineage=read(path/'ENGINEERING_CONTINUATION.json') or {}
        inherited=lineage.get('inherited_epoch',0)
        value['current_training']=history_summary(history,timing_after_epoch=inherited)
        if lineage:
            value['current_training'].update(inherited_epochs=inherited,optimized_epochs_completed=max(0,len(history)-inherited))
        value['current_training']['saved_training_status']=read(path/'training_status.json')
        value['current_training']['live']=read(path/'live_progress.json')
        check=read(path.parent/(path.name+'_input_check.json'))
        value['current_training']['input_check']=check
        if check:value['current_training']['input_check_eta_hours']=check['eta_seconds']/3600
        speed,basis=timing(history,value['current_training']['live'],inherited)
        if speed is None and pilot_speed:
            speed,basis=pilot_speed,'小规模吞吐验收临时外推；首个新完整epoch后改实测'
        value['current_training']['epoch_timing_basis']=basis
        if speed and value['current_training']['seconds_per_epoch'] is None:
            value['current_training'].update(seconds_per_epoch=speed,eta_to_epoch_cap_hours=(150-len(history))*speed/3600,
                eta_if_no_further_improvement_hours=min(max(0,150-len(history)),max(0,5-value['current_training'].get('stale_epochs',0)))*speed/3600,
                eta_note='临时吞吐外推；早停情景假设此后无改善，不是保证完成时间；完整新epoch后替换')
    if config:
        timings={};completed=0;cap_seconds=0.;rates=[]
        for job in config['jobs']:
            index=read(root/'runs'/job['name']/'index.json')
            if not index:continue
            path=Path(index['output']);h=read(path/'history.json') or []
            summary=history_summary(h)
            lineage=read(path/'ENGINEERING_CONTINUATION.json') or {}
            speed,_=timing(h,read(path/'live_progress.json'),lineage.get('inherited_epoch',0))
            if speed:
                timings[job['layer']]=speed;rates.append(speed)
        fallback=sum(rates)/len(rates) if rates else pilot_speed
        unmeasured=[]
        for job in config['jobs']:
            index=read(root/'runs'/job['name']/'index.json')
            path=Path(index['output']) if index else None
            result=read(path/'result.json') if path else None
            saved=read(path/'training_status.json') if path else None
            if result and result.get('checkpoint_reload_verified') and saved and saved.get('status')=='completed':
                completed+=1;continue
            h=read(path/'history.json') if path else None
            epochs=h[-1]['epoch'] if h else 0
            rate=timings.get(job['layer'],fallback)
            if job['layer'] not in timings:unmeasured.append(job['name'])
            if rate is not None:cap_seconds+=max(0,150-epochs)*rate
        value.update(runs_completed=completed,runs_planned=len(config['jobs']),
            queue_eta_to_epoch_caps_hours=cap_seconds/3600 if fallback else None,
            queue_eta_note='按每组最多150轮测算；早停可大幅缩短，无法预先承诺终点。未实测层临时借用其他同形状层耗时',
            queue_eta_jobs_using_other_layer_timing=unmeasured)
        if not rates and pilot_speed:
            value['queue_eta_note']='仅按小规模吞吐验收外推150轮上限；全量随机IO可能变慢，首个新完整epoch后改实测；早停时刻未知'
        if fallback and value.get('status')=='running':
            value['estimated_finish_at_epoch_caps_beijing']=(datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(seconds=cap_seconds)).isoformat(timespec='seconds')
    return value

"""Real S2/S8 disposable benchmark of exact frozen-loop execution candidates."""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .baseline import load_reference,digest
from .data import POOL
from .execution_pipeline import execution_override,epoch_function,PrefetchLoader,TrainMetrics
from .fast_pipeline import tensor_digest
from .transport_bench import loaders,model
from .packed_pipeline import packed_move
from .train_monitor import write


class TinyPacked(torch.utils.data.Dataset):
    def __len__(self):return 37
    def __getitem__(self,i):
        return dict(packed_current=torch.full((2,3),float(i),dtype=torch.float16),
                    packed_future=torch.full((2,3),float(i+1),dtype=torch.float16),
                    a_t=torch.zeros(7),sample_index=i)


def prefetch_rng_probe():
    records=[]
    for enabled in (False,True):
        torch.manual_seed(83);gen=torch.Generator().manual_seed(42)
        loader=torch.utils.data.DataLoader(TinyPacked(),batch_size=8,shuffle=True,generator=gen,num_workers=4,pin_memory=True)
        if enabled:loader=PrefetchLoader(loader)
        epochs=[]
        for _ in range(2):
            order=[];values=[]
            for batch in loader:
                order.extend(batch['sample_index'].tolist())
                if enabled:result={k:batch[k] for k in ('z_t','a_t','delta_z')}
                else:result=packed_move(batch,'cuda')
                values.append(tensor_digest(result))
            if sorted(order)!=list(range(37)):raise ValueError('Prefetch missed/duplicated an index')
            epochs.append(dict(order=order,values=values,cpu_rng=tensor_digest(torch.get_rng_state()),
                               cuda_rng=tensor_digest(torch.cuda.get_rng_state_all()),generator=tensor_digest(gen.get_state())))
        records.append(epochs)
    if records[0]!=records[1]:raise ValueError('Full-epoch prefetch tail/order/RNG differs')
    return True


class Limited:
    def __init__(self,base,batches):self.base=base;self.batches=batches;self.order=[];self.start=None
    def __len__(self):return self.batches
    def __iter__(self):
        it=iter(self.base)
        try:
            for i in range(self.batches):
                if i==16:torch.cuda.synchronize();self.start=time.perf_counter()
                value=next(it);self.order.extend(value['sample_index'].tolist());yield value
        finally:
            if hasattr(it,'close'):it.close()
            elif getattr(it,'_shutdown_workers',None):it._shutdown_workers()


def run_case(ref,plan,saved,layer,deferred,prefetch,out,measure=96,validation_batches=64,validation_prefetch=None):
    train=ref['train'];m,opt=model(ref,saved)
    gen=torch.Generator();gen.set_state(saved['loader_rng'])
    train_loader,val_loader=loaders(ref,plan,layer,4,True,gen)
    if prefetch:
        train_loader=PrefetchLoader(train_loader)
        if validation_prefetch is not False:val_loader=PrefetchLoader(val_loader)
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    limited=Limited(train_loader,16+measure);inputs=[];losses=[];grads=[];norms=[];counter=0;first=None
    original_forward=train.reference_forward;original_update=train.reference_update;clip=torch.nn.utils.clip_grad_norm_
    def forward(*args,**kwargs):
        nonlocal counter
        if counter<16:inputs.append(tensor_digest(args[1]))
        result=original_forward(*args,**kwargs)
        if counter<16:losses.append(tensor_digest(result[1]))
        counter+=1
        return result
    def clipping(parameters,*args,**kwargs):
        parameters=list(parameters)
        if counter<=16:grads.append(tensor_digest([p.grad for p in parameters]))
        return clip(parameters,*args,**kwargs)
    def update(*args,**kwargs):
        nonlocal first
        with patch.object(torch.nn.utils,'clip_grad_norm_',clipping):norm=original_update(*args,**kwargs)
        if norm is not None and counter<=16:
            norms.append(tensor_digest(norm))
            first=dict(inputs=inputs[:],losses=losses[:],gradients=grads[:],norms=norms[:],
                weights=tensor_digest(m.state_dict()),optimizer=tensor_digest(opt.state_dict()),
                cpu_rng=tensor_digest(torch.get_rng_state()),cuda_rng=tensor_digest(torch.cuda.get_rng_state_all()),
                loader_rng=tensor_digest(gen.get_state()))
        if counter%16==0:write(out/'live.json',dict(layer=layer,deferred=deferred,prefetch=prefetch,batches=counter,total=16+measure))
        return norm
    torch.cuda.reset_peak_memory_stats()
    try:
        with patch.object(train,'move',packed_move),execution_override(ref,deferred=deferred,prefetch=prefetch,validation_prefetch=validation_prefetch),\
             patch.object(train,'reference_forward',forward),patch.object(train,'reference_update',update):
            f=epoch_function(train,deferred)
            totals,count,steps,norm=f(m,opt,limited,SimpleNamespace(batch_size=32,accumulate=16,beta=1.),
                {'train':range(49447)},'cuda',ref['loss'],ref['config'].LossType.BOTH,forward,update)
            torch.cuda.synchronize();seconds=time.perf_counter()-limited.start if limited.start else None
            final=dict(weights=tensor_digest(m.state_dict()),optimizer=tensor_digest(opt.state_dict()),
                       cpu_rng=tensor_digest(torch.get_rng_state()),cuda_rng=tensor_digest(torch.cuda.get_rng_state_all()),
                       loader_rng=tensor_digest(gen.get_state()),totals=totals.tolist(),count=count,steps=steps)
            val=Limited(val_loader,validation_batches);began=time.perf_counter()
            metrics=train.evaluate(m,val,'cuda',ref['loss'],ref['config'].LossType.BOTH,1.)
            torch.cuda.synchronize();val_seconds=time.perf_counter()-began
        return dict(layer=layer,deferred=deferred,prefetch=prefetch,validation_prefetch=validation_prefetch,first_update_parity=first,
            all_updates_parity=final,order=limited.order,validation_order=val.order,validation=metrics,
            train_seconds=seconds,train_samples_per_second=(count-512)/seconds if seconds else None,
            validation_seconds=val_seconds,validation_samples_per_second=metrics['samples']/val_seconds,
            estimated_epoch_seconds=49447/((count-512)/seconds)+12089/(metrics['samples']/val_seconds) if seconds else None,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    finally:
        del m,opt,train_loader,val_loader,limited;POOL.close();gc.collect();torch.cuda.empty_cache()


def exact(left,right):
    for key in ('first_update_parity','all_updates_parity','order','validation_order','validation'):
        if left[key]!=right[key]:raise ValueError('Exact parity failed: '+key)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('base','manifest','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);state=dict(status='running',phase='preflight')
    write(a.output/'status.json',state)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
        torch.set_num_threads(4);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);source_sha=digest(a.checkpoint)
        ref=load_reference(a.base/'night-code-1e80242789',a.base/'runtime/vla-corrector-reference')
        prefetch_rng_probe()
        plan=json.loads(a.manifest.read_text());cases=[]
        for name,deferred,prefetch in [('before',False,False),('metrics',True,False),('prefetch',False,True),('combined',True,True),('after',False,False)]:
            state.update(phase='s2_frozen_loop_parity_and_speed',case=name);write(a.output/'status.json',state)
            r=run_case(ref,plan,saved,'block_02',deferred,prefetch,a.output);r['name']=name
            cases.append(r);write(a.output/(name+'.json'),r);exact(cases[0],r)
        baseline=min(r['estimated_epoch_seconds'] for r in cases if r['name'] in ('before','after'))
        best=min((r for r in cases if r['name'] not in ('before','after')),key=lambda r:r['estimated_epoch_seconds'])
        speedup=baseline/best['estimated_epoch_seconds']
        state.update(phase='s8_frozen_loop_parity_and_speed');write(a.output/'status.json',state)
        s8=[]
        for d,pref in [(False,False),(best['deferred'],best['prefetch'])]:
            s8.append(run_case(ref,plan,saved,'block_08',d,pref,a.output,measure=64))
        exact(*s8)
        write(a.output/'s8.json',s8)
        if digest(a.checkpoint)!=source_sha:raise ValueError('Original checkpoint changed')
        result=dict(status='passed' if speedup>=1.08 else 'no_robust_gain',selected=best['name'],
            deferred=best['deferred'],prefetch=best['prefetch'],speedup_vs_best_baseline=speedup,
            baseline_epoch_seconds=baseline,estimated_epoch_seconds=best['estimated_epoch_seconds'],
            s8_speedup=s8[0]['estimated_epoch_seconds']/s8[1]['estimated_epoch_seconds'],
            input_target_gradient_optimizer_rng_metrics_exact=True,source_checkpoint_sha256=source_sha,
            full_iterator_rng_and_tail_exact=True,
            inherited_epoch=saved['epoch'],manifest_sha256=digest(a.manifest),pipeline_sha256=digest(Path(__file__).with_name('execution_pipeline.py')),
            cases=[{k:v for k,v in r.items() if k not in ('first_update_parity','all_updates_parity','order','validation_order')} for r in cases],
            scope='Disposable frozen-loop execution benchmark; no official training updates or best models retained')
        write(a.output/'result.json',result);state.update(status=result['status'],phase='complete');write(a.output/'status.json',state)
    except BaseException as e:
        state.update(status='error',error=type(e).__name__+': '+str(e));write(a.output/'status.json',state);raise


if __name__=='__main__':main()

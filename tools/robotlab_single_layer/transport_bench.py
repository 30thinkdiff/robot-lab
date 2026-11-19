"""Bounded, disposable real-data benchmark against the active four-worker pipeline."""
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .baseline import load_reference,model_config,digest
from .data import POOL,EpisodeDataset
from .fast_pipeline import Indexed,mlp_collate,tensor_digest
from .packed_pipeline import packed_groups,packed_move
from .train_monitor import write
from .manifest import validate_plan


def loaders(reference,plan,layer,workers,packed,generator):
    from torch.utils.data import ConcatDataset
    groups={s:ConcatDataset([EpisodeDataset(r,layer) for r in sorted(plan['records'],key=lambda r:r['episode']) if r['split']==s])
            for s in ('train','validation')}
    # Exercise the actual frozen trainer's Subset(all-episode ConcatDataset)
    # representation, not merely a filtered per-split concatenation.
    groups,info,_=reference['train'].select_split(groups,'episode',.1,42)
    if info['samples']!={'train':49447,'validation':12089}:raise ValueError('Wrong actual split')
    if packed:groups=packed_groups(groups)
    groups={k:Indexed(v) for k,v in groups.items()}
    values=reference['train'].reference_loaders(groups,32,generator,num_workers=workers,pin_memory=True)
    if not packed:
        for loader in values:loader.collate_fn=mlp_collate
    return values


def model(reference,saved,optimizer=True):
    cls=reference['models'].precision_class(reference['model_cls'],'amp-bfloat16')
    m,opt,_=reference['train'].reference_components(cls,model_config(reference),'cuda',.0003,.0001,150)
    m.load_state_dict(saved['state_dict'],strict=True)
    if optimizer:opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    return m,opt


def run_case(reference,plan,saved,workers,packed,output,*,layer='block_02',measure=96):
    train=reference['train'];m,opt=model(reference,saved)
    gen=torch.Generator();gen.set_state(saved['loader_rng'])
    train_loader,val_loader=loaders(reference,plan,layer,workers,packed,gen)
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    move=packed_move if packed else train.move
    m.train();opt.zero_grad(set_to_none=True)
    loss_type=reference['config'].LossType.BOTH;order=[];inputs=[];losses=[];gradients=[];norms=[]
    original_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters);gradients.append(tensor_digest([p.grad for p in parameters]))
        return original_clip(parameters,*args,**kwargs)
    iterator=iter(train_loader);started=None;wait_seconds=0.;measured_samples=0;events=[]
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(16+measure):
            if i==16:torch.cuda.synchronize();started=time.perf_counter()
            begin=time.perf_counter();batch=next(iterator);wait=time.perf_counter()-begin
            order.extend(batch['sample_index'].tolist())
            if i>=16:
                a,b,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)];a.record()
            value=move(batch,'cuda',False)
            if i>=16:b.record()
            if i<16:inputs.append(tensor_digest(value))
            _,loss,_,_=train.reference_forward(m,value,reference['loss'],loss_type,1.)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
            if i<16:losses.append(tensor_digest(loss))
            scaled_loss=loss*len(batch['a_t'])/512;boundary=(i+1)%16==0
            if i<16:
                with patch.object(torch.nn.utils,'clip_grad_norm_',clip):norm=train.reference_update(m,opt,scaled_loss,boundary)
                if norm is not None:norms.append(tensor_digest(norm))
            else:
                train.reference_update(m,opt,scaled_loss,boundary);c.record();events.append((a,b,c))
                measured_samples+=len(batch['a_t']);wait_seconds+=wait
            if i==15:
                parity=dict(inputs=inputs,order=order[:],losses=losses,gradients=gradients,norms=norms,
                    weights=tensor_digest(m.state_dict()),optimizer=tensor_digest(opt.state_dict()),
                    cpu_rng=tensor_digest(torch.get_rng_state()),cuda_rng=tensor_digest(torch.cuda.get_rng_state_all()),
                    loader_rng=tensor_digest(gen.get_state()))
            if i%16==15:write(output/'live.json',dict(layer=layer,workers=workers,packed=packed,batches=i+1,total=16+measure))
        torch.cuda.synchronize();elapsed=time.perf_counter()-started if started else None
        return dict(layer=layer,workers=workers,packed=packed,parity=parity,order=order,
            samples_per_second=measured_samples/elapsed if elapsed else None,measured_samples=measured_samples,
            elapsed_seconds=elapsed,loader_wait_seconds=wait_seconds,
            gpu_transfer_and_restore_seconds=sum(a.elapsed_time(b) for a,b,c in events)/1000,
            gpu_compute_interval_seconds=sum(b.elapsed_time(c) for a,b,c in events)/1000,
            timing_note='CPU wait and CUDA intervals may overlap; do not sum them as disjoint wall-time shares',
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    finally:
        if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()
        del iterator,train_loader,val_loader,m,opt
        POOL.close();gc.collect();torch.cuda.empty_cache()


def validation(reference,plan,saved,workers,packed):
    m,opt=model(reference,saved,False);del opt
    gen=torch.Generator();gen.set_state(saved['loader_rng'])
    ls=loaders(reference,plan,'block_02',workers,packed,gen)
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    iterator=iter(ls[1]);train=reference['train'];move=packed_move if packed else train.move
    m.eval()
    try:
        with torch.no_grad():
            first=move(next(iterator),'cuda');first_hash=tensor_digest(first)
            train.reference_forward(m,first,reference['loss'],reference['config'].LossType.BOTH,1.)
        torch.cuda.synchronize();begin=time.perf_counter()
        with patch.object(train,'move',move):
            metrics=train.evaluate(m,itertools.islice(iterator,64),'cuda',reference['loss'],reference['config'].LossType.BOTH,1.)
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        return dict(workers=workers,packed=packed,first_batch_sha256=first_hash,metrics=metrics,
                    seconds=seconds,samples_per_second=metrics['samples']/seconds)
    finally:
        if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()
        del iterator,ls,m;POOL.close();gc.collect();torch.cuda.empty_cache()


def half_extremes_gpu_check():
    x=np.arange(65536,dtype=np.uint16).view(np.float16);x=x[np.isfinite(x)].copy()
    a=torch.from_numpy(x).reshape(1,-1,1);b=a.flip(1).contiguous()
    batch=dict(packed_current=a,packed_future=b,a_t=torch.zeros(1,7))
    expected={'z_t':a.float(),'a_t':batch['a_t'],'delta_z':b.float()-a.float()}
    actual=packed_move(batch,'cuda')
    if tensor_digest(actual)!=tensor_digest(expected):raise ValueError('Extreme finite half values differ on GPU')
    return True


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('base','manifest','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='preflight');write(args.output/'status.json',state)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
        torch.set_num_threads(4);plan=json.loads(args.manifest.read_text());validate_plan(plan)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True);source_hash=digest(args.checkpoint)
        if (saved['protocol']['model_type'],saved['protocol']['batch_size'],saved['protocol']['accumulate'])!=('mlp',32,16):raise ValueError('Wrong protocol')
        ref=load_reference(args.base/'night-code-1e80242789',args.base/'runtime/vla-corrector-reference')
        half_extremes_gpu_check();cases=[]
        for name,workers,packed in [('current_before',4,False),('packed4',4,True),('packed8',8,True),('current_after',4,False)]:
            state.update(phase='s2_equivalence_and_throughput',case=name);write(args.output/'status.json',state)
            r=run_case(ref,plan,saved,workers,packed,args.output);r['name']=name;cases.append(r);write(args.output/(name+'.json'),r)
            if r['parity']!=cases[0]['parity'] or r['order']!=cases[0]['order']:raise ValueError('Exact input/target/loss/gradient/update/RNG/order failed: '+name)
        best=max((r for r in cases if r['packed']),key=lambda r:r['samples_per_second'])
        base=max(r['samples_per_second'] for r in cases if not r['packed']);speedup=best['samples_per_second']/base
        if speedup<1.1:raise RuntimeError('Less than10% reliable gain; retain current training pipeline')
        state.update(phase='s8_exact_parity');write(args.output/'status.json',state)
        s8=[run_case(ref,plan,saved,w,packed,args.output,layer='block_08',measure=0) for w,packed in [(4,False),(best['workers'],True)]]
        if s8[0]['parity']!=s8[1]['parity']:raise ValueError('S8 parity failed')
        state.update(phase='validation_exact_metrics');write(args.output/'status.json',state)
        vals=[validation(ref,plan,saved,w,packed) for w,packed in [(4,False),(best['workers'],True)]]
        if vals[0]['metrics']!=vals[1]['metrics'] or vals[0]['first_batch_sha256']!=vals[1]['first_batch_sha256']:raise ValueError('Validation parity failed')
        if digest(args.checkpoint)!=source_hash:raise ValueError('Source checkpoint modified')
        result=dict(status='passed',selected_workers=best['workers'],pipeline_sha256=digest(Path(__file__).with_name('packed_pipeline.py')),
            manifest_sha256=digest(args.manifest),checkpoint_sha256=source_hash,inherited_epoch=saved['epoch'],
            s2_input_parity=True,s8_input_parity=True,gradient_parity=True,optimizer_update_parity=True,rng_parity=True,
            sample_order_parity=True,validation_parity=True,future_target_only=True,half_extremes_gpu_exact=True,
            speedup_vs_best_baseline=speedup,baseline_samples_per_second=base,selected_samples_per_second=best['samples_per_second'],
            validation=vals,estimated_epoch_seconds=49447/best['samples_per_second']+12089/vals[1]['samples_per_second'],
            candidates=[{k:v for k,v in r.items() if k not in ('order','parity')} for r in cases],
            benchmark_sha256=digest(__file__),scope='Disposable engineering probe; no official training update or new weights retained')
        write(args.output/'result.json',result);state.update(status='passed',phase='complete');write(args.output/'status.json',state)
    except BaseException as e:
        state.update(status='error',error=type(e).__name__+': '+str(e));write(args.output/'status.json',state);raise


if __name__=='__main__':main()

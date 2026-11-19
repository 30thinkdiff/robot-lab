"""Isolated real-checkpoint, real-data equality and end-to-end throughput pilot."""
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset

from .baseline import load_reference,digest,model_config
from .data import EpisodeDataset,POOL
from .manifest import validate_plan
from .fast_pipeline import Indexed,set_collate,tensor_digest,model_inputs
from .train_monitor import write


def groups(plan,layer):
    return {split:Indexed(ConcatDataset([EpisodeDataset(r,layer) for r in plan['records'] if r['split']==split]))
            for split in ('train','validation')}


def run_case(reference,plan,saved,workers,mode,output,*,measured_batches=64,only_parity=False,layer='block_02'):
    datasets=groups(plan,layer)
    config=model_config(reference)
    cls=reference['models'].precision_class(reference['model_cls'],'amp-bfloat16')
    model,opt,_=reference['train'].reference_components(cls,config,'cuda',.0003,.0001,150)
    model.load_state_dict(saved['state_dict'],strict=True);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    gen=torch.Generator();gen.set_state(saved['loader_rng'])
    loaders=set_collate(reference['train'].reference_loaders(datasets,32,gen,num_workers=workers,pin_memory=True),mode)
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    loss_type=reference['config'].LossType.BOTH;model.train();opt.zero_grad(set_to_none=True)
    train=reference['train'];inputs=[];order=[];losses=[];grads=[];norms=[]
    original_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters);grads.append(tensor_digest([p.grad for p in parameters]))
        return original_clip(parameters,*args,**kwargs)
    n_batches=16 if only_parity else 16+measured_batches
    iterator=iter(loaders[0]);started=None;measured_samples=0
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    try:
        for i in range(n_batches):
            if i==16:torch.cuda.synchronize();started=time.perf_counter()
            batch=next(iterator);order.extend(batch['sample_index'].tolist())
            if i<16:inputs.append(tensor_digest(model_inputs(batch)))
            value=train.move(batch,'cuda',False)
            _,loss,_,_=train.reference_forward(model,value,reference['loss'],loss_type,1.)
            if not torch.isfinite(loss):raise FloatingPointError('Benchmark nonfinite loss')
            if i<16:losses.append(tensor_digest(loss))
            boundary=(i+1)%16==0
            scaled_loss=loss*len(batch['a_t'])/512  # exact frozen full-group operation order
            if i<16:
                with patch.object(torch.nn.utils,'clip_grad_norm_',clip):norm=train.reference_update(model,opt,scaled_loss,boundary)
                if norm is not None:norms.append(tensor_digest(norm))
            else:
                train.reference_update(model,opt,scaled_loss,boundary);measured_samples+=len(batch['a_t'])
            if i==15:
                parity=dict(inputs=inputs,first512_order=order[:],losses=losses,gradient_sha256=grads,
                    grad_norm_sha256=norms,parameters_sha256=tensor_digest(model.state_dict()),
                    optimizer_sha256=tensor_digest(opt.state_dict()),
                    cpu_rng_sha256=tensor_digest(torch.get_rng_state()),cuda_rng_sha256=tensor_digest(torch.cuda.get_rng_state_all()),
                    loader_rng_sha256=tensor_digest(gen.get_state()))
            if i%16==15:write(output/'live.json',dict(layer=layer,workers=workers,mode=mode,batches=i+1,total=n_batches))
        torch.cuda.synchronize();seconds=time.perf_counter()-started if started else None
        result=dict(layer=layer,workers=workers,collate=mode,parity=parity,order=order,
            samples_per_second=measured_samples/seconds if seconds else None,
            measured_seconds=seconds,measured_samples=measured_samples,
            overall_seconds=time.perf_counter()-begin,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    finally:
        # Benchmark is bounded and closes its workers before the next trial.
        if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()
        del iterator,loaders,model,opt,datasets
        POOL.close();gc.collect();torch.cuda.empty_cache()
    return result


def validation_case(reference,plan,saved,workers,mode):
    datasets=groups(plan,'block_02');config=model_config(reference)
    cls=reference['models'].precision_class(reference['model_cls'],'amp-bfloat16')
    model=cls(config).cuda().eval();model.load_state_dict(saved['state_dict'],strict=True)
    generator=torch.Generator();generator.set_state(saved['loader_rng'])
    loaders=set_collate(reference['train'].reference_loaders(datasets,32,generator,num_workers=workers,pin_memory=True),mode)
    torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    iterator=iter(loaders[1]);train=reference['train'];loss_type=reference['config'].LossType.BOTH
    try:
        first=next(iterator);input_sha=tensor_digest(model_inputs(first))
        with torch.no_grad():train.reference_forward(model,train.move(first,'cuda'),reference['loss'],loss_type,1.)
        torch.cuda.synchronize();started=time.perf_counter()
        metrics=train.evaluate(model,itertools.islice(iterator,32),'cuda',reference['loss'],loss_type,1.)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        return dict(workers=workers,mode=mode,metrics=metrics,first_batch_sha256=input_sha,
                    seconds=seconds,samples_per_second=metrics['samples']/seconds)
    finally:
        if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()
        del iterator,loaders,model,datasets
        POOL.close();gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('base','manifest','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='preflight');write(a.output/'status.json',state)
    try:
        import subprocess
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
        torch.set_num_threads(4)
        plan=json.loads(a.manifest.read_text());validate_plan(plan)
        checkpoint_sha=digest(a.checkpoint);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if saved['protocol']['model_type']!='mlp' or saved['protocol']['batch_size']!=32 or saved['protocol']['accumulate']!=16:
            raise ValueError('Expected actual MLP micro32/accumulate16 checkpoint')
        ref=load_reference(a.base/'night-code-1e80242789',a.base/'runtime/vla-corrector-reference')
        cases=[]
        for name,workers,mode in [('baseline_cold',2,'legacy'),('workers4',4,'mlp_no_history'),('workers8',8,'mlp_no_history'),('baseline_warm',2,'legacy')]:
            state.update(phase='real_s2_train_benchmark',case=name);write(a.output/'status.json',state)
            row=run_case(ref,plan,saved,workers,mode,a.output);row['name']=name;cases.append(row)
            write(a.output/(name+'.json'),row)
            if row['parity']!=cases[0]['parity'] or row['order']!=cases[0]['order']:raise ValueError('Exact real input/loss/gradient/update/RNG/order parity failed: '+name)
        baseline=max(r['samples_per_second'] for r in cases if r['collate']=='legacy')
        best=max((r for r in cases if r['collate']=='mlp_no_history'),key=lambda r:r['samples_per_second'])
        speedup=best['samples_per_second']/baseline
        if speedup<1.1:raise RuntimeError('No robust speed gain over warm baseline; do not switch')
        state.update(phase='real_s8_exact_parity');write(a.output/'status.json',state)
        s8=[]
        for workers,mode in [(2,'legacy'),(best['workers'],'mlp_no_history')]:
            s8.append(run_case(ref,plan,saved,workers,mode,a.output,only_parity=True,layer='block_08'))
        if s8[0]['parity']!=s8[1]['parity']:raise ValueError('S8 input/gradient/update/RNG parity failed')
        state.update(phase='validation_throughput_and_metrics');write(a.output/'status.json',state)
        validation=[validation_case(ref,plan,saved,w,m) for w,m in [(2,'legacy'),(best['workers'],'mlp_no_history')]]
        if validation[0]['metrics']!=validation[1]['metrics'] or validation[0]['first_batch_sha256']!=validation[1]['first_batch_sha256']:
            raise ValueError('Validation inputs or metrics differ')
        if digest(a.checkpoint)!=checkpoint_sha:raise ValueError('Source checkpoint changed')
        result=dict(status='passed',selected_workers=best['workers'],collate='mlp_no_history',
            manifest_sha256=digest(a.manifest),checkpoint_sha256=checkpoint_sha,inherited_epoch=saved['epoch'],
            pipeline_sha256=digest(Path(__file__).with_name('fast_pipeline.py')),
            benchmark_sha256=digest(__file__),s2_input_parity=True,s8_input_parity=True,gradient_parity=True,
            optimizer_update_parity=True,rng_parity=True,sample_order_parity=True,
            speedup_vs_best_baseline=speedup,baseline_samples_per_second=baseline,
            selected_samples_per_second=best['samples_per_second'],
            validation=validation,estimated_epoch_seconds=49447/best['samples_per_second']+12089/validation[1]['samples_per_second'],
            candidates=[{k:v for k,v in r.items() if k not in ('parity','order')} for r in cases],
            s8_parity=[r['parity'] for r in s8],
            scope='isolated engineering throughput pilot; copied model/optimizer states discarded, not formal training epochs')
        write(a.output/'result.json',result);state.update(status='passed',phase='complete');write(a.output/'status.json',state)
    except BaseException as e:
        state.update(status='error',error=type(e).__name__+': '+str(e));write(a.output/'status.json',state);raise


if __name__=='__main__':main()

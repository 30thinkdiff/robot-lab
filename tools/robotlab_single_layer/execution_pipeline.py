"""Scoped execution candidates; frozen math/checkpoint/state-machine code stays pinned.

Only train-metric collection is AST-replaced (three exact statement patterns).
Values are reduced on CPU in the original float64 batch order, never on GPU.
"""
import ast
import copy
from contextlib import ExitStack,contextmanager
import inspect
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import torch

from .baseline import PINNED,digest
from .packed_pipeline import packed_move


class TrainMetrics:
    def __init__(self):self.rows=[];self.counts=[]
    def append(self,total,mse,cos,n):
        # CUDA mse_loss may return a scalar backed by the full B*L*D temporary
        # (192MiB here). detach alone keeps that storage alive. Stack COPYs just
        # the three scalar values into an owned 12-byte FP32 buffer; no reduction.
        self.rows.append(torch.stack(tuple(v.detach() for v in (total,mse,cos))))
        self.counts.append(n)
    def flush(self):
        sums=np.zeros(3)
        if self.rows:
            values=torch.stack(self.rows).cpu().numpy()
            for row,n in zip(values,self.counts):
                sums+=np.array([float(v) for v in row])*n
        self.rows.clear();self.counts.clear()
        return sums


def transform_metrics(function,*,helper=False):
    """Reject source drift instead of applying a broad training-loop rewrite."""
    counters={'init':0,'collect':0,'flush':0}
    class Replace(ast.NodeTransformer):
        def visit_Assign(self,node):
            text=ast.unparse(node)
            if text=='totals, count = (np.zeros(3), 0)':
                counters['init']+=1
                return [node,*ast.parse('_slc_metrics = _slc_TrainMetrics()').body]
            if (not helper and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id=='val' and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Name) and node.value.func.id=='evaluate'):
                counters['flush']+=1
                return [*ast.parse('totals = _slc_metrics.flush()').body,node]
            return self.generic_visit(node)
        def visit_AugAssign(self,node):
            if ast.unparse(node)=='totals += np.array([total.item(), mse.item(), cos.item()]) * n':
                counters['collect']+=1
                return ast.parse('_slc_metrics.append(total, mse, cos, n)').body[0]
            return self.generic_visit(node)
        def visit_Return(self,node):
            if helper:
                counters['flush']+=1
                return [*ast.parse('totals = _slc_metrics.flush()').body,node]
            return node
    result=Replace().visit(copy.deepcopy(function))
    if counters!={'init':1,'collect':1,'flush':1}:raise ValueError('Frozen metric statement patterns changed: '+str(counters))
    return result


def _function(train,node):
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace=dict(train.__dict__)
    exec(compile(module,str(train.__file__)+':scoped_execution_candidate','exec'),namespace)
    f=namespace[node.name]
    return types.FunctionType(f.__code__,train.__dict__,f.__name__,f.__defaults__)


def epoch_function(train,deferred):
    """Extract the actual frozen minibatch loop for disposable parity/perf tests."""
    if digest(train.__file__)!=PINNED['corrector_train.py']:raise ValueError('Frozen source SHA changed')
    run=next(n for n in ast.parse(Path(train.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    outer=next(n for n in run.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    start=next(i for i,n in enumerate(outer.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='model.train()')
    end=next(i for i,n in enumerate(outer.body) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('val = evaluate('))
    f=ast.parse('def _slc_epoch(model, optimizer, train_loader, args, groups, device, loss_fn, loss_type, forward_fn, update_fn):\n step = 0\n return totals, count, step, grad_norm').body[0]
    f.body=f.body[:1]+copy.deepcopy(outer.body[start:end])+f.body[-1:]
    if deferred:f=transform_metrics(f,helper=True)
    return _function(train,f)


@torch.no_grad()
def evaluate_deferred(model,loader,device,loss_fn,loss_type,beta,forward_fn,move):
    model.eval();rows=[];counts=[];zeros=[]
    for batch in loader:
        batch=move(batch,device,getattr(model,'uses_history',False))
        pred,total,mse,cos=forward_fn(model,batch,loss_fn,loss_type,beta)
        target=batch['delta_z']
        values=torch.stack((total,mse,cos,target.square().mean(),pred.square().mean().sqrt()))
        # Same per-batch finite check before zero-target handling or any update.
        if not torch.isfinite(values).all():raise FloatingPointError('验证指标出现NaN/Inf')
        n=len(pred);rows.append(values);counts.append(n)
        zeros.append((target.reshape(n,-1).norm(dim=1)==0).sum())
    sums=np.zeros(5,dtype=np.float64);count=0
    if rows:
        values=torch.stack(rows).cpu().numpy()
        for row,n in zip(values,counts):
            sums+=np.array([float(v) for v in row])*n;count+=n
        zero_targets=sum(int(v) for v in torch.stack(zeros).cpu().numpy())
    else:zero_targets=0
    names=('loss','mse','cosine_loss','zero_prediction_mse','mean_batch_pred_rms')
    return {**dict(zip(names,(sums/count).tolist())),'samples':count,'zero_targets':zero_targets}


class PrefetchLoader:
    """One GPU batch ahead; full iterator exhaustion preserves loader RNG state."""
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getattr__(self,name):return getattr(self.base,name)
    def __iter__(self):
        iterator=iter(self.base);stream=torch.cuda.Stream()
        def take():
            try:cpu=next(iterator)
            except StopIteration:return None
            with torch.cuda.stream(stream):
                batch=packed_move(cpu,'cuda')
                if 'sample_index' in cpu:batch['sample_index']=cpu['sample_index']
            return batch
        try:
            ready=take()
            while ready is not None:
                current=torch.cuda.current_stream();current.wait_stream(stream)
                batch=ready
                for value in batch.values():
                    if isinstance(value,torch.Tensor) and value.is_cuda:value.record_stream(current)
                ready=take()
                yield batch
        finally:
            stream.synchronize()
            if getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()


def moved_or_packed(batch,device,history=False):
    if 'packed_current' in batch:return packed_move(batch,device,history)
    if history or set(batch)-{'z_t','a_t','delta_z','sample_index'}:raise ValueError('Unexpected prefetched batch')
    result={key:batch[key] for key in ('z_t','a_t','delta_z')}
    expected=torch.device(device)
    if expected.type=='cuda' and expected.index is None:expected=torch.device('cuda',torch.cuda.current_device())
    if any(v.device!=expected or v.dtype!=torch.float32 for v in result.values()):
        raise ValueError('Prefetched model inputs/targets must retain original FP32/device')
    return result


@contextmanager
def execution_override(reference,*,deferred,prefetch,validation_prefetch=None):
    train=reference['train']
    if digest(train.__file__)!=PINNED['corrector_train.py']:raise ValueError('Frozen source changed')
    with ExitStack() as stack:
        if prefetch:
            load=train.reference_loaders
            def loaders(*args,**kwargs):
                pair=load(*args,**kwargs)
                return (PrefetchLoader(pair[0]),PrefetchLoader(pair[1]) if validation_prefetch is not False else pair[1])
            stack.enter_context(patch.object(train,'reference_loaders',loaders))
            stack.enter_context(patch.object(train,'move',moved_or_packed))
        if deferred:
            stack.enter_context(patch.object(train,'_slc_TrainMetrics',TrainMetrics,create=True))
            node=transform_metrics(next(n for n in ast.parse(Path(train.__file__).read_text()).body
                                       if isinstance(n,ast.FunctionDef) and n.name=='run'))
            stack.enter_context(patch.object(train,'run',_function(train,node)))
            def evaluate(model,loader,device,loss_fn,loss_type,beta,forward_fn=None):
                return evaluate_deferred(model,loader,device,loss_fn,loss_type,beta,
                    forward_fn or train.reference_forward,train.move)
            stack.enter_context(patch.object(train,'evaluate',evaluate))
        yield

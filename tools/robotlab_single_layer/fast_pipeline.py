"""MLP-only collation and loader concurrency; tensor math stays in frozen code."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def mlp_collate(samples):
    for s in samples:
        if set(s)-{'z_t','z_hist','a_t','delta_z','sample_index'}:
            raise ValueError('Unexpected sample keys; not an approved MLP sample')
        if 'z_hist' in s and tuple(s['z_hist'].shape)!=(1,*s['z_t'].shape):
            raise ValueError('Cannot discard multi-frame history')
    return default_collate([{k:v for k,v in s.items() if k!='z_hist'} for s in samples])


class Indexed(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return dict(self.base[index],sample_index=index)


def set_collate(loaders,mode):
    if mode not in ('legacy','mlp_no_history'):raise ValueError('Unknown collate mode')
    for loader in loaders:
        if mode=='mlp_no_history':loader.collate_fn=mlp_collate
    return loaders


def tensor_digest(value):
    """Include dtype/shape and all bytes. Handles CUDA and BF16 without casting."""
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().cpu().contiguous()
            h.update(str((str(v.dtype),tuple(v.shape))).encode())
            h.update(memoryview(v.reshape(-1).view(torch.uint8).numpy()).cast('B'))
        elif isinstance(x,dict):
            for key in sorted(x,key=lambda k:(type(k).__name__,repr(k))):
                h.update(repr(key).encode());walk(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for row in x:walk(row)
        else:h.update(repr(x).encode())
    walk(value)
    return h.hexdigest()


@contextmanager
def loader_override(reference,workers):
    if workers not in (2,4,8):raise ValueError('Only benchmarked worker counts2/4/8')
    train=reference['train'];original=train.reference_loaders
    def load(groups,batch_size,generator,*,num_workers=0,pin_memory=False):
        if num_workers!=workers:raise ValueError('Recorded loader protocol and actual worker count differ')
        return set_collate(original(groups,batch_size,generator,num_workers=workers,pin_memory=pin_memory),'mlp_no_history')
    with patch.object(train,'reference_loaders',load):yield


def require_acceptance(pipeline,manifest):
    from .baseline import digest
    receipt=json.loads(Path(pipeline['receipt']).read_text())
    if digest(pipeline['receipt'])!=pipeline['receipt_sha256']:raise ValueError('Pipeline acceptance changed')
    if (receipt.get('status')!='passed' or not receipt.get('selected_workers')==pipeline['workers']
        or receipt.get('pipeline_sha256')!=digest(__file__)
        or receipt.get('manifest_sha256')!=digest(manifest)):
        raise ValueError('Pipeline receipt does not bind this code/data/worker setting')
    for key in ('s2_input_parity','s8_input_parity','gradient_parity','optimizer_update_parity','rng_parity','sample_order_parity'):
        if receipt.get(key) is not True:raise ValueError('Pipeline equivalence not passed: '+key)
    if receipt['speedup_vs_best_baseline']<1.1:raise ValueError('No measured throughput benefit')


def model_inputs(batch):
    return {k:batch[k] for k in ('z_t','a_t','delta_z')}

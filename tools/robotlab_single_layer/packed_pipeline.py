"""Lossless FP16 transport; restore the exact FP32 model inputs and targets.

No feature quantization, model precision, sampling, or optimizer change. Future
features are transport-only supervision: never returned to the model as inputs.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import ConcatDataset,Dataset,Subset
from torch.utils.data._utils.collate import default_collate

from . import data
from .baseline import digest


class PackedEpisode(Dataset):
    def __init__(self,original):
        if not isinstance(original,data.EpisodeDataset):raise TypeError('Original single-layer episode required')
        self.original=original

    def __len__(self):return len(self.original)

    def __getitem__(self,index):
        ds=self.original;t,due=ds.pairs[index]
        raw=data.POOL.get(Path(ds.spec['raw_path'])/(ds.layer+'.fp16.npy'))
        current=np.array(raw[t],dtype=np.float16,copy=True).reshape(2048,768)
        future=np.array(raw[due],dtype=np.float16,copy=True).reshape(2048,768)
        actions=data.POOL.get(Path(ds.spec['connector_path'])/'actions.npy')
        action=np.array(actions[t],dtype=np.float32,copy=True)
        if not all(np.isfinite(v).all() for v in (current,future,action)):
            raise FloatingPointError('Nonfinite sample; no filtering or rescaling')
        return dict(packed_current=torch.from_numpy(current),packed_future=torch.from_numpy(future),
                    a_t=torch.from_numpy(action))


def packed_groups(groups):
    memo={}
    def wrap(dataset):
        if id(dataset) in memo:return memo[id(dataset)]
        if isinstance(dataset,data.EpisodeDataset):value=PackedEpisode(dataset)
        elif isinstance(dataset,ConcatDataset):value=ConcatDataset([wrap(v) for v in dataset.datasets])
        elif isinstance(dataset,Subset):value=Subset(wrap(dataset.dataset),dataset.indices)
        else:raise TypeError('Only frozen EpisodeDataset/ConcatDataset/Subset wrappers are accepted')
        memo[id(dataset)]=value
        return value
    return {split:wrap(dataset) for split,dataset in groups.items()}


def packed_move(batch,device,history=False):
    if history:raise ValueError('Packed transport is only accepted for the current-frame MLP')
    if set(batch)-{'packed_current','packed_future','a_t','sample_index'}:raise ValueError('Unexpected packed sample')
    current,future=batch['packed_current'],batch['packed_future']
    if current.dtype!=torch.float16 or future.dtype!=torch.float16 or current.shape!=future.shape:
        raise ValueError('Expected the unchanged FP16 stored feature pair')
    if batch['a_t'].dtype!=torch.float32:raise ValueError('Original FP32 action condition required')
    # Conversion precedes subtraction. Subtracting in FP16 would change targets.
    z=current.to(device,non_blocking=True).float()
    target_future=future.to(device,non_blocking=True).float()
    return dict(z_t=z,a_t=batch['a_t'].to(device,non_blocking=True),delta_z=target_future-z)


@contextmanager
def loader_override(reference,workers):
    if workers not in (4,8):raise ValueError('Only explicitly benchmarked worker counts4/8')
    train=reference['train'];original=train.reference_loaders
    def load(groups,batch_size,generator,*,num_workers=0,pin_memory=False):
        if num_workers!=workers:raise ValueError('Recorded and actual worker counts differ')
        loaders=original(packed_groups(groups),batch_size,generator,num_workers=workers,pin_memory=pin_memory)
        for loader in loaders:loader.collate_fn=default_collate
        return loaders
    with patch.object(train,'reference_loaders',load),patch.object(train,'move',packed_move):yield


def require_acceptance(pipeline,manifest):
    r=json.loads(Path(pipeline['receipt']).read_text())
    if digest(pipeline['receipt'])!=pipeline['receipt_sha256']:raise ValueError('Acceptance receipt changed')
    if (r.get('status')!='passed' or r.get('pipeline_sha256')!=digest(__file__)
        or r.get('manifest_sha256')!=digest(manifest) or r.get('selected_workers')!=pipeline['workers']
        or pipeline.get('kind')!='packed_fp16_transport'):
        raise ValueError('Packed transport receipt is not bound to this code/data/worker setting')
    for key in ('s2_input_parity','s8_input_parity','gradient_parity','optimizer_update_parity',
                'rng_parity','sample_order_parity','validation_parity','future_target_only'):
        if r.get(key) is not True:raise ValueError('Missing exact acceptance: '+key)
    if r['speedup_vs_best_baseline']<1.1:raise ValueError('No robust measured performance benefit')

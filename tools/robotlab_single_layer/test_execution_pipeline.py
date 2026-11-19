"""Metrics retain exact batch-order CPU reductions, gradient and RNG semantics."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset,Dataset
from unittest.mock import patch

from .test_core import reference
from .execution_pipeline import execution_override,epoch_function,TrainMetrics
from .fast_pipeline import tensor_digest


class Tiny(torch.nn.Module):
    def __init__(self,cfg=None):super().__init__();self.linear=torch.nn.Linear(3,3)
    def forward(self,z,a):return self.linear(z)+a[:,None,:]
    def num_parameters(self):return sum(p.numel() for p in self.parameters())


class ExecutionTests(unittest.TestCase):
    def test_full_acceptance_rejects_mismatched_policy_or_missing_rng_proof(self):
        from .execution_gate import require_acceptance
        from .baseline import digest
        from . import execution_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            receipt=root/'receipt.json';selected=dict(deferred=True,prefetch=True,validation_prefetch=False)
            value=dict(status='passed',pipeline_sha256=digest(execution_pipeline.__file__),manifest_sha256=digest(manifest),
                selected=selected,full_epoch_counts_exact=True,all_97_updates_parameters_optimizer_rng_metrics_exact=True,
                s8_exact=True,full_iterator_rng_and_tail_exact=True,full_s2_speedup=1.1,s8_short_speedup=1.04)
            receipt.write_text(json.dumps(value));cfg=dict(receipt=str(receipt),receipt_sha256=digest(receipt),selected=selected)
            require_acceptance(cfg,manifest)
            with self.assertRaises(ValueError):require_acceptance(dict(cfg,selected=dict(selected,prefetch=False)),manifest)
            value['full_iterator_rng_and_tail_exact']=False;receipt.write_text(json.dumps(value));cfg['receipt_sha256']=digest(receipt)
            with self.assertRaises(ValueError):require_acceptance(cfg,manifest)

    def test_scalar_buffer_does_not_retain_full_backing_storage(self):
        large=torch.arange(4096,dtype=torch.float32);scalar=large[127]
        sink=TrainMetrics();sink.append(scalar,scalar,scalar,1)
        self.assertEqual(scalar.untyped_storage().nbytes(),4096*4)
        self.assertEqual(sink.rows[0].untyped_storage().nbytes(),3*4)
        self.assertTrue(np.array_equal(sink.flush(),np.array([127.,127.,127.])))

    def test_ordered_scalar_reduction_is_not_replaced_by_gpu_sum(self):
        rows=[(1e8,2.,.1),(-1e8,3.,.2),(1.,4.,.3)]
        sink=TrainMetrics();expected=np.zeros(3)
        for i,row in enumerate(rows):
            values=[torch.tensor(v) for v in row];sink.append(*values,i+1)
            expected+=np.array([v.item() for v in values])*(i+1)
        self.assertTrue(np.array_equal(sink.flush(),expected));self.assertFalse(sink.rows)

    def test_actual_frozen_epoch_tail_updates_metrics_weights_optimizer_rng_exact(self):
        ref=reference();train=ref['train'];torch.manual_seed(67)
        samples=[dict(z_t=torch.randn(2,3),a_t=torch.randn(3),delta_z=torch.randn(2,3)) for _ in range(11)]
        samples[2]['delta_z'].zero_()
        initial=Tiny().state_dict();records=[]
        for deferred in (False,True):
            m=Tiny();m.load_state_dict(initial);opt=torch.optim.AdamW(m.parameters(),lr=.0003)
            gen=torch.Generator().manual_seed(73);torch.manual_seed(91)
            loader=DataLoader(samples,batch_size=3,shuffle=True,generator=gen)
            with execution_override(ref,deferred=deferred,prefetch=False):
                f=epoch_function(train,deferred)
                result=f(m,opt,loader,SimpleNamespace(batch_size=3,accumulate=2,beta=1.),
                         {'train':samples},'cpu',ref['loss'],ref['config'].LossType.BOTH,
                         train.reference_forward,train.reference_update)
                validation=train.evaluate(m,DataLoader(samples,batch_size=3),'cpu',ref['loss'],ref['config'].LossType.BOTH,1.)
                records.append(dict(metrics=result[0].copy(),count=result[1],step=result[2],norm=result[3].clone(),
                    weights=tensor_digest(m.state_dict()),optimizer=tensor_digest(opt.state_dict()),validation=validation,
                    rng=tensor_digest(torch.get_rng_state()),loader_rng=tensor_digest(gen.get_state())))
        self.assertTrue(np.array_equal(records[0].pop('metrics'),records[1].pop('metrics')))
        self.assertTrue(torch.equal(records[0].pop('norm'),records[1].pop('norm')))
        self.assertEqual(records[0],records[1])

    def test_nonfinite_validation_raises_on_the_same_batch(self):
        ref=reference();torch.manual_seed(81);m=Tiny()
        sample=dict(z_t=torch.randn(1,2,3),a_t=torch.randn(1,3),delta_z=torch.full((1,2,3),float('nan')))
        for deferred in (False,True):
            with execution_override(ref,deferred=deferred,prefetch=False):
                with self.assertRaises(FloatingPointError):
                    ref['train'].evaluate(m,[sample],'cpu',ref['loss'],ref['config'].LossType.BOTH,1.)

    def test_frozen_run_pause_resume_and_best_checkpoint_payload_exact(self):
        from .training import arguments
        from .test_core import SOURCE
        ref=reference();train=ref['train'];torch.manual_seed(31)
        class Episode(Dataset):
            def __init__(self,ep,n):
                self.ep_idx=np.full(n,ep);self.rows=[dict(z_t=torch.randn(2,3),a_t=torch.randn(3),delta_z=torch.randn(2,3)) for _ in range(n)]
            def __len__(self):return len(self.rows)
            def __getitem__(self,i):return self.rows[i]
        episodes={'train':Episode(1,11),'validation':Episode(2,5)}
        groups={k:ConcatDataset([v]) for k,v in episodes.items()}
        records=[dict(episode=int(v.ep_idx[0]),split=k,frames=len(v),pairs=len(v),files={}) for k,v in episodes.items()]
        payloads=[]
        with tempfile.TemporaryDirectory() as temp:
            for deferred in (False,True):
                out=Path(temp)/str(deferred)
                args=arguments(Path(temp)/'unused.json',SOURCE,out,device='cpu')
                args.manifest=None;args.batch_size=3;args.accumulate=2;args.loader_workers=0
                args.epochs=3;args.patience=3;args.cpu_threads=2;args.predictor_precision='float32'
                args.stop_after_epoch=1;out.mkdir()
                with patch.object(train,'reference_api',return_value=(ref['config'],None,Tiny,ref['loss'],ref['hashes'])),\
                     patch.object(train,'datasets',return_value=(groups,records,(2,3,3,10.))),\
                     execution_override(ref,deferred=deferred,prefetch=False),contextlib.redirect_stdout(io.StringIO()):
                    train.run(args)
                    first=torch.load(out/'latest.pt',weights_only=True)
                    self.assertEqual(first['epoch'],1)
                    args.resume=True;args.stop_after_epoch=0
                    train.run(args)
                state=torch.load(out/'latest.pt',weights_only=True)
                for row in state['history']:row.pop('seconds')
                payloads.append(tensor_digest(state))
        self.assertEqual(payloads[0],payloads[1])


if __name__=='__main__':unittest.main()

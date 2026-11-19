"""Queue/ETA regressions; fixtures are not real training performance evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .train_queue import jobs,attempt,child_environment
from .train_monitor import history_summary,observe,status


class TrainingQueueTests(unittest.TestCase):
    def test_fixed_six_jobs_and_fresh_child_import(self):
        values=jobs()
        self.assertEqual([r['name'] for r in values],['S2_s42','S8_s42','S2_s43','S8_s43','S2_s44','S8_s44'])
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{},clear=True):
            result=subprocess.run([sys.executable,'-c','import robotlab_single_layer.train_queue'],
                cwd=temp,env=child_environment(),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_early_stop_etas_are_conditional_and_best_is_total_loss(self):
        self.assertIsNone(history_summary([])['eta_to_epoch_cap_hours'])
        h=[dict(epoch=i,step=97*i,seconds=100,validation=dict(loss=v,mse=10-i)) for i,v in enumerate([3.,2.,2.1,2.2],1)]
        value=history_summary(h)
        self.assertEqual(value['best_epoch'],2)
        self.assertEqual(value['stale_epochs'],2)
        self.assertEqual(value['eta_if_no_further_improvement_hours'],300/3600)
        self.assertEqual(value['eta_to_epoch_cap_hours'],14600/3600)

    def test_incomplete_startup_preserved_and_latest_resume_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=jobs()[0]
            first,resume,done=attempt(root,job);self.assertFalse(resume or done)
            first.mkdir();(first/'protocol.json').write_text('{}')
            second,resume,done=attempt(root,job)
            self.assertNotEqual(first,second);self.assertTrue((first/'protocol.json').exists())
            second.mkdir();(second/'latest.pt').write_bytes(b'synthetic checkpoint marker')
            (second/'protocol.json').write_text('{}')
            third,resume,done=attempt(root,job)
            self.assertEqual(second,third);self.assertTrue(resume);self.assertFalse(done)

    def test_continuation_timing_excludes_inherited_epochs_but_keeps_best(self):
        from .train_monitor import write
        history=[dict(epoch=i,step=97*i,seconds=1080 if i<=7 else 400,
                      validation=dict(loss=1/i)) for i in range(1,9)]
        self.assertEqual(history_summary(history,timing_after_epoch=7)['seconds_per_epoch'],400)
        self.assertIsNone(history_summary(history[:7],timing_after_epoch=7)['seconds_per_epoch'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'runs'/'S2_s42'/'attempt_001';run.mkdir(parents=True)
            receipt=root/'receipt.json';write(receipt,dict(status='passed',estimated_epoch_seconds=450))
            write(root/'queue_config.json',dict(jobs=jobs(),pipeline=dict(receipt=str(receipt))))
            write(root/'queue_status.json',dict(status='paused',current_output=str(run)))
            write(run.parent/'index.json',dict(output=str(run)))
            write(run/'ENGINEERING_CONTINUATION.json',dict(inherited_epoch=7))
            write(run/'history.json',history[:7])
            first=status(root)
            self.assertEqual(first['current_training']['seconds_per_epoch'],450)
            self.assertEqual(first['current_training']['best_epoch'],7)
            self.assertIn('临时外推',first['current_training']['epoch_timing_basis'])
            write(run/'history.json',history)
            later=status(root)
            self.assertEqual(later['current_training']['seconds_per_epoch'],400)
            self.assertEqual(later['current_training']['optimized_epochs_completed'],1)

    def test_observer_preserves_parameters_loss_gradients_and_rng(self):
        import torch
        from .test_core import reference
        ref=reference();train=ref['train'];torch.manual_seed(81)
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(3,3)
            def forward(self,z,a):return self.linear(z)+a[:,None,:]
        first=Tiny();second=copy.deepcopy(first)
        batch=dict(z_t=torch.randn(2,2,3),a_t=torch.randn(2,3),delta_z=torch.randn(2,2,3))
        loss_type=ref['config'].LossType.BOTH;expected=[];actual=[]
        def run(model,records):
            opt=torch.optim.AdamW(model.parameters(),lr=.0003)
            for _ in range(3):
                out=train.reference_forward(model,batch,ref['loss'],loss_type,1.)
                norm=train.reference_update(model,opt,out[1],True)
                records.append((out[0].detach().clone(),out[1].detach().clone(),norm.detach().clone()))
            model.eval()
            with torch.no_grad():train.reference_forward(model,batch,ref['loss'],loss_type,1.)
        rng=torch.get_rng_state().clone();run(first,expected)
        with tempfile.TemporaryDirectory() as temp:
            with observe(ref,Path(temp)):run(second,actual)
            live=json.loads((Path(temp)/'live_progress.json').read_text())
            self.assertEqual(live['train_samples_after_backward'],6)
            self.assertEqual(live['epoch_optimizer_updates'],3)
            self.assertEqual(live['validation_samples_forwarded'],2)
        for a,b in zip(expected,actual):
            for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        for a,b in zip(first.parameters(),second.parameters()):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()

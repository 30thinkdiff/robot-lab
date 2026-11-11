"""混合检测/引导的协议拒绝、目标来源、惰性调用及原OGG数学等价性。"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
import corrector_hybrid as hybrid
import corrector_models as original


class TinyGuide(torch.nn.Module):
    uses_history=True
    def __init__(self,sequence):
        super().__init__();self.uses_action_sequence=sequence
        self.cfg=SimpleNamespace(action_steps=10,action_dim=7)
        self.map=torch.nn.Linear(7,3,bias=False);self.calls=0
        self.eval().requires_grad_(False)
    def forward(self,visual,action):
        self.calls+=1
        if self.uses_action_sequence:action=action.mean(1)
        return visual[:,-1]+self.map(action)[:,None,:]


def protocols():
    single={'model_type':'transformer','encoder_policy_revision':'fixed','source_sha':'ref',
        'data_manifest_sha256':'data','fps':10.,'k_step':10,'prediction_seconds':1.,
        'h_window':1,'anchor_history':3,'precision':'amp-bfloat16','records':[{'episode':1}],
        'model':{'token_dim':960,'action_dim':7,'d_model':768,'n_heads':12,'n_layers':5,'dropout':0.}}
    seq=copy.deepcopy(single);seq['model_type']='transformer-action';seq['action_steps']=10
    seq['model']['action_steps']=10
    return single,seq


class HybridCorrectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.z=torch.randn(1,4,3);self.target=torch.randn(1,4,3)
        self.x=torch.randn(1,50,32);self.v=torch.randn_like(self.x);self.t=torch.tensor([.5])

    def test_protocol_compatibility_and_data_timebase(self):
        a,b=protocols();report=hybrid.compatible(a,b)
        self.assertEqual(report['online_due_steps'],20)
        self.assertEqual(report['guide_action_offsets'],list(range(0,20,2)))
        for key,bad in [('encoder_policy_revision','other'),('prediction_seconds',.5),('fps',20),('records',[])]:
            changed=copy.deepcopy(b);changed[key]=bad
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):hybrid.compatible(a,changed)

    def test_different_feature_dimension_and_action_length_rejected(self):
        a,b=protocols();b['model']['token_dim']=768
        with self.assertRaisesRegex(ValueError,'token_dim'):hybrid.compatible(a,b)
        a,b=protocols();b['action_steps']=9
        with self.assertRaisesRegex(ValueError,'A10'):hybrid.compatible(a,b)

    def test_single_and_sequence_match_original_guidance_exactly(self):
        for sequence in [False,True]:
            guide=TinyGuide(sequence)
            offsets=list(range(0,20,2)) if sequence else [0]
            actual,stats=hybrid.guidance_update(guide,self.z,self.x,self.v,self.t,self.target,offsets)
            if sequence:
                expected,loss=original.guided_sequence_velocity(guide,self.z[:,None],self.x,self.v,self.t,self.target,2)
            else:
                expected,loss=original.guided_history_velocity(guide,self.z[:,None],self.x,self.v,self.t,self.target)
            self.assertTrue(torch.equal(actual,expected));self.assertEqual(stats['ogg_loss'],loss)
            self.assertGreater(stats['gradient_l2'],0)
            self.assertTrue(all(p.grad is None for p in guide.parameters()))
            for index in range(50):
                if index not in offsets:self.assertTrue(torch.equal(actual[:,index],self.v[:,index]))
            self.assertTrue(torch.equal(actual[:,:,7:],self.v[:,:,7:]))

    def test_target_must_have_same_shape_no_broadcast(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            hybrid.guidance_update(TinyGuide(True),self.z,self.x,self.v,self.t,self.target[:,:1],list(range(0,20,2)))

    def test_frozen_eval_required(self):
        g=TinyGuide(False);g.train()
        with self.assertRaisesRegex(ValueError,'冻结'):
            hybrid.guidance_update(g,self.z,self.x,self.v,self.t,self.target,[0])

    def test_guide_only_called_during_triggered_recovery_query(self):
        obj=hybrid.HybridCorrector.__new__(hybrid.HybridCorrector)
        obj.guide=TinyGuide(True);obj.guide_offsets=list(range(0,20,2))
        obj.original_denoise=lambda *a: self.v
        obj.denoise_index=0;obj.every=3;obj.active=False;obj.last_event=None
        for _ in range(10):obj.denoise_step(None,None,self.x,self.t)
        self.assertEqual(obj.guide.calls,0)
        obj.active=True;obj.denoise_index=0;obj.last_event='event_24';obj.target_now=self.target
        obj.z_now=self.z;obj.eta=1.;obj.guide_s=0.;obj.guidance_calls=0;obj.step=24;obj.label='B'
        rows=[];obj.log=rows.append
        for _ in range(10):obj.denoise_step(None,None,self.x,self.t)
        self.assertEqual(obj.guide.calls,4);self.assertEqual(obj.guidance_calls,4)
        self.assertEqual([r['denoise'] for r in rows],[0,3,6,9])
        self.assertTrue(all(r['event']=='event_24' and r['gradient_l2']>0 for r in rows))
        obj.active=False
        for _ in range(10):obj.denoise_step(None,None,self.x,self.t)
        self.assertEqual(obj.guide.calls,4)

    def test_no_guide_without_detector_target(self):
        obj=hybrid.HybridCorrector.__new__(hybrid.HybridCorrector)
        obj.original_denoise=lambda *a:self.v
        obj.active=True;obj.denoise_index=0;obj.every=3;obj.last_event=None
        with self.assertRaisesRegex(RuntimeError,'禁止调用引导器'):
            obj.denoise_step(None,None,self.x,self.t)

    def test_correction_target_is_detector_difference_without_guide_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=hybrid.HybridCorrector.__new__(hybrid.HybridCorrector)
            obj.directory=Path(tmp);obj.step=24;obj.k=20;obj.z_now=self.z;obj.label='B'
            obj.wall_start=time.perf_counter();obj.target_files=[]
            obj.guide=TinyGuide(True);obj.log=lambda row:None
            expected=torch.randn_like(self.z);observed=torch.randn_like(self.z)
            target=obj.detector_target(expected,observed)
            self.assertTrue(torch.equal(target,expected-observed));self.assertEqual(obj.guide.calls,0)
            saved=torch.load(obj.directory/'event_0024.pt',weights_only=True)
            self.assertTrue(torch.equal(saved['target'],target))
            self.assertEqual(obj.target_files[0]['target_source'],'detector_expected_minus_observed')

    def test_metadata_hash_mismatch_stops_before_feature_reinterpretation(self):
        a,b=protocols()
        a['records'][0]['files']={'metadata.json':'wrong'}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'features/episode_1';p.mkdir(parents=True);(p/'metadata.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'metadata'):
                hybrid.feature_evidence(Path(tmp),a,b)


if __name__=='__main__':unittest.main()

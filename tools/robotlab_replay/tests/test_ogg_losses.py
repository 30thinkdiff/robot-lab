"""CPU autograd tests against actual frozen deployed cosine source."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch=None

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
FROZEN=ROOT/'outputs/corrector_tracking_fix_20261114/dev/j36m-35ecdd7488bf'


def frozen_function(path,names,namespace):
    tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(body)!=len(names):raise ValueError('frozen functions missing')
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['guided_velocity']


@unittest.skipIf(torch is None,'requires existing Torch CPU environment')
@unittest.skipUnless((FROZEN/'scripts/j36m/online.py').is_file() and (FROZEN/'studies/policy_generalization/corrector/online_practice.py').is_file(),
                    'requires local audited frozen source snapshot; not distributed in Git')
class LossTests(unittest.TestCase):
    def setUp(self):
        from corrector_ogg_losses import mlp_factory,joint_factory,components,LAYERS
        self.mlp_factory,self.joint_factory,self.components=mlp_factory,joint_factory,components
        torch.manual_seed(12)
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__();self.cfg=SimpleNamespace(action_dim=7);self.fc=torch.nn.Linear(7,12);self.calls=0
            def forward(self,z,a):self.calls+=1;return self.fc(a).reshape(-1,3,4)+z*.1
        class J(torch.nn.Module):
            def __init__(self):super().__init__();self.fc=torch.nn.Linear(7,12);self.calls=0
            def forward(self,v):
                self.calls+=1;y=self.fc(v['action']).reshape(-1,3,4)
                return {'delta_'+k:y*(i+1) for i,k in enumerate(LAYERS)}
        self.m=M().eval().requires_grad_(False);self.j=J().eval().requires_grad_(False)
        self.x=torch.randn(2,50,32);self.v=torch.randn_like(self.x);self.t=torch.tensor([.8,.3]);self.z=torch.randn(2,3,4)
        self.target=torch.randn_like(self.z);self.targets={k:self.target*(i+1) for i,k in enumerate(LAYERS)}
        self.cal={'weights':dict(zip(LAYERS,[.42,.29,.29])),'norm_floor':dict.fromkeys(LAYERS,1e-8)}
        ns={'torch':torch,'F':F,'LAYERS':LAYERS,'np':__import__('numpy')}
        self.original_j=frozen_function(FROZEN/'scripts/j36m/online.py',{'guided_velocity'},dict(ns))
        self.original_m=frozen_function(FROZEN/'studies/policy_generalization/corrector/online_practice.py',{'visual_mismatch','guided_velocity'},dict(ns))

    def test_cosine_is_bitwise_unchanged_and_one_forward(self):
        for model,factory,original,args in [(self.m,self.mlp_factory,self.original_m,(self.z,self.x,self.v,self.t,self.target)),
            (self.j,self.joint_factory,self.original_j,({'connector':self.z},self.x,self.v,self.t,self.targets,self.cal))]:
            expected=original(model,*args);model.calls=0
            actual=factory(original,'cosine')(model,*args)
            self.assertTrue(torch.equal(expected[0],actual[0]));self.assertEqual(expected[1],actual[1]);self.assertEqual(model.calls,1)

    def test_raw_mse_and_sum_have_expected_values(self):
        p=torch.tensor([[[1.,2.]]],requires_grad=True);t=torch.tensor([[[3.,0.]]])
        m,terms=self.components(p,t,'mse');c,_=self.components(p,t,'cosine');b,_=self.components(p,t,'mse_cosine')
        self.assertEqual(float(m.detach()),4.);torch.testing.assert_close(b,m+c);self.assertEqual(terms['raw_mse'],4.)

    def test_both_modes_gradient_support_frozen_inputs_and_finite(self):
        for mode in ['mse','mse_cosine']:
            for model,factory,original,args in [(self.m,self.mlp_factory,self.original_m,(self.z,self.x,self.v,self.t,self.target)),
                (self.j,self.joint_factory,self.original_j,({'connector':self.z},self.x,self.v,self.t,self.targets,self.cal))]:
                model.calls=0;before=self.v.clone();out,_=factory(original,mode)(model,*args)
                self.assertTrue(torch.isfinite(out).all());self.assertGreater(float((out-self.v).norm()),0)
                self.assertTrue(torch.equal(out[:,1:],self.v[:,1:]));self.assertTrue(torch.equal(out[:,0,7:],self.v[:,0,7:]));self.assertEqual(model.calls,1)
                self.assertTrue(torch.equal(before,self.v));self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_joint_weights_reliable_layers_and_invalid_target(self):
        self.cal['weights']['block_02']=0
        out,r=self.joint_factory(self.original_j,'mse')(self.j,{'connector':self.z},self.x,self.v,self.t,self.targets,self.cal)
        self.assertEqual(r['valid_layers'],['block_08','connector'])
        self.assertAlmostEqual(r['loss'],sum(r['per_layer_loss'].values())/2,places=5)
        target={k:torch.zeros_like(v) for k,v in self.targets.items()}
        out,r=self.joint_factory(self.original_j,'mse')(self.j,{'connector':self.z},self.x,self.v,self.t,target,self.cal)
        self.assertFalse(r['updated']);self.assertTrue(torch.equal(out,self.v))
        with self.assertRaises(ValueError):self.components(torch.ones(1,2,3),torch.ones(1,3,3),'mse')

    def test_lvm_code_and_frozen_sources_are_not_modified(self):
        source=(ROOT/'scripts/corrector_loss_ablation.py').read_text()
        self.assertIn("patch.object(obj,name,value)",source)
        self.assertNotIn("(legacy,'visual_mismatch'",source)
        self.assertNotIn('write_text', (ROOT/'scripts/corrector_ogg_losses.py').read_text())

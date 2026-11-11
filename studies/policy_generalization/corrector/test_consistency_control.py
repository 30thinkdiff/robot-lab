"""固定触发规则、校准隔离及闭环换计划不抹除真实历史。"""
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_consistency_control import FrozenTriggerRule,ConsistencyLVM
import corrector_online_runtime


class Predictor(torch.nn.Module):
    uses_history=True;uses_action_sequence=True
    cfg=SimpleNamespace(token_dim=3,action_dim=7,action_steps=10)
    def forward(self,z,a):return torch.zeros_like(z[:,-1])


class Policy:
    def __init__(self):
        self.config=SimpleNamespace(rtc_config=None);self.model=SimpleNamespace(denoise_step=lambda *a:None)
        self.chunks=0;self.reset()
    def reset(self):self._queues={'action':deque(maxlen=50)}
    def select_action(self,batch):
        if not self._queues['action']:
            self.chunks+=1;self._queues['action'].extend([torch.full((1,7),float(self.chunks)) for _ in range(50)])
        return self._queues['action'].popleft()


class ConsistencyControlTests(unittest.TestCase):
    def config(self):return {'threshold':.5,'alpha':1.,'consecutive':5,'cooldown_steps':20}

    def test_persistence_and_control_clock_cooldown(self):
        rule=FrozenTriggerRule(self.config());events=[]
        for t in range(60):
            if rule.receive(1.,t):events.append(t)
        self.assertEqual(events,[4,28,52])

    def test_missing_scores_break_consecutive_count(self):
        rule=FrozenTriggerRule(self.config())
        for t in range(4):self.assertFalse(rule.receive(1.,t))
        self.assertFalse(rule.receive(1.,20))
        for t in range(21,24):self.assertFalse(rule.receive(1.,t))
        self.assertTrue(rule.receive(1.,24))

    def test_nonfinite_and_reversed_clock_rejected(self):
        rule=FrozenTriggerRule(self.config())
        with self.assertRaises(ValueError):rule.receive(float('nan'),0)
        rule.receive(.1,1)
        with self.assertRaises(ValueError):rule.receive(.1,1)

    def test_rule_recomputed_from_reference_before_held_out_tests(self):
        p=ROOT/'studies/policy_generalization/corrector/manifests/closed_loop_calibration_20261111.json'
        c=json.loads(p.read_text());values=sorted(c['source']['scores']);i=(len(values)-1)*.95;lo=int(i);hi=min(lo+1,len(values)-1)
        self.assertEqual(values[lo]*(hi-i)+values[hi]*(i-lo),c['rule']['threshold'])
        self.assertNotIn(c['source']['seed'],c['test_conditions']['seeds'])
        self.assertNotIn(0,c['test_conditions']['initial_states'])
        self.assertFalse(c['test_result_tuning_allowed'])

    def test_interrupt_discards_plan_but_keeps_true_past(self):
        protocol={'prediction_seconds':1.,'h_window':1,'fps':10.,'action_steps':10,'k_step':10,'anchor_history':3}
        factory=lambda **kw:SimpleNamespace(**kw)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(corrector_online_runtime,'monitor_api',return_value=(factory,None,'sha')), \
             patch.object(corrector_online_runtime.OnlineCorrector,'encode',lambda self,b:torch.ones(1,2,3)):
            p=Policy();r=ConsistencyLVM(p,Predictor().eval(),protocol,Path('.'),Path(tmp),'executed_lvm',
                                       self.config(),lambda raw:torch.tensor(raw)[None])
            r.reset()
            for t in range(35):
                action=r.select_action({});r.outer_action=action[0].numpy().copy()
                r.controller_step(lambda raw:None,r.outer_action)
                self.assertEqual(r.executed.submitted,t+1)
            self.assertEqual(r.events,1);self.assertEqual(p.chunks,2)
            self.assertEqual(r.executed.observations,35)
            self.assertEqual(r.executed.latest['origin_step'],14)
            self.assertEqual(r.guidance_calls,0)
            self.assertEqual(r.raw_commands[27]['raw_controller_action'][0],1.)
            self.assertEqual(r.raw_commands[28]['raw_controller_action'][0],2.)
            self.assertEqual([t for t,_ in r.executed.actions],list(range(15,35)))


if __name__=='__main__':unittest.main()

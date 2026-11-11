"""到期真实动作采样、时序、重置与被动控制门禁。"""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from corrector_executed_monitor import ExecutedHistory,PassiveExecutedMonitor,lazy_controller_capture


class Model(torch.nn.Module):
    def forward(self,z,a):
        self.last_action=a.clone();self.last_z=z.clone()
        return z[:,-1]+a[:,:,:3].mean(1)[:,None,:]


class ExecutedMonitorTests(unittest.TestCase):
    def test_true_past_actions_and_observation_endpoints(self):
        m=Model();h=ExecutedHistory(m)
        for t in range(22):
            z=torch.full((1,2,3),float(t))
            row=h.observe(t,z)
            if t<20:self.assertIsNone(row['executed_score'])
            else:
                indices=list(range(t-20,t,2))
                self.assertEqual(row['action_indices'],indices)
                self.assertEqual(m.last_action[0,:,0].tolist(),indices)
                self.assertEqual(float(m.last_z[0,0,0,0]),t-20)
                self.assertEqual(row['action_window'],[t-20,t-1])
                self.assertLess(max(indices),t)
            h.submit(t,torch.full((1,7),float(t)))

    def test_missing_duplicate_or_future_action_rejected(self):
        h=ExecutedHistory(Model());z=torch.zeros(1,2,3)
        with self.assertRaisesRegex(RuntimeError,'观测之后'):h.submit(0,torch.ones(1,7))
        h.observe(0,z)
        with self.assertRaisesRegex(RuntimeError,'Z_t'):h.observe(1,z)
        h.submit(0,torch.ones(1,7))
        with self.assertRaisesRegex(RuntimeError,'控制提交'):h.submit(0,torch.ones(1,7))

    def test_reset_has_no_cross_episode_visual_or_actions(self):
        h=ExecutedHistory(Model())
        for t in range(21):
            h.observe(t,torch.full((1,2,3),float(t)));h.submit(t,torch.ones(1,7))
        h.reset();self.assertEqual(len(h.actions),0);self.assertEqual(len(h.frames),0)
        self.assertIsNone(h.observe(0,torch.zeros(1,2,3))['executed_score'])

    def test_missing_history_is_not_padded(self):
        h=ExecutedHistory(Model())
        for t in range(20):
            h.observe(t,torch.ones(1,2,3));h.submit(t,torch.ones(1,7))
        h.actions.popleft()
        with self.assertRaisesRegex(RuntimeError,'禁止填零'):h.observe(20,torch.ones(1,2,3))

    def test_common_evaluation_warmup_is_explicit(self):
        h=ExecutedHistory(Model())
        for t in range(25):
            row=h.observe(t,torch.ones(1,2,3));h.submit(t,torch.ones(1,7))
            self.assertEqual(row['eligible_executed'],t>=20)
            self.assertEqual(row['eligible_common'],t>=24)

    def test_controller_capture_excludes_reset_and_rejects_changed_actions(self):
        import numpy as np
        obj=PassiveExecutedMonitor.__new__(PassiveExecutedMonitor)
        obj.in_reset=True;obj.settling_commands=0
        self.assertEqual(obj.controller_step(lambda a:'settled',np.zeros(7)),'settled')
        self.assertEqual(obj.settling_commands,1)
        obj.in_reset=False;obj.selected={'step':0};obj.outer_action=np.zeros(7,dtype=np.float32)
        obj.baseline_actions=[[1.]*7]
        with self.assertRaisesRegex(RuntimeError,'改变了已核验H50'):
            obj.controller_step(lambda a:self.fail('不得提交改变的控制'),np.zeros(7))

    def test_lazy_environment_is_hooked_after_first_reset(self):
        class Controller:
            def step(self,a):return ('stepped',a)
        class Env:
            _env=None
            def reset(self):
                if self._env is None:self._env=Controller()
                self._env.step('settling')
                return 'reset_done'
        class Runtime:
            in_reset=False
            def __init__(self):self.policy_actions=[];self.settling=0
            def controller_step(self,original,a):
                if self.in_reset:self.settling+=1
                else:self.policy_actions.append(a)
                return original(a)
        env=Env();runtime=Runtime()
        self.assertIsNone(env._env)
        with lazy_controller_capture(env,runtime):
            self.assertEqual(env.reset(),'reset_done')
            self.assertEqual(env._env.step('policy0'),('stepped','policy0'))
            env.reset()
            env._env.step('policy1')
            self.assertEqual(runtime.policy_actions,['policy0','policy1'])
            self.assertEqual(runtime.settling,1)
        env._env.step('outside')
        self.assertEqual(runtime.policy_actions,['policy0','policy1'])


if __name__=='__main__':unittest.main()

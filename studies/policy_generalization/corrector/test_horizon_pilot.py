"""实际前H缓存消费、推理次数与条件对齐检查；不加载真实VLA。"""
from collections import deque
from contextlib import ExitStack, nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('horizon_pilot', ROOT/'scripts/smolvla_horizon_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class FakePolicy:
    def __init__(self, h):
        self.config = SimpleNamespace(n_action_steps=h, num_steps=10)
        self.model = SimpleNamespace(sample_actions=self.sample, denoise_step=lambda: None)
        self.count = 0
        self.reset()

    def reset(self):
        self._queues = {'action': deque(maxlen=self.config.n_action_steps)}

    def sample(self, batch):
        self.count += 1
        for _ in range(10):
            self.model.denoise_step()
        # 每块值不同，验证不能误执行上一块剩余动作。
        return torch.arange(50*7).reshape(1, 50, 7)+self.count*1000

    def _get_action_chunk(self, batch):
        return self.model.sample_actions(batch)

    def select_action(self, batch):
        if not self._queues['action']:
            out = self._get_action_chunk(batch)
            self._queues['action'].extend(out.transpose(0, 1)[:self.config.n_action_steps])
        return self._queues['action'].popleft()


class HorizonPilotTests(unittest.TestCase):
    def test_each_horizon_reinfers_using_new_batch_after_exact_h_actions(self):
        for h in pilot.HORIZONS:
            with self.subTest(h=h):
                policy = FakePolicy(h)
                audit = pilot.QueueAudit(policy, h, Path('.'), warmup=0)
                audit.warm = True
                seen = []
                original = audit.original_chunk
                def chunk(batch):
                    seen.append(batch['step'])
                    return original(batch)
                audit.original_chunk = chunk
                with ExitStack() as stack:
                    stack.enter_context(patch.object(torch.cuda, 'synchronize'))
                    stack.enter_context(patch.object(policy, '_get_action_chunk', audit.chunk))
                    stack.enter_context(patch.object(policy.model, 'sample_actions', audit.sample))
                    stack.enter_context(patch.object(policy.model, 'denoise_step', audit.denoise_step))
                    values = [int(audit.select({'step': s})[0, 0]) for s in range(2*h+1)]
                audit.check()
                self.assertEqual(seen, [0, h, 2*h])
                self.assertEqual(audit.calls, 3)
                self.assertEqual(audit.denoise, 30)
                self.assertEqual(values[h], 2000)
                self.assertEqual(values[-1], 3000)

    def test_rejects_old_queue_at_new_episode_or_boundary(self):
        policy = FakePolicy(2)
        audit = pilot.QueueAudit(policy, 2, Path('.'))
        audit.warm = True
        policy._queues['action'].append(torch.zeros(1, 7))
        with self.assertRaisesRegex(RuntimeError, '旧队列'):
            audit.select({})

    def test_rejects_wrong_chunk_size_and_wrong_action(self):
        policy = FakePolicy(2)
        audit = pilot.QueueAudit(policy, 2, Path('.'))
        audit.active = True
        audit.original_chunk = lambda *a: torch.zeros(1, 2, 7)
        with patch.object(torch.cuda, 'synchronize'), self.assertRaisesRegex(RuntimeError, '非完整50'):
            audit.chunk({})
        audit.warm = True
        audit.active = False
        audit.original_select = lambda *a: torch.ones(1, 7)
        with self.assertRaisesRegex(RuntimeError, '实际动作块调用'):
            audit.select({})

    def test_continuity_separates_chunk_boundary_and_gripper(self):
        actions = [[0,0,0,0,0,0,-1], [1,0,0,0,0,0,-1],
                   [4,0,0,0,2,0,1], [5,0,0,0,2,0,1]]
        result = pilot.continuity(actions, 2)
        self.assertEqual(result['boundary_n'], 1)
        self.assertEqual(result['boundary_translation_delta_mean'], 3)
        self.assertEqual(result['within_translation_delta_mean'], 1)
        self.assertEqual(result['gripper_sign_switches'], 1)
        self.assertIsNone(pilot.continuity(actions, 1)['within_translation_delta_mean'])

    def test_old_process_detection_and_idle_gpu_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'123').mkdir()
            (root/'123/cmdline').write_bytes(b'python\0/remote/corrector_guardian.py\0run\0')
            self.assertEqual(pilot.old_processes(root)[0]['pid'], 123)
        with patch.object(pilot, 'old_processes', return_value=[{'pid': 123}]):
            with self.assertRaisesRegex(RuntimeError, '旧/其他实验'):
                pilot.preflight()
        with patch.object(pilot, 'old_processes', return_value=[]), \
             patch.object(pilot.subprocess, 'check_output', side_effect=[
                 'NVIDIA GeForce RTX 4090, 2400, 24564, 43', '123, python, 2300']):
            with self.assertRaisesRegex(RuntimeError, 'GPU非空闲'):
                pilot.preflight()

    def test_training_gate_requires_all_fifteen_completed(self):
        with patch.object(pilot, 'read', return_value={'status': 'completed', 'epochs_completed': 50}):
            self.assertEqual(len(pilot.prior_training(Path('.'))), 15)
        with patch.object(pilot, 'read', return_value={'status': 'running'}):
            with self.assertRaisesRegex(RuntimeError, '旧训练未完成'):
                pilot.prior_training(Path('.'))

    def test_summary_rejects_unpaired_initial_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'trials').mkdir()
            for n, h in enumerate([50, 20]):
                result = {k: 1. for k in ['env_steps', 'chunk_calls', 'sample_actions_calls', 'denoise_forward_calls',
                    'policy_s', 'rollout_wall_s', 'peak_torch_allocated_gib', 'peak_torch_reserved_gib']}
                result.update(success=True, continuity={}, initial_state_sha256='same',
                              first_input_sha256=str(n), first_noise_sha256='same')
                for key in ['all_translation_delta_mean', 'boundary_translation_delta_mean',
                            'all_rotation_delta_mean', 'boundary_rotation_delta_mean', 'gripper_sign_switches']:
                    result['continuity'][key] = 0.
                (root/'trials'/f'{n}.json').write_text(json.dumps({'status': 'completed',
                    'case': {'method': f'H{h}', 'condition': 'task1', 'seed': 1000}, 'result': result, 'wall_s': 1.}))
            with self.assertRaisesRegex(RuntimeError, '不一致'):
                pilot.pilot_summary(root)


if __name__ == '__main__':
    unittest.main()

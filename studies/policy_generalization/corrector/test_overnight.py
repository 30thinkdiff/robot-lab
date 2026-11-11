"""真实轨迹诊断的时钟/角色/早期前缀和有限夜间统计。"""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_overnight_metrics import model_inputs,prediction_metrics,TrajectoryDiagnostics
from analyse_corrector_overnight import auc,correlation,analyse
import corrector_overnight_queue as queue
from corrector_overnight_eval import initial_state_count


class Toy(torch.nn.Module):
    uses_history=True
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
    def forward(self,z,a):return torch.full_like(z[:,-1],2.)+self.weight


class OvernightTests(unittest.TestCase):
    def test_initial_states_reuse_previously_working_lerobot_getter(self):
        from types import ModuleType
        from unittest.mock import Mock
        legacy=Mock(side_effect=RuntimeError('不能调用旧默认torch.load入口'))
        suite=SimpleNamespace(get_task_init_states=legacy)
        module=ModuleType('lerobot.envs.libero');module.get_task_init_states=Mock(return_value=list(range(50)))
        with patch.dict(sys.modules,{'lerobot.envs.libero':module}):self.assertEqual(initial_state_count(suite,9),50)
        legacy.assert_not_called();module.get_task_init_states.assert_called_once_with(suite,9,is_libero_plus=False)

    def test_no_future_visual_and_control20_data10_alignment(self):
        frames=torch.arange(40.).reshape(40,1,1);actions=torch.arange(40.)[:,None].expand(-1,7)
        v,a,target=model_inputs(frames,actions,[4,6],3,10)
        self.assertEqual(v[:, :,0,0].tolist(),[[0,2,4],[2,4,6]])
        self.assertEqual(a[:,:,0].tolist(),[list(range(4,24,2)),list(range(6,26,2))])
        self.assertEqual(target[:,0,0].tolist(),[20,20])

    def test_metrics_and_zero_energy_not_faked(self):
        p=torch.ones(2,2,3);t=torch.stack([torch.zeros(2,3),torch.ones(2,3)*2])
        m=prediction_metrics(p,t)
        self.assertEqual(m['mse'].tolist(),[1,1]);self.assertEqual(m['target_energy'].tolist(),[0,4])
        self.assertAlmostEqual(float(m['cosine_loss'][1]),0,places=6)

    def test_actual_actions_and_plan_changes_are_separate(self):
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d=TrajectoryDiagnostics({'h3':Toy()},{'h3':{'h_window':3,'action_steps':10}},Path(tmp))
                for t in range(70):
                    d.observe(torch.full((1,128,960),t*.1))
                    d.submit(torch.full((1,7),float(t)),torch.full((1,7),float(t)),
                             [torch.full((1,7),float(t+i)) for i in range(1,50)],t//50)
                rows=d.evaluate(audit_sample=True)
                executed=next(r for r in rows if r['condition_role']=='executed' and r['scope']=='full')
                self.assertLess(executed['mse'],1e-10);self.assertEqual(executed['windows'],23)
                self.assertTrue(all(r['early_prefix_complete'] for r in rows))
                self.assertIn('planned_changed',{r['condition_role'] for r in rows})
                with gzip.open(Path(tmp)/'prediction_windows.jsonl.gz','rt') as file:windows=[json.loads(l) for l in file]
                changed=[r for r in windows if r['role']=='planned' and not r['same_plan']]
                self.assertTrue(changed);self.assertTrue(all(r['condition_max_abs_difference']==0 for r in changed))
                self.assertFalse(any(r['condition_honored'] for r in changed))
                self.assertEqual(min(r['due'] for r in windows),24)
                self.assertTrue((Path(tmp)/'audit_h3.npz').is_file())
                with gzip.open(Path(tmp)/'action_conditions.json.gz','rt') as file:commands=json.load(file)
                self.assertEqual(len(commands['actual_normalized_commands']),70)
                with self.assertRaises(RuntimeError):d.submit(torch.zeros(1,7),torch.zeros(1,7),[],1)
        finally:torch.set_num_threads(threads)

    def test_auc_ties_and_degenerate_outcomes(self):
        self.assertEqual(auc([0,1],[0,1]),1);self.assertEqual(auc([1,1],[0,1]),.5)
        self.assertIsNone(auc([1,2],[1,1]));self.assertIsNone(correlation([1,1,1],[0,1,0]))
        self.assertAlmostEqual(correlation([0,1,0,1],[0,1,0,1]),1)

    def test_pending_records_excluded_and_early_prefix_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'trials').mkdir()
            for i,success in enumerate([True,False]):
                result={k:1 for k in ['env_steps','chunk_calls','policy_s','monitor_s','rollout_wall_s','shared_capture_s',
                    'diagnostic_seconds','event_replans','guidance_calls','peak_torch_allocated_gib','eligible_steps','scored_steps']}
                result.update(success=success,prediction_summary=[{'model':'m','condition_role':'executed','scope':'early',
                    'early_prefix_complete':bool(i),'mse':i+1,'cosine_loss':.2}])
                r={'case':{'id':str(i),'condition':'libero_object/task_1/init_10','seed':1004+i,'method':'H50'},'result':result,'status':'completed'}
                (root/'trials'/f'{i}.json').write_text(json.dumps(r))
            (root/'trials/pending.json').write_text(json.dumps({'status':'running'}))
            summary=analyse(root);self.assertEqual(summary['completed'],2)
            import csv
            with (root/'analysis/prediction_success_relationship.csv').open() as file:rows=list(csv.DictReader(file))
            self.assertTrue(all(int(r['episodes'])==1 and int(r['excluded_incomplete_early'])==1 for r in rows))

    def test_supervisor_retries_only_twice_then_preserves_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=SimpleNamespace(pid=999,returncode=137,poll=lambda:137)
            args=SimpleNamespace(deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
            with patch.object(queue,'ROOT',root),patch.object(queue.resource,'setrlimit'),patch.object(queue.signal,'signal'), \
                 patch.object(queue.subprocess,'Popen',return_value=child) as launch,patch.object(queue.time,'sleep'):
                with self.assertRaises(RuntimeError):queue.run(args)
            self.assertEqual(launch.call_count,3)
            state=json.loads((root/'queue_status.json').read_text())
            self.assertEqual(state['retries'],2);self.assertEqual(state['status'],'error')
            self.assertTrue((root/'results/STOP').exists())

    def test_resource_pause_is_at_boundary_and_records_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results').mkdir()
            (root/'results/pipeline_status.json').write_text(json.dumps({'status':'paused'}))
            polls=iter([None,0]);child=SimpleNamespace(pid=999,returncode=0,poll=lambda:next(polls,0))
            args=SimpleNamespace(deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
            with patch.object(queue,'ROOT',root),patch.object(queue.resource,'setrlimit'),patch.object(queue.signal,'signal'), \
                 patch.object(queue.subprocess,'Popen',return_value=child),patch.object(queue.time,'sleep'), \
                 patch.object(queue,'resources',return_value={'system_free_gib':4,'data_free_gib':8,'available_ram_gib':60}):
                queue.run(args)
            self.assertTrue((root/'results/STOP').exists())
            state=json.loads((root/'queue_status.json').read_text())
            self.assertEqual(state['status'],'paused');self.assertTrue(state['resource_pause_requested'])


if __name__=='__main__':unittest.main()

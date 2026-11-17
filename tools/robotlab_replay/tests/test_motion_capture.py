import json
from pathlib import Path
import sys
import tempfile
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from collections import deque
import numpy as np

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from motion_capture import MotionCapture,ActionDecoder,rotation_increment,nominal_trajectory,pose_error
from analyse_motion_archive import analyze
from motion_pilot import selection, frozen_audit


class FakeController:
    name='OSC_POSE';use_delta=True;control_dim=6
    input_min=-1.;input_max=1.;output_min=np.array([-.05]*3+[-.5]*3);output_max=-output_min
    kp=np.ones(6)*150;kd=np.ones(6)*24.5
    def __init__(self,data):self.data=data;self.goal_pos=np.zeros(3);self.goal_ori=np.eye(3)
    def set_goal(self,action):
        self.goal_pos=self.data.site_xpos[0]+np.asarray(action[:3])*.05
        self.goal_ori=rotation_increment(np.asarray(action[3:])*.5)@self.data.site_xmat[0].reshape(3,3)
        return 'same-goal-return'
    def run_controller(self):return np.r_[150*(self.goal_pos-self.data.site_xpos[0]),np.zeros(4)]


class FakeRobot:
    eef_site_id=0;_ref_joint_pos_indexes=np.arange(7);_ref_joint_vel_indexes=np.arange(7)
    _ref_gripper_joint_pos_indexes=np.array([7,8]);_ref_gripper_joint_vel_indexes=np.array([7,8])
    _ref_joint_actuator_indexes=np.arange(7)
    def __init__(self,data):self.controller=FakeController(data);self.data=data;self.torques=np.zeros(7)
    def control(self,action,policy_step=False):
        if policy_step:self.controller.set_goal(action[:6])
        self.torques=np.clip(self.controller.run_controller(),-.01,.01)
        self.data.ctrl[:7]=self.torques
        return self.torques


class FakeRaw:
    def __init__(self):
        self.data=SimpleNamespace(time=0.,qpos=np.zeros(9),qvel=np.zeros(9),site_xpos=np.zeros((1,3)),
            site_xmat=np.eye(3).reshape(1,9),ctrl=np.zeros(7),ncon=0,actuator_force=np.zeros(7),qfrc_actuator=np.zeros(9))
        self.sim=SimpleNamespace(data=self.data);self.robots=[FakeRobot(self.data)];self.calls=0
    def step(self,action):
        self.calls+=1
        for j in range(2):
            self.robots[0].control(action,policy_step=j==0)
            self.data.qvel[:7]+=self.data.ctrl*.025;self.data.qpos+=self.data.qvel*.025
            self.data.site_xpos[0]=self.data.qpos[:3];self.data.time+=.025
            self.data.actuator_force[:]=self.data.ctrl;self.data.qfrc_actuator[:7]=self.data.ctrl
        self.last_return=(self.data.qpos.copy(),0.,False,{'test':True})
        return self.last_return


class FakeEnv:
    def __init__(self):self._env=None
    def reset(self):self._env=FakeRaw();return 'original-reset-object'


class FakePolicy:
    def __init__(self):self.queue=deque();self.calls=0;self.actions=[]
    def _get_action_chunk(self,batch):
        self.calls+=1
        self.last_chunk=(np.arange(350,dtype=np.float32).reshape(1,50,7)+self.calls)/1000
        return self.last_chunk
    def select_action(self,batch):
        if not self.queue:self.queue.extend(self._get_action_chunk(batch)[0,:2])
        self.last_selected=self.queue.popleft()[None];return self.last_selected


def rollout(env,policy,n=5):
    env.reset();returned=[]
    for _ in range(n):
        action=policy.select_action({});value=env._env.step(action[0])
        assert value is env._env.last_return
        returned.append(value[0].copy())
    return returned


class MotionTests(unittest.TestCase):
    def test_hf_statistics_symlink_is_supported_but_arbitrary_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'hub/models--test--policy/snapshots/revision';folder.mkdir(parents=True)
            blob=root/'hub/blobs/ab/stats';blob.parent.mkdir(parents=True)
            values=np.r_[np.zeros(7,dtype='<f4'),np.ones(7,dtype='<f4')].tobytes()
            header=json.dumps({'action.mean':{'dtype':'F32','shape':[7],'data_offsets':[0,28]},
                'action.std':{'dtype':'F32','shape':[7],'data_offsets':[28,56]}}).encode()
            blob.write_bytes(struct.pack('<Q',len(header))+header+values)
            config={'steps':[{'registry_name':'unnormalizer_processor','config':{'norm_map':{'ACTION':'MEAN_STD'},
                'features':{'action':{'shape':[7]}}},'state_file':'stats.safetensors'},
                {'registry_name':'device_processor'}]}
            (folder/'policy_postprocessor.json').write_text(json.dumps(config));link=folder/'stats.safetensors';link.symlink_to(blob)
            np.testing.assert_array_equal(ActionDecoder.from_snapshot(folder).decode(np.ones(7)),np.ones(7))
            link.unlink();outside=root/'outside.safetensors';outside.write_bytes(blob.read_bytes());link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError,'escapes'):ActionDecoder.from_snapshot(folder)

    def test_instrumented_fake_control_preserves_outputs_steps_forward_count_and_restores(self):
        reference_env=FakeEnv();reference_policy=FakePolicy();reference=rollout(reference_env,reference_policy)
        env=FakeEnv();policy=FakePolicy();old_reset=env.reset;old_chunk=policy._get_action_chunk;old_select=policy.select_action
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/'motion'
            with MotionCapture(env,policy,folder,ActionDecoder(np.zeros(7),np.ones(7)),method='H20',provenance={'test':'fake'},substep_stride=1):
                actual=rollout(env,policy)
            for a,b in zip(actual,reference):np.testing.assert_array_equal(a,b)
            self.assertEqual(policy.calls,reference_policy.calls);self.assertEqual(env._env.calls,5)
            self.assertEqual(env.reset,old_reset);self.assertEqual(policy._get_action_chunk,old_chunk);self.assertEqual(policy.select_action,old_select)
            rows=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['slot'] for r in rows],[0,1,0,1,0])
            self.assertTrue(all(len(r['goals'])==1 and r['controller_updates']==2 and len(r['substeps'])==2 for r in rows))
            self.assertTrue(all(r['clipped_updates']==2 for r in rows))
            self.assertAlmostEqual(rows[-1]['after']['sim_time_s'],.25)
            self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'completed')
            self.assertIn('not a physical prediction',json.loads((folder/'protocol.json').read_text())['nominal_semantics'])

    def test_exception_restores_hooks_and_preserves_partial_output(self):
        env=FakeEnv();policy=FakePolicy();old=env.reset
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'motion'
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                with MotionCapture(env,policy,path,ActionDecoder(np.zeros(7),np.ones(7)),method='j_ogg',provenance={}):
                    rollout(env,policy,1);raise RuntimeError('deliberate')
            self.assertEqual(env.reset,old)
            self.assertEqual(json.loads((path/'status.json').read_text())['status'],'interrupted_or_error')
            self.assertEqual(len((path/'steps.jsonl').read_text().splitlines()),1)
            with self.assertRaises(FileExistsError):
                with MotionCapture(env,policy,path,ActionDecoder(np.zeros(7),np.ones(7)),method='H50',provenance={}):pass

    def test_wrong_decoder_stops_before_control_and_second_reset_is_rejected(self):
        env=FakeEnv();policy=FakePolicy()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'decoder differs'):
                with MotionCapture(env,policy,Path(t)/'wrong',ActionDecoder(np.ones(7),np.ones(7)),method='H50',provenance={}):
                    env.reset();a=policy.select_action({});env._env.step(a[0])
            self.assertEqual(env._env.calls,0)
            with self.assertRaisesRegex(RuntimeError,'one episode'):
                with MotionCapture(env,FakePolicy(),Path(t)/'reset',ActionDecoder(np.zeros(7),np.ones(7)),method='H50',provenance={}):
                    env.reset();env.reset()

    def test_budget_and_unsupported_controller_stop_without_added_forward(self):
        env=FakeEnv();policy=FakePolicy()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(RuntimeError,'byte budget'):
                with MotionCapture(env,policy,Path(t)/'budget',ActionDecoder(np.zeros(7),np.ones(7)),method='H50',provenance={},max_bytes=2048):
                    rollout(env,policy)
            self.assertEqual(env._env.calls,0)

    def test_rotation_composition_clipping_and_tracking_vs_accumulation(self):
        config={'type':'OSC_POSE','control_delta':True,'input_min':-1,'input_max':1,'output_min':[-.05]*3+[-.5]*3,'output_max':[.05]*3+[.5]*3}
        command=np.array([.2,0,0,0,0,0,-1.]);nominal=nominal_trajectory([0,0,0],np.eye(3),[command]*10,config)
        self.assertAlmostEqual(nominal[-1]['position_m'][0],.1)
        actual={'eef_position_m':[.06,0,0],'eef_orientation_matrix':np.eye(3).tolist()}
        self.assertAlmostEqual(pose_error(nominal[-1],actual)['position_m'],.04)
        r=rotation_increment([0,np.pi/2,0])@rotation_increment([np.pi/2,0,0])
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
        self.assertFalse(np.allclose(r,rotation_increment([np.pi/2,np.pi/2,0])))
        self.assertAlmostEqual(nominal_trajectory([0,0,0],np.eye(3),[[100,0,0,0,0,0,0]],config)[0]['position_m'][0],.05)

    def test_pilot_selection_is_bounded_and_never_silently_adds_methods_or_seeds(self):
        self.assertEqual(selection(['libero_object/task_9/init_2:1000'],['H50','j_ogg']),
            [('libero_object/task_9/init_2',1000,'H50'),('libero_object/task_9/init_2',1000,'j_ogg')])
        for cases,methods in [(['plus/default:1000'],['H50']),(['libero_object/task_9/init_2:1000'],['H50','H50']),
            (['libero_object/task_9/init_2:1000']*2,['H50'])]:
            with self.assertRaises(ValueError):selection(cases,methods)

    def test_unknown_controller_refused_and_controller_hooks_restored_after_error(self):
        env=FakeEnv();old_reset=env.reset
        def absolute_reset():
            value=old_reset();env._env.robots[0].controller.use_delta=False;return value
        env.reset=absolute_reset
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'relative'):
                with MotionCapture(env,FakePolicy(),Path(t)/'absolute',ActionDecoder(np.zeros(7),np.ones(7)),method='H50',provenance={}):env.reset()
            self.assertEqual(env.reset,absolute_reset)
            self.assertEqual(env._env.calls,0)


ARCHIVES=list((HERE/'.local/motion_archive').glob('historical_*/manifest.json'))
@unittest.skipUnless(ARCHIVES,'requires fetched real action blocks, no simulated evidence')
class RealMotionArchiveTests(unittest.TestCase):
    def test_real_six_cases_decode_queue_and_commands_match_without_inventing_states(self):
        report=analyze(ARCHIVES[0].parent)
        self.assertEqual(len(report['cases']),6);self.assertEqual(report['total_chunks'],36)
        self.assertTrue(all(r['selected_normalized_max_abs_error']==0 for r in report['cases']))
        self.assertTrue(all(r['trajectory_gap'] is None and not r['actual_pose_series_available'] for r in report['cases']))
        self.assertTrue(all(r['consumed_slots']==r['steps'] for r in report['cases']))


if __name__=='__main__':unittest.main()

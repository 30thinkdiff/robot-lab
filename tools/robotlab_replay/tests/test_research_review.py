import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from corrector_multilayer import ReadOnlyVisionTap,unwrap_tensor,write_frame,aligned_pairs
from corrector_casebank_review import summarize_instances,runtime_summary,put_group
from collections import defaultdict


class ReviewTests(unittest.TestCase):
    def test_state_seed_and_missing_outcome_are_not_conflated(self):
        rows=[{'condition':'task1/init0','seed':s,'cohort':'x','replay_complete':False,'trace':True,
            'baseline':False,'commands':False,'numeric_ogg':False,'signature':None,'id':str(s)} for s in [1000,1001]]
        result=summarize_instances([[0],[1]],rows)
        self.assertEqual((result['cases'],result['physical_states'],result['evaluation_seed_value_count']),(2,1,2))
        self.assertEqual(result['local_complete_cases'],0)

    def test_duplicates_are_rejected_instead_of_picking_favorable_outcome(self):
        g=defaultdict(dict);r={'cohort':'x','phase':'formal','condition':'task0/init0','seed':1,'method':'ogg'}
        put_group(g,r,0)
        with self.assertRaises(ValueError):put_group(g,r,1)

    def test_runtime_uses_recent_real_costs_and_current_written_bytes(self):
        rows=[{'started_utc':f'{i:03d}','wall_s':i+1,'bytes':1024*(i+1),
               'case':{'condition':'task0/init0','seed':1,'method':'H50','config':{'phase':'discovery'}}} for i in range(40)]
        r=runtime_summary({'last_trials':rows,'root_bytes':1000,'gpu':'','execution':{'planned':50}})
        self.assertEqual(r['recent_n'],30);self.assertEqual(r['recent_mean_seconds'],25.5)
        self.assertEqual(r['remaining_trials'],10);self.assertAlmostEqual(r['estimated_remaining_hours'],255/3600)


class FeatureSchemaTests(unittest.TestCase):
    def record(self,episode,frame):
        return {'episode':episode,'frame_index':frame,'timestamp':frame/10,'data_fps':10.,'camera_keys':['a','b'],
            'stored_shapes':{'block_02':[2,64,768]},'dtype':'float16','pooling':{'grid':8},'feature_sites':'block pre norm'}

    def test_temporal_pairs_respect_episode_and_one_second_not_twenty_data_frames(self):
        records=[self.record(1,0),self.record(2,10),self.record(1,10),self.record(1,20)]
        self.assertEqual(aligned_pairs(records),[(0,2),(2,3)])
        records[2]['timestamp']=.5
        with self.assertRaises(ValueError):aligned_pairs(records)

    def test_protocol_and_duplicate_frames_cannot_silently_pair(self):
        records=[self.record(1,0),self.record(1,10)];records[1]['camera_keys'].reverse()
        with self.assertRaises(ValueError):aligned_pairs(records)
        with self.assertRaises(ValueError):aligned_pairs([self.record(1,0),self.record(1,0)])

    def test_layer_indices_and_shapes_are_explicit(self):
        with self.assertRaises(ValueError):ReadOnlyVisionTap(None,[1,1])
        with self.assertRaises(ValueError):ReadOnlyVisionTap(None,[-1])
        with self.assertRaises(ValueError):unwrap_tensor(np.zeros((4,3)))
        a=np.ones((1,4,3));self.assertIs(unwrap_tensor((a,None)),a)


@unittest.skipUnless(importlib.util.find_spec('torch'),'本机无Torch；在远端已有CPU Torch环境运行这些接口检查')
class RealTorchHookTests(unittest.TestCase):
    def make_wrapper(self):
        import torch
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Module();self.encoder.layers=torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(3)])
                self.post_layernorm=torch.nn.LayerNorm(4)
            def forward(self,x):
                for layer in self.encoder.layers:x=layer(x)
                return self.post_layernorm(x)
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__();self.vlm=torch.nn.Module();self.vlm.vision_model=Vision();self.vlm.connector=torch.nn.Linear(4,3)
            def get_vlm_model(self):return self.vlm
            def embed_image(self,x):return self.vlm.connector(self.vlm.vision_model(x))
        return Wrapper().eval().requires_grad_(False)

    def test_hook_preserves_original_torch_output_and_is_removed(self):
        import torch
        w=self.make_wrapper();x=torch.arange(64).float().reshape(1,16,4);original=w.embed_image(x)
        with ReadOnlyVisionTap(w,[0,1,2]) as tap:
            with tap.capture('a') as captured:result=w.embed_image(x)
            self.assertTrue(torch.equal(result,original));self.assertTrue(torch.equal(captured['connector'],result))
            self.assertEqual(set(captured),{'block_00','block_01','block_02','vision_postnorm','connector'})
            self.assertNotEqual(captured['block_00'].data_ptr(),x.data_ptr())
        self.assertTrue(all(not m._forward_hooks for m in w.modules()))
        self.assertTrue(all(p.grad is None for p in w.parameters()))

    def test_exception_memory_guard_and_training_rejection_remove_hooks(self):
        import torch
        w=self.make_wrapper()
        with self.assertRaises(RuntimeError):
            with ReadOnlyVisionTap(w,[0],max_view_bytes=1) as tap:
                with tap.capture('a'):w.embed_image(torch.zeros(1,16,4))
        self.assertTrue(all(not m._forward_hooks for m in w.modules()))
        with self.assertRaises(ValueError):
            with ReadOnlyVisionTap(w.train(),[0]):pass

    def test_real_cpu_tensors_cache_without_pickle_and_pairs_have_explicit_views(self):
        import torch
        w=self.make_wrapper();views={}
        with ReadOnlyVisionTap(w,[0,2]) as tap:
            for camera in ['a','b']:
                with tap.capture(camera) as captured:w.embed_image(torch.ones(1,16,4))
                views[camera]=captured
        with tempfile.TemporaryDirectory() as temp:
            row=write_frame(temp,1,0,0.,views,['a','b'],grid=2)
            with np.load(Path(temp)/row['file'],allow_pickle=False) as a:
                self.assertEqual(a['block_00'].shape,(2,4,4));self.assertEqual(a['block_00'].dtype,np.float16)
            with self.assertRaises(FileExistsError):write_frame(temp,1,0,0.,views,['a','b'],grid=2)
            with self.assertRaises(ValueError):write_frame(temp,1,1,.9,views,['a','b'],grid=2)


if __name__=='__main__':unittest.main()

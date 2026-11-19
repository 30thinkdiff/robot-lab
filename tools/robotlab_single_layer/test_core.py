"""CPU checks. Synthetic fixtures are explicitly not real GPU/robot experiments."""
import copy
import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from . import baseline, data
from .manifest import build_manifest, validate_plan
from .model import LayerShapeAdapter
from .online import corrector_class
from .replay_contract import case_config, recording_requirements
from .training import arguments, require_gates
from .extraction_plan import episode_commands
from .metadata import check_connector_frame
from .extraction import boundary, next_attempt, validate_rows, publish, exclusive_worker, configure_runtime
from .progress import estimate,status as read_status
from .job import worker_environment

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('ROBOTLAB_SLC_SOURCE_ROOT', str(ROOT / 'outputs/source_audit/vla-corrector')))
BASELINE = Path(os.environ.get('ROBOTLAB_SLC_BASELINE_DIR', str(ROOT / 'outputs/corrector_j36m_repository/night-code-1e80242789')))
TAP = ROOT / 'scripts/corrector_multilayer.py'
torch.set_num_threads(2)


def reference():
    return baseline.load_reference(BASELINE, SOURCE)


def plan():
    old = json.loads(Path(__file__).with_name('historical_mlp.json').read_text())
    return dict(revision=baseline.DATA_REVISION, policy_revision=baseline.POLICY_REVISION,
        policy_weight_sha256=baseline.POLICY_WEIGHT, fps=10., k_step=10, anchor_history=3,
        raw_shape=[2, 1024, 768], raw_dtype='float16', pooling='none', layer_indices_zero_based=[2, 8],
        records=[dict(r, raw_path='/not-read/raw/episode_' + str(r['episode']),
                      connector_path='/not-read/deep/episode_' + str(r['episode'])) for r in old['records']])


def fixture(root):
    """Sparse NPY test files with production shape, synthetic values only."""
    ep, n = 813, 13
    raw, deep = root / 'raw', root / 'deep'
    raw.mkdir(); deep.mkdir()
    for layer in baseline.SHAPES:
        a = np.lib.format.open_memmap(raw / (layer + '.fp16.npy'), mode='w+', dtype='<f2', shape=(n, 2, 1024, 768))
        a[2] = 1; a[12] = 4
        a.flush(); a._mmap.close()
    for name, value in dict(actions=np.arange(n * 7, dtype=np.float16).reshape(n, 7),
        episode_index=np.full(n, ep, dtype=np.int64), frame_index=np.arange(n),
        timestamp=np.arange(n, dtype=np.float32) / 10.).items():
        np.save(deep / (name + '.npy'), value)
    meta = dict(format='siglip_frame_cache_v2', split='train', num_frames=n, fps=10.,
        policy_revision=baseline.POLICY_REVISION, camera_keys=baseline.CAMERAS,
        image_keys=baseline.IMAGES, action_representation='checkpoint_processor_normalized',
        raw_provenance=dict(sources=[dict(sha256='f' * 64)]))
    (deep / 'metadata.json').write_text(json.dumps(meta))
    (deep / 'COMPLETE.json').write_text(json.dumps(dict(files={p.name: baseline.digest(p) for p in deep.iterdir()})))
    marker = dict(format='multilayer_raw_fp16_v1', episode=ep, num_frames=n, fps=10.,
        pooling='none', dtype='float16', layer_indices_zero_based=[2, 8], camera_keys=baseline.CAMERAS,
        source_image_keys=baseline.IMAGES, shape_per_layer=[n, 2, 1024, 768],
        policy_revision=baseline.POLICY_REVISION, policy_weight_sha256=baseline.POLICY_WEIGHT,
        k_data_frames=10, deep_cache_complete_sha256=baseline.digest(deep / 'COMPLETE.json'),
        frame_index_and_timestamp_source='deep_cache_path',
        action_source='deep_cache_path/actions.npy; checkpoint_processor_normalized',
        source_parquet_sha256='f' * 64,
        files={p.name: dict(sha256=baseline.digest(p), size_bytes=p.stat().st_size) for p in raw.iterdir()})
    (raw / 'COMPLETE.json').write_text(json.dumps(marker))
    return dict(episode=ep, frames=n, split='train', raw_path=str(raw), connector_path=str(deep))


class ProtocolTests(unittest.TestCase):
    def test_full_historical_split_and_pairs(self):
        ref = reference(); p = plan(); validate_plan(p)
        groups = {split: ConcatDataset([data.EpisodeDataset(r, 'block_02') for r in p['records'] if r['split'] == split])
                  for split in ('train', 'validation')}
        _, split, _ = ref['train'].select_split(groups, 'episode', .1, 42)
        old = json.loads(Path(__file__).with_name('historical_mlp.json').read_text())
        self.assertEqual(split['samples'], {'train': 49447, 'validation': 12089})
        self.assertEqual(split['index_sha256'], old['split']['index_sha256'])
        self.assertEqual(split['shared_episodes'], [])

    def test_reject_duplicates_split_swaps_and_resolution(self):
        for kind in ('duplicate', 'split', 'shape', 'timebase'):
            p = plan()
            if kind == 'duplicate': p['records'].append(p['records'][0])
            elif kind == 'split': p['records'][0]['split'] = 'validation'
            elif kind == 'shape': p['raw_shape'] = [2, 64, 768]
            else: p['fps'] = 20.
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_plan(p)

    def test_training_defaults_and_gate(self):
        original = arguments('m.json', SOURCE, '/tmp/not-created')
        self.assertEqual((original.batch_size, original.accumulate), (32, 16))
        a = arguments('m.json', SOURCE, '/tmp/not-created', microbatch=2)
        self.assertEqual((a.batch_size * a.accumulate, a.epochs, a.patience), (512, 150, 5))
        self.assertEqual((a.loss_type, a.beta, a.predictor_precision), ('both', 1., 'amp-bfloat16'))
        with self.assertRaises(ValueError): arguments('m', SOURCE, '/tmp/n', microbatch=3)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'gates.json'; p.write_text('{}')
            with self.assertRaises(ValueError): require_gates(p, 'block_02', reference(), Path(__file__))

    def test_replay_metadata_explicit_and_no_fabricated_motion(self):
        self.assertTrue(case_config('block_02', 'ogg', 'a' * 64)['model_label'].startswith('S2'))
        self.assertFalse(recording_requirements()['replay_import_completed'])
        with self.assertRaises(ValueError): case_config('connector', 'ogg', 'a' * 64)

    def test_extraction_only_plans_and_pins_successful_exporter(self):
        exporter = Path(os.environ.get('ROBOTLAB_SLC_EXPORTER', str(ROOT / 'outputs/corrector_tracking_fix_20261114/dev/j36m-35ecdd7488bf/scripts/corrector_multilayer_pilot.py')))
        commands = episode_commands(plan(), exporter=exporter, runtime='/not-run/runtime',
            source_root=SOURCE, connector_root='/not-run/connector', data_root='/not-run/data', old_manifest='/not-run/manifest')
        self.assertEqual(len(commands), 454)
        self.assertTrue(all(r['action'] == 'planned_not_started' and '--save-fp16' in r['command'] for r in commands))
        with self.assertRaises(ValueError):
            episode_commands(plan(), exporter=__file__, runtime='unused', source_root=SOURCE,
                connector_root='unused', data_root='unused', old_manifest='unused')


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.spec = fixture(Path(self.tmp.name))
        self.expected_files = json.loads((Path(self.spec['connector_path']) / 'COMPLETE.json').read_text())['files']
        self.history_patch = patch.object(data, 'historical_record', return_value={'files': self.expected_files})
        self.history_patch.start()

    def tearDown(self):
        self.history_patch.stop(); data.POOL.close(); self.tmp.cleanup()

    def test_full_resolution_sha_and_target(self):
        data.verify_episode(self.spec, 'block_02')
        ds = data.EpisodeDataset(self.spec, 'block_02'); item = ds[0]
        self.assertEqual(ds.pairs, [(2, 12)])
        self.assertEqual(item['z_t'].shape, (2048, 768))
        self.assertTrue(torch.all(item['z_t'] == 1))
        self.assertTrue(torch.all(item['delta_z'] == 3))
        self.assertTrue(torch.equal(item['a_t'], torch.arange(14, 21).float()))
        for bad in (-2, 1):
            with self.assertRaises(IndexError): ds[bad]

    def test_future_visual_and_actions_do_not_enter_inputs(self):
        ds = data.EpisodeDataset(self.spec, 'block_08'); before = ds[0]
        raw = np.load(Path(self.spec['raw_path']) / 'block_08.fp16.npy', mmap_mode='r+')
        raw[12] = 8; raw.flush(); raw._mmap.close()
        a = np.load(Path(self.spec['connector_path']) / 'actions.npy', mmap_mode='r+')
        a[3:] = -100; a.flush(); a._mmap.close()
        after = ds[0]
        self.assertTrue(torch.equal(before['z_t'], after['z_t']))
        self.assertTrue(torch.equal(before['a_t'], after['a_t']))
        self.assertFalse(torch.equal(before['delta_z'], after['delta_z']))
        with self.assertRaises(ValueError): data.verify_episode(self.spec, 'block_08')

    def test_timestamp_offset_rejected_even_with_valid_new_hash(self):
        deep, raw = Path(self.spec['connector_path']), Path(self.spec['raw_path'])
        np.save(deep / 'timestamp.npy', np.arange(13) / 10. + .05)
        marker = json.loads((deep / 'COMPLETE.json').read_text())
        marker['files']['timestamp.npy'] = baseline.digest(deep / 'timestamp.npy')
        self.expected_files['timestamp.npy'] = marker['files']['timestamp.npy']
        (deep / 'COMPLETE.json').write_text(json.dumps(marker))
        rm = json.loads((raw / 'COMPLETE.json').read_text()); rm['deep_cache_complete_sha256'] = baseline.digest(deep / 'COMPLETE.json')
        (raw / 'COMPLETE.json').write_text(json.dumps(rm))
        with self.assertRaises(ValueError): data.verify_episode(self.spec, 'block_02')

    def test_map_pool_bounded_and_read_only(self):
        pool = data.MapPool(2)
        try:
            for name in ('frame_index.npy', 'timestamp.npy', 'actions.npy'):
                value = pool.get(Path(self.spec['connector_path']) / name)
                self.assertFalse(value.flags.writeable)
                self.assertLessEqual(len(pool.maps), 2)
        finally: pool.close()

    def test_new_completion_marker_cannot_hide_changed_original_actions(self):
        deep, raw = Path(self.spec['connector_path']), Path(self.spec['raw_path'])
        np.save(deep / 'actions.npy', np.zeros((13, 7), dtype=np.float16))
        marker = json.loads((deep / 'COMPLETE.json').read_text())
        marker['files']['actions.npy'] = baseline.digest(deep / 'actions.npy')
        (deep / 'COMPLETE.json').write_text(json.dumps(marker))
        rm = json.loads((raw / 'COMPLETE.json').read_text()); rm['deep_cache_complete_sha256'] = baseline.digest(deep / 'COMPLETE.json')
        (raw / 'COMPLETE.json').write_text(json.dumps(rm))
        with self.assertRaises(ValueError): data.verify_episode(self.spec, 'block_02')

    def test_explicit_metadata_subset_keeps_original_provenance(self):
        deep, raw = Path(self.spec['connector_path']), Path(self.spec['raw_path'])
        q, scales = np.zeros((13,128,960),dtype=np.int8), np.ones((13,128),dtype=np.float16)
        for name, values in [('z_q.npy',q),('z_scale.npy',scales)]:
            np.save(deep/name, values); self.expected_files[name]=baseline.digest(deep/name)
        original = {'files': self.expected_files}
        (deep/'SOURCE_COMPLETE.json').write_text(json.dumps(original))
        (deep/'COMPLETE.json').unlink()
        for name in ('z_q.npy','z_scale.npy'): (deep/name).unlink()
        refs = dict(format='connector_quantized_frame_sha256_v1',episode=813,frames=13,
            source_q_npy_sha256=self.expected_files['z_q.npy'],source_scale_npy_sha256=self.expected_files['z_scale.npy'],
            source_complete_sha256=baseline.digest(deep/'SOURCE_COMPLETE.json'),
            frame_sha256=[hashlib.sha256(q[i].tobytes()+scales[i].tobytes()).hexdigest() for i in range(13)])
        (deep/'connector_frame_hashes.json').write_text(json.dumps(refs))
        marker=dict(format='single_layer_conditioning_metadata_v1',episode=813,frames=13,
            source_complete_sha256=baseline.digest(deep/'SOURCE_COMPLETE.json'),
            files={p.name:baseline.digest(p) for p in deep.iterdir()})
        (deep/'METADATA_COMPLETE.json').write_text(json.dumps(marker))
        rm=json.loads((raw/'COMPLETE.json').read_text());rm['deep_cache_complete_sha256']=baseline.digest(deep/'SOURCE_COMPLETE.json')
        (raw/'COMPLETE.json').write_text(json.dumps(rm))
        with patch('robotlab_single_layer.metadata.historical_record',return_value={'files':self.expected_files}):
            data.verify_episode(self.spec,'block_02')
            self.assertFalse((deep/'COMPLETE.json').exists())
            refs['source_q_npy_sha256']='0'*64
            (deep/'connector_frame_hashes.json').write_text(json.dumps(refs))
            marker['files']['connector_frame_hashes.json']=baseline.digest(deep/'connector_frame_hashes.json')
            (deep/'METADATA_COMPLETE.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):data.verify_episode(self.spec,'block_02')

    def test_frame_hash_checks_exact_original_quantization(self):
        tree=ast.parse((SOURCE/'src/siglip_dynamics/extract.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_quantize_per_token')
        ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<upstream-quantizer>','exec'),ns)
        quantize=ns['_quantize_per_token'];z=np.random.default_rng(42).normal(size=(128,960)).astype(np.float32)
        q,scale=quantize(z);expected=hashlib.sha256(q.tobytes()+scale.tobytes()).hexdigest()
        self.assertEqual(check_connector_frame(quantize,z,{'frame_sha256':[expected]},0),expected)
        z[0,0]+=10
        with self.assertRaises(ValueError):check_connector_frame(quantize,z,{'frame_sha256':[expected]},0)


class NumericalTests(unittest.TestCase):
    def test_full_architecture_meta_and_shape(self):
        ref = reference()
        with torch.device('meta'):
            m = baseline.make_predictor(ref, 'meta')
            y = LayerShapeAdapter(m)(torch.empty(1, 2, 1024, 768), torch.empty(1, 7))
        self.assertEqual(tuple(y.shape), (1, 2, 1024, 768))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 37266176)
        self.assertEqual(len(m.blocks), 4)
        self.assertTrue(all(b.fc1.in_features == b.fc1.out_features == 2048 for b in m.blocks))

    def test_exact_upstream_loss_including_zero_vectors(self):
        ref = reference(); typ = ref['config'].LossType.BOTH
        p, y = torch.randn(2, 4, 768), torch.randn(2, 4, 768)
        total, mse, cos = ref['loss'](p, y, typ, 1.)
        self.assertTrue(torch.equal(total, torch.nn.functional.mse_loss(p, y) + (1 -
            torch.nn.functional.cosine_similarity(p.flatten(1), y.flatten(1), dim=-1).mean())))
        zero = torch.zeros_like(p)
        self.assertEqual(ref['loss'](zero, zero, typ, 1.)[0].item(), 1.)

    def test_weighted_accumulation_matches_single_batch_including_tail(self):
        ref = reference()
        class Tiny(nn.Module):
            def __init__(self, cfg=None): super().__init__(); self.fc = nn.Linear(4, 3)
            def forward(self, z, a): return self.fc(z) + a[:, None, :3]
        torch.manual_seed(5)
        whole, opt1, _ = ref['train'].reference_components(Tiny, None, 'cpu', .0003, .0001, 150)
        micro, opt2, _ = ref['train'].reference_components(Tiny, None, 'cpu', .0003, .0001, 150)
        micro.load_state_dict(whole.state_dict())
        z, a, target = torch.randn(5, 2, 4), torch.randn(5, 7), torch.randn(5, 2, 3)
        typ = ref['config'].LossType.BOTH
        full = ref['loss'](whole(z, a), target, typ, 1.)[0]
        ref['train'].reference_update(whole, opt1, full, True)
        for i, sl in enumerate((slice(0, 3), slice(3, 5))):
            loss = ref['loss'](micro(z[sl], a[sl]), target[sl], typ, 1.)[0]
            ref['train'].reference_update(micro, opt2, loss * len(z[sl]) / 5, i == 1)
        self.assertTrue(all(torch.allclose(p, q, atol=1e-7, rtol=1e-6) for p, q in zip(whole.parameters(), micro.parameters())))

    def test_actual_mlp_backward_ogg_and_checkpoint_reload(self):
        # Actual production widths; small token count is a CPU numeric fixture,
        # not a reduced production architecture or full-resolution GPU claim.
        ref = reference(); torch.manual_seed(8)
        model = baseline.make_predictor(ref)
        z, target = torch.randn(1, 4, 768), torch.randn(1, 4, 768)
        x, velocity, t = torch.randn(1, 50, 32), torch.randn(1, 50, 32), torch.tensor([.7])
        train_loss = ref['loss'](model(z, x[:, 0, :7]), target, ref['config'].LossType.BOTH, 1.)[0]
        train_loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.action_proj.weight.grad.norm()), 0.)
        model.zero_grad(set_to_none=True); model.eval().requires_grad_(False)
        updated, loss = ref['runtime'].guided_velocity(model, z, x, velocity, t, target, 1.)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertGreater(float((updated - velocity).norm()), 0.)
        self.assertTrue(torch.equal(updated[:, 1:], velocity[:, 1:]))
        self.assertTrue(torch.equal(updated[:, 0, 7:], velocity[:, 0, 7:]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.pt'; torch.save(model.state_dict(), path)
            other = baseline.make_predictor(ref).eval().requires_grad_(False)
            other.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            with torch.no_grad(): self.assertTrue(torch.equal(model(z, x[:, 0, :7]), other(z, x[:, 0, :7])))


class FakePredictor(nn.Module):
    def __init__(self):
        super().__init__(); self.cfg = SimpleNamespace(token_dim=768, action_dim=7); self.linear = nn.Linear(7, 768)
    def forward(self, z, a):
        direction = torch.linspace(.3, 1.3, 768)
        return (2 * direction[None, None] + .1 * self.linear(a)[:, None]).expand_as(z)


class FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(rtc_config=None, image_features=baseline.CAMERAS, n_action_steps=50, num_steps=10)
        self._queues = {'action': deque()}; self.calls = 0
        self.model = SimpleNamespace(denoise_step=lambda masks, cache, x, t: x * .2)
    def select_action(self, batch):
        if not self._queues['action']:
            self.calls += 1
            x = torch.linspace(-1., 1., 50 * 32).reshape(1, 50, 32) + .01 * self.calls
            for i in range(10):
                v = self.model.denoise_step(None, None, x, torch.tensor([1. - .1 * i]))
                x = x - .1 * v
            self._queues['action'].extend(x[:, i, :7].clone() for i in range(50))
        return self._queues['action'].popleft()


def fixture_encode(controller, batch):
    t = controller.step
    return (torch.linspace(.3, 1.3, 768) * (.05 * t if t < 45 else -.05 * t)).reshape(1, 1, 768).expand(1, 4, 768)


class ExtractionRecoveryTests(unittest.TestCase):
    def test_fresh_resume_child_gets_package_path_without_shell_setup(self):
        import subprocess,sys
        with patch.dict(os.environ,{},clear=True),tempfile.TemporaryDirectory() as temp:
            env=worker_environment()
            r=subprocess.run([sys.executable,'-c','import robotlab_single_layer.progress'],
                             cwd=temp,env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(env['OMP_NUM_THREADS'],'4')

    def test_original_extractor_receives_explicit_runtime_source(self):
        import sys
        before=list(sys.path)
        try:
            with patch.dict(os.environ,{},clear=True):
                configure_runtime('/task')
                self.assertEqual(os.environ['LEROBOT_SRC'],'/task/runtime/lerobot/src')
                self.assertEqual(sys.path[:2],['/task/runtime/lerobot/src','/task/runtime/vla-corrector-reference/src'])
                self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
        finally:sys.path[:]=before

    def test_eta_uses_actual_completed_episode_wall_time(self):
        first=estimate(1000,0,0,[])
        self.assertIsNone(first['eta_seconds'])
        value=estimate(1000,400,100,[dict(frames=200,wall_seconds=20),dict(frames=200,wall_seconds=30)])
        self.assertEqual(value['remaining_frames'],500)
        self.assertEqual(value['frames_per_second'],8)
        self.assertEqual(value['eta_seconds'],62.5)
        self.assertEqual(estimate(1000,1000,0,[dict(frames=200,wall_seconds=20)])['eta_seconds'],0)

    def test_stale_running_is_not_reported_as_live(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json';path.write_text(json.dumps(dict(status='running',pid=99999999)))
            value=read_status(temp)
            self.assertFalse(value['process_alive'])
            self.assertEqual(value['status'],'process_missing_requires_resume')

    def test_boundary_reserves_complete_episode_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stop=root/'STOP'; payload=1000
            required=payload+5*2**30+64*2**20
            self.assertIsNone(boundary(root,payload,stop,required))
            self.assertEqual(boundary(root,payload,stop,required-1),'insufficient_disk')
            stop.touch()
            self.assertEqual(boundary(root,payload,stop,required),'stop_requested')

    def test_partial_attempt_retained_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=next_attempt(root,813); (first/'partial').write_bytes(b'keep')
            second=next_attempt(root,813)
            self.assertNotEqual(first,second)
            self.assertEqual((first/'partial').read_bytes(),b'keep')
            with exclusive_worker(root):
                with self.assertRaises(BlockingIOError):
                    with exclusive_worker(root):pass

    def test_source_time_and_frame_alignment(self):
        rows=[dict(episode_index=813,frame_index=i,timestamp=i/10) for i in range(13)]
        validate_rows(rows,813,13,np.arange(13)/10)
        for key,value in (('episode_index',814),('frame_index',3),('timestamp',.7)):
            changed=copy.deepcopy(rows);changed[2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_rows(changed,813,13,np.arange(13)/10)

    def test_publish_requires_both_layers_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);attempt=root/'attempt';attempt.mkdir();final=root/'final'
            with patch('robotlab_single_layer.extraction.verify_episode') as verified:
                publish(attempt,final,dict(episode=813))
                self.assertTrue(final.is_dir())
                self.assertFalse(attempt.exists())
                self.assertEqual([v.args[1] for v in verified.call_args_list],['block_02','block_08'])
                with self.assertRaises(FileExistsError):publish(root/'other',final,{})


class OnlineTests(unittest.TestCase):
    def check_mode(self, mode):
        ref = reference(); extended = corrector_class(ref, TAP)
        protocol = dict(h_window=1, anchor_history=3, k_step=10, fps=10., prediction_seconds=1., visual_layer='block_02')
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for variant in ('original', 'off', 'on'):
                torch.manual_seed(12)
                policy, predictor = FakePolicy(), FakePredictor().eval().requires_grad_(False)
                cls = ref['runtime'].OnlineCorrector if variant == 'original' else extended
                Fixture = type('Fixture', (cls,), {'encode': fixture_encode})
                extra = {} if variant == 'original' else dict(layer='block_02', diagnostics=variant == 'on')
                directory = Path(temp) / variant
                controller = Fixture(policy, predictor, protocol, mode, SOURCE, directory, **extra)
                with controller.installed(): actions = [policy.select_action({}).clone() for _ in range(75)]
                logs = [json.loads(l) for l in (directory / 'monitor_steps.jsonl').read_text().splitlines()]
                results.append((torch.stack(actions), controller, logs))
            first = results[0]
            self.assertGreater(sum(bool(r.get('trigger')) for r in first[2]), 0)
            if mode != 'monitor': self.assertGreater(first[1].events, 0)
            if mode == 'ogg': self.assertGreater(first[1].guidance_calls, 0)
            for value in results[1:]:
                self.assertTrue(torch.equal(first[0], value[0]))
                self.assertEqual(first[1].state.dynamic_threshold, value[1].state.dynamic_threshold)
                self.assertEqual(first[1].state.clean_history, value[1].state.clean_history)
                self.assertEqual(first[1].events, value[1].events)
                self.assertEqual(first[1].guidance_calls, value[1].guidance_calls)
                self.assertEqual([p[0] for p in first[1].pending], [p[0] for p in value[1].pending])
                self.assertEqual(first[1].policy.calls, value[1].policy.calls)
                if first[1].target_now is not None:
                    self.assertTrue(torch.equal(first[1].target_now, value[1].target_now))
                for a, b in zip(first[1].pending, value[1].pending):
                    self.assertTrue(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
            if mode == 'monitor':
                plain = FakePolicy()
                self.assertTrue(torch.equal(first[0], torch.stack([plain.select_action({}) for _ in range(75)])))
            scored = [r for r in results[2][2] if r.get('score') is not None]
            self.assertEqual([r['step'] for r in scored], list(range(24, 75)))
            self.assertTrue(all(r['prediction_origin_step'] == r['step'] - 20 for r in scored))
            self.assertEqual(len(results[2][1].observed_prediction_origins), len(results[2][1].pending))
            details = [r for r in results[2][2] if r.get('event') == 'single_layer_ogg_details']
            self.assertEqual(len(details), results[2][1].guidance_calls)
            if mode == 'ogg': self.assertTrue(any(r['action_gradient_l2'] > 0 for r in details))

    def test_ogg_state_queue_logging_on_off_match_original(self): self.check_mode('ogg')
    def test_lvm_state_queue_logging_on_off_match_original(self): self.check_mode('lvm')
    def test_monitor_state_queue_logging_on_off_match_original(self): self.check_mode('monitor')

    def test_readonly_hook_preserves_forward_and_camera_order(self):
        class Add(nn.Module):
            def __init__(self, value): super().__init__(); self.value = value
            def forward(self, x): return x + self.value
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm = nn.Module(); self.vlm.vision_model = nn.Module()
                self.vlm.vision_model.encoder = nn.Module()
                self.vlm.vision_model.encoder.layers = nn.ModuleList([Add(i + 1) for i in range(9)])
                self.vlm.vision_model.post_layernorm = nn.Identity()
                self.vlm.connector = nn.Linear(768, 960)
            def get_vlm_model(self): return self.vlm
            def embed_image(self, image):
                for layer in self.vlm.vision_model.encoder.layers: image = layer(image)
                image = self.vlm.vision_model.post_layernorm(image)
                return self.vlm.connector(image[:, :64])
        ref = reference(); cls = corrector_class(ref, TAP)
        wrapper = Wrapper().eval().requires_grad_(False)
        # Existing frozen loading may keep a BF16 action expert while the
        # actual visual encoder and Connector remain FP32. Do not cast it.
        wrapper.lm_expert=nn.Linear(2,2,dtype=torch.bfloat16).eval().requires_grad_(False)
        images = [torch.ones(1, 1024, 768), torch.ones(1, 1024, 768) * 2]
        policy = FakePolicy(); policy.model.vlm_with_expert = wrapper
        policy.prepare_images = lambda batch: (images, [torch.ones(1, dtype=torch.bool)] * 2)
        with torch.no_grad(): before = wrapper.embed_image(images[0])
        with tempfile.TemporaryDirectory() as temp:
            for layer, expected in (('block_02', 6), ('block_08', 45)):
                protocol = dict(h_window=1, anchor_history=3, k_step=10, fps=10., prediction_seconds=1., visual_layer=layer)
                c = cls(policy, FakePredictor().eval().requires_grad_(False), protocol, 'monitor', SOURCE, Path(temp) / layer, layer=layer)
                z = c.encode({}).reshape(1, 2, 1024, 768)
                self.assertTrue(torch.equal(z[:, 0], images[0] + expected))
                self.assertTrue(torch.equal(z[:, 1], images[1] + expected))
                self.assertTrue(all(len(m._forward_hooks) == 0 for m in wrapper.modules()))
            wrapper.vlm.vision_model.register_parameter('dtype_probe',nn.Parameter(torch.zeros(1,dtype=torch.bfloat16),requires_grad=False))
            with self.assertRaises(ValueError):c.encode({})
            self.assertEqual(wrapper.vlm.vision_model.dtype_probe.dtype,torch.bfloat16)
        with torch.no_grad(): self.assertTrue(torch.equal(before, wrapper.embed_image(images[0])))

    def test_control_changes_rejected(self):
        ref = reference(); cls = corrector_class(ref, TAP)
        protocol = dict(h_window=1, anchor_history=3, k_step=10, fps=10., prediction_seconds=1., visual_layer='block_08')
        with tempfile.TemporaryDirectory() as temp:
            for kw in (dict(cooldown=20), dict(eta=2), dict(force_step=40)):
                with self.subTest(kw=kw), self.assertRaises(ValueError):
                    cls(FakePolicy(), FakePredictor().eval().requires_grad_(False), protocol, 'ogg', SOURCE, temp, layer='block_08', **kw)
            with self.assertRaises(ValueError):
                cls(FakePolicy(), FakePredictor().eval().requires_grad_(False), protocol, 'ogg', SOURCE, temp, layer='block_02')


if __name__ == '__main__':
    unittest.main()

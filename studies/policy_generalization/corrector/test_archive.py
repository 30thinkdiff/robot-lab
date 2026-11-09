"""断连、部分复制、损坏检测；不需要GPU，不把构造数据当模型实验。"""
from pathlib import Path
import importlib.util
import io
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "corrector_data_workbench", Path(__file__).resolve().parents[3] / "scripts/corrector_data_workbench.py"
)
workbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workbench)
publish, verified = workbench.publish, workbench.verified


class ArchiveTests(unittest.TestCase):
    def test_finished_features_do_not_require_deleted_raw_or_loading_encoder(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);final=root/'features/episode_813';final.mkdir(parents=True)
            (final/'metadata.json').write_text(json.dumps({'policy_revision':workbench.POLICY_SHA,
                'corrector_revision':workbench.CORRECTOR_SHA}))
            workbench.write_json(final/'COMPLETE.json',{'files':{'metadata.json':workbench.digest(final/'metadata.json')}})
            args=SimpleNamespace(output_root=root,episodes=[813])
            with patch.object(workbench,'ensure_source',side_effect=AssertionError('不应加载模型')):
                workbench.extract(args)
    def test_cpu_extraction_requires_explicit_opt_in_and_cuda_never_silently_falls_back(self):
        from unittest.mock import patch
        with self.assertRaises(RuntimeError):
            workbench.extraction_device("cpu", False)
        self.assertEqual(workbench.extraction_device("cpu", True), "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                workbench.extraction_device("cuda", True)

    def test_decoded_image_values_preserved_and_writable(self):
        from PIL import Image
        image = Image.new('RGB', (2, 2), color=(134, 135, 129))
        blob = io.BytesIO(); image.save(blob, format='PNG')
        result = workbench.decode_rgb(blob.getvalue())
        self.assertTrue(result.flags.writeable)
        self.assertEqual(result[0, 0].tolist(), [134, 135, 129])

    def test_legacy_dtype_keyword_preserves_requested_precision(self):
        from unittest.mock import Mock
        loader = Mock(return_value='model')
        self.assertEqual(workbench.load_vlm_compat(loader, 'public-model', torch_dtype='bfloat16'), 'model')
        loader.assert_called_once_with('public-model', dtype='bfloat16')
        with self.assertRaises(ValueError):
            workbench.load_vlm_compat(loader, 'public-model', torch_dtype='bfloat16', dtype='float32')

    def test_complete_policy_disables_duplicate_backbone_and_keeps_bfloat16_visual_weights(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.safetensors').write_bytes(b'loader-contract-fixture')
            vlm=torch.nn.Linear(2,2,bias=False)
            weights=torch.tensor([[.123456,.987654],[1.23456,-.55555]])
            with torch.no_grad():vlm.weight.copy_(weights)
            policy=SimpleNamespace(model=SimpleNamespace(vlm_with_expert=SimpleNamespace(vlm=vlm)))
            loader=Mock(return_value=policy);cfg=SimpleNamespace(load_vlm_weights=True,device='auto')
            result=workbench.load_complete_policy(loader,str(root),cfg,'cpu',revision=None)
            self.assertIs(result,policy)
            self.assertFalse(cfg.load_vlm_weights)
            self.assertEqual(cfg.device,'cpu')
            self.assertTrue(loader.call_args.kwargs['strict'])
            self.assertTrue(loader.call_args.kwargs['local_files_only'])
            torch.testing.assert_close(vlm.weight,weights.bfloat16(),atol=0,rtol=0)
            with self.assertRaises(FileNotFoundError):
                workbench.load_complete_policy(loader,str(root/'missing'),cfg,'cpu')

    def test_complete_policy_snapshot_prefers_cache_and_does_not_ping_hub(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.safetensors').write_bytes(b'cache-fixture')
            with patch('huggingface_hub.snapshot_download',return_value=str(root)) as download:
                self.assertEqual(workbench.cached_policy_snapshot(True),str(root))
                download.assert_called_once_with(workbench.POLICY_ID,revision=workbench.POLICY_SHA,
                                                  token=False,local_files_only=True)

    def test_extractor_adapter_is_local_scoped_and_never_requests_base_model_weights(self):
        import sys
        import types
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        calls=[]
        class Config:
            @classmethod
            def from_pretrained(cls,*a,**kw):
                calls.append(('policy_config',kw));return SimpleNamespace(load_vlm_weights=True)
        class Assets:
            @classmethod
            def from_pretrained(cls,*a,**kw):
                calls.append(('config_or_processor',kw));return object()
        class Policy:
            @classmethod
            def from_pretrained(cls,path,**kw):
                calls.append(('full_policy',kw))
                self.assertFalse(kw['config'].load_vlm_weights)
                self.assertTrue(kw['strict'])
                Assets.from_pretrained('cached-config')
                return SimpleNamespace(model=SimpleNamespace(vlm_with_expert=SimpleNamespace(vlm=torch.nn.Linear(2,2))))
        class Extractor:
            def __init__(self,path,*,device,local_files_only):
                self.policy=Policy.from_pretrained(path,local_files_only=local_files_only,revision=None)
                self.model_dtype=next(self.policy.model.vlm_with_expert.vlm.parameters()).dtype
        # 仅替代依赖以验证调用合同；没有模拟真实VLA权重加载或视觉性能。
        modules={}
        for name,attrs in {
            'transformers':{'AutoConfig':Assets,'AutoProcessor':Assets},
            'lerobot.policies.smolvla.configuration_smolvla':{'SmolVLAConfig':Config},
            'lerobot.policies.smolvla.modeling_smolvla':{'SmolVLAPolicy':Policy},
        }.items():
            module=types.ModuleType(name);module.__dict__.update(attrs);modules[name]=module
        original=Policy.from_pretrained
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,modules):
            (Path(tmp)/'model.safetensors').write_bytes(b'fixture')
            encoder=workbench.cached_visual_extractor(Extractor,tmp,'cpu')
            self.assertEqual(encoder.model_dtype,torch.bfloat16)
        self.assertEqual(Policy.from_pretrained,original)
        self.assertTrue(all(kw['local_files_only'] for _,kw in calls))

    def test_incomplete_copy_can_finish_and_then_detect_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "local"
            local.mkdir()
            (local / "data.npy").write_bytes(b"complete binary fixture")
            destination = root / "archive"
            partial = root / "archive.partial"
            partial.mkdir()
            (partial / "data.npy").write_bytes(b"incomplete")
            self.assertFalse(verified(destination))
            self.assertFalse(verified(partial))
            publish(local, destination)
            self.assertTrue(verified(destination))
            with self.assertRaises(FileExistsError):
                publish(local, destination)
            (destination / "data.npy").write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                verified(destination)

    def test_extra_old_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "local"
            local.mkdir()
            (local / "data.npy").write_bytes(b"current")
            partial = root / "archive.partial"
            partial.mkdir()
            (partial / "stale.npy").write_bytes(b"stale")
            with self.assertRaises(RuntimeError):
                publish(local, root / "archive")
            self.assertFalse(verified(root / "archive"))

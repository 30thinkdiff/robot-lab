"""固定上游算法的训练接入检查；构造缓存，不当作真实视觉/机器人结果。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("corrector_training_entry", ROOT / "scripts/corrector_train.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SOURCE = ROOT / "outputs/source_audit/vla-corrector"


def fake_cache(root):
    for ep, split in ((813, "train"), (825, "validation")):
        path = root / f"episode_{ep}"
        path.mkdir(parents=True)
        rng = np.random.default_rng(ep)
        np.save(path / "z_q.npy", rng.integers(-10, 11, (7, 2, 4), dtype=np.int8))
        np.save(path / "z_scale.npy", np.full((7, 2), .01, dtype=np.float16))
        np.save(path / "actions.npy", rng.normal(size=(7, 7)).astype(np.float16))
        np.save(path / "frame_index.npy", np.arange(7))
        np.save(path / "episode_index.npy", np.full(7, ep, dtype=np.int32))
        runner.archive.write_json(path / "metadata.json", {"num_frames": 7, "l_tokens": 2, "d_dim": 4,
            "action_dim": 7, "fps": 10, "split": split, "corrector_revision": runner.archive.CORRECTOR_SHA,
            "scope": "synthetic fixture, not encoder features",
            "policy_revision": runner.archive.POLICY_SHA, "action_representation": "checkpoint_processor_normalized"})
        runner.archive.write_json(path / "COMPLETE.json", {
            "files": {p.name: runner.archive.digest(p) for p in path.iterdir()}})


@unittest.skipUnless((SOURCE / ".git").exists(), "需要本地固定上游源码")
class TrainingEntryTests(unittest.TestCase):
    def args(self, root, name):
        from argparse import Namespace
        return Namespace(mode="fit", features=root / "features", source_root=SOURCE,
                         output=root / name, local_cache=root / "staged", device="cpu", profile="pilot",
                         k_step=1, epochs=3, patience=5, batch_size=2, accumulate=2, lr=3e-4,
                         weight_decay=1e-4, loss_type="both", beta=1., seed=42, cpu_threads=1,
                         resume=False, stop_after_epoch=0, implementation="reference")

    def test_unfinished_student_tasks_do_not_fall_back_or_write_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / "features")
            args = self.args(root, "student")
            args.implementation = "student"
            args.output.mkdir()
            # 用未完成函数源码替身验证保护逻辑；用户日后补齐任务后此测试仍有效。
            with patch.object(runner.inspect, "getsource", return_value="def task():\n    raise NotImplementedError('TODO')\n"):
                with self.assertRaises(NotImplementedError):
                    runner.run(args)
            self.assertFalse((args.output / "protocol.json").exists())
            self.assertFalse((args.output / "latest.pt").exists())

    def test_student_hooks_are_really_used_by_training_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / "features")
            args = self.args(root, "student_calls")
            args.implementation = "student"
            args.output.mkdir()
            calls = []
            def forward(*a, **kw):
                calls.append(torch.is_grad_enabled())
                return runner.reference_forward(*a, **kw)
            # 测试替身只验证调用线路，不把未填写的student代码算作已完成。
            with patch.object(runner, "training_functions", return_value=(
                    runner.reference_components, runner.reference_loaders, forward, runner.reference_update)):
                runner.run(args)
            self.assertIn(True, calls)
            self.assertIn(False, calls)

    def test_reference_loss_is_original_and_preserves_batch(self):
        before = list(sys.path)
        cfg, _, _, loss, _ = runner.reference_api(SOURCE)
        self.assertEqual(sys.path, before)
        pred = torch.tensor([[[1., 0.]], [[0., -3.]]])
        target = torch.tensor([[[1., 0.]], [[0., 1.]]])
        total, mse, cos = loss(pred, target, cfg.LossType.BOTH, 1.)
        self.assertEqual((total.item(), mse.item(), cos.item()), (5., 4., 1.))
        self.assertTrue(loss.__code__.co_filename.endswith("siglip_dynamics/train.py"))

    def test_notebook_sync_preserves_outputs_and_refreshes_script_snapshot(self):
        spec = importlib.util.spec_from_file_location("sync_corrector", ROOT / "scripts/sync_corrector_notebook.py")
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebook.ipynb"
            source = "".join(f"{key} = 'old snapshot'\n" for key in sync.SOURCES)
            output = [{"output_type": "stream", "name": "stdout", "text": ["saved real run\n"]}]
            path.write_text(json.dumps({"cells": [{"id": "corrector-data-1", "cell_type": "code",
                "source": source.splitlines(True), "execution_count": 17, "outputs": output}]}))
            sync.sync(path)
            cell = json.loads(path.read_text())["cells"][0]
            self.assertEqual(cell["outputs"], output)
            self.assertEqual(cell["execution_count"], 17)
            import ast
            parsed = {node.targets[0].id: ast.literal_eval(node.value)
                      for node in ast.parse("".join(cell["source"])).body}
            self.assertEqual(parsed["TRAIN_SOURCE"], (ROOT / "scripts/corrector_train.py").read_text())

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / "features")
            full, split = self.args(root, "full"), self.args(root, "split")
            for a in (full, split):
                a.output.mkdir()
            runner.run(full)
            split.stop_after_epoch = 1
            runner.run(split)
            split.resume, split.stop_after_epoch = True, 0
            runner.run(split)
            a = torch.load(full.output / "latest.pt", weights_only=True)
            b = torch.load(split.output / "latest.pt", weights_only=True)
            self.assertEqual(a["step"], b["step"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            self.assertEqual(a["epoch"], 3)
            for name in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], atol=0, rtol=0)
            self.assertTrue(json.loads((split.output / "result.json").read_text())["checkpoint_reload_verified"])
            (split.output / "history.json").unlink()
            (split.output / "metrics.csv").unlink()
            runner.run(split)  # 已完成全部epoch，也应能从latest重建分析产物。
            self.assertEqual(len(json.loads((split.output / "history.json").read_text())), 3)
            self.assertTrue((split.output / "metrics.csv").is_file())
            with self.assertRaises(RuntimeError):
                split.k_step = 2
                runner.run(split)

    def test_accumulation_matches_full_batch_including_short_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / "features")
            small, large = self.args(root, "small"), self.args(root, "large")
            small.stop_after_epoch = large.stop_after_epoch = 1
            large.batch_size, large.accumulate = 4, 1
            # 用SGD隔离累积权重数学：Adam在接近零的梯度上会放大float32求和次序误差。
            # 实际AdamW保存恢复由上一项测试覆盖。
            with patch.object(runner.torch.optim, "AdamW", torch.optim.SGD):
                for a in (small, large):
                    a.output.mkdir(); runner.run(a)
            a = torch.load(small.output / "latest.pt", weights_only=True)
            b = torch.load(large.output / "latest.pt", weights_only=True)
            self.assertEqual(a["step"], 2)  # 6个样本→4+2两个更新，尾组按实际样本数归一化。
            for name in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], atol=2e-6, rtol=2e-5)

    def test_corrupt_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / "features")
            path = root / "features/episode_813/actions.npy"
            path.write_bytes(b"corrupted")
            _, dataset, _, _, _ = runner.reference_api(SOURCE)
            with self.assertRaises(RuntimeError):
                runner.datasets(root / "features", root / "staged", dataset, 1)


if __name__ == "__main__":
    unittest.main()

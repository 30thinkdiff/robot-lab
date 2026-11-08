"""预测报告验证冻结索引和训练集均值基线；构造缓存不是真实视觉成绩。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

import numpy as np

from .test_training_entry import ROOT, SOURCE, fake_cache, runner
from . import test_training_entry

spec = importlib.util.spec_from_file_location('prediction_report', ROOT / 'scripts/corrector_prediction_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@unittest.skipUnless((SOURCE / '.git').exists(), '需要固定上游审计源码')
class PredictionReportTests(unittest.TestCase):
    def test_intermediate_snapshot_is_evaluated_and_foreign_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / 'features')
            args = test_training_entry.TrainingEntryTests().args(root, 'run')
            args.snapshot_every, args.stop_after_epoch = 1, 1
            args.output.mkdir(); runner.run(args)
            output = root / 'report'; output.mkdir()
            inputs = Namespace(run=args.output, features=args.features, source_root=SOURCE,
                               output=output, raw_root=None, checkpoint=args.output / 'snapshots/epoch_0000.pt')
            # 无需训练完整结束，也不借用best文件代替选中的初始模型。
            (args.output / 'result.json').unlink()
            report.predict(inputs)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['checkpoint_epoch'], 0)
            self.assertEqual(summary['optimizer_step'], 0)
            inputs.checkpoint = args.output / 'snapshots/epoch_0001.pt'
            with self.assertRaises(ValueError): report.predict(inputs)  # 不混写两份权重报告。
            inputs.output = root / 'other'; inputs.output.mkdir()
            import torch
            foreign = torch.load(inputs.checkpoint, weights_only=True)
            foreign['feature_protocol']['seed'] = 999
            inputs.checkpoint = root / 'foreign.pt'; torch.save(foreign, inputs.checkpoint)
            with self.assertRaises(ValueError): report.predict(inputs)

    def test_report_uses_frozen_validation_and_rejects_changed_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cache(root / 'features')
            args = test_training_entry.TrainingEntryTests().args(root, 'run')
            args.stop_after_epoch = 1
            args.output.mkdir(); runner.run(args)
            output = root / 'report'; output.mkdir()
            inputs = Namespace(run=args.output, features=args.features, source_root=SOURCE, output=output, raw_root=None)
            report.predict(inputs)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['sample_count'], 6)
            self.assertEqual(summary['first_pair'], {'episode': 825, 'frame': 0, 'future_frame': 1})
            self.assertEqual(summary['split']['shared_episodes'], [])
            values = np.load(output / 'first_pair.npz')
            self.assertEqual(values['predicted_delta'].shape, (2, 4))
            _, cls, _, _, _ = runner.reference_api(SOURCE)
            groups, _, _ = runner.datasets(args.features, None, cls, 1)
            mean = np.mean([item['delta_z'].numpy() for item in groups['train']], axis=0)
            expected = np.mean([((mean-item['delta_z'].numpy())**2).mean() for item in groups['validation']])
            self.assertAlmostEqual(summary['metrics']['train_mean_mse'], float(expected), places=6)
            frozen = np.load(args.output / 'split_indices.npz')
            np.savez_compressed(args.output / 'split_indices.npz', train=frozen['train'], validation=frozen['validation'][::-1])
            with self.assertRaises(ValueError):
                report.predict(inputs)


if __name__ == '__main__':
    unittest.main()

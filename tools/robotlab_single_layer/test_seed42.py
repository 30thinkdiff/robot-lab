"""CPU-only scope and trustworthy completion gates."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from . import train_queue as q
from .seed42_pipeline import trained
from .train_monitor import write


class Seed42Tests(unittest.TestCase):
    def test_scope_uses_only_seed42_and_rejects_other_subsets(self):
        receipt=dict(status='completed',episodes=454,frames=66984,train_pairs=49447,validation_pairs=12089,
            manifest_sha256='h',data_adapter_sha256='h',training_adapter_sha256='h')
        cfg=dict(training_scope='seed42_only',jobs=q.jobs()[:2],pinned_files={},validation_receipt='receipt',manifest='manifest',output='/unused')
        with patch.object(q,'digest',return_value='h'),patch.object(q,'read',side_effect=lambda p:receipt if str(p)=='receipt' else None):
            self.assertTrue(q.validate_config(cfg))
            for jobs in (q.jobs(),q.jobs()[2:4],q.jobs()[:1]):
                with self.assertRaises(ValueError):q.validate_config(dict(cfg,jobs=jobs))
    def test_base_logger_survives_scoped_legacy_factory_patch(self):
        from .test_core import reference,TAP
        from .online import corrector_class
        ref=reference();cls=corrector_class(ref,TAP)
        instance=cls.__new__(cls);instance.diagnostics=False;instance.trace=False
        with patch.object(ref['runtime'],'OnlineCorrector',lambda *a:None):
            instance.log({'step':0})

    def test_paused_model_cannot_start_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'runs/S8_s42/attempt_001';run.mkdir(parents=True)
            write(run.parent/'index.json',dict(output=str(run)));write(run/'training_status.json',dict(status='paused'))
            self.assertIsNone(trained(root,'S8_s42'))
            write(run/'training_status.json',dict(status='completed'))
            with self.assertRaises(RuntimeError):trained(root,'S8_s42')
            write(run/'result.json',dict(checkpoint_reload_verified=True));self.assertEqual(trained(root,'S8_s42'),run)


if __name__=='__main__':unittest.main()

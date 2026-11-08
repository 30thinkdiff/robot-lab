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

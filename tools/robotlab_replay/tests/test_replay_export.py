import json,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_replay_export import export


@unittest.skipUnless((ROOT/'tools/robotlab_replay/.local/import_batches/casebank_07369cf020fbc70d/results').exists(),'requires real Plus capture')
class RealExport(unittest.TestCase):
    def setUp(self):self.source=ROOT/'tools/robotlab_replay/.local/import_batches/casebank_07369cf020fbc70d/results'
    def test_selected_real_trial_snapshot_keeps_source_counts_and_refuses_overwrite(self):
        case=json.loads((self.source/'plan.json').read_text())['cases'][0]
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'one.zip'
            # Actual output is ~0.1MiB; only simulate the production 5GiB
            # reserve so this real-data check can run on a small laptop disk.
            with patch('corrector_replay_export.shutil.disk_usage',return_value=SimpleNamespace(free=10*2**30)):
                result=export(self.source,output,[case['id']])
            self.assertEqual(result['completed'],1);self.assertFalse(output.with_suffix('.zip.partial').exists())
            with zipfile.ZipFile(output) as z:
                execution=json.loads(z.read('results/execution.json'))
                self.assertEqual(execution['completed'],1);self.assertEqual(execution['source_execution']['completed'],17)
                self.assertFalse(any(n.endswith(('.pt','.pkl')) for n in z.namelist()))
                self.assertEqual(len([n for n in z.namelist() if n.startswith('results/trials/')]),1)
            original=output.read_bytes()
            with self.assertRaises(FileExistsError):export(self.source,output,[case['id']])
            self.assertEqual(output.read_bytes(),original)
    def test_unknown_id_and_source_directory_write_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'bad.zip'
            with self.assertRaises(ValueError):export(self.source,output,['not-completed'])
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):export(self.source,self.source/'forbidden.zip')

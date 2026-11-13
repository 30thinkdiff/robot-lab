"""Acceptance on the actual, completed 120-rollout archive; no synthetic curves."""
import json,sys,unittest,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from data import Catalog
from casebank import inventory


@unittest.skipUnless((HERE/'.local/casebank_downloads/expanded_j_20261113_derived.zip').exists(),'requires actual completed J archive')
class ExpandedAcceptance(unittest.TestCase):
    def test_true_action_video_sync_and_retention_coverage(self):
        c=Catalog()
        try:
            rows=[r for r in c.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_expanded_ordinary_20261113_derived']
            self.assertEqual(len(rows),120)
            groups=[g for g in inventory(c)['groups'] if g['study']=='replay_casebank_expanded_ordinary_20261113_derived']
            self.assertEqual(len(groups),60)
            self.assertTrue(all(all(m['replay_complete'] for m in g['methods'].values()) for g in groups))
            for task,init,seed in [(7,2,1000),(9,2,1000),(1,0,1001),(9,1,1001)]:
                r=next(r for r in rows if (r['task'],r['init'],r['seed'],r['method'])==(task,init,seed,'j_ogg'))
                d=c.details(r['id']);self.assertEqual(d['video']['alignment'],'recording_verified')
                self.assertEqual(d['coverage']['fraction'],1.);self.assertEqual(d['coverage']['max_gap_steps'],0)
                self.assertEqual(len(d['queue_trace']),r['steps'])
                self.assertTrue(all('submitted_action' in s and 'block_actual' in s for s in d['steps']))
                self.assertIn('inferred',d['actual_protocol']['derivation']['controller_return_flags'])
        finally:c.close()
    def test_derivation_keeps_original_trial_bytes_and_checkpoint_exclusion(self):
        import hashlib
        with zipfile.ZipFile(HERE/'.local/casebank_downloads/expanded_j_20261113_derived.zip') as z:
            self.assertFalse(any(n.endswith(('.pt','.pkl')) for n in z.namelist()))
            for n in z.namelist():
                if not n.startswith('results/trials/'):continue
                r=json.loads(z.read(n));original=z.read('results/original_trials/'+r['case']['id']+'.json')
                self.assertEqual(hashlib.sha256(original).hexdigest(),r['derivation']['original_trial_sha256'])


@unittest.skipUnless((HERE/'.local/casebank_downloads/plus_smoke_20261113.zip').exists(),'requires real Plus smoke archive')
class PlusAcceptance(unittest.TestCase):
    def test_real_seventeen_configs_and_recorded_guidance(self):
        with zipfile.ZipFile(HERE/'.local/casebank_downloads/plus_smoke_20261113.zip') as z:
            records=[json.loads(z.read(n)) for n in z.namelist() if n.startswith('results/trials/')]
            self.assertEqual(len(records),17)
            self.assertTrue(all(r['status']=='completed' for r in records))
            for r in records:
                if not r['result']['guidance_calls']:continue
                base=f"results/attempts/{r['case']['id']}/attempt_001/"
                rows=[json.loads(line) for line in z.read(base+'monitor_steps.jsonl').splitlines()]
                gradients=[row for row in rows if 'gradient_l2' in row]
                self.assertEqual(len(gradients),r['result']['guidance_calls'])
                self.assertTrue(all(g['original_update_formula_exact_equal'] and g['gradient_l2']>=0 for g in gradients))
                retained=[n for n in z.namelist() if n.startswith(base+'tensors/')]
                self.assertEqual(len(retained),2)
        c=Catalog()
        try:
            rows=[r for r in c.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_expanded_plus_smoke_20261113_002']
            self.assertEqual(len(rows),17)
            for r in rows:self.assertEqual(c.details(r['id'])['video']['alignment'],'recording_verified')
        finally:c.close()

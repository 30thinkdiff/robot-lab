import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from corrector_followup_analysis import analyse
SNAPSHOT=ROOT/'outputs/gpu_followup_20261116/20261116T080300Z.json'


@unittest.skipUnless(SNAPSHOT.exists(),'requires final real 96-trial snapshot')
class FinalHorizonAnalysis(unittest.TestCase):
    def setUp(self):self.data=json.loads(SNAPSHOT.read_text())

    def test_real_denominators_and_matched_pairs(self):
        result,rows,pairs=analyse(self.data)
        self.assertEqual(len(rows),160);self.assertEqual(len(pairs),32)
        self.assertEqual([result['methods'][m]['successes'] for m in ['H50','H20','H10','H5','H1']],[7,20,19,15,6])
        for method in ['H10','H5','H1']:
            p=result['paired_vs_H20'][method]
            self.assertEqual(sum(p.values()),32)
            self.assertEqual(p['both_success']+p['new_only_success'],result['methods'][method]['successes'])

    def test_missing_or_duplicate_is_not_failure(self):
        self.data['original_baselines'].pop()
        with self.assertRaisesRegex(ValueError,'matrix'):analyse(self.data)
        self.data=json.loads(SNAPSHOT.read_text());self.data['original_baselines'].append(copy.deepcopy(self.data['original_baselines'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):analyse(self.data)

    def test_partial_and_signature_mismatch_not_promoted_to_final(self):
        self.data['phases']['plus_horizons']['execution']['status']='running'
        with self.assertRaisesRegex(ValueError,'all 96'):analyse(self.data)
        self.data=json.loads(SNAPSHOT.read_text());self.data['phases']['plus_horizons']['cases'][0]['matches_original_H20_signatures']=False
        with self.assertRaisesRegex(ValueError,'signature'):analyse(self.data)

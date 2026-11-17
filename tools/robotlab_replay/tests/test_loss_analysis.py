import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_loss_analysis import analyse
DATA=ROOT/'outputs/ogg_loss_20261116'


@unittest.skipUnless((DATA/'snapshot.json').exists(),'requires completed real OGG loss pilot')
class RealLossAnalysis(unittest.TestCase):
    def setUp(self):self.data=json.loads((DATA/'snapshot.json').read_text())

    def test_real_24_cases_same_first_candidate_and_actual_gradients(self):
        result,rows,guides=analyse(self.data,DATA)
        self.assertEqual(len(rows),24);self.assertEqual(len(result['first_guide_checks']),8)
        self.assertEqual(len(guides),280)
        for method,success in [('mlp_ogg',3),('j_ogg',2)]:
            for mode in ['cosine','mse','mse_cosine']:
                self.assertEqual(result['methods'][method][mode]['successes'],success)

    def test_different_initial_condition_not_paired(self):
        self.data['stages'][2]['trials'][0]['actual_protocol']['first']['state_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'signatures'):analyse(self.data,DATA)

    def test_missing_or_other_slot_update_is_not_accepted(self):
        self.data['stages'][0]['trials'][0]['guidance'][0]['per_action_clean_update_l2'][0][1]=.01
        with self.assertRaisesRegex(ValueError,'multi-action'):analyse(self.data,DATA)

    def test_no_guidance_is_valid_but_not_fake_candidate_evidence(self):
        # An explicit unit-test mutation, never exported as real experimental data.
        key=('libero_object/task_1/init_0','j_ogg');baseline=None
        for stage in self.data['stages']:
            for t in stage['trials']:
                c=t['record']['case']
                if (c['condition'],c['method'])!=key:continue
                if baseline is None:baseline=t['commands']
                t['commands']=baseline;t['guidance']=[];t['guidance_details']=[];t['record']['result']['guidance_calls']=0
        result,_,_=analyse(self.data,DATA)
        check=next(r for r in result['first_guide_checks'] if (r['condition'],r['method'])==key)
        self.assertTrue(check['no_guidance_all_modes']);self.assertIsNone(check['first_candidate_exact'])

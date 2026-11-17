import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_loss_extension_analysis import merge
OLD=ROOT/'outputs/ogg_loss_20261116'
NEW=ROOT/'outputs/ogg_loss_extension_20261116'
SELECTION=ROOT/'outputs/autodl_setup/ogg_loss_extension_20261116/selection.json'


@unittest.skipUnless((NEW/'snapshot.json').exists(),'requires completed real extension')
class RealExtensionAnalysis(unittest.TestCase):
    def test_actual_96_disjoint_pairs_and_losses(self):
        result,rows,patterns=merge(OLD,NEW,SELECTION)
        self.assertEqual(len(rows),96);self.assertEqual(len(patterns),16)
        self.assertEqual(result['test_conditions'],8);self.assertEqual(result['initial_state_signatures'],6)
        self.assertEqual(result['active_first_candidate_groups'],30);self.assertEqual(result['no_guidance_matched_groups'],2)
        self.assertEqual([result['methods']['j_ogg'][m]['successes'] for m in ['cosine','mse','mse_cosine']],[7,9,8])
        self.assertEqual(result['paired_vs_cosine']['j_ogg']['mse']['new_only_success'],2)
        self.assertEqual(result['paired_vs_cosine']['mlp_ogg']['mse']['cosine_only_success'],1)

    def test_unexpected_seed_matrix_cannot_be_merged(self):
        data=json.loads(SELECTION.read_text());data['seeds']=[1000,9999]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'selection.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'matrix'):merge(OLD,NEW,p)

    def test_all_imported_extension_videos_and_reviewed_groups(self):
        sys.path.insert(0,str(ROOT/'tools/robotlab_replay'))
        from data import Catalog
        from casebank import inventory
        c=Catalog()
        try:
            groups=[g for g in inventory(c)['groups'] if g['study'].startswith('reviewed_ogg_losses_')]
            self.assertEqual(len(groups),16)
            self.assertEqual(len({g['condition'] for g in groups}),8)
            for g in groups:
                self.assertTrue(all(m['replay_complete'] for m in g['methods'].values()))
            runs=[r for r in c.runs.values() if r['_protocol'].get('study_id','').startswith('replay_casebank_ogg_loss_extension_')]
            self.assertEqual(len(runs),72)
            for r in runs:self.assertEqual(c.details(r['id'])['video']['alignment'],'recording_verified')
            self.assertEqual(c.errors,[])
        finally:c.close()

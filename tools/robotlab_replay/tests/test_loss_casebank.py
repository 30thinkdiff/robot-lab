import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from data import Catalog
from casebank import inventory
from loss_casebank import build,install,validate_reference,protocol_signature
from representative_cases import comparison_groups
DATA=HERE.parents[1]/'outputs/ogg_loss_20261116'


@unittest.skipUnless((DATA/'snapshot.json').exists(),'requires real completed loss trials')
class LossViews(unittest.TestCase):
    def setUp(self):self.c=Catalog()
    def tearDown(self):self.c.close()

    def test_four_real_groups_all_six_methods_complete_without_copying_trials(self):
        n=len(self.c.runs);groups=[g for g in inventory(self.c)['groups'] if g['study'].startswith('reviewed_ogg_losses_pilot_')]
        self.assertEqual(len(groups),4)
        for group in groups:
            self.assertEqual(len(group['methods']),6)
            self.assertTrue(all(r['replay_complete'] for r in group['methods'].values()))
        self.assertEqual(len(self.c.runs),n);self.assertEqual(self.c.errors,[])

    def test_install_is_idempotent_and_unknown_sha_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.c.state=Path(folder)
            self.assertEqual(install(self.c,DATA)['referenced_trials'],24)
            self.assertEqual(install(self.c,DATA)['copied_trials'],0)
            self.assertEqual(len(comparison_groups(self.c)),4)
            p=self.c.state/'comparison_sets/ogg_loss_pilot_ordinary.json'
            value=json.loads(p.read_text());next(iter(value['groups'][0]['methods'].values()))['trial_sha256']='0'*64
            p.write_text(json.dumps(value));groups=comparison_groups(self.c)
            self.assertEqual(len(groups),2);self.assertTrue(any('SHA mismatch' in v for v in self.c.errors))

    def test_loss_alias_and_changed_queue_cannot_be_mislabeled(self):
        r=next(r for r in self.c.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_ogg_loss_ordinary_mse_20261116' and r['method']=='mlp_ogg')
        with self.assertRaisesRegex(ValueError,'alias mismatch'):validate_reference('mlp_ogg@cosine',{'loss_mode':'cosine'},r)
        modified=dict(r);modified['_record']=copy.deepcopy(r['_record']);modified['_record']['case']['config']['steps']=20
        with self.assertRaisesRegex(ValueError,'queue'):protocol_signature(modified)

    def test_export_preserves_loss_mode_and_actual_checkpoint_identity(self):
        from server import export_report
        r=next(r for r in self.c.runs.values() if r['ogg_loss_mode']=='mse' and r['method']=='mlp_ogg')
        with tempfile.TemporaryDirectory() as folder:
            self.c.state=Path(folder);result=export_report(self.c,[r['id']],20)
            manifest=json.loads((Path(result['directory'])/'provenance.json').read_text())
            source=manifest['sources'][0]
            self.assertEqual(source['ogg_guidance_objective']['loss_mode'],'mse')
            self.assertEqual(source['method_configuration']['ogg_loss_mode'],'mse')
            self.assertEqual(source['checkpoint_hashes']['legacy/mlp']['checkpoint_sha256'],r['_protocol']['legacy_weights']['mlp']['checkpoint_sha256'])

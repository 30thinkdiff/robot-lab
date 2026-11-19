"""Real local recovery-case import and safe source selection checks."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(REPO/'scripts')]
from import_crosssuite import selection
from data import Catalog
from casebank import inventory
from corrector_casebank_import import install_bundle

METRICS=REPO/'studies/policy_generalization/results/20261117/crosssuite/trials.csv'
ARCHIVE=HERE/'.local/casebank_downloads/crosssuite90_947fd6be5dc01368.zip'
STUDY='replay_casebank_crosssuite_20261117_v1_transfer'


@unittest.skipUnless(METRICS.exists(),'formal cross-suite metrics not present')
class SelectionTests(unittest.TestCase):
    def test_exact_two_seeds_four_methods(self):
        ids=selection(METRICS)
        self.assertEqual(len(ids),8)
        self.assertTrue(all(len(k)==20 and len(v)==64 for k,v in ids.items()))

    def test_incomplete_or_duplicate_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with METRICS.open() as f:
                reader=csv.DictReader(f);fields=reader.fieldnames
                rows=[r for r in reader if r['condition']=='libero_90/task_29/init_1']
            path=Path(temp)/'bad.csv'
            for broken in (rows[:-1],rows[:-1]+[rows[0]]):
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(broken)
                with self.assertRaises(ValueError):selection(path)


@unittest.skipUnless(ARCHIVE.exists(),'private eight-video archive is not distributed')
class RealImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=Catalog()
    @classmethod
    def tearDownClass(cls):cls.catalog.close()

    def test_original_trial_hashes_and_eight_verified_videos(self):
        chosen=selection(METRICS)
        runs=[r for r in self.catalog.runs.values() if r['_protocol'].get('study_id')==STUDY]
        self.assertEqual(len(runs),8)
        for r in runs:
            raw=r['_source'].read(r['member'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),chosen[r['_record']['case']['id']])
            d=self.catalog.details(r['id'])
            self.assertEqual(d['video']['alignment'],'recording_verified')
            self.assertTrue(d['video']['sha256_verified'])
            self.assertEqual(len(d['metrics']['initial_signature']),3)
            self.assertTrue(all(len(h)==64 for h in d['metrics']['initial_signature']))
            self.assertEqual(len(d['steps']),r['steps'])
            self.assertEqual(r['condition'],'libero_90/task_29/init_1')
            if r['method'].startswith('mlp'):self.assertEqual(d['coverage']['fraction'],1)

    def test_complete_pattern_groups_and_same_condition_pairs(self):
        groups=[g for g in inventory(self.catalog)['groups'] if g['study']==STUDY]
        self.assertEqual(len(groups),2)
        for g in groups:
            self.assertEqual(set(g['methods']),{'H50','H20','mlp_lvm','mlp_ogg'})
            self.assertTrue(all(r['replay_complete'] for r in g['methods'].values()))
            pattern=''.join('1' if g['methods'][m]['success'] else '0' for m in ['H50','H20','mlp_lvm','mlp_ogg'])
            self.assertEqual(pattern,'0101' if g['seed']==1000 else '0001')
            run=g['methods']['mlp_ogg']
            pairs=self.catalog.pairs(run['id'])
            self.assertEqual({r['method'] for r in pairs},{'H50','H20','mlp_lvm'})
            detail=self.catalog.details(run['id'])
            self.assertEqual(sum(e['kind']=='ogg' for e in detail['events']),12)
            self.assertTrue(all(self.catalog.details(r['id'])['metrics']['initial_signature']==detail['metrics']['initial_signature']
                                for r in g['methods'].values()))

    def test_reimport_is_idempotent_and_preserves_original_manifest(self):
        manifest=HERE/'.local/import_batches/casebank_947fd6be5dc01368/manifest.json'
        before=manifest.read_bytes()
        receipt=install_bundle(ARCHIVE,HERE/'.local')
        self.assertTrue(receipt['already_imported'])
        self.assertEqual(manifest.read_bytes(),before)


if __name__=='__main__':unittest.main()

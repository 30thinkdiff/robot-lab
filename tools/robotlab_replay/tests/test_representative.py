import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from representative_cases import FOUR, select_groups, comparison_groups
from import_representative import snapshot_selection, ordinary_manifest
from data import Catalog
from casebank import inventory

SNAPSHOT = HERE.parents[1] / 'outputs/corrector_progress_snapshots/20261113T161432Z.zip'


class SelectionTests(unittest.TestCase):
    def test_missing_not_failure_unknown_rejected_duplicates_rejected(self):
        records = [{'case': {'condition':'test/a', 'seed':1, 'method':m, 'id':str(i)},
            'status':'completed', 'result':{'success':True}} for i,m in enumerate(FOUR)]
        self.assertEqual(len(select_groups(records[:-1], {})), 0)
        self.assertEqual(select_groups(records, {})[0]['pattern'], '1111')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_groups(records + [records[0]], {})
        records[0]['result']['success'] = None
        with self.assertRaisesRegex(ValueError, 'boolean'):
            select_groups(records, {})

    def test_choose_pattern_first_then_uncovered_category_not_all_seeds(self):
        records=[]; conditions={}
        for name, category in [('a','camera'),('b','camera'),('c','lighting')]:
            conditions[name]={'category':category}
            for seed in [1,2,3]:
                for m in FOUR:
                    records.append({'case':{'condition':name,'seed':seed,'method':m,'id':name+m+str(seed)},
                        'status':'completed','result':{'success':True}})
        selected=select_groups(records, conditions)
        self.assertEqual([(g['condition'],g['seed']) for g in selected],[('a',1),('c',1)])


@unittest.skipUnless(SNAPSHOT.exists(), 'requires actual final metadata snapshot')
class RealSelectionTests(unittest.TestCase):
    def test_real_plus_patterns_categories_and_existing_repeat(self):
        requests=snapshot_selection(SNAPSHOT)
        self.assertEqual([len(r['trials']) for r in requests],[48,6])
        self.assertEqual({g['pattern'] for g in requests[0]['selection']['groups']},
            {'0000','1111','0011','1101','0010'})
        self.assertEqual(len({g['category'] for g in requests[0]['selection']['groups']}),8)
        self.assertEqual(requests[1]['selection']['groups'][0]['seed'],1002)
        self.assertEqual(requests, snapshot_selection(SNAPSHOT))

    def test_ordinary_manifest_is_reference_only_and_tampering_fails_closed(self):
        c=Catalog()
        try:
            manifest,evidence=ordinary_manifest(c,SNAPSHOT)
            self.assertEqual({g['pattern'] for g in manifest['groups']},{'0000','0011','1100','1111'})
            self.assertEqual(sum(g['replay_complete'] for g in manifest['groups']),3)
            count=len(c.runs)
            with tempfile.TemporaryDirectory() as temp:
                state=Path(temp);folder=state/'comparison_sets';folder.mkdir()
                (folder/manifest['evidence_file']).write_bytes(evidence)
                p=folder/'selected.json';p.write_text(json.dumps(manifest))
                old=c.state;c.state=state
                try:
                    self.assertEqual(len(comparison_groups(c)),4)
                    bad=copy.deepcopy(manifest);bad['groups'][0]['seed']+=999
                    p.write_text(json.dumps(bad));self.assertEqual(comparison_groups(c),[])
                    self.assertTrue(any('seed mismatch' in e for e in c.errors))
                    p.write_text(json.dumps(manifest));(folder/manifest['evidence_file']).write_bytes(b'{}')
                    self.assertEqual(comparison_groups(c),[])
                    self.assertEqual(len(c.runs),count)
                finally:c.state=old
        finally:c.close()


@unittest.skipUnless((HERE/'.local/comparison_sets/ordinary_j_mlp.json').exists(), 'requires completed real representative import')
class ImportedAcceptance(unittest.TestCase):
    def test_all_imported_videos_actions_scores_and_comparison_views(self):
        c=Catalog()
        try:
            groups=inventory(c)['groups']
            ordinary=[g for g in groups if g['study']=='reviewed_ordinary_j_mlp_20261114']
            self.assertEqual(len(ordinary),4)
            self.assertEqual(sum(all(v['replay_complete'] for v in g['methods'].values()) for g in ordinary),3)
            plus=[r for r in c.runs.values() if r['_protocol'].get('study_id','').startswith('replay_casebank_representative_')]
            self.assertEqual(len(plus),54)
            for r in plus:
                d=c.details(r['id'])
                self.assertEqual(d['video']['alignment'],'recording_verified',r['id'])
                self.assertEqual(len(d['queue_trace']),r['steps'])
                self.assertTrue(all('submitted_action' in s for s in d['steps']))
                if r['method'].startswith(('j_','mlp_')):
                    self.assertTrue(any(s['score'] is not None for s in d['steps']))
            self.assertEqual(c.errors,[])
        finally:c.close()


if __name__ == '__main__':unittest.main()

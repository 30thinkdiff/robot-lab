import copy,json,sys,tempfile,unittest,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_plus_horizon import source_plan,corrected_plan,registry,audit_api


class HorizonFollowupTests(unittest.TestCase):
    def test_baseline_flags_are_fixed_before_original_plan_id_creation(self):
        def make(conditions,seeds,methods,protocol):return {'methods':methods,'protocol':protocol}
        methods={m:dict(v,uses_corrector=True,corrector_sha256=None) for m,v in registry().items()}
        result=corrected_plan(make,['c'],[1000],methods,{}, {'frozen':'test'})
        self.assertTrue(all(not c['uses_corrector'] and c['corrector_sha256'] is None for c in result['methods'].values()))
        self.assertTrue(all('model' not in c and 'mode' not in c for c in result['methods'].values()))
        self.assertFalse(result['protocol']['ogg_enabled']);self.assertFalse(result['protocol']['lvm_enabled'])
        self.assertTrue(all(c['uses_corrector'] for c in methods.values()))
        with self.assertRaises(ValueError):corrected_plan(make,['c'],[1001],methods,{}, {})

    def test_reused_audit_functions_load_without_importing_model_or_environment(self):
        audit,continuity=audit_api(ROOT/'scripts/smolvla_horizon_pilot.py')
        self.assertEqual(audit.__name__,'QueueAudit')
        result=continuity([[0]*7,[.1]*7,[.2]*7],1)
        self.assertEqual(result['boundary_n'],2);self.assertIsNone(result['within_translation_delta_mean'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.py';p.write_text('raise RuntimeError("never execute")')
            with self.assertRaisesRegex(ValueError,'SHA'):audit_api(p)

    def test_real_32_condition_plan_and_single_seed_are_preserved_exactly(self):
        archive=ROOT/'outputs/corrector_progress_snapshots/20261113T161432Z.zip'
        if not archive.exists():self.skipTest('requires real final Plus snapshot')
        with zipfile.ZipFile(archive) as z:raw=z.read('plus/plus/plan.json')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'plan.json';p.write_bytes(raw);plan=source_plan(p)
            self.assertEqual(len(plan['protocol']['conditions']),32)
            self.assertEqual(len([(c,1000,m) for c in plan['protocol']['conditions'] for m in registry()]),96)
            bad=copy.deepcopy(plan);bad['cases'][0]['seed']=1009;p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):source_plan(p)


if __name__=='__main__':unittest.main()

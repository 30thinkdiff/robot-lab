import ast
import io
import json
from pathlib import Path
import struct
import sys
import unittest
import tempfile
from unittest.mock import patch
import zipfile
from fractions import Fraction
from contextlib import nullcontext
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[1];REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(HERE))
from corrector_casebank_plan import discover,cases,groups,methods
from corrector_casebank_import import validate_numeric_npz,install_bundle
from casebank import inventory
from data import Catalog


class CasebankUnitTests(unittest.TestCase):
    def test_import_rejects_cross_platform_backslash_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('results/..\\outside.json','{}')
            with self.assertRaisesRegex(ValueError,'非法成员路径'):install_bundle(archive,root/'state')
            self.assertFalse((root/'outside.json').exists())
    def test_remote_pts_use_actual_pyav_timestamps_without_ffprobe(self):
        tree=ast.parse((REPO/'scripts/corrector_casebank_eval.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode_video_pts')
        namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<actual-source-pts>','exec'),namespace)
        frames=[SimpleNamespace(pts=i,time_base=Fraction(1,100)) for i in [0,5,17]]
        module=SimpleNamespace(open=lambda p:nullcontext(SimpleNamespace(decode=lambda **kw:iter(frames))))
        with patch.dict(sys.modules,{'av':module}):self.assertEqual(namespace['decode_video_pts']('fake.mp4'),[0.,.05,.17])
        frames[1].pts=None
        with patch.dict(sys.modules,{'av':module}),self.assertRaises(RuntimeError):namespace['decode_video_pts']('fake.mp4')
    def test_remote_python_c_bootstrap_does_not_require_file(self):
        namespace={'__name__':'remote_bootstrap_test'}
        exec(compile((REPO/'scripts/corrector_casebank_remote.py').read_text(),'<remote-python-c>','exec'),namespace)
        with patch.object(sys,'argv',['remote','remote-inspect']),patch.dict(namespace,{'inspect_remote':lambda:{'gpu':'mock'}}):
            with patch('sys.stdout',new=io.StringIO()) as out:namespace['main']()
            self.assertEqual(json.loads(out.getvalue())['gpu'],'mock')

    def test_fifteen_methods_include_fair_new_rule_pair_and_all_architectures(self):
        m=methods();self.assertEqual(len(m),15)
        self.assertEqual(m['H20']['action_steps'],20)
        self.assertEqual(m['hybrid_t1_a10_ogg']['guide'],'transformer-a10-h1')
        self.assertEqual(m['planned_lvm']['model'],m['executed_lvm']['model'])
        self.assertTrue(all(c['video_policy']=='all' for c in m.values()))

    def test_npz_rejects_object_dtype_and_accepts_numeric_without_numpy(self):
        def archive(dtype,payload):
            header=repr({'descr':dtype,'fortran_order':False,'shape':(1,)}).encode();header+=b' '*((-len(header)-11)%64)+b'\n'
            data=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+payload
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:z.writestr('values.npy',data)
            return out.getvalue()
        validate_numeric_npz(archive('<f4',struct.pack('<f',1.)))
        with self.assertRaises(ValueError):validate_numeric_npz(archive('|O',b'pickle?'))
        with self.assertRaises(ValueError):validate_numeric_npz(archive('<f4',b''))

    def test_remote_payload_covers_static_imports_and_is_bounded(self):
        from corrector_casebank_remote import payload,FILES
        raw,code=payload();self.assertLess(len(raw),2**20);self.assertEqual(len(code),12)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertIn('corrector_overnight_eval.py',z.namelist())
            for name in FILES:ast.parse(z.read(name))


@unittest.skipUnless((REPO/'outputs/corrector_audit_20261111/results_bundle.zip').exists(),'需真实旧实验')
class RealCasebankTests(unittest.TestCase):
    def test_imported_real_four_patterns_have_verified_video_and_measured_queue(self):
        c=Catalog()
        try:
            rows=[g for g in inventory(c)['groups'] if g['study']!='legacy_five_models' and all(m in g['methods'] for m in ['H50','H20','mlp_ogg'])]
            if not rows:self.skipTest('需取回真实完整补测，不使用模拟回放代替')
            patterns={''.join('1' if g['methods'][m]['success'] else '0' for m in ['H50','H20','mlp_ogg']) for g in rows}
            self.assertTrue({'001','000','011','111'}<=patterns)
            for g in rows:
                for method in ['H50','H20','mlp_ogg']:
                    r=g['methods'][method];self.assertTrue(r['replay_complete']);d=c._detail(r['id'])
                    self.assertEqual(d['video']['alignment'],'recording_verified')
                    self.assertEqual(len(d['queue_trace']),d['run']['steps'])
                    self.assertIn('block_actual',d['steps'][61]);self.assertIn('submitted_action',d['steps'][61])
        finally:c.close()
    def test_actual_patterns_and_frozen_660_case_selection(self):
        s=discover(REPO/'outputs/corrector_audit_20261111/results_bundle.zip')
        counts={r['pattern']:r['matched_cases'] for r in s['inventory']}
        self.assertEqual(counts,{'001':3,'000':4,'011':8,'111':42,'010':1,'101':2,'100':0,'110':0})
        self.assertEqual(len(s['anchors']),11);self.assertEqual(len(groups(s)),44)
        plan=cases(s);self.assertEqual(len(plan),660);self.assertEqual(len({c['id'] for c in plan}),660)
        self.assertEqual({c['config']['anchor_pattern'] for c in plan[:12]},{'001','000','011','111'})
        self.assertEqual({c['method'] for c in plan[:12]},{'H50','H20','mlp_ogg'})
        self.assertEqual(sum(c['config']['phase']=='confirmation' for c in plan),495)
        frozen=json.loads((REPO/'studies/policy_generalization/corrector/manifests/replay_casebank_20261112.json').read_text())
        self.assertEqual(s,frozen)

    def test_replay_case_inventory_keeps_seed_pairing_and_does_not_claim_completeness(self):
        c=Catalog()
        try:
            data=inventory(c);legacy=[r for r in data['groups'] if r['study']=='legacy_five_models'];self.assertEqual(len(legacy),60)
            groups_by_pattern={}
            for row in legacy:
                triple=[row['methods'][m] for m in ['H50','H20','mlp_ogg']]
                p=''.join('1' if r['success'] else '0' for r in triple);groups_by_pattern[p]=groups_by_pattern.get(p,0)+1
                self.assertFalse(all(r['replay_complete'] for r in triple))
            self.assertEqual(groups_by_pattern['001'],3);self.assertEqual(groups_by_pattern['111'],42)
        finally:c.close()


if __name__=='__main__':unittest.main()

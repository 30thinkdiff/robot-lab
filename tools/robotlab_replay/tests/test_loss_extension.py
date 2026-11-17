import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from corrector_loss_ablation import selected_pairs,ORDINARY,PLUS
SELECTION=ROOT/'outputs/autodl_setup/ogg_loss_extension_20261116/selection.json'


@unittest.skipUnless(SELECTION.exists(),'requires actual bounded extension plan')
class ExtensionSelection(unittest.TestCase):
    def test_reuse_old_24_and_only_schedule_missing_72(self):
        data=json.loads(SELECTION.read_text());new=[];old=[]
        for domain in ['ordinary','plus']:
            pairs=selected_pairs(SELECTION,domain);self.assertEqual(len(pairs),6)
            new.extend((domain,*p) for p in pairs);old.extend((domain,*p) for p in data['reused_pairs'][domain])
        self.assertEqual(len(new)*6,72);self.assertEqual(len(old)*6,24)
        self.assertFalse(set(new)&set(old));self.assertEqual(len(set((p[0],p[1]) for p in new+old)),8)
        self.assertEqual(selected_pairs(None,'ordinary'),ORDINARY);self.assertEqual(selected_pairs(None,'plus'),PLUS)

    def test_duplicate_unknown_seed_and_missing_physical_cell_rejected(self):
        original=json.loads(SELECTION.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selection.json'
            for variant in ['duplicate','seed','missing']:
                d=copy.deepcopy(original)
                if variant=='duplicate':d['new_pairs']['ordinary'].append(d['new_pairs']['ordinary'][0])
                if variant=='seed':d['new_pairs']['ordinary'][0][1]=9999
                if variant=='missing':d['new_pairs']['ordinary'].pop()
                path.write_text(json.dumps(d))
                with self.assertRaises(ValueError):selected_pairs(path,'ordinary')

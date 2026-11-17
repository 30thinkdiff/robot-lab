import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from data import Source,Catalog
from motion_data import load_motion

ARCHIVE=HERE.parents[1]/'outputs/gpu_followup_20261116/motion_representatives.zip'


@unittest.skipUnless(ARCHIVE.exists(),'requires real motion capture bundle')
class RealMotionData(unittest.TestCase):
    def setUp(self):
        self.source=Source(ARCHIVE)
        name=next(n for n in self.source.names if '/trials/' in n)
        self.record=self.source.json(name)
        self.base=f"results/attempts/{self.record['case']['id']}/attempt_001/"
        self.n=self.record['result']['env_steps']

    def tearDown(self):self.source.close()

    def test_all_twelve_real_recordings_and_video_phase(self):
        c=Catalog()
        try:
            runs=[r for r in c.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_motion_diagnostic_v1']
            self.assertEqual(len(runs),12)
            total=0
            for r in runs:
                d=c.details(r['id']);m=d['motion']
                self.assertTrue(m['available'],m['message']);self.assertEqual(d['video']['alignment'],'recording_verified')
                self.assertEqual(len(m['rows']),r['steps']);total+=len(m['rows'])
                for s in m['rows']:
                    self.assertEqual(s['submitted_command'],d['steps'][s['step']]['submitted_action'])
                    self.assertAlmostEqual(s['after']['sim_time_s']-s['before']['sim_time_s'],.05,places=6)
                self.assertIn('末次after无视频帧',m['phase'])
            self.assertEqual(total,2687)
        finally:c.close()

    def test_missing_does_not_invent_states(self):
        self.assertFalse(load_motion(self.source,self.base,{'result':{}},self.n,20)['available'])

    def test_wrong_sha_case_and_fps_fail_closed(self):
        bad=copy.deepcopy(self.record);bad['result']['motion_capture']['files_sha256']['motion/steps.jsonl']='0'*64
        self.assertFalse(load_motion(self.source,self.base,bad,self.n,20)['available'])
        bad=copy.deepcopy(self.record);bad['case']['seed']+=1
        self.assertFalse(load_motion(self.source,self.base,bad,self.n,20)['available'])
        self.assertFalse(load_motion(self.source,self.base,self.record,self.n,10)['available'])

    def test_export_selected_state_and_whole_episode_errors(self):
        from server import export_report
        c=Catalog()
        try:
            r=next(r for r in c.runs.values() if r['_protocol'].get('study_id')=='replay_casebank_motion_diagnostic_v1')
            with tempfile.TemporaryDirectory() as folder:
                c.state=Path(folder)
                result=export_report(c,[r['id']],20)
                p=Path(result['directory']);payload=json.loads((p/'A_motion_step.json').read_text())
                self.assertEqual(payload['motion_step']['step'],20)
                self.assertEqual(payload['source']['motion']['verified'],True)
                self.assertEqual(len((p/'A_motion_errors.csv').read_text().splitlines()),r['steps']+1)
                self.assertIn('After-action error',(p/'A_motion.svg').read_text())
        finally:c.close()

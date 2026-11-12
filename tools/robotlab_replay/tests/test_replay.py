import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
import subprocess
import signal
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from data import Catalog,Source,recording_evidence,REPO
from server import parse_range,export_report,chart_svg
from diagnostics import summarize,heatmaps,alignment_steps


class UnitTests(unittest.TestCase):
    def test_missing_outcome_and_arrays_remain_missing(self):
        d={'run':{'status':'completed','success':None,'method':'lvm'},'steps':[{'step':0,'events':[]}],
           'coverage':None,'events':[],'video':{'alignment':'unavailable','message':'视频未记录'},'source':{}}
        summary=summarize(d)
        self.assertTrue(any(c['title']=='任务结局未记录' for c in summary['cards']))
        self.assertFalse(any(c['title']=='任务结果：成功' for c in summary['cards']))
        self.assertTrue(heatmaps(d,0)['missing_commands']);self.assertTrue(heatmaps(d,0)['missing_updates'])
        with self.assertRaises(ValueError):heatmaps(d,-1)

    def test_manual_alignment_rejects_invalid_anchors_and_retains_offset(self):
        details=[{'steps':[{}]*100,'control_fps':20},{'steps':[{}]*120,'control_fps':20}]
        steps,anchor=alignment_steps(details,30,{'a':30,'b':40,'label':'人工事件位置'})
        self.assertEqual(steps,[30,40]);self.assertEqual(anchor['mode'],'manual_event_offset')
        self.assertEqual(alignment_steps(details,0,{'a':30,'b':10,'label':'测试'})[0],[0,-20])
        self.assertEqual(alignment_steps(details,130,{'a':30,'b':10,'label':'测试'})[0],[130,110])
        with self.assertRaises(ValueError):alignment_steps([details[0],{**details[1],'control_fps':10}],0,{'a':30,'b':40,'label':'不同频率'})
        for bad in [{'a':True,'b':1,'label':'x'},{'a':-1,'b':1,'label':'x'},
                    {'a':100,'b':1,'label':'x'},{'a':1,'b':120,'label':'x'},
                    {'a':1,'b':2,'label':''},{'a':1,'b':2,'label':'x'*121},
                    {'a':1,'b':2,'label':'x','path':'/etc/passwd'}]:
            with self.assertRaises(ValueError):alignment_steps(details,0,bad)

    def test_range_request_semantics(self):
        self.assertEqual(parse_range(None,100),(0,99,False))
        self.assertEqual(parse_range('bytes=4-9',100),(4,9,True))
        self.assertEqual(parse_range('bytes=90-',100),(90,99,True))
        self.assertEqual(parse_range('bytes=-10',100),(90,99,True))
        self.assertEqual(parse_range('bytes=95-999',100),(95,99,True))
        for bad in ['bytes=100-','bytes=9-1','bytes=0-1,4-6','bytes=-0','nonsense']:
            with self.assertRaises(ValueError):parse_range(bad,100)

    def test_source_path_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'source';folder.mkdir();(root/'secret').write_text('secret')
            (folder/'link').symlink_to(root/'secret');s=Source(folder)
            for bad in ['../secret','/etc/passwd','link']:
                with self.assertRaises(ValueError):s.read(bad)
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../secret','bad')
            s=Source(archive)
            with self.assertRaises(ValueError):s.read('../secret')
            s.close()

    def test_minimal_missing_trace_and_video_degrades(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'trials').mkdir()
            case={'id':'example','condition':'libero_object/task_2/init_2','method':'H50','seed':1000,'config':{'action_steps':50}}
            (folder/'plan.json').write_text(json.dumps({'protocol':{'fps':20,'score_start':2},'cases':[case]}))
            (folder/'trials/example.json').write_text(json.dumps({'case':case,'status':'completed','result':{'success':False,'env_steps':3,'chunk_calls':1}}))
            c=Catalog([folder],state=folder/'derived');d=c.details(next(iter(c.runs)))
            self.assertIsNone(d['coverage']);self.assertFalse(d['video']['available']);self.assertEqual(d['steps'][0]['monitor_state'],'未启用Corrector')
            self.assertTrue(all(s['score'] is None and 'submitted_action' not in s for s in d['steps']))
            self.assertEqual(d['steps'][0]['block_derived'],1);c.close()

    def test_invalid_queue_reconstruction_is_not_presented(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'trials').mkdir()
            case={'id':'x','condition':'libero_object/task_0/init_0','method':'ogg','seed':1,'config':{'action_steps':50}}
            (folder/'plan.json').write_text(json.dumps({'protocol':{},'cases':[case]}))
            (folder/'trials/x.json').write_text(json.dumps({'case':case,'status':'completed','result':{'env_steps':10,'chunk_calls':9}}))
            c=Catalog([folder],state=folder/'derived');d=c.details(next(iter(c.runs)))
            self.assertFalse(any('block_derived' in s for s in d['steps']));c.close()

    def test_unverified_recording_does_not_claim_exact_sync(self):
        with tempfile.TemporaryDirectory() as tmp:self.assertFalse(recording_evidence(Path(tmp))['verified'])

    def test_checkpoint_cannot_be_served_as_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'trials').mkdir();(root/'weights.pt').write_bytes(b'not a video')
            case={'id':'x','condition':'libero_object/task_0/init_0','method':'H50','seed':1,'config':{}}
            (root/'plan.json').write_text(json.dumps({'cases':[case],'protocol':{}}))
            (root/'trials/x.json').write_text(json.dumps({'case':case,'status':'completed','video_artifacts':[{'path':'weights.pt'}],'result':{'env_steps':1}}))
            c=Catalog([root],state=root/'derived');self.assertIsNone(c.video_path(next(iter(c.runs.values()))));c.close()

    def test_server_ctrl_c_closes_and_does_not_leave_service(self):
        child=subprocess.Popen([sys.executable,str(HERE/'server.py'),'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            first=child.stdout.readline();self.assertIn('http://127.0.0.1:',first)
            child.send_signal(signal.SIGINT);self.assertEqual(child.wait(timeout=8),0)
        finally:
            if child.poll() is None:child.kill();child.wait()
            child.stdout.close();child.stderr.close()


@unittest.skipUnless((REPO/'outputs/hybrid_pilot_20261111/results.zip').exists(),'真实实验归档不随公共代码分发')
class RealExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=Catalog()
    @classmethod
    def tearDownClass(cls):cls.catalog.close()
    def hybrid(self,method):
        return next(r for r in self.catalog.runs.values() if r['experiment']=='混合 Corrector' and r['method']==method and r['task']==1 and r['init']==0 and not r['forced'])

    def test_real_inventory_and_no_index_errors(self):
        self.assertGreaterEqual(len(self.catalog.runs),1200);self.assertEqual(self.catalog.errors,[])
        self.assertTrue(any(r['forced'] for r in self.catalog.runs.values()))

    def test_success_failure_and_strict_pairing(self):
        a,b=self.hybrid('A'),self.hybrid('B');self.assertTrue(a['success']);self.assertFalse(b['success'])
        self.assertEqual(a['steps'],168);self.assertEqual(b['steps'],280)
        self.assertIn(a['id'],[r['id'] for r in self.catalog.pairs(b['id'])])
        self.assertTrue(all(r['condition']==b['condition'] and r['seed']==b['seed'] for r in self.catalog.pairs(b['id'])))

    def test_different_condition_export_is_explicitly_exploratory(self):
        a=self.hybrid('B');b=next((r for r in self.catalog.runs.values() if r['experiment']=='Replay完整成败组合样本库' and r['task']==0 and r['init']==0 and r['method']=='H50'),None)
        if b is None:self.skipTest('需完整Case Bank真实回合')
        with tempfile.TemporaryDirectory() as tmp,patch.object(self.catalog,'state',Path(tmp)):
            export=export_report(self.catalog,[a['id'],b['id']],61)
            manifest=json.loads((Path(export['directory'])/'provenance.json').read_text())
            self.assertEqual(manifest['comparison']['type'],'exploratory_different_conditions')
            self.assertFalse(manifest['comparison']['same_task_and_init'])
            self.assertEqual(manifest['sources'][0]['condition'],a['condition'])
            self.assertEqual(manifest['sources'][1]['condition'],b['condition'])
            self.assertEqual(manifest['sources'][0]['selected_environment_step'],61)
            self.assertIn('探索性', (Path(export['directory'])/'SUMMARY.md').read_text())

    def test_step61_guidance_target_and_source_preserved(self):
        d=self.catalog.details(self.hybrid('B')['id']);row=d['steps'][61]
        self.assertEqual(sum(e['kind']=='ogg' for e in row['events']),4)
        self.assertEqual(sum(e['kind']=='replan' for e in row['events']),1)
        self.assertAlmostEqual(row['score'],0.49109506607055664)
        self.assertTrue(any(e['data'].get('target_shape')==[1,128,960] for e in row['events']))
        self.assertEqual(len(row['submitted_action']),7)
        self.assertTrue(any('gradient_l2' in e['data'] for e in row['events'] if e['kind']=='ogg'))
        # Same first detector target/score before changing guide; no invented mechanism.
        a=self.catalog.details(self.hybrid('A')['id'])['steps'][61]
        self.assertEqual(a['score'],row['score'])
        target=lambda s:next(e['data']['target_sha256'] for e in s['events'] if e['kind']=='target')
        self.assertEqual(target(a),target(row))

    def test_real_heatmaps_use_saved_commands_and_recorded_action_support(self):
        for method,support in [('A',{0}),('B',set(range(0,20,2)))]:
            d=self.catalog.details(self.hybrid(method)['id']);h=heatmaps(d,61)
            self.assertEqual(len(h['commands']),25)
            for row in h['commands']:self.assertEqual(row['values'],d['steps'][row['step']]['submitted_action'])
            self.assertEqual(len(h['updates']),4)
            for row in h['updates']:
                self.assertEqual(len(row['values']),50)
                self.assertEqual({i for i,v in enumerate(row['values']) if v!=0},support)
            self.assertIn('不是实际执行位移',h['update_units'])

    def test_diagnostic_summary_states_evidence_without_inventing_cause(self):
        c=self.catalog.details(self.hybrid('C')['id']);cards=summarize(c)['cards']
        gap=next(v for v in cards if v['title']=='存在监测空窗')
        self.assertEqual(gap['step'],51);self.assertIn('155/256',gap['text'])
        failure=next(v for v in cards if v['title']=='任务结果：失败')
        self.assertIn('不能自动认定',failure['text']);self.assertFalse(summarize(c)['full_visual_tensor_available'])
        r=next(r for r in self.catalog.runs.values() if r['model']=='MLP' and r['method']=='ogg' and not r['forced'])
        old=self.catalog.details(r['id']);self.assertTrue(heatmaps(old,61)['missing_updates'])

    @unittest.skipUnless((HERE/'.local/imported_video/manifest.json').exists(),'需真实视频')
    def test_manual_anchor_export_retains_each_real_frame_and_handles_before_start(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(self.catalog,'state',Path(tmp)):
            a,b=self.hybrid('B')['id'],self.hybrid('A')['id']
            export=export_report(self.catalog,[a,b],61,alignment={'a':61,'b':85,'label':'测试不同位置，不是阶段真值'})
            folder=Path(export['directory']);m=json.loads((folder/'provenance.json').read_text())
            self.assertEqual(m['environment_steps'],[61,85]);self.assertIn('manual',m['alignment_basis'])
            self.assertAlmostEqual(m['sources'][1]['frame_export']['pts_seconds'],4.25)
            self.assertTrue((folder/'B_step_0085.png').is_file());self.assertTrue((folder/'B_heatmaps.json').is_file())
            missing=export_report(self.catalog,[a,b],0,alignment={'a':61,'b':10,'label':'测试负偏移'})
            m=json.loads((Path(missing['directory'])/'provenance.json').read_text())
            self.assertEqual(m['environment_steps'],[0,-51]);self.assertFalse(m['sources'][1]['step_available'])
            self.assertNotIn('frame_export',m['sources'][1]);self.assertFalse(any(n.startswith('B_step_') for n in missing['files']))

    def test_exported_gap_chart_labels_clipped_threshold_without_filling_gaps(self):
        d=self.catalog.details(self.hybrid('C')['id']);svg=chart_svg(d,61)
        self.assertIn('fill="#fbf0df"',svg);self.assertIn('Initial threshold 10 exceeds displayed axis',svg)
        self.assertIn('requested step61',svg)

    def test_exported_ended_run_is_explicit(self):
        d=self.catalog.details(self.hybrid('A')['id']);svg=chart_svg(d,200)
        self.assertIn('requested step200',svg);self.assertIn('run already ended at 168 steps',svg)

    def test_a10_monitor_gap_coverage_from_real_trace(self):
        d=self.catalog.details(self.hybrid('C')['id']);cov=d['coverage']
        self.assertLess(cov['fraction'],.8);self.assertEqual(cov['max_gap_steps'],19)
        self.assertEqual(cov['scores'],sum(s['score'] is not None for s in d['steps'][d['first_due']:]))
        self.assertTrue(any(s['monitor_state']=='无到期评分' for s in d['steps']))
        self.assertTrue(any(e['kind']=='skip' for e in d['events']))

    def test_full_a10_cohort_reproduces_previous_audited_coverage(self):
        runs=[r for r in self.catalog.runs.values() if r['model']=='A10-H1' and r['method']=='ogg' and not r['forced'] and r['experiment']=='Object10 五模型']
        self.assertEqual(len(runs),60)
        coverage=[self.catalog.details(r['id'])['coverage'] for r in runs]
        self.assertEqual(sum(c['eligible_score_steps'] for c in coverage),8256)
        self.assertEqual(sum(c['scores'] for c in coverage),4873)
        self.assertEqual(max(c['max_gap_steps'] for c in coverage),19)

    def test_old_experiments_with_no_local_video_and_no_gradients(self):
        r=next(r for r in self.catalog.runs.values() if r['model']=='MLP' and r['method']=='ogg' and not r['forced'] and not self.catalog.video_path(r))
        d=self.catalog.details(r['id']);self.assertFalse(d['video']['available'])
        guides=[e for e in d['events'] if e['kind']=='ogg'];self.assertTrue(guides)
        self.assertTrue(all('gradient_l2' not in e['data'] for e in guides))

    def test_real_source_hash_matches_after_all_reads(self):
        r=self.hybrid('B');d=self.catalog.details(r['id'])
        self.assertEqual(hashlib.sha256(r['_source'].read(r['member'])).hexdigest(),d['source']['trial_sha256'])

    @unittest.skipUnless((HERE/'.local/imported_video/manifest.json').exists(),'需先只读导入3个真实视频')
    def test_actual_video_pts_and_recording_order(self):
        d=self.catalog.details(self.hybrid('B')['id']);v=d['video']
        self.assertEqual(v['alignment'],'recording_verified');self.assertEqual(len(v['pts']),280)
        self.assertAlmostEqual(v['pts'][61],3.05);self.assertTrue(v['sha256_verified'])
        self.assertTrue(all(a<b for a,b in zip(v['pts'],v['pts'][1:])))

    @unittest.skipUnless((HERE/'.local/imported_video/manifest.json').exists(),'需真实视频')
    def test_recording_proof_is_not_reused_for_unbound_video(self):
        original=self.hybrid('B');video=self.catalog.video_path(original)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'trials').mkdir();shutil.copyfile(video,root/'video.mp4')
            case={'id':'x','condition':'libero_object/task_1/init_0','method':'H50','seed':1000,'config':{'action_steps':50}}
            sha=hashlib.sha256(video.read_bytes()).hexdigest()
            (root/'plan.json').write_text(json.dumps({'cases':[case],'protocol':{'fps':20}}))
            (root/'trials/x.json').write_text(json.dumps({'case':case,'status':'completed','result':{'env_steps':280,'chunk_calls':6},'video_artifacts':[{'path':'video.mp4','sha256':sha}]}))
            c=Catalog([root],state=HERE/'.local');detail=c.details(next(iter(c.runs)))
            self.assertEqual(detail['video']['alignment'],'ordinal_only')
            self.assertFalse(detail['video']['recording_evidence_bound_to_video_sha']);c.close()

    @unittest.skipUnless((HERE/'.local/imported_video/manifest.json').exists(),'需真实视频')
    def test_export_real_frame_has_provenance_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.catalog,'state',Path(tmp)):
                a,b=self.hybrid('A')['id'],self.hybrid('B')['id']
                one=export_report(self.catalog,[b,a],61,'人工复盘：比较同一目标下的引导变化')
                two=export_report(self.catalog,[b,a],61,'独立导出')
                self.assertNotEqual(one['export'],two['export'])
                folder=Path(one['directory']);self.assertTrue((folder/'A_step_0061.png').is_file())
                manifest=json.loads((folder/'provenance.json').read_text())
                self.assertEqual(manifest['sources'][0]['frame_export']['frame_index'],61)
                self.assertAlmostEqual(manifest['sources'][0]['frame_export']['pts_seconds'],3.05)
                self.assertNotIn('/root/autodl-tmp',json.dumps(manifest))
                self.assertIn('环境步', (folder/'SUMMARY.md').read_text())
                import csv
                with (folder/'A_events.csv').open() as f:events=list(csv.DictReader(f))
                guides=[r for r in events if r['environment_step']=='61' and r['type']=='ogg']
                self.assertEqual(len(guides),4);self.assertTrue(all(r['gradient_l2'] and r['ogg_loss'] for r in guides))
                for f in manifest['files']:self.assertEqual(hashlib.sha256((folder/f['name']).read_bytes()).hexdigest(),f['sha256'])


@unittest.skipUnless(any((HERE/'.local/import_batches').glob('*/manifest.json')),'需先只读追加真实基线视频')
class RealImportReviewTests(unittest.TestCase):
    def test_requested_seven_sources_have_independent_video_bindings(self):
        from import_review import selection,validate
        requested=selection();self.assertEqual(len(requested),7)
        batches=list((HERE/'.local/import_batches').glob('*.zip'))
        manifest=validate(batches[-1],requested)
        self.assertEqual(len(manifest['videos']),7)
        self.assertEqual(manifest['skipped'],[])
        c=Catalog()
        try:
            for item in requested:
                r=next(r for r in c.runs.values() if r['member']==item['member'])
                d=c.details(r['id']);self.assertEqual(d['video']['alignment'],'recording_verified')
                self.assertEqual(len(d['video']['pts']),item['env_steps'])
            hybrid=next(r for r in c.runs.values() if r['method']=='C' and r['task']==1 and r['init']==0 and not r['forced'])
            self.assertEqual(c.details(hybrid['id'])['video']['alignment'],'recording_verified')
            # 新增正式A10绑定不能挤掉相同视频的原Hybrid C绑定。
            alias=next(r for r in c.runs.values() if r['model']=='T-H1' and r['method']=='H20' and r['task']==9 and r['init']==2)
            self.assertEqual(c.details(alias['id'])['video']['alignment'],'ordinal_only')
        finally:c.close()

    def test_import_rejects_unrequested_alias_and_unknown_member(self):
        from import_review import selection,validate
        requested=selection();archive=list((HERE/'.local/import_batches').glob('*.zip'))[-1]
        with self.assertRaises(ValueError):validate(archive,requested[1:])
        with tempfile.TemporaryDirectory() as tmp:
            altered=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(archive) as src,zipfile.ZipFile(altered,'w') as dst:
                for name in src.namelist():dst.writestr(name,src.read(name))
                dst.writestr('../escape.txt',b'bad')
            with self.assertRaises(ValueError):validate(altered,requested)

    def test_snapshot_fair_comparison_uses_common_conditions(self):
        snapshots=list((REPO/'outputs/corrector_overnight_20261111').glob('snapshot_20261112_105732.zip'))
        if not snapshots:self.skipTest('需真实2486回合快照')
        from analyse_snapshot import analyse
        with tempfile.TemporaryDirectory() as tmp:
            summary=analyse(snapshots[0],Path(tmp)/'derived')
            self.assertEqual(summary['execution']['status'],'paused')
            self.assertEqual(summary['execution']['reason'],'time_budget')
            self.assertEqual(summary['execution']['completed'],2486)
            core=[r for r in summary['paired_tables'] if r['scope']=='core_six']
            four=[r for r in summary['paired_tables'] if r['scope']=='matched_four']
            self.assertEqual([r['completed'] for r in core],[100]*6)
            self.assertEqual([r['successes'] for r in core],[71,94,78,84,86,85])
            self.assertEqual([r['completed'] for r in four],[571]*4)
            self.assertEqual([r['successes'] for r in four],[393,529,436,477])


if __name__=='__main__':unittest.main()

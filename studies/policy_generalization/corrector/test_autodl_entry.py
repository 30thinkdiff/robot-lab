"""只检查部署/调度合同，不在CPU测试中安装环境、联网或执行真实机器人。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import corrector_autodl as runner
import package_corrector_autodl as packer


class AutoDLTests(unittest.TestCase):
    def test_dead_proxy_stops_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));log=Mock()
            with patch.dict(runner.os.environ, {'https_proxy':'http://127.0.0.1:17890'}), \
                    patch.object(runner.socket, 'create_connection', side_effect=ConnectionRefusedError()), \
                    patch.object(runner.wb, 'prepare_with_managed_torch') as install:
                with self.assertRaisesRegex(ConnectionError,'代理端口不可达'):
                    runner.prepare(args,log)
                install.assert_not_called()

    def test_proxy_check_closes_probe_and_skips_unspecified_proxy(self):
        log=Mock()
        with patch.object(runner.socket,'create_connection') as connect:
            runner.check_proxy_listener(None,log)
            connect.assert_not_called()
            runner.check_proxy_listener('http://127.0.0.1:17890',log)
            connect.assert_called_once_with(('127.0.0.1',17890),timeout=5)
            connect.return_value.__exit__.assert_called_once()

    def args(self,root):
        args=runner.parser().parse_args(['night'])
        args.workdir=root/'work';args.output=root/'results';args.source_root=root/'source'
        args.checkpoint=root/'best.pt';args.checkpoint.write_bytes(b'fixture-not-real-checkpoint')
        args.checkpoint_sha256=runner.wb.digest(args.checkpoint)
        return args

    def smoke_result(self,args,*,guidance=4,horizon=20,checkpoint_sha=None):
        _,root=runner.evaluation_command(args,'smoke')
        (root/'trials').mkdir(parents=True)
        record={'case':{'id':'fixture','config':{'corrector_sha256':checkpoint_sha or runner.check_checkpoint(args)}},
                'attempt':1,'status':'completed','result':{'event_replans':1,'guidance_calls':guidance}}
        (root/'trials/fixture.json').write_text(json.dumps(record))
        directory=root/'attempts/fixture/attempt_001';directory.mkdir(parents=True)
        (directory/'actual_protocol.json').write_text(json.dumps({'monitor_horizon_steps':horizon}))
        protocol={'runtime_source_sha256':runner.wb.digest(ROOT/'scripts/corrector_online_runtime.py'),
                  'learner_core_sha256':runner.wb.digest(ROOT/'studies/policy_generalization/corrector/online_practice.py')}
        (root/'plan.json').write_text(json.dumps({'protocol':protocol}))
        return root

    def test_batch_has_fifty_paired_trials_and_no_forced_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp))
            smoke,_=runner.evaluation_command(args,'smoke')
            batch,_=runner.evaluation_command(args,'batch')
            self.assertIn('--debug-force-step',smoke)
            self.assertNotIn('--debug-force-step',batch)
            self.assertNotIn('--max-steps',batch)
            self.assertEqual(len(args.methods)*len(args.initial_state_ids)*len(args.seeds)*len(args.task_ids),50)
            self.assertEqual(batch[batch.index('--control-implementation')+1],'student')

    def test_smoke_requires_actual_guidance_correct_horizon_and_model(self):
        for guidance,horizon,sha in [(0,20,None),(4,10,None),(4,20,'wrong')]:
            with self.subTest(guidance=guidance,horizon=horizon),tempfile.TemporaryDirectory() as tmp:
                args=self.args(Path(tmp));self.smoke_result(args,guidance=guidance,horizon=horizon,checkpoint_sha=sha)
                with self.assertRaises(RuntimeError):runner.verify_smoke(args)
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));root=self.smoke_result(args)
            self.assertEqual(runner.verify_smoke(args),root)

    def test_batch_uses_offline_cache_only_after_smoke_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));log=Mock()
            with self.assertRaises(RuntimeError):runner.evaluate(args,'batch',log)
            log.run.assert_not_called()
            self.smoke_result(args)
            _,out=runner.evaluation_command(args,'batch');(out/'analysis').mkdir(parents=True)
            (out/'analysis/summary.json').write_text('{}')
            runner.evaluate(args,'batch',log)
            self.assertEqual(log.run.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')

    def test_checkpoint_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));args.checkpoint_sha256='wrong'
            with self.assertRaises(ValueError):runner.check_checkpoint(args)

    def test_package_is_flat_importable_and_hash_verified(self):
        import zipfile
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bundle.zip'
            manifest=packer.build(archive)
            with zipfile.ZipFile(archive) as z:
                self.assertTrue(all(Path(n).name==n for n in z.namelist()))
                z.extractall(root/'unpacked')
            runner.verify_bundle(root/'unpacked')
            result=subprocess.run([sys.executable,root/'unpacked/corrector_autodl.py','--help'],
                                  cwd=root,capture_output=True,text=True,check=True)
            self.assertIn('night',result.stdout)
            self.assertIn('corrector_online_practice.py',manifest['files'])
            (root/'unpacked/corrector_online_runtime.py').write_text('changed')
            with self.assertRaises(RuntimeError):runner.verify_bundle(root/'unpacked')

    def test_diagnostic_archive_excludes_credentials_weights_and_redacts_tokens(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));(args.workdir/'logs').mkdir(parents=True)
            token='hf_'+'a'*30
            (args.workdir/'logs/test.log').write_text('token='+token)
            (args.workdir/'.codex').mkdir()
            (args.workdir/'.codex/auth.json').write_text('private credential fixture')
            args.output.mkdir()
            (args.output/'model.pt').write_bytes(b'not collected')
            (args.output/'summary.json').write_text('{}')
            output=runner.diagnostics(args)
            with zipfile.ZipFile(output) as z:
                self.assertNotIn('model.pt',' '.join(z.namelist()))
                self.assertNotIn('auth.json',' '.join(z.namelist()))
                self.assertNotIn(token,z.read('logs/test.log').decode())
                self.assertEqual(z.read('results/summary.json'),b'{}')


if __name__=='__main__':unittest.main()

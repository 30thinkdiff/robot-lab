import json
from pathlib import Path
import tempfile
import unittest

from .batch_evaluation import make_plan, run_plan, set_recording_fps, summarize


class BatchEvaluationTests(unittest.TestCase):
    def test_overall_rate_weights_episodes_and_records_coverage(self):
        plan = make_plan(["a", "b"], [1, 2, 3], {"H50": {}}, {})
        records = [{"case": plan["cases"][i], "status": "completed",
                    "result": {"success": value, "chunk_calls": 2}}
                   for i, value in [(0, True), (1, True), (2, False), (3, True)]]
        with tempfile.TemporaryDirectory() as tmp:
            summarize(records, tmp, plan)
            total = json.loads((Path(tmp) / "summary.json").read_text())["overall_by_method"][0]
            self.assertEqual(total["success_rate"], 3 / 4)
            self.assertEqual(total["completion_rate"], 4 / 6)
            self.assertEqual(total["mean_chunk_calls"], 2)
            self.assertIsNone(total["mean_monitor_s"])

    def test_recording_fps_only_changes_instance_metadata(self):
        class Env:
            metadata = {"render_fps": 80}
        env = Env()
        env.envs = [Env()]
        set_recording_fps(env, 20)
        self.assertEqual(env.metadata["render_fps"], 20)
        self.assertEqual(env.envs[0].metadata["render_fps"], 20)
        self.assertEqual(Env.metadata["render_fps"], 80)

    def test_error_is_not_a_robot_failure(self):
        plan = make_plan(["view_5"], [1, 2, 3], {"H50": {}}, {})
        records = [{"case": plan["cases"][0], "status": "completed", "result": {"success": False}},
                   {"case": plan["cases"][1], "status": "error", "error": "disconnect"}]
        with tempfile.TemporaryDirectory() as tmp:
            rows = summarize(records, tmp, plan)
            self.assertEqual(rows[0]["completed"], 1)
            self.assertEqual(rows[0]["errors"], 1)
            self.assertEqual(rows[0]["pending"], 1)
            self.assertEqual(rows[0]["success_rate"], 0.)

    def test_resume_skips_verified_trials_but_rejects_damaged_video(self):
        plan = make_plan(["view_5"], [1], {"H50": {}}, {"scope": "test fixture, not policy performance"})
        calls = []
        def callback(case, directory):
            calls.append(case["id"])
            video = directory / "fixture.mp4"
            video.write_bytes(b"not a real video; filesystem contract fixture only")
            return {"success": True, "video_paths": [str(video)]}
        with tempfile.TemporaryDirectory() as tmp:
            run_plan(tmp, plan, {"H50": callback})
            run_plan(tmp, plan, {"H50": callback})
            self.assertEqual(len(calls), 1)
            video = next(Path(tmp).rglob("fixture.mp4"))
            video.write_bytes(b"damaged")
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"H50": callback})

    def test_interrupted_trial_is_retried_with_new_attempt(self):
        plan = make_plan(["v"], [1], {"H50": {}}, {})
        def interrupted(case, directory):
            (directory / "partial.log").write_text("some real progress would go here")
            raise RuntimeError("connection lost")
        def finished(case, directory):
            video = directory / "fixture.mp4"
            video.write_bytes(b"filesystem fixture, no real rollout")
            return {"success": False, "video_paths": [str(video)]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"H50": interrupted})
            status = Path(tmp) / "trials" / f"{plan['cases'][0]['id']}.json"
            self.assertEqual(json.loads(status.read_text())["status"], "error")
            run_plan(tmp, plan, {"H50": finished})
            self.assertEqual(json.loads(status.read_text())["attempt"], 2)
            self.assertEqual(len(list(Path(tmp).rglob("partial.log"))), 1)

    def test_corrector_cannot_silently_fall_back_to_baseline(self):
        plan = make_plan(["v"], [1], {"event_replan": {"uses_corrector": True}}, {})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {})
            with self.assertRaises(RuntimeError):
                run_plan(tmp, plan, {"event_replan": lambda *_: None})
            self.assertFalse((Path(tmp) / "plan.json").exists())

    def test_duplicate_records_rejected(self):
        plan = make_plan(["v"], [1], {"H50": {}}, {})
        record = {"case": plan["cases"][0], "status": "running"}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            summarize([record, record], tmp, plan)

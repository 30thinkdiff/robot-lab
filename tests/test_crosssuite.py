"""CPU checks use the real suite inventory, without importing a GPU stack."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('crosssuite', ROOT / 'scripts/corrector_crosssuite_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inventory():
    # Tiny synthetic metadata are for unit tests only. Deployment uses the
    # separately collected real native-suite inventory and verifies every asset.
    return {name: {'horizon': cap, 'tasks': [dict(task_id=i, name=name + '_' + str(i),
        language=name + ' ' + str(i), init_count=50, bddl_exists=True, init_exists=True)
        for i in range(90 if name == 'libero_90' else 10)]} for name, cap in module.CAPS.items()}


class CrosssuiteTests(unittest.TestCase):
    def test_declared_counts_and_four_way_pairing(self):
        wanted, _ = module.schedule(inventory())
        self.assertEqual(len(wanted), 880)
        self.assertEqual(len(set((c, s, m) for c, s, m, p in wanted)), 880)
        self.assertEqual([sum(p == phase for _, _, _, p in wanted) for phase in ('primary', 'ninety_subset', 'extra_init')], [480, 160, 240])
        for i in range(0, len(wanted), 4):
            group = wanted[i:i + 4]
            self.assertEqual(tuple(x[2] for x in group), module.METHODS)
            self.assertEqual(len(set((x[0], x[1]) for x in group)), 1)

    def test_early_prefix_covers_all_primary_tasks(self):
        wanted, c = module.schedule(inventory())
        first = wanted[:120]
        self.assertEqual({c[x[0]]['suite'] for x in first}, set(module.PRIMARY))
        self.assertEqual(len(set(x[0] for x in first)), 30)
        self.assertFalse(any('libero_object/' in x[0] for x in wanted))

    def test_object_overlap_stops_before_gpu(self):
        inv = inventory()
        inv['libero_goal']['tasks'][0]['name'] = inv['libero_object']['tasks'][0]['name']
        with self.assertRaises(ValueError):
            module.schedule(inv)

    def test_wrong_native_cap_or_asset_stops(self):
        for attribute, value in [('horizon', 280), ('missing', False)]:
            inv = inventory()
            if attribute == 'horizon':
                inv['libero_10']['horizon'] = value
            else:
                inv['libero_10']['tasks'][0]['init_exists'] = value
            with self.assertRaises(ValueError):
                module.schedule(inv)

    def test_config_instance_only_and_long_cap(self):
        cfg = SimpleNamespace(task='libero_object', task_ids=[3], episode_length=280, fps=20,
                              is_libero_plus=False, control_mode='relative', hard_reset=False, init_states=True)
        original_type = type(cfg)
        case = {'config': {'task_id': 3, 'suite': 'libero_10', 'episode_length': 520}}
        self.assertIs(module.configure_instance(cfg, case), cfg)
        self.assertEqual(type(cfg), original_type)
        self.assertEqual((cfg.task, cfg.episode_length), ('libero_10', 520))
        self.assertEqual(cfg.fps, 20)
        with self.assertRaises(ValueError):
            module.configure_instance(cfg, case)

    def test_bridge_separate_from_formal(self):
        wanted, conditions = module.schedule(inventory(), bridge=True)
        self.assertEqual(len(wanted), 4)
        self.assertEqual(set(conditions), {'libero_object/task_1/init_0'})
        self.assertTrue(all(x[1] == 1001 and x[3] == 'bridge' for x in wanted))

    def test_init_receipt_reads_selected_index_before_native_increment(self):
        raw = SimpleNamespace(init_state_id=0, _reset_stride=1, _init_states=list(range(50)))
        output = object()
        calls = []
        def reset(seed):
            calls.append((raw.init_state_id, seed))
            raw.init_state_id += raw._reset_stride
            return output
        actual, receipt = module.reset_with_init_receipt(raw, reset, seed=1000)
        self.assertIs(actual, output)
        self.assertEqual(calls, [(0, 1000)])
        self.assertEqual(receipt['init'], 0)
        self.assertEqual(receipt['init_cursor_after_reset'], 1)
        raw.init_state_id = 50
        _, receipt = module.reset_with_init_receipt(raw, reset, seed=1001)
        self.assertEqual(receipt['init'], 0)
        self.assertEqual(receipt['init_cursor_after_reset'], 51)

    def test_materialization_hash_includes_native_suite_and_cap(self):
        batch_spec = importlib.util.spec_from_file_location('batch', ROOT / 'studies/policy_generalization/corrector/batch_evaluation.py')
        batch = importlib.util.module_from_spec(batch_spec)
        batch_spec.loader.exec_module(batch)
        wanted, conditions = module.schedule(inventory())
        methods = {m: {'uses_corrector': m.startswith('mlp_'), 'corrector_sha256': 'f' * 64 if m.startswith('mlp_') else None} for m in module.METHODS}
        plan = module.materialize(batch.make_plan, wanted, conditions, methods, {'scope': 'test'})
        again = module.materialize(batch.make_plan, wanted, conditions, methods, {'scope': 'test'})
        self.assertEqual(plan, again)
        self.assertEqual(len(set(x['id'] for x in plan['cases'])), 880)
        for c in plan['cases']:
            self.assertEqual(c['config']['episode_length'], module.CAPS[c['config']['suite']])
            self.assertEqual(c['config']['uses_corrector'], c['method'].startswith('mlp_'))

    def test_partial_quartet_not_scored_as_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'trials').mkdir()
            for i, method in enumerate(module.METHODS[:3]):
                module.write(root / 'trials' / (str(i) + '.json'), {
                    'case': {'config': {'suite': 'libero_goal', 'phase': 'primary'},
                             'condition': 'libero_goal/task_0/init_0', 'seed': 1000, 'method': method},
                    'status': 'completed', 'result': {'success': i == 0}})
            result = module.summarize(root)
            self.assertEqual(result['completed'], 3)
            self.assertEqual(result['complete_quartets'], 0)
            self.assertEqual(result['partial_quartets'], 1)
            self.assertTrue(all(x['paired_success_rate'] is None for x in result['by_suite_phase']))

    def test_replay_cap_uses_declared_native_suite(self):
        import_spec = importlib.util.spec_from_file_location('casebank_import', ROOT / 'scripts/corrector_casebank_import.py')
        importer = importlib.util.module_from_spec(import_spec)
        import_spec.loader.exec_module(importer)
        case = {'condition':'libero_10/task_0/init_0','config':{'suite':'libero_10','episode_length':520}}
        self.assertEqual(importer.declared_episode_cap(case, {'horizon':None}), 520)
        self.assertEqual(importer.declared_episode_cap({'config':{}}, {}), 280)
        case['config']['episode_length'] = 280
        with self.assertRaises(ValueError):
            importer.declared_episode_cap(case, {})


@unittest.skipUnless(list((ROOT / 'outputs/autodl_setup/crosssuite_20261117').glob('audited_*.json')),
                     'read-only real transfer audit is not distributed with public code')
class RealCrosssuiteAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from corrector_crosssuite_analysis import analyse
        cls.analyse = staticmethod(analyse)
        path = sorted((ROOT / 'outputs/autodl_setup/crosssuite_20261117').glob('audited_*.json'))[-1]
        cls.data = json.loads(path.read_text())
        cls.summary, cls.tables = analyse(cls.data)

    def test_real_complete_paired_matrix_and_main_denominators(self):
        self.assertEqual(self.summary['formal_trials'], 880)
        self.assertEqual(self.summary['paired_quartets'], 220)
        self.assertEqual(self.summary['task_init_conditions'], 110)
        self.assertEqual(self.summary['tasks_per_suite'], {k: 10 for k in module.PRIMARY + ('libero_90',)})
        self.assertEqual(self.summary['overall_main_descriptive']['H20'], {
            'all_inits_successes': 138, 'all_inits_n': 180, 'primary_successes': 93, 'primary_n': 120})

    def test_real_comparison_counts_match_method_totals(self):
        methods = {(r['cohort'], r['suite'], r['method']): r for r in self.tables['methods']}
        for r in self.tables['comparisons']:
            base = methods[r['cohort'], r['suite'], r['baseline']]
            candidate = methods[r['cohort'], r['suite'], r['candidate']]
            self.assertEqual(candidate['successes'] - base['successes'], r['candidate_only_success'] - r['baseline_only_success'])
            self.assertEqual(r['matched'], r['both_success'] + r['both_failure'] + r['candidate_only_success'] + r['baseline_only_success'])

    def test_real_coverage_excludes_warmup_and_guide_updates_present(self):
        self.assertEqual(self.summary['coverage'], {'scores': 106501, 'eligible_steps': 106501,
                                                   'max_gap_steps': 0, 'warmup_excluded_steps': 24})
        self.assertEqual(self.summary['ogg']['guidance_calls'], 3848)
        self.assertEqual(self.summary['ogg']['nonzero_gradient_calls'], 3848)
        self.assertTrue(all(r['logged_post_reset_cursor'] == r['init'] + 1 for r in self.tables['trials']))

    def test_duplicate_or_incomplete_trial_is_rejected(self):
        import copy
        data = copy.deepcopy(self.data)
        data['transfer']['trials'][-1] = data['transfer']['trials'][0]
        with self.assertRaises(ValueError):
            self.analyse(data)
        data['transfer']['trials'].pop()
        with self.assertRaises(ValueError):
            self.analyse(data)

    def test_changed_pair_signature_is_rejected(self):
        import copy
        data = copy.deepcopy(self.data)
        data['transfer']['trials'][0]['first']['noise_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.analyse(data)


if __name__ == '__main__':
    unittest.main()

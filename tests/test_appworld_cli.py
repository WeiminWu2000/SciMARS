"""Public CLI boundaries: no paid dry-runs, no overwrites, grade after commit."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mah2.appworld_cli import main


class AppWorldCLITests(unittest.TestCase):
    def args(self, root):
        task = root/'world/data/tasks/abc_1'
        task.mkdir(parents=True, exist_ok=True)
        (task/'specs.json').write_text(json.dumps({'instruction': 'Inspect public state.'}))
        return ['--task', 'abc_1', '--appworld-root', str(root/'world'),
                '--appworld-python', sys.executable, '--output', str(root/'run')]

    def test_dry_run_needs_no_credentials_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            with patch('mah2.gateway.Gateway', side_effect=AssertionError('No calls')), contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(args), 0)
            self.assertFalse(json.loads(stream.getvalue())['paid_calls'])
            self.assertFalse((root/'run').exists())

    def test_rejects_existing_output_invalid_budget_and_task_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            for suffix in (['--budget', '2'], ['--task', '../abc_1']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(args + suffix)
            (root/'run').mkdir()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_official_grade_follows_commit_and_failure_keeps_artifacts(self):
        class Gateway:
            charged = .01
            calls = []
            def __init__(self, *args, **kwargs): pass
            def set_stage(self, *args): pass
        class Team:
            def __init__(self, *args, **kwargs): pass
            def run(self): return {'runnable': True, 'completed': True, 'task_id': 'abc_1', 'experiment': 'fixture'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root) + ['--method', 'base', '--execute', '--grade']
            def grade(outcome, destination):
                self.assertTrue((destination/'commit.json').is_file())
                self.assertIsNone(json.loads((destination/'commit.json').read_text())['official_success'])
                raise RuntimeError('fixture grader failure')
            with patch('mah2.gateway.Gateway', Gateway), patch('mah2.gateway.load_api_key', return_value='fixture'), patch('mah2.appworld_adapter.Team', Team), patch('mah2.appworld_adapter.grade_selected', side_effect=grade), patch.dict('os.environ'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
            result = json.loads((root/'run/result.json').read_text())
            self.assertIn('fixture grader failure', result['error'])
            self.assertIsNone(result['official_success'])
            self.assertEqual(result['charged_or_reserved_usd'], .01)


    def test_settings_apply_fixes_memory_shares_and_store(self):
        from mah2.appworld_cli import settings, METHODS
        config = json.loads((Path(__file__).resolve().parents[1]/'config/appworld.json').read_text())
        for method in METHODS:
            c = settings(config, method, memory_store='/store')
            self.assertTrue(c['completion_contract'] and c['dsstar']['completion_contract'])
            self.assertTrue(c['dsstar']['prefer_completed_selection'])
            self.assertEqual(c['execution_output_tokens'], 4096)
            self.assertEqual('memory_store' in c, method.startswith('memory'))
            self.assertEqual('proposal_c' in c['dsstar']['stage_shares'], method == 'memory_harness2')
            self.assertEqual(c['outer_cycles'], 2)
            self.assertEqual(c['max_team_steps'], 8)
        paper = settings(config, 'harness2', paper_settings=True)
        self.assertNotIn('completion_contract', paper)
        self.assertNotIn('completion_contract', paper['dsstar'])
        self.assertNotIn('completion_contract', config['dsstar'])  # input not mutated

    def test_dry_run_for_every_method(self):
        for method in ('base', 'harness2', 'integrate', 'mass', 'memory', 'memory_harness2'):
            with tempfile.TemporaryDirectory() as tmp:
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(main(self.args(Path(tmp)) + ['--method', method]), 0)
                report = json.loads(stream.getvalue())
                self.assertEqual(report['method'], 'integrate' if method == 'mass' else method)

    def test_memory_store_only_for_memory_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'store').mkdir(); (root/'store/graph.jsonl').write_text('')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(self.args(root) + ['--method', 'harness2', '--memory-store', str(root/'store')])
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main(self.args(root) + ['--method', 'memory', '--memory-store', str(root/'store')])
            self.assertEqual(json.loads(stream.getvalue())['memory_store'], str((root/'store').resolve()))

    def test_each_method_dispatches_to_its_executor(self):
        from mah2 import appworld_cli
        from mah2.appworld_optimizer import AppWorldExperiment
        outcome = {'runnable': True, 'completed': True, 'task_id': 'abc_1', 'experiment': 'x'}
        seen = {}
        def recursive(config, root, method, packet, data, library, depth, cls):
            seen['cls'] = cls; seen['shares'] = config['stage_shares']; seen['depth'] = depth
            (root/'successor').mkdir(parents=True)
            (root/'successor/outcome.json').write_text(json.dumps(outcome))
            (root/'commit.json').write_text('{}')
        class Gateway:
            charged = 0; calls = []
            def set_stage(self, *a): pass
        config = json.loads((Path(__file__).resolve().parents[1]/'config/appworld.json').read_text())
        for method in ('harness2', 'memory_harness2'):
            with tempfile.TemporaryDirectory() as tmp, patch('mah2.dsstar.recursion.run_recursive', recursive):
                c = appworld_cli.settings(config, method)
                got, extra = appworld_cli.run_method(method, Gateway(), c, Path(tmp), 'abc_1', {'task_inst': 't'}, 1.0)
            self.assertEqual(got, outcome)
            self.assertEqual(seen['depth'], 2)
            if method == 'harness2':
                self.assertIs(seen['cls'], AppWorldExperiment)
            else:
                self.assertIn('candidate_c', seen['cls'].SELECTABLE)
                self.assertIn('proposal_c', seen['shares'])
        teams = []
        class Team:
            def __init__(self, gateway, config, directory, task, harness=None, workflow=''):
                teams.append((type(self).__name__, config.get('memory_store'), workflow))
                self.directory = Path(directory)
            def run(self):
                self.directory.mkdir(parents=True)
                (self.directory/'trace.jsonl').write_text('')
                (self.directory/'outcome.json').write_text(json.dumps(outcome))
                return outcome
        class MemoryTeam(Team): pass
        with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.Team', Team), \
                patch('mah2.memory.team.appworld_team_class', return_value=MemoryTeam):
            appworld_cli.run_method('base', Gateway(), appworld_cli.settings(config, 'base'), Path(tmp)/'b', 'abc_1', {}, 1.0)
            appworld_cli.run_method('memory', Gateway(), appworld_cli.settings(config, 'memory', memory_store='/s'),
                                    Path(tmp)/'m', 'abc_1', {}, 1.0)
        self.assertEqual(teams, [('Team', None, ''), ('MemoryTeam', '/s', '')])
        def mass(gateway, config, packet, output, search, seed, budget, episode_adapter, footer, design_override):
            directory = search/'episode_0'
            episode_adapter(gateway, None, packet, output, directory, config, 'WORKFLOW TEXT')
            return {'selected': str(directory)}
        teams.clear()
        with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.Team', Team), patch('mah2.mass_rhi.run', mass):
            got, extra = appworld_cli.run_method('integrate', Gateway(), appworld_cli.settings(config, 'integrate'),
                                                 Path(tmp), 'abc_1', {'task_inst': 't'}, 1.0)
        self.assertEqual(got, outcome)
        self.assertEqual(teams, [('Team', None, 'WORKFLOW TEXT')])

if __name__ == '__main__':
    unittest.main()

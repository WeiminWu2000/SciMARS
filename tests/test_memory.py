"""Memory-graph methods for the DS-STAR (ScienceAgentBench) and AppWorld teams, no model calls.

The rollout/experiment tests execute real sandboxed Python (bubblewrap + a science
environment); set MAH2_TEST_SCIENCE to a venv with pandas, or they are skipped.
"""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from mah2.memory import team as tm
from mah2.memory import policy as mp

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = str(Path(os.environ.get('MAH2_TEST_SCIENCE', '/projects/p32327/Multi_agent_harness2_glm/.venv-science')).resolve())
HAVE_SANDBOX = shutil.which('bwrap') and Path(SCIENCE, 'bin/python').exists()

CODE = ("import pandas as pd, os\nos.makedirs('pred_results', exist_ok=True)\n"
        "a = pd.read_csv('./data/a.csv'); a['double'] = a['value'] * 2\n"
        "a.to_csv('pred_results/out.csv', index=False); print(a.shape)\n")


def note(kind, body, target, rel='derived_from'):
    return {'type': kind, 'body': body, 'links': [{'rel': rel, 'target': target}]}


def team_reply(prompt, role):
    """Scripted DS-STAR roles; each also writes a memory note."""
    if 'Analyzer boundary' in prompt:
        return ("```python\nprint(open('./data/a.csv').read())\n```\n<memory>" +
                json.dumps([note('interpretation', 'a.csv has columns id,value', 'context/data/a.csv')]) + '</memory>')
    if role == 'planner':
        return 'Double the value column and write pred_results/out.csv.<memory>' + json.dumps(
            [note('hypothesis', 'doubling value is the requested transform', 'context/data/a.csv'),
             {'type': 'finding', 'body': 'ungrounded claim', 'links': []}]) + '</memory>'
    if role == 'verifier':
        return 'Yes, the output has the required columns.<memory>' + json.dumps(
            [note('finding', 'output verified', 'context/data/a.csv', 'supports')]) + '</memory>'
    if role == 'router':
        return 'Add Step'
    return '```python\n' + CODE + '```\n<memory>[{"type": "hypothesis", "body": "no extra assumptions", ' \
                                  '"links": [{"rel": "relates_to", "target": "context/data/a.csv"}]}]</memory>'


def config():
    c = json.loads((ROOT / 'config/study.json').read_text())['dsstar']
    c.update(science_environment=SCIENCE, allowance_usd=1.0, input_per_million=0, output_per_million=0,
             public_input_reads=False, public_probes=False, require_public_checks=False, public_selection=False,
             rollout_seconds=600, execution_seconds=60, scoped_glm_coordination=False)
    c['stage_shares'] = dict(tm.MEMORY_SHARES)
    return c


class UnitTests(unittest.TestCase):
    def test_notes_are_removed_before_use(self):
        clean, notes, error = tm.split_notes('Add Step\n<memory>[{"type":"finding"}]</memory>')
        self.assertEqual((clean, error), ('Add Step', None))
        self.assertEqual(notes, [{'type': 'finding'}])
        clean, notes, error = tm.split_notes('Yes<memory>not json</memory>')
        self.assertEqual(clean, 'Yes')
        self.assertIsNone(notes)
        self.assertIn('not valid JSON', error)
        self.assertEqual(tm.split_notes('```python\nx=1\n```'), ('```python\nx=1\n```', None, None))

    def test_grounding_and_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data'
            (data / 'sub').mkdir(parents=True)
            (data / 'a.csv').write_text('id,value\n1,2\n')
            (data / 'blob.bin').write_bytes(b'\x00\x01' * 10)
            policy = tm.initial('science')['memory']
            m = tm.TeamMemory(Path(tmp) / 'graph', policy, 'double the value')
            m.seed_directory(data)
            self.assertEqual(m.graph.nodes['context/data/blob.bin']['type'], 'dataset')
            self.assertIn('[binary]', m.graph.nodes['context/data/blob.bin']['body'])
            result = m.ingest('analyzer', [note('interpretation', 'value is numeric', 'context/data/a.csv'),
                                           note('finding', 'bad', 'context/missing'),
                                           {'type': 'dataset', 'body': 'agents cannot add datasets'}])
            self.assertEqual(len(result['written']), 1)
            self.assertEqual(len(result['errors']), 2)
            self.assertEqual(m.graph.nodes[result['written'][0]['key']]['author'], 'analyzer')
            ctx = m.context('planner')  # subscribed to finding/open_question; reads interpretation
            self.assertIn(result['written'][0]['key'], [n['key'] for n in ctx['memory_routed']])
            self.assertEqual(ctx['memory_duty'], policy['write']['planner'])
            m.observe('e0003', 'stdout', source='sha')
            with self.assertRaises(ValueError):
                m.graph.add(key='log/exec/e0003', type='observation', body='edit', source='x', author='coder')

    def test_scopes_of_three_editors_and_composer(self):
        exp = tm.MemoryHarness2Mixin()
        base = tm.initial('science')
        roles = copy.deepcopy(base)
        roles['roles']['coder'] = 'check columns'
        memory = copy.deepcopy(base)
        memory['memory']['read']['coder']['k'] = 3
        memory['memory']['version'] = 2
        exp.validate_harness(roles, base, 'composer', 'proposal_a')
        exp.validate_harness(memory, base, 'composer', 'proposal_c')
        with self.assertRaises(ValueError):
            exp.validate_harness(memory, base, 'composer', 'proposal_a')
        with self.assertRaises(ValueError):
            exp.validate_harness(roles, base, 'composer', 'proposal_c')
        unversioned = copy.deepcopy(memory)
        unversioned['memory']['version'] = 1
        with self.assertRaises(ValueError):
            exp.validate_harness(unversioned, base, 'composer', 'proposal_c')
        exp._candidate_c = memory
        both = copy.deepcopy(memory)
        both['roles']['coder'] = 'check columns'
        exp.validate_harness(both, base, 'composer', 'composer')
        invented = copy.deepcopy(memory)
        invented['memory']['read']['coder']['k'] = 5
        with self.assertRaises(ValueError):
            exp.validate_harness(invented, base, 'composer', 'composer')

    def test_appworld_memory_fields_reach_prompt_and_notes_are_ingested(self):
        MemoryTeam = tm.appworld_team_class()

        class Gateway:
            payload = None

            def request(self, payload):
                self.payload = payload
                return {'choices': [{'message': {'content': json.dumps({'plan': 'inspect', 'memory': [
                    note('experiment_proposal', 'list playlists next', 'log/exec/e0000')]})}}]}
        with tempfile.TemporaryDirectory() as tmp:
            g = Gateway()
            team = MemoryTeam(g, {}, Path(tmp) / 'run', 'fixture')
            team.info = {'instruction': 'make a playlist'}
            team.event(kind='execution', code='print(1)', output='1', completed=False)
            reply = team.ask('planner', 'plan')
            self.assertEqual(reply, {'plan': 'inspect'})
            prompt = json.loads(g.payload['messages'][1]['content'])
            self.assertIn('memory_routed', prompt)
            self.assertIn('grounding rule', prompt['memory_format'])
            self.assertIn('context/task', team.memory.graph.nodes)
            written = [n for n in team.memory.graph.nodes.values() if n['author'] == 'planner']
            self.assertEqual(len(written), 1)
            events = [json.loads(x) for x in (Path(tmp) / 'run/trace.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['kind'], 'memory')


@unittest.skipUnless(HAVE_SANDBOX, 'needs bwrap and a science environment')
class SandboxTests(unittest.TestCase):
    def data(self, tmp):
        data = Path(tmp) / 'data'
        data.mkdir()
        (data / 'a.csv').write_text('id,value\n1,10\n2,20\n')
        return data

    def test_memory_rollout_runs_and_strips_notes(self):
        class Client:
            def complete(self, prompt, role=None):
                return team_reply(prompt, role)
        with tempfile.TemporaryDirectory() as tmp:
            data = self.data(tmp)
            r = tm.MemoryRollout(Client(), tm.initial('science'), config(), Path(tmp) / 'run', data)
            outcome = r.run({'task_inst': 'Double value into pred_results/out.csv', 'output_fname': 'pred_results/out.csv'})
            self.assertTrue(outcome['runnable'], outcome)
            self.assertNotIn('<memory>', (Path(tmp) / 'run/submission.py').read_text())
            summary = json.loads((Path(tmp) / 'run/memory_summary.json').read_text())
            self.assertGreater(summary['stats']['notes_accepted'], 2)
            self.assertGreater(summary['stats']['notes_rejected'], 0)
            self.assertGreater(summary['stats']['observations'], 1)
            kinds = [json.loads(x)['kind'] for x in (Path(tmp) / 'run/trace.jsonl').read_text().splitlines()]
            self.assertIn('memory', kinds)

    def test_memory_harness2_round_three_editors_and_composer(self):
        def editor(prompt, stage):
            view = json.JSONDecoder().raw_decode(prompt[prompt.index('EVIDENCE:\n') + 10:])[0]
            h = copy.deepcopy(view['base_harness'])
            cite = {'evidence': ['reference:e0001'], 'public_check': 'trace', 'expected_behavior': 'better'}
            if stage == 'proposal_a':
                h['roles']['coder'] = 'Check output columns before writing.'
                return json.dumps(dict(harness=h, **cite))
            if stage == 'proposal_b':
                return json.dumps({'harness': h, 'evidence': []})
            if stage == 'proposal_c':
                h['memory']['read']['verifier']['k'] = 8
                h['memory']['version'] += 1
                return json.dumps(dict(harness=h, **cite))
            c = view['candidates']['c']['harness']
            h['memory'] = c['memory']
            return json.dumps(dict(harness=h, evidence=['candidate_c:e0001'], public_check='trace',
                                   expected_behavior='better'))

        class Client:
            def __init__(self, c, budget, ledger, stage='plain'):
                self.stage = stage

            def complete(self, prompt, role=None):
                if self.stage in ('proposal_a', 'proposal_b', 'proposal_c', 'composer'):
                    return editor(prompt, self.stage)
                if self.stage == 'reflector':
                    return '{"experience": null}'
                return team_reply(prompt, role)
        with tempfile.TemporaryDirectory() as tmp:
            data = self.data(tmp)
            Exp = tm.science_experiment_class()
            exp = Exp(config(), Path(tmp) / 'round_1', client_factory=Client)
            commit = exp.run('whole_team', {'task_inst': 'Double value into pred_results/out.csv',
                                            'output_fname': 'pred_results/out.csv'}, data, [])
            root = Path(tmp) / 'round_1'
            decisions = {s: json.loads((root / f'{s}_decision.json').read_text())
                         for s in ('proposal_a', 'proposal_b', 'proposal_c', 'composer')}
            self.assertTrue(all(d['valid'] for d in decisions.values()), decisions)
            for label in ('reference', 'candidate_a', 'candidate_b', 'candidate_c'):
                self.assertTrue((root / label / 'memory_summary.json').exists())
            final = json.loads((root / 'selected_harness.json').read_text())
            self.assertEqual(final['memory']['read']['verifier']['k'], 8)
            self.assertEqual(final['memory']['version'], 2)
            self.assertTrue(commit['runnable'])
            mp.validate_memory(final['memory'])


if __name__ == '__main__':
    unittest.main()

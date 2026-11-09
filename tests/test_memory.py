"""AppWorld shared-memory tests with scripted replies and no model calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from mah2.memory import team as tm
from mah2.memory import policy as mp

def note(kind, body, target, rel='derived_from'):
    return {'type': kind, 'body': body, 'links': [{'rel': rel, 'target': target}]}


class UnitTests(unittest.TestCase):
    def test_grounding_and_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = tm.initial('appworld')['memory']
            m = tm.TeamMemory(Path(tmp) / 'graph', policy, 'playlist')
            m.seed_task('Create a playlist')
            self.assertEqual(m.graph.nodes['context/task']['type'], 'dataset')
            result = m.ingest('analyzer', [note('interpretation', 'playlist requested', 'context/task'),
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
        base = tm.initial('appworld')
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


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mah2.appworld_adapter import Team
from mah2.dsstar.harness import initial


class AppWorldAdapterTests(unittest.TestCase):
    def test_scoped_editor_patch_preserves_memory_and_passes_validation(self):
        from mah2.memory.team import initial as memory_initial, appworld_experiment_class
        from types import SimpleNamespace
        base=memory_initial('appworld')
        reply={'patch':{'coordination':{'version':2,'coder_receives_question':True}},
               'evidence':['reference:e0001'],'rationale':'Deliver the public question'}
        with tempfile.TemporaryDirectory() as tmp:
            cls=appworld_experiment_class(); obj=cls.__new__(cls)
            obj.root=Path(tmp);obj.c={'editor_component_replies':True}
            obj.client=lambda stage:SimpleNamespace(complete=lambda *a,**k:json.dumps(reply))
            ev={'base_harness':base,'runs':{'reference':{'outcome':{},'trace':[{'id':'e0001','kind':'execution'}]}}}
            candidate,record=obj.edit('proposal_b','',ev,'coordination')
            self.assertTrue(record['valid'],record)
            self.assertEqual(candidate['memory'],base['memory'])
            self.assertEqual(candidate['roles'],base['roles'])
            self.assertTrue(candidate['coordination']['coder_receives_question'])
            self.assertLess(len(json.dumps(reply)),len(json.dumps(base))/3)

    def test_editor_patch_rejects_scope_escape_and_unknown_fields(self):
        from mah2.appworld_optimizer import materialize_edit
        from mah2.memory.team import initial as memory_initial
        base=memory_initial('appworld')
        for patch in ({'memory':{'version':2}}, {'coordination':{'invented':True}}):
            with self.assertRaises(ValueError):
                materialize_edit({'patch':patch},base,'proposal_b')
        with self.assertRaises(ValueError):
            materialize_edit({'patch':{},'harness':base},base,'composer')
        self.assertEqual(materialize_edit({'patch':{}},base,'proposal_a'),base)

    def test_memory_patch_preserves_unedited_role_policy(self):
        from mah2.appworld_optimizer import materialize_edit
        from mah2.memory.team import initial as memory_initial
        from mah2.memory.policy import validate
        base=memory_initial('appworld')
        candidate=materialize_edit({'patch':{'memory':{'version':2,'read':{'planner':{'k':8}}}}},base,'proposal_c')
        validate(candidate,base,'memory')
        self.assertEqual(candidate['memory']['read']['coder'],base['memory']['read']['coder'])
        self.assertEqual(candidate['memory']['read']['planner']['k'],8)
        self.assertEqual(base['memory']['read']['planner']['k'],6)

    def test_mass_custom_workflow_survives_released_parser_and_revision(self):
        from mah2 import mass_rhi
        class Checked(Exception): pass
        class Gateway:
            charged=0.
        def check(cfg,tasks):
            query=mass_rhi.pipeline.ROOT/f'tasks/initial/query{tasks[0]}_ourTeam.txt'
            template=mass_rhi.rhi._read_query_template(query,
                start_marker=mass_rhi.rhi.DEFAULT_START_MARKER,
                end_marker='END OF EDITABLE TEAM WORKFLOW')
            self.assertEqual(template.task_text_for_eval,'public task')
            self.assertIn('inspect APIs',template.design_block)
            self.assertNotIn('fixed API restrictions',template.design_block)
            revised=mass_rhi.rhi._normalize_generated_design_block(
                template.design_block+'\nrepair handoffs\n'+template.after_design,
                start_marker=mass_rhi.rhi.DEFAULT_START_MARKER,
                end_marker='END OF EDITABLE TEAM WORKFLOW',after_design=template.after_design)
            self.assertIn('repair handoffs',revised)
            self.assertNotIn('fixed API restrictions',revised)
            raise Checked()
        with tempfile.TemporaryDirectory() as tmp, patch.object(mass_rhi.pipeline,'search',side_effect=check):
            with self.assertRaises(Checked):
                mass_rhi.run(Gateway(),{'model':'fixture','mass_rhi':{'iterations':2}},
                    {'task_inst':'public task'},Path(tmp),Path(tmp)/'search',1,1.,
                    footer='fixed API restrictions',design_override='inspect APIs')

    def test_manifest_has_disjoint_scenarios_and_balanced_categories(self):
        root=Path(__file__).resolve().parents[1]
        m=json.loads((root/'config/appworld_tasks.json').read_text())
        self.assertEqual(len(m['debug']),5)
        self.assertEqual(len(m['evaluation']),20)
        self.assertEqual(len({r['scenario'] for r in m['evaluation']}),20)
        self.assertFalse({r['scenario'] for r in m['debug']} & {r['scenario'] for r in m['evaluation']})
        for domain in ('music','finance'):
            self.assertEqual(sum(r['domain']==domain for r in m['evaluation']),10)
        self.assertTrue(all(r['split']=='test_normal' for r in m['evaluation']))
        self.assertTrue(all(r['split']=='dev' for r in m['debug']))

    def scripted_team(self, replies, config, memory=False):
        """Team over a scripted gateway and an in-memory world (no AppWorld, no API)."""
        class Gateway:
            def __init__(self):self.payloads=[]
            def request(self,payload):
                self.payloads.append(payload)
                return {'choices':[{'message':{'content':replies.pop(0)}}]}
        class World:
            executed=[]
            def __init__(self,*a,**k):self.info={'instruction':'fixture'}
            def execute(self,code):
                import ast;ast.parse(code)  # as the real World.execute
                World.executed.append(code)
                return {'output':'ok','completed':'complete_task' in code}
            def close(self):pass
        if memory:
            from mah2.memory.team import appworld_team_class
            cls=appworld_team_class()
        else:
            cls=Team
        return Gateway(),World,cls,config

    def test_syntax_error_in_code_reaches_debugger_instead_of_aborting(self):
        # Observed: SyntaxError from ast.parse ended whole rollouts (e.g. "unterminated string literal").
        replies=['{"code":"print(1)"}','{"plan":"p"}','{"code":"print(\'x)"}',
                 '{"code":"apis.supervisor.complete_task()"}']
        g,World,cls,config=self.scripted_team(replies,{'recoverable_step_errors':True,'max_team_steps':1})
        with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.World',World):
            outcome=cls(g,config,Path(tmp)/'run','fixture').run()
            trace=[json.loads(x) for x in (Path(tmp)/'run/trace.jsonl').read_text().splitlines()]
        self.assertTrue(outcome['completed'],outcome)
        self.assertIsNone(outcome['error'])
        rejected=[e for e in trace if e['kind']=='execution' and e.get('executed') is False]
        self.assertEqual(len(rejected),1)
        self.assertIn('SyntaxError',rejected[0]['output'])
        self.assertIn('debugger',[e.get('receiver') for e in trace if e['kind']=='handoff'])
        self.assertNotIn("print('x)",World.executed)

    def test_syntax_error_aborts_without_flag(self):
        replies=['{"code":"print(\'x)"}']
        g,World,cls,config=self.scripted_team(replies,{'max_team_steps':1})
        with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.World',World):
            outcome=cls(g,config,Path(tmp)/'run','fixture').run()
        self.assertIn('SyntaxError',outcome['error'])

    def test_truncated_reply_is_requested_again_once(self):
        # Observed: replies cut off at max_tokens raised JSONDecodeError and ended the rollout.
        for memory in (False,True):
            replies=['{"code":"print(1)"}','{"plan":"unterminated','{"plan":"p"}',
                     '{"code":"apis.supervisor.complete_task()"}']
            g,World,cls,config=self.scripted_team(replies,{'recoverable_step_errors':True,'max_team_steps':1},memory)
            with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.World',World):
                outcome=cls(g,config,Path(tmp)/'run','fixture').run()
                trace=[json.loads(x) for x in (Path(tmp)/'run/trace.jsonl').read_text().splitlines()]
            self.assertTrue(outcome['completed'],(memory,outcome))
            self.assertEqual(sum(e['kind']=='reply_retry' for e in trace),1)
            self.assertIn('not one complete JSON object',g.payloads[2]['messages'][-1]['content'])
            # A second consecutive failure still ends the rollout (bounded, paid retries).
            replies=['{"code":"print(1)"}','{"plan":"bad','{"plan":"bad again']
            g,World,cls,config=self.scripted_team(replies,{'recoverable_step_errors':True,'max_team_steps':1},memory)
            with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.World',World):
                outcome=cls(g,config,Path(tmp)/'run','fixture').run()
            self.assertIn('JSONDecodeError',outcome['error'])

    def test_completion_contract_only_with_flag(self):
        from mah2.appworld_adapter import system_prompt, SYSTEM
        self.assertEqual(system_prompt({}),SYSTEM)
        contract=system_prompt({'completion_contract':True})
        self.assertIn('complete_task() with no answer',contract)
        self.assertIn('never a sentence',contract)

    def test_evidence_view_keeps_final_complete_task_call(self):
        # Observed: 1400-char head excerpts hid the final complete_task call from the selector.
        from mah2.appworld_optimizer import AppWorldExperiment
        code='x = 1\n'*400+"apis.supervisor.complete_task(answer='long prose')"
        ev={'runs':{'reference':{'outcome':{},'trace':[{'id':'e0001','kind':'execution','code':code}]}}}
        obj=AppWorldExperiment.__new__(AppWorldExperiment)
        obj.c={};self.assertNotIn('complete_task',json.dumps(obj.view(ev)))
        obj.c={'evidence_head_tail':True}
        self.assertIn("complete_task(answer='long prose')",obj.view(ev)['runs']['reference']['trace'][0]['code'])

    def test_selection_prefers_completed_runs_and_retries_invalid_reply(self):
        # Observed: selectors committed runs that never called complete_task; no such run succeeded.
        from mah2.appworld_optimizer import AppWorldExperiment
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for label,completed in (('reference',False),('candidate_a',True),('candidate_b',True),('successor',False)):
                (root/label).mkdir()
                (root/label/'trace.jsonl').write_text(json.dumps({'id':'e0001','kind':'execution','code':'x','output':'y'})+'\n')
                (root/label/'outcome.json').write_text(json.dumps({'runnable':True,'completed':completed}))
            replies=['{"selected":"reference","evidence":["reference:e0001"]}',
                     '{"selected":"candidate_b","evidence":["candidate_b:e0001"]}']
            prompts=[]
            obj=AppWorldExperiment.__new__(AppWorldExperiment)
            obj.root=root;obj.c={'prefer_completed_selection':True}
            obj.client=lambda stage:SimpleNamespace(complete=lambda p,role=None:(prompts.append(p),replies.pop(0))[1])
            choices={k:{'h':k} for k in ('reference','candidate_a','candidate_b','successor')}
            fallback=('successor',choices['successor'],{'runnable':True,'completed':False})
            label,harness,outcome=obj.select({'task_inst':'t'},choices,fallback)
            self.assertEqual(label,'candidate_b')
            self.assertNotIn('"reference"',prompts[0].split('\n',1)[1].split('"runs"')[1][:20])
            self.assertIn('Correction required',prompts[1])
            # Exactly one completed candidate: committed without a selector call.
            (root/'candidate_b'/'outcome.json').write_text(json.dumps({'runnable':True,'completed':False}))
            obj.client=lambda stage:self.fail('selector must not be called')
            self.assertEqual(obj.select({'task_inst':'t'},choices,fallback)[0],'candidate_a')

    def test_world_start_timeout_fails_rollout_not_task(self):
        # Observed: TimeoutError in World.__init__ (outside Team.run's handling) killed 4 whole runs.
        for failures,expect_completed in ((1,True),(2,False)):
            replies=['{"code":"apis.supervisor.complete_task()"}']
            g,World,cls,config=self.scripted_team(replies,{'max_team_steps':1})
            starts=[]
            class SlowWorld(World):
                def __init__(self,*a,**k):
                    starts.append(1)
                    if len(starts)<=failures:raise TimeoutError('AppWorld worker response timed out')
                    super().__init__(*a,**k)
            with tempfile.TemporaryDirectory() as tmp, patch('mah2.appworld_adapter.World',SlowWorld):
                outcome=cls(g,config,Path(tmp)/'run','fixture').run()
            self.assertEqual(outcome['completed'],expect_completed)
            if not expect_completed:
                self.assertFalse(outcome['runnable'])
                self.assertIn('TimeoutError',outcome['error'])
                self.assertEqual(len(starts),2)

    def test_handoff_contract_and_public_history_reach_receiver(self):
        class Gateway:
            payload=None
            def request(self,payload):
                self.payload=payload
                return {'choices':[{'message':{'content':'{"plan":"inspect"}'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            h=initial();h['coordination']['version']=2
            h['coordination']['contracts']['verifier.planner']={
                'version':2,'fields':['verification'],'sender_instruction':'cite observations',
                'receiver_instruction':'repair the observed failure'}
            g=Gateway();team=Team(g,{},Path(tmp)/'run','fixture',h)
            team.info={'instruction':'fixture'}
            team.state['verification']='observed missing field'
            team.event(kind='execution',code='print(1)',output='1',completed=False)
            team.ask('planner','plan')
            prompt=json.loads(g.payload['messages'][1]['content'])
            self.assertEqual(prompt['handoff']['verification'],'observed missing field')
            self.assertIn('repair the observed failure',prompt['contracts'])
            self.assertEqual(prompt['recent_public_interactions'][0]['output'],'1')
            self.assertNotIn('grade',prompt)


if __name__=='__main__':unittest.main()

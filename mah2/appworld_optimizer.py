"""AppWorld evidence adapter over the canonical Harness² round and recursion."""
import copy
import json
from types import SimpleNamespace

from mah2.appworld_adapter import COMPLETION_CONTRACT, Team, parse_json, write_json
from mah2.dsstar.harness import validate
from mah2.dsstar.optimizer import Experiment, validate_hybrid_edit
from mah2.dsstar.provider import Budget


def materialize_edit(record, base, stage):
    """Expand scoped patches on the host, then apply the existing full validators.

    Dictionaries merge, lists/strings replace. New keys are permitted only inside
    coordination.contracts; the harness validator checks interface/field names.
    """
    if 'patch' not in record:
        return record['harness']
    if 'harness' in record:
        raise ValueError('Return patch or harness, not both')
    patch = record['patch']
    scopes = {'proposal_a': {'roles'}, 'proposal_b': {'coordination'},
              'proposal_c': {'memory'}, 'composer': set(base)}
    if not isinstance(patch, dict) or not set(patch) <= scopes.get(stage, set()):
        raise ValueError('Patch changes a component outside the editor scope')
    def merge(current, update, path=()):
        if isinstance(current, dict):
            if not isinstance(update, dict):
                raise ValueError('Object required at ' + '.'.join(path))
            result = copy.deepcopy(current)
            for key, value in update.items():
                if key not in result:
                    if path != ('coordination', 'contracts'):
                        raise ValueError('Unknown patch field: ' + '.'.join((*path, key)))
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = merge(result[key], value, (*path, key))
            return result
        return copy.deepcopy(update)
    return merge(base, patch)


class AppWorldExperiment(Experiment):
    task_domain = "AppWorld"
    gateway = None
    task_id = None

    RULES = {
        'proposal_a':'Change role procedures only. Preserve the complete coordination object.',
        'proposal_b':'Change exactly one existing coordination route/contract. Preserve all role procedures. Bump the coordination version for changed contracts.',
        'composer':'Keep, combine, revise or reject the proposals using their probe traces. Roles may be revised; coordination must exactly equal the base or candidate_b coordination.',
    }

    def client(self, stage):
        order = [s for s in self.STAGES if s in self.c['stage_shares']]
        prefix = sum(self.c['stage_shares'][s] for s in order[:order.index(stage)+1])
        cap = max(0, self.c['allowance_usd']*prefix-sum(b.charged for b in self.budgets.values()))
        b = Budget(cap)
        self.budgets[stage] = b
        g = self.gateway
        g.set_stage(self.root.name+'/'+stage, cap)
        start = g.charged
        def complete(prompt, role=None):
            try:
                response = g.request({'messages':[{'role':'user','content':prompt}],
                                      'max_tokens':4096,
                                      'response_format':{'type':'json_object'}})
                return response['choices'][0]['message'].get('content') or ''
            finally:
                b.charged = g.charged-start
        return SimpleNamespace(complete=complete)

    def run_team(self, stage, harness, packet, data):
        write_json(self.root/(stage+'_harness.json'), harness)
        self.client(stage)  # Assign the prefix allowance before executing the team.
        start = self.gateway.charged
        try:
            return Team(self.gateway, self.c, self.root/stage, self.task_id, harness).run()
        finally:
            self.budgets[stage].charged = self.gateway.charged-start

    def view(self, ev):
        # Head and tail: the decisive end of code/output (e.g. the complete_task call) survives.
        head_tail = self.c.get('evidence_head_tail')
        def cut(v):
            if not isinstance(v, str) or len(v) <= 1400:
                return v
            return v[:600]+' [...EXCERPT...] '+v[-800:] if head_tail else v[:1400]+' [EXCERPT]'
        view = {k:v for k,v in ev.items() if k != 'runs'}
        view['runs'] = {}
        for label, run in ev['runs'].items():
            # Remove prompts (they repeat the task) but keep addressable outputs.
            rows = []
            for e in run['trace']:
                if e['kind'] == 'handoff':
                    continue
                row = {k:cut(v) for k,v in e.items()}
                rows.append(row)
            view['runs'][label] = dict(outcome=run['outcome'], trace=rows[-18:])
        return view

    def edit(self, stage, prompt, ev, scope):
        client = self.client(stage)
        mission = (f'Improve the {self.task_domain} execution harness, not the task answer. No task-specific answers, '
                   'code solutions, IDs or account details in role procedures or contracts. '
                   'Roles: analyzer, planner, coder, verifier, router, debugger, finalizer. '
                   'The current harness schema must be preserved. A no-op is valid. '+self.scope_rule(stage, self.RULES.get(stage, ''))+
                   '\nCite observed run:event IDs for changes. Never claim official correctness. '
                   'Return {"harness": <complete harness>, "evidence": ["reference:e0001", ...], '
                   '"rationale":"mechanism and uncertainty"}. To inspect an event first return '
                   '{"read":{"stage":"reference","event":"e0001"}}.\n'+json.dumps(self.view(ev)))
        if self.c.get('editor_component_replies'):
            mission += ('\nOUTPUT OVERRIDE: return {"patch": <changed fields only>, '
                        '"evidence": ["run:event", ...], "rationale": "..."}. '
                        'Do not repeat unchanged roles or memory. Objects merge with the base; '
                        'lists and strings replace their fields. An empty patch is a no-op. '
                        'Only your permitted component may change. Increment component and '
                        'contract versions as required. To remove a contract, use the full '
                        'harness format instead. All original scope checks still apply.')
        (self.root/(stage+'_prompt.txt')).write_text(mission)
        valid_refs = {label+':'+e['id'] for label,r in ev['runs'].items() for e in r['trace']}
        record = {}
        for _ in range(4):
            try:
                record = parse_json(client.complete(mission, role=stage))
                if 'read' in record:
                    request = record['read'];label = request['stage']
                    if label not in ev['runs']:
                        raise ValueError('Only supplied runs may be inspected')
                    full = [json.loads(x) for x in (self.root/label/'trace.jsonl').read_text().splitlines()]
                    match = [e for e in full if e['id']==request['event']]
                    if len(match)!=1:raise ValueError('Unknown event')
                    mission += '\nRequested event: '+json.dumps(match[0]);continue
                harness = self.validate_harness(materialize_edit(record, ev['base_harness'], stage), ev['base_harness'],
                                                'coordination' if stage=='proposal_b' else 'composer', stage)
                validate_hybrid_edit(stage, harness, ev)
                if harness != ev['base_harness']:
                    refs = record.get('evidence', [])
                    if not refs or not set(refs)<=valid_refs:
                        raise ValueError('Changed harness requires valid trace citations')
                    if stage=='composer' and harness['coordination']!=ev['base_harness']['coordination']:
                        if not any(r.startswith('candidate_b:') for r in refs):
                            raise ValueError('Coordination adoption requires candidate_b probe evidence')
                record['valid'] = True;break
            except (ValueError, KeyError, TypeError) as exc:
                record = dict(valid=False,error=str(exc));mission += '\nCorrection required: '+str(exc)
            except Exception as exc:
                record = dict(valid=False,error=f'{type(exc).__name__}: {exc}');break
        if not record.get('valid'):
            record['valid'] = False;harness = copy.deepcopy(ev['base_harness'])
        record['harness'] = harness
        write_json(self.root/(stage+'_decision.json'),record)
        return harness,record

    def select(self, packet, choices, fallback):
        from mah2.dsstar.optimizer import evidence
        choices = {k:v for k,v in choices.items() if k in self.SELECTABLE}
        runs = {k:evidence(self.root/k) for k in choices}
        viable = {k:v for k,v in choices.items() if runs[k]['outcome']['runnable']}
        if not viable:return fallback
        guided = self.c.get('prefer_completed_selection')
        if guided:
            # Public rule: a run that never reached complete_task cannot have finished the task
            # (0 of 118 incomplete audited rollouts succeeded), so completed runs come first.
            completed = {k:v for k,v in viable.items() if runs[k]['outcome'].get('completed')}
            viable = completed or viable
            if fallback[0] not in viable:
                label = next(k for k in ('successor','reference','candidate_a','candidate_b','candidate_c') if k in viable)
                fallback = label,viable[label],runs[label]['outcome']
            if len(viable) == 1:
                write_json(self.root/'selection.json',dict(valid=True,selected=fallback[0],
                           rationale='only completed runnable candidate; no selector call'))
                return fallback
        prompt = (f'Select the best current-round {self.task_domain} execution using only public task requirements '
                  'and recorded observations. Completion claims and lack of errors are not proof. '
                  'No official tests or scores are available. Return {"selected":"label", '
                  '"evidence":["label:event"],"rationale":"specific observed requirements and uncertainty"}.\n'+
                  (COMPLETION_CONTRACT+' Check how each run called complete_task against this contract.\n'
                   if self.c.get('completion_contract') else '')+
                  json.dumps(self.view(dict(task=packet,runs={k:runs[k] for k in viable}))))
        (self.root/'selection_prompt.txt').write_text(prompt)
        client = self.client('selector')
        refs = {k+':'+e['id'] for k in viable for e in runs[k]['trace']}
        for attempt in range(2 if guided else 1):
            try:
                record = parse_json(client.complete(prompt,role='selector'))
                selected = record['selected']
                if selected not in viable or not record.get('evidence') or not set(record['evidence'])<=refs:
                    raise ValueError('Invalid selected candidate or citations; choose one of '
                                     + ', '.join(viable) + ' and cite label:event IDs shown above')
                record['valid'] = True
                result = selected,viable[selected],runs[selected]['outcome']
                break
            except (ValueError, KeyError, TypeError) as exc:
                record = dict(valid=False,error=f'{type(exc).__name__}: {exc}',fallback=fallback[0])
                result = fallback
                prompt += '\nCorrection required: '+str(exc)
            except Exception as exc:
                record = dict(valid=False,error=f'{type(exc).__name__}: {exc}',fallback=fallback[0])
                result = fallback
                break
        write_json(self.root/'selection.json',record)
        return result

    def reflect(self, ev, packet, data, library):
        prompt = (f'Extract one general lesson from the selected {self.task_domain} execution and its revisions. '
                  'Do not include task IDs, filenames, account details or answer values. '
                  'Return {"lesson":null} or {"lesson":{"cue":"...","procedure":"...",'
                  '"limitations":"...","evidence":["run:event"]}}.\n'+json.dumps(self.view(ev)))
        try:
            record = parse_json(self.client('reflector').complete(prompt,role='reflector'))
            lesson = record.get('lesson')
            refs = {k+':'+e['id'] for k,r in ev['runs'].items() for e in r['trace']}
            if lesson and set(lesson.get('evidence',[]))<=refs and lesson.get('evidence'):
                library.append(lesson)
            write_json(self.root/'reflection.json',record)
        except Exception as exc:
            write_json(self.root/'reflection.json',dict(error=f'{type(exc).__name__}: {exc}'))

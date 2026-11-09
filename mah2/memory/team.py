"""Shared memory and memory-policy adaptation for the AppWorld team.

Roles add grounded notes through a memory list in their JSON replies. The host
routes evidence under the harness policy. Each execution uses a fresh graph or
its own copy of a frozen development store; evaluation writes do not transfer.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from mah2.memory import policy as mp
from mah2.memory.graph import MemoryError_, MemoryGraph
from mah2.dsstar import harness as core_harness

MAX_NOTES = 2
MAX_NOTE_BODY = 400
NAMESPACE = {'interpretation': 'context', 'open_question': 'context', 'hypothesis': 'decisions',
             'experiment_proposal': 'decisions', 'recommendation': 'status', 'finding': 'work',
             'observation': 'work', 'dataset': 'context'}
AGENT_TYPES = ('interpretation', 'hypothesis', 'finding', 'open_question', 'recommendation',
               'experiment_proposal')

WRITE = {
    'appworld': {
        'analyzer': 'Record each API signature, response field and account fact you confirmed as an '
                    'interpretation derived_from the execution observation that showed it.',
        'planner': 'Record the remaining requirements and the next step as an experiment_proposal linked to the '
                   'observations it relies on.',
        'coder': 'Record each state change you made or confirmed as a finding derived_from its execution '
                 'observation.',
        'verifier': 'Record each requirement checked against observed state as a finding that supports or '
                    'contradicts the relevant observation.',
        'router': 'Record a finalize/continue decision as a recommendation linked to the finding it relies on.',
        'debugger': 'Record the root cause of each API error and the corrected usage as a finding derived_from '
                    'the failing execution observation.',
        'finalizer': 'Record the submitted answer or final state and its supporting evidence as a finding.',
    },
}

NOTE_FORMAT = ('Each note: {"type": one of ' + ', '.join(AGENT_TYPES) + ', "body": "concise reusable fact", '
               '"links": [{"rel": one of ' + ', '.join(mp.RELATIONS) + ', "target": "<existing key>"}]}. '
               'Every note must link to at least one existing key shown in memory (grounding rule); '
               f'at most {MAX_NOTES} short notes (at most {MAX_NOTE_BODY} characters each, longer text is cut). '
               'Record only facts later roles will need and cannot see in recent interactions. Notes are optional.')


def initial(profile: str) -> dict:
    h = mp.initial(memory=True)
    h['memory']['write'] = dict(WRITE[profile])
    if profile == 'appworld':
        # Persistent procedures (recommendation) and API findings are routed to every role.
        for read in h['memory']['read'].values():
            read['types'] = list(dict.fromkeys(read['types'] + ['recommendation', 'finding']))
    return h


def core(h):
    return {'roles': h['roles'], 'coordination': h['coordination']}


class TeamMemory:
    """One execution's graph plus routing and note intake under the harness memory policy."""

    def __init__(self, root: Path, policy: dict, query: str = '', base: Path | None = None):
        """`base`: a frozen persistent store (graph.jsonl) copied in read-only; this run adds a layer."""
        root = Path(root)
        if base is not None:
            root.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(Path(base) / 'graph.jsonl', root / 'graph.jsonl')
        self.graph = MemoryGraph(root)
        self.base_seq = self.graph.seq
        self.policy = policy
        self.query = query[:300]
        self.seen = {r: self.base_seq for r in mp.ROLES}  # feeds carry this run's notes only
        self.counter = 0
        self.stats = dict(notes_accepted=0, notes_rejected=0, observations=0)

    def seed_task(self, text: str):
        self.graph.add(key='context/task', type='dataset', body=text[:5000], author='host', source='task')

    def observe(self, event_id: str, body: str, source: str):
        key = f'log/exec/{event_id.lower()}'
        if key in self.graph.nodes:
            return
        self.graph.add(key=key, type='observation', body=body[:3000] or '(no output)', author='host',
                       source=source)
        self.stats['observations'] += 1

    def context(self, role: str, exclude=()) -> dict:
        """Routed and fed memory for `role`; `exclude` holds keys the prompt already shows verbatim."""
        read = self.policy['read'][role]
        exclude = set(exclude)
        query = ' '.join(x for x in (read['query'], role, self.query) if x)
        routed = [n for n in self.graph.search(query, read['k'] + len(exclude), read['types'], read['namespaces'],
                                               body=300) if n['key'] not in exclude][:read['k']]
        feed = self.graph.feed(self.seen.get(role, 0), self.policy['subscriptions'][role],
                               exclude_author=role, limit=3)
        feed = [n for n in feed if n['key'] not in {r['key'] for r in routed} | exclude]
        self.seen[role] = self.graph.seq
        for n in feed:
            n['body'] = n['body'][:300]
        return dict(memory_routed=routed, memory_feed=feed, memory_duty=self.policy['write'][role])

    def ingest(self, role: str, notes) -> dict:
        written, errors = [], []
        if notes is None:
            return dict(written=written, errors=errors)
        if not isinstance(notes, list):
            notes = [notes]
        for note in notes[:MAX_NOTES]:
            try:
                if not isinstance(note, dict):
                    raise MemoryError_('a note must be a JSON object')
                kind = note.get('type')
                if kind not in AGENT_TYPES:
                    raise MemoryError_(f'note type must be one of {AGENT_TYPES}')
                body = note.get('body')
                if not isinstance(body, str) or not body.strip():
                    raise MemoryError_('note body must be non-empty text')
                links = note.get('links') or note.get('edges') or []
                if not isinstance(links, list):
                    raise MemoryError_('links must be a list')
                self.counter += 1
                key = f'{NAMESPACE[kind]}/{role}-{self.counter:03d}'
                node = self.graph.add(key=key, type=kind, body=body[:MAX_NOTE_BODY], author=role,
                                      edges=[dict(rel=l.get('rel'), target=l.get('target')) if isinstance(l, dict)
                                             else {} for l in links])
                written.append(dict(key=node['key'], type=kind, edges=node['edges']))
                self.stats['notes_accepted'] += 1
            except MemoryError_ as exc:
                errors.append(str(exc))
                self.stats['notes_rejected'] += 1
        if len(notes) > MAX_NOTES:
            errors.append(f'only the first {MAX_NOTES} notes were considered')
        return dict(written=written, errors=errors)

    def summary(self, limit: int = 12) -> dict:
        nodes = [MemoryGraph.summary(n, 160) for n in self.graph.log[self.base_seq:]
                 if n['author'] != 'host'][-limit:]
        check = self.graph.check()
        check['orphans'] = len(check['orphans'])  # count only; keys are in the trace
        return dict(check=check, stats=self.stats, recent_agent_nodes=nodes)


# --------------------------------------------------------------- AppWorld team
def appworld_team_class():
    from mah2.appworld_adapter import Team, parse_json

    class MemoryTeam(Team):
        """AppWorld team with the shared memory graph; prompts gain memory fields."""

        def __init__(self, gateway, config, directory, task, harness=None, workflow=''):
            super().__init__(gateway, config, directory, task, harness or initial('appworld'), workflow)
            base = config.get('memory_store')
            self.memory = TeamMemory(self.root / 'memory', self.h['memory'], base=Path(base) if base else None)

        def event(self, **kw):
            rec = super().event(**kw)
            if kw.get('kind') == 'execution':
                self.memory.observe(rec['id'], f"completed={kw.get('completed')}\ncode:\n{kw.get('code', '')[-800:]}\n"
                                    f"output:\n{kw.get('output', '')[-2000:]}", source='execute')
            return rec

        def ask(self, role, instruction):
            if 'context/task' not in self.memory.graph.nodes:
                # The prompt already carries the full task; the node only grounds notes.
                self.memory.query = str(self.info.get('instruction', ''))[:300]
                self.memory.seed_task('Task instruction: ' + str(self.info.get('instruction', '')))
            fields = {
                'analyzer': (), 'planner': ('execution', 'plans', 'routing'),
                'coder': ('plans', 'code', 'execution'),
                'verifier': ('plans', 'code', 'execution'),
                'router': ('plans', 'execution'),
                'debugger': ('code', 'execution'),
                'finalizer': ('plans', 'execution', 'verification'),
            }[role]
            handoff = {k: self.state.get(k) for k in fields}
            coord = self.h['coordination']
            if ((role == 'planner' and coord['planner_receives_diagnosis']) or
                    (role == 'router' and coord['router_receives_diagnosis'])):
                handoff['verification'] = self.state['verification']
            instructions = []
            for edge, contract in coord['contracts'].items():
                sender, receiver = edge.split('.')
                if role == sender:
                    instructions.append(contract['sender_instruction'])
                if role == receiver:
                    handoff.update({k: self.state.get(k) for k in contract['fields']})
                    instructions.append(contract['receiver_instruction'])
            recent = [e for e in self.events if e['kind'] == 'execution'][-6:]
            ctx = self.memory.context(role, exclude=['context/task'] + [f"log/exec/{e['id'].lower()}" for e in recent])
            prompt = dict(role=role, task=self.info, instruction=instruction,
                          role_procedure=self.h['roles'][role], workflow=self.workflow,
                          handoff=handoff, contracts=instructions,
                          recent_public_interactions=[
                              dict(id=e['id'], code=e['code'], output=e['output'][-6000:]) for e in recent],
                          memory_routed=ctx['memory_routed'], memory_feed=ctx['memory_feed'],
                          memory_duty=ctx['memory_duty'],
                          memory_format=('Optionally add "memory" as the LAST key of your JSON reply, after the '
                                         'requested fields. ' + NOTE_FORMAT))
            self.event(kind='handoff', receiver=role, prompt=prompt)
            salvage = []

            def parse(text):
                try:
                    return parse_json(text)
                except (ValueError, json.JSONDecodeError):
                    # A reply cut off inside its trailing memory list keeps the requested fields.
                    cut = max(text.rfind(',"memory"'), text.rfind(', "memory"'))
                    if cut <= 0:
                        raise
                    salvage.append('reply truncated inside memory; notes dropped')
                    return parse_json(text[:cut] + '}')

            reply = self.request_reply(role, prompt, self.token_cap(role), parse)
            result = self.memory.ingest(role, reply.pop('memory', None))
            result['errors'].extend(salvage)
            self.event(kind='memory', role=role, memory_routed=[n['key'] for n in ctx['memory_routed']],
                       memory_feed=[n['key'] for n in ctx['memory_feed']], written=result['written'],
                       errors=result['errors'])
            return reply

        def run(self):
            outcome = super().run()
            (self.root / 'memory_summary.json').write_text(json.dumps(self.memory.summary(), indent=2))
            return outcome

    return MemoryTeam


# --------------------------------------------------------------- Harness² with a memory editor
MEMORY_SURFACE = ('\n# MEMORY POLICY SURFACE\nThe harness also has a "memory" object controlling the team\'s shared '
                  'memory graph: write (per-role recording duty text), read (per-role routed retrieval: node types, '
                  'namespaces, k in [0,12], query hint), subscriptions (per-role node types pushed when other roles '
                  'add them) and version (increment it whenever memory changes). Node types: '
                  + ', '.join(mp.NODE_TYPES) + '. Namespaces: ' + ', '.join(mp.NAMESPACES)
                  + '. The graph rules (ontology, grounding, immutability, host-logged execution observations) are '
                    'fixed. Trace events of kind "memory" show what each role received and wrote, and rejected notes; '
                    'evidence.memory_graphs summarizes each run\'s graph.')
SCOPES = {
    'proposal_a': 'Preserve the entire memory object exactly.',
    'proposal_b': 'Preserve the entire memory object exactly.',
    'proposal_c': ('MEMORY EDITOR: revise the memory policy only (write duties, read routing, subscriptions; '
                   'increment memory.version). Preserve every role instruction and the entire coordination object '
                   'exactly. Fix observed memory failures: missing or irrelevant routed notes, rejected notes, '
                   'unused memory, roles that lacked information another role recorded. A no-op is valid.'),
    'composer': ('Memory must exactly equal either the base memory or the candidate_c memory; adopting candidate_c '
                 'memory requires candidate_c evidence.'),
}
MEMORY_PROMPT = ('Diagnose how the team used its shared memory graph in the existing trace (memory events, rejected '
                 'notes, routed and missing information) and propose one measured revision of the memory policy. '
                 'A no-op is valid when memory use is already adequate.')
MEMORY_SHARES = {'reference': .22, 'proposal_a': .06, 'proposal_b': .06, 'proposal_c': .06, 'candidate_a': .11,
                 'candidate_b': .11, 'candidate_c': .11, 'composer': .08, 'successor': .08, 'selector': .08,
                 'reflector': .03}


class MemoryHarness2Mixin:
    STAGES = ('reference', 'proposal_a', 'proposal_b', 'proposal_c', 'candidate_a', 'candidate_b', 'candidate_c',
              'composer', 'successor', 'selector', 'reflector')
    SELECTABLE = ('reference', 'candidate_a', 'candidate_b', 'candidate_c', 'successor')
    memory_profile = 'appworld'

    def validate_harness(self, candidate, base, scope, stage):
        if set(candidate) != {'roles', 'coordination', 'memory'}:
            raise ValueError('harness keys must be roles, coordination, memory')
        result = core_harness.validate(core(candidate), core(base), scope)
        result['memory'] = mp.validate_memory(candidate['memory'], base['memory'])
        if stage in ('proposal_a', 'proposal_b') and result['memory'] != base['memory']:
            raise ValueError(f'{stage} must preserve the memory policy exactly')
        if stage == 'proposal_c' and core(result) != core(base):
            raise ValueError('Memory edit must preserve every role instruction and the coordination exactly')
        if stage == 'composer':
            c = getattr(self, '_candidate_c', None)
            allowed = [base['memory']] + ([c['memory']] if c else [])
            if result['memory'] not in allowed:
                raise ValueError('Composer memory must exactly equal the base memory or the candidate_c memory')
        return result

    def scope_rule(self, stage, rule):
        if stage == 'proposal_c':
            return SCOPES[stage] + MEMORY_SURFACE
        return (rule + ' ' + SCOPES.get(stage, '')).strip() + MEMORY_SURFACE

    def memory_graphs(self, labels):
        graphs = {}
        for label in labels:
            path = self.root / label / 'memory_summary.json'
            if path.exists():
                graphs[label] = json.loads(path.read_text())
        return graphs

    def run(self, method, packet, data, library, starting_harness=None, round_index=1, round_count=1,
            retained_candidates=None):
        from mah2.dsstar import optimizer as O
        self.data = Path(data).resolve()
        base = copy.deepcopy(starting_harness) if starting_harness is not None else initial(self.memory_profile)
        if 'memory' not in base:
            base['memory'] = initial(self.memory_profile)['memory']
        self._candidate_c = None
        self.run_team('reference', base, packet, data)
        ev = dict(method=method, round_index=round_index, round_count=round_count, task=packet, base_harness=base,
                  library=O.retrieve(library, packet['task_inst']),
                  runs={'reference': O.evidence(self.root / 'reference')})
        ev['memory_graphs'] = self.memory_graphs(['reference'])
        if self.c.get('public_input_reads'):
            ev['public_input_inventory'] = self.inspect_input({'path': '.'}, 'inventory', max_entries=80)
        if method == 'whole_team':
            prompts = [O.upstream.propose(O.BP, dim, 'ds_star', k=round_count, index=round_index,
                                          round_t=round_index if round_count > 1 else None) for dim in ('INV', 'DOM')]
            scopes = ['invariant', 'domain']
        else:
            prompts = ['Propose an evidence-linked repair within exactly one agent role. Preserve every interface. '
                       'If an interface is implicated, describe that dependency rather than changing it '
                       'unilaterally. A no-op is valid.', '']
            scopes = ['local', 'coordination']
        prompts[1] = ('Diagnose an observed coordination failure and propose one measured handoff or feedback-route '
                      'repair. Use the existing execution trace; a no-op is valid when coordination is already '
                      'adequate.')
        scopes[1] = 'coordination'
        a, da = self.edit('proposal_a', prompts[0], ev, scopes[0])
        b, db = self.edit('proposal_b', prompts[1], ev, scopes[1])
        m, dm = self.edit('proposal_c', MEMORY_PROMPT, ev, 'composer')
        self._candidate_c = m
        for label, harness in (('candidate_a', a), ('candidate_b', b), ('candidate_c', m)):
            self.run_team(label, harness, packet, data)
            ev['runs'][label] = O.evidence(self.root / label)
        ev['memory_graphs'] = self.memory_graphs(['reference', 'candidate_a', 'candidate_b', 'candidate_c'])
        ev['candidates'] = {'a': da, 'b': db, 'c': dm}
        prompt = ((O.upstream.compose_round(O.BP, 'ds_star', t=round_index, k=round_count) if round_count > 1
                   else O.upstream.aggregate(O.BP, 'ds_star')) if method == 'whole_team' else
                  'Recursively improve the CURRENT harness from its reference and three independent candidate probes. '
                  'KEEP_BASE preserves the current harness, not the original. Check producer-consumer compatibility '
                  'and cite probe evidence for every retained coordination or memory change. Retain, revise, merge '
                  'or reject candidates. Agreement is not scientific correctness. Preserve uncertainty. A no-op is '
                  'valid.')
        prompt += (' Three probes were run: candidate_a (roles), candidate_b (coordination), candidate_c (memory '
                   'policy). Coordination must equal the base or candidate_b coordination; memory must equal the base '
                   'or candidate_c memory, citing candidate_c evidence when adopted.')
        successor, decision = self.edit('composer', prompt, ev, 'composer')
        if successor['memory'] != base['memory'] and not any(
                r.startswith('candidate_c:') for r in decision.get('evidence', [])):
            successor = dict(successor, memory=copy.deepcopy(base['memory']))
            decision['memory_reverted'] = 'memory adoption lacked candidate_c evidence'
            (self.root / 'composer_decision.json').write_text(json.dumps(dict(decision, harness=successor), indent=2))
        tested = [('reference', base), ('candidate_a', a), ('candidate_b', b), ('candidate_c', m)]
        final = self.execute_successor(successor, packet, data, tested)
        selected, adopted = 'successor', successor
        if self.c.get('runnable_fallback') and not final['runnable']:
            for label, harness in tested:
                if ev['runs'][label]['outcome']['runnable']:
                    selected, adopted, final = label, harness, ev['runs'][label]['outcome']
                    break
        if self.c.get('public_selection'):
            choices = dict(reference=base, candidate_a=a, candidate_b=b, candidate_c=m, successor=successor)
            if (self.root / 'successor_reuse.json').exists():
                choices.pop('successor')
            selected, adopted, final = self.select(packet, choices, (selected, adopted, final))
        commit = dict(method=method, runnable=final['runnable'], official_success=None,
                      selected_stage=selected if final['runnable'] else None,
                      program_sha256=final.get('program_sha256'), fallback=selected if selected != 'successor' else 'none')
        if (self.root / 'successor_reuse.json').exists():
            commit['successor_reused_from'] = json.loads((self.root / 'successor_reuse.json').read_text())['source_stage']
        (self.root / 'selected_harness.json').write_text(json.dumps(adopted, indent=2))
        (self.root / 'commit.json').write_text(json.dumps(commit, indent=2))
        ev['runs']['successor'] = O.evidence(self.root / 'successor')
        if (self.root / 'successor_reuse.json').exists():
            ev['successor_reuse'] = json.loads((self.root / 'successor_reuse.json').read_text())
        ev['adopted_harness'] = adopted
        ev['selected_stage'] = selected
        if (self.root / 'selection.json').exists():
            ev['selection'] = json.loads((self.root / 'selection.json').read_text())
        ev['composition'] = decision
        try:
            self.reflect(ev, packet, data, library)
        except (ValueError, TypeError, KeyError) as exc:
            (self.root / 'reflection_error.txt').write_text(type(exc).__name__ + ': ' + str(exc))
        (self.root / 'library_after.json').write_text(json.dumps(library, indent=2))
        (self.root / 'costs.json').write_text(json.dumps({s: dict(cap=b.cap, charged=b.charged)
                                                          for s, b in self.budgets.items()}, indent=2))
        return commit


def appworld_experiment_class():
    from mah2.appworld_optimizer import AppWorldExperiment
    MemoryTeam = appworld_team_class()

    class AppWorldMemoryExperiment(MemoryHarness2Mixin, AppWorldExperiment):
        memory_profile = 'appworld'
        RULES = dict(AppWorldExperiment.RULES, proposal_c='')

        def run_team(self, stage, harness, packet, data):
            (self.root / (stage + '_harness.json')).write_text(json.dumps(harness, indent=2))
            self.client(stage)
            start = self.gateway.charged
            try:
                return MemoryTeam(self.gateway, self.c, self.root / stage, self.task_id, harness).run()
            finally:
                self.budgets[stage].charged = self.gateway.charged - start

    return AppWorldMemoryExperiment

"""Editable memory component of the team harness, and full-harness scope checks.

A memory-enabled harness is {"roles", "coordination", "memory"}. The memory
policy controls how each role uses the shared memory graph:

  write          per-role recording duty (what to record, which node types, which edges)
  read           per-role retrieval routed into its handoff: node types, namespaces, k, query hint
  subscriptions  per-role node types pushed to it when other roles add them (state routing)

The graph service itself (ontology, grounding rule, immutability, search) is
infrastructure and is not editable.
"""
from __future__ import annotations

import copy

from mah2.dsstar.harness import ROLES, initial as initial_core, validate as validate_core

# Closed epistemic ontology of the active context graph (arXiv:2607.13220).
NODE_TYPES = ('dataset', 'observation', 'interpretation', 'hypothesis', 'finding', 'open_question',
              'recommendation', 'experiment_proposal')
# Mycelium room namespaces.
NAMESPACES = ('context', 'work', 'decisions', 'status', 'procedures', 'log')
# Provenance edges (ACG) plus Mycelium typed relations.
RELATIONS = ('derived_from', 'supports', 'contradicts', 'generated_by', 'supersedes', 'depends_on',
             'part_of', 'relates_to')
MAX_K = 12


def initial_memory():
    return dict(
        version=1,
        write={
            'analyzer': 'Record data facts as interpretation nodes (derived_from the dataset nodes) and gaps as '
                        'open_question nodes under context/.',
            'planner': 'Record each batch strategy as an experiment_proposal and the modelling bet as a hypothesis '
                       'under decisions/, linked to the evidence they rely on.',
            'coder': 'Record each fit/validation result as an observation under work/ (source = script or output '
                     'path) and link it to the hypothesis or proposal it tests.',
            'verifier': 'Record each confirmed defect or check outcome as a finding that supports or contradicts '
                        'the node it checks.',
            'router': 'Record routing decisions as a recommendation under status/ when they change direction.',
            'debugger': 'Record each repaired defect as a finding under procedures/, derived_from the defect.',
            'finalizer': 'Record the delivered model and its validation as a finding under status/.',
        },
        read={r: dict(types=list(t), namespaces=[], k=6, query='') for r, t in {
            'analyzer': ('dataset', 'observation', 'open_question'),
            'planner': ('finding', 'hypothesis', 'observation', 'open_question', 'interpretation'),
            'coder': ('experiment_proposal', 'hypothesis', 'finding'),
            'verifier': ('observation', 'experiment_proposal', 'hypothesis'),
            'router': ('finding', 'recommendation', 'open_question'),
            'debugger': ('finding', 'observation'),
            'finalizer': ('finding', 'observation'),
        }.items()},
        subscriptions={
            'analyzer': ['observation'], 'planner': ['finding', 'open_question'],
            'coder': ['experiment_proposal', 'recommendation'], 'verifier': ['observation'],
            'router': ['finding'], 'debugger': ['finding'], 'finalizer': ['finding']})


def validate_memory(memory: dict, base: dict | None = None) -> dict:
    if not isinstance(memory, dict) or set(memory) != {'version', 'write', 'read', 'subscriptions'}:
        raise ValueError('memory must have exactly version, write, read, subscriptions')
    if type(memory['version']) is not int or memory['version'] < 1:
        raise ValueError('invalid memory version')
    for part in ('write', 'read', 'subscriptions'):
        if not isinstance(memory[part], dict) or set(memory[part]) != set(ROLES):
            raise ValueError(f'memory.{part} must define every role')
    for role in ROLES:
        if not isinstance(memory['write'][role], str) or len(memory['write'][role]) > 2000:
            raise ValueError('memory.write entries must be strings of at most 2000 characters')
        read = memory['read'][role]
        if not isinstance(read, dict) or set(read) != {'types', 'namespaces', 'k', 'query'}:
            raise ValueError('memory.read entries need types, namespaces, k, query')
        if not set(read['types']) <= set(NODE_TYPES) or not set(read['namespaces']) <= set(NAMESPACES):
            raise ValueError(f'unknown node type or namespace; types {NODE_TYPES}, namespaces {NAMESPACES}')
        if type(read['k']) is not int or not 0 <= read['k'] <= MAX_K:
            raise ValueError(f'memory.read k must be an integer in [0, {MAX_K}]')
        if not isinstance(read['query'], str) or len(read['query']) > 500:
            raise ValueError('memory.read query must be a string of at most 500 characters')
        if not isinstance(memory['subscriptions'][role], list) or not set(memory['subscriptions'][role]) <= set(NODE_TYPES):
            raise ValueError('memory.subscriptions entries must be lists of node types')
    if base is not None and memory != base and memory['version'] <= base['version']:
        raise ValueError('memory version must increase when the memory policy changes')
    return copy.deepcopy(memory)


def initial(memory: bool = False) -> dict:
    h = initial_core()
    if memory:
        h['memory'] = initial_memory()
    return h


def core(h):
    return {'roles': h['roles'], 'coordination': h['coordination']}


def validate(candidate: dict, base: dict, scope: str = 'composer') -> dict:
    """Validate a (possibly memory-enabled) harness against `base` for an edit scope.

    scopes: roles, coordination, memory, integrated, composer.
    """
    has_memory = 'memory' in base
    if set(candidate) != ({'roles', 'coordination', 'memory'} if has_memory else {'roles', 'coordination'}):
        raise ValueError('harness keys must be ' + ('roles, coordination, memory' if has_memory else 'roles, coordination'))
    core_scope = 'coordination' if scope == 'coordination' else 'composer'
    result = validate_core(core(candidate), core(base), core_scope)
    if has_memory:
        result['memory'] = validate_memory(candidate['memory'], base['memory'])
    if scope == 'roles' and (result['coordination'] != base['coordination'] or
                             result.get('memory') != base.get('memory')):
        raise ValueError('Role edit must preserve coordination and memory exactly')
    if scope == 'coordination' and (result['roles'] != base['roles'] or result.get('memory') != base.get('memory')):
        raise ValueError('Coordination edit must preserve every role procedure and the memory policy exactly')
    if scope == 'memory' and (result['roles'] != base['roles'] or result['coordination'] != base['coordination']):
        raise ValueError('Memory edit must preserve roles and coordination exactly')
    return result

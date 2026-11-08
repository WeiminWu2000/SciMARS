"""Build a persistent AppWorld memory store from tasks disjoint from the evaluation set.

    python scripts/build_appworld_memory.py --output runs/appworld_store              # plan, no calls
    python scripts/build_appworld_memory.py --output runs/appworld_store --execute    # paid GLM calls
    mah2 appworld --method memory_harness2 --memory-store runs/appworld_store/store ...

The memory team runs each store task in turn, starting from the store built so far. The store grows in
two ways: it imports that run's agent notes (and the host nodes they cite) under a per-task key prefix,
and one neutral curator call distils at most four generic procedures, stored as `recommendation` nodes
under procedures/. No grader is called, so only public observations enter the store. Evaluation runs get
a read-only copy and write their own layer on top, so nothing passes between evaluation tasks.

Store tasks: config/appworld_tasks.json "memory_store" (scenario-disjoint from "evaluation").
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Neutral curator: it names no topic, so nothing learned from evaluation failures steers the store.
CURATOR = ('You maintain a shared procedural memory for an AppWorld agent team. Below are notes the team '
           'recorded while solving one task, and whether the run reached completion (completion is not '
           'correctness). Write at most four GENERIC procedures, supported by these notes, that would help on '
           'different future tasks. Never include names, IDs, amounts, dates, passwords or answers specific to '
           'this task. Return JSON {"procedures": [{"body": "...", "links": ["<note key>", ...]}]}; links must be '
           'keys from the notes. Return {"procedures": []} when nothing generic is supported.\n')


def prefixed(key, tag):
    namespace, rest = key.split('/', 1)
    return f'{namespace}/{tag}-{rest.replace("/", "-")}'[:len(namespace) + 1 + 64]


def merge(store, run_graph, start, tag):
    """Import agent notes of one run (and the host nodes they cite) into the store."""
    latest = {n['key']: n for n in run_graph.log[start:]}
    agent = [n for n in latest.values() if n['author'] not in ('host', 'curator')]
    needed = {e['target'] for n in agent for e in n['edges'] if e['target'] in latest}
    mapping = {}
    for node in sorted(latest.values(), key=lambda n: n['seq']):
        if node['author'] == 'host' and node['key'] not in needed:
            continue
        edges = [dict(rel=e['rel'], target=mapping.get(e['target'], e['target'])) for e in node['edges']]
        edges = [e for e in edges if e['target'] in store.nodes]
        if not edges and node['type'] not in ('dataset', 'observation'):
            continue
        key = prefixed(node['key'], tag)
        store.add(key=key, type=node['type'], body=node['body'], edges=edges,
                  source=node.get('source') or (tag if node['type'] in ('dataset', 'observation') else None),
                  author=node['author'], stage=tag)
        mapping[node['key']] = key
    return [mapping[n['key']] for n in agent if n['key'] in mapping]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True)
    p.add_argument('--budget-per-task', type=float, default=0.2)
    p.add_argument('--config', default=str(ROOT / 'config/appworld.json'))
    p.add_argument('--appworld-root', default=os.environ.get('APPWORLD_ROOT', str(ROOT / 'appworld_data')))
    p.add_argument('--appworld-python', default=os.environ.get('MAH2_APPWORLD_PYTHON', str(ROOT / '.venv-appworld/bin/python')))
    p.add_argument('--paper-settings', action='store_true', help='Build without the config "fixes" block')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    manifest = json.loads((ROOT / 'config/appworld_tasks.json').read_text())
    tasks = manifest['memory_store']['tasks']
    evaluation = {t['scenario'] for t in manifest['evaluation']}
    overlap = {t.split('_')[0] for t in tasks} & evaluation
    if overlap:
        raise SystemExit(f'store tasks share scenarios with evaluation: {sorted(overlap)}')
    print(f'{len(tasks)} store tasks: {tasks}')
    if not a.execute:
        print('Plan only; add --execute (GLM calls, at most $%.2f).' % (a.budget_per_task * len(tasks)))
        return 0
    os.environ['APPWORLD_ROOT'] = str(Path(a.appworld_root).resolve())
    os.environ['MAH2_APPWORLD_PYTHON'] = str(Path(a.appworld_python).absolute())
    from mah2.appworld_adapter import parse_json
    from mah2.appworld_cli import settings
    from mah2.gateway import Gateway, load_api_key
    from mah2.memory.graph import MemoryError_, MemoryGraph
    from mah2.memory.team import appworld_team_class
    load_api_key()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = settings(json.loads(Path(a.config).read_text()), 'memory', a.paper_settings)
    store = MemoryGraph(out / 'store')
    MemoryTeam = appworld_team_class()
    log = []
    for i, task in enumerate(tasks, 1):
        tag = f'd{i}'
        gateway = Gateway(out / 'api' / tag, config, cap=a.budget_per_task)
        gateway.set_stage(tag, a.budget_per_task * .8)
        cfg = dict(config, memory_store=str(out / 'store') if store.nodes else None)
        row = dict(task=task, tag=tag)
        try:
            team = MemoryTeam(gateway, cfg, out / 'runs' / task, task)
            start = team.memory.base_seq
            outcome = team.run()
            keys = merge(store, team.memory.graph, start, tag)
            row.update(completed=outcome['completed'], error=outcome['error'], imported_notes=len(keys))
            notes = [dict(key=k, type=store.nodes[k]['type'], body=store.nodes[k]['body']) for k in keys]
            if notes:
                gateway.set_stage(tag + '/curator', a.budget_per_task)
                reply = gateway.request({'messages': [{'role': 'user', 'content': CURATOR + json.dumps(
                    dict(completed=outcome['completed'], notes=notes), ensure_ascii=False)}],
                    'max_tokens': 1500, 'response_format': {'type': 'json_object'}})
                procedures = parse_json(reply['choices'][0]['message'].get('content') or '{}').get('procedures', [])
                added = []
                for n, proc in enumerate(procedures[:4], 1):
                    try:
                        links = [k for k in proc.get('links', []) if k in keys]
                        node = store.add(key=f'procedures/{tag}-p{n}', type='recommendation',
                                         body=str(proc.get('body', ''))[:600], author='curator', stage=tag,
                                         edges=[dict(rel='derived_from', target=k) for k in links])
                        added.append(node['key'])
                    except MemoryError_ as exc:
                        row.setdefault('curator_rejected', []).append(str(exc))
                row['procedures'] = added
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            row['usd'] = gateway.charged
        log.append(row)
        print(json.dumps(row), flush=True)
    (out / 'build.json').write_text(json.dumps(dict(tasks=tasks, rows=log, check=store.check(),
                                                    usd=sum(r['usd'] for r in log)), indent=2))
    print(json.dumps(store.check()['by_type']))
    return 0


if __name__ == '__main__':
    sys.exit(main())

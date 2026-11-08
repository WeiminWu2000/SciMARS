"""Shared memory graph for the seven-role team (host-side service).

Combines Mycelium's room memory (keyed markdown memories in namespaces, typed
relations, backlinks, integrity checks, search, change subscriptions;
github.com/mycelium-io/mycelium) with the active-context-graph rules of
arXiv:2607.13220 (closed epistemic node ontology, provenance edges, grounding
of every entry in existing nodes, immutable observations, automatic provenance
logging of executions). Search is lexical BM25 (Mycelium uses on-device
embeddings; this keeps the sandbox offline and deterministic).

In SciMARS the host routes memory into each role's prompt and takes notes from
role replies (see mah2.memory.team); roles never edit or delete history directly.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

from .policy import NAMESPACES, NODE_TYPES, RELATIONS

KEY_RE = re.compile(r'^(%s)/[a-z0-9][a-z0-9._-]{0,63}(/[a-z0-9][a-z0-9._-]{0,63}){0,3}$' % '|'.join(NAMESPACES))
SOURCE_TYPES = ('dataset', 'observation')  # may be grounded by a source instead of an edge
MAX_BODY = 6000
TOKEN = re.compile(r'[a-z0-9_]+')


class MemoryError_(ValueError):
    pass


def tokens(text):
    return TOKEN.findall(text.lower())


class MemoryGraph:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.nodes = {}
        self.log = []
        path = self.root / 'graph.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                self._apply(json.loads(line), persist=False)

    # ------------------------------------------------------------ storage
    def _apply(self, row, persist=True):
        node = self.nodes.get(row['key'])
        row = dict(row, version=(node['version'] + 1) if node else 1)
        self.nodes[row['key']] = row
        self.log.append(row)
        if persist:
            with (self.root / 'graph.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            md = self.root / 'rooms' / (row['key'] + '.md')
            md.parent.mkdir(parents=True, exist_ok=True)
            front = {k: row[k] for k in ('key', 'type', 'author', 'version', 'seq', 'stage', 'source') if row.get(k) is not None}
            edges = ''.join(f'\n  - {e["rel"]}: {e["target"]}' for e in row['edges'])
            md.write_text('---\n' + ''.join(f'{k}: {v}\n' for k, v in front.items()) +
                          (f'edges:{edges}\n' if edges else '') + '---\n' + row['body'] + '\n')
        return row

    def add(self, *, key, type, body, edges=(), source=None, author='unknown', stage=None):
        with self.lock:
            if not isinstance(key, str) or not KEY_RE.match(key):
                raise MemoryError_(f'key must look like <namespace>/<slug>, namespace in {NAMESPACES}, '
                                   'lowercase [a-z0-9._-]')
            if type not in NODE_TYPES:
                raise MemoryError_(f'type must be one of {NODE_TYPES}')
            if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
                raise MemoryError_(f'body must be non-empty text of at most {MAX_BODY} characters')
            clean = []
            for e in edges or []:
                if not isinstance(e, dict) or e.get('rel') not in RELATIONS:
                    raise MemoryError_(f'edge relation must be one of {RELATIONS}')
                if e.get('target') not in self.nodes:
                    raise MemoryError_(f'edge target {e.get("target")!r} does not exist (grounding rule)')
                if e['target'] == key:
                    raise MemoryError_('a node cannot link to itself')
                clean.append(dict(rel=e['rel'], target=e['target']))
            if not clean and not (type in SOURCE_TYPES and source):
                raise MemoryError_('grounding rule: every entry must link to an existing node '
                                   '(dataset/observation entries may instead name a source file or job)')
            existing = self.nodes.get(key)
            if existing and existing['type'] in SOURCE_TYPES:
                raise MemoryError_('dataset and observation nodes are immutable; add a new node that '
                                   'derives from or supersedes it')
            if existing and existing['type'] != type:
                raise MemoryError_('a key keeps its node type; use a new key')
            row = dict(key=key, type=type, body=body, edges=clean, source=source, author=author,
                       stage=stage, seq=len(self.log), time=time.time())
            return self._apply(row)

    # -------------------------------------------------------------- reads
    def backlinks(self, key):
        return [dict(source=k, rel=e['rel']) for k, n in self.nodes.items() for e in n['edges'] if e['target'] == key]

    def get(self, key):
        with self.lock:
            if key not in self.nodes:
                raise MemoryError_(f'no node {key!r}')
            return dict(self.nodes[key], backlinks=self.backlinks(key))

    def ls(self, prefix='', type=None):
        with self.lock:
            return [self.summary(n) for k, n in sorted(self.nodes.items())
                    if k.startswith(prefix or '') and (type is None or n['type'] == type)]

    @staticmethod
    def summary(n, body=240):
        text = n['body'] if len(n['body']) <= body else n['body'][:body] + '...'
        return dict(key=n['key'], type=n['type'], author=n['author'], version=n['version'], stage=n.get('stage'),
                    edges=n['edges'], body=text)

    def search(self, query='', k=6, types=(), namespaces=(), exclude_authors=(), body=600):
        with self.lock:
            pool = [n for n in self.nodes.values()
                    if (not types or n['type'] in types)
                    and (not namespaces or n['key'].split('/', 1)[0] in namespaces)
                    and n['author'] not in exclude_authors]
            if not pool or k <= 0:
                return []
            docs = [Counter(tokens(n['key'] + ' ' + n['type'] + ' ' + n['body'])) for n in pool]
            q = tokens(query)
            avg = sum(sum(d.values()) for d in docs) / len(docs)
            df = Counter(t for d in docs for t in set(d))

            def bm25(d):
                length = sum(d.values())
                return sum(math.log(1 + (len(docs) - df[t] + .5) / (df[t] + .5)) * d[t] * 2.2 /
                           (d[t] + 1.2 * (.25 + .75 * length / max(avg, 1))) for t in q if t in d)
            # Ties (and empty queries) favour recent nodes.
            ranked = sorted(zip(pool, docs), key=lambda x: (-bm25(x[1]), -x[0]['seq']))
            return [self.summary(n, body) for n, _ in ranked[:k]]

    def feed(self, since, types, exclude_author=None, limit=8):
        with self.lock:
            rows = [n for n in self.nodes.values() if n['seq'] >= since and n['type'] in types
                    and n['author'] != exclude_author]
            return [self.summary(n, 600) for n in sorted(rows, key=lambda n: n['seq'])[-limit:]]

    def check(self):
        with self.lock:
            linked = {e['target'] for n in self.nodes.values() for e in n['edges']} | \
                     {k for k, n in self.nodes.items() if n['edges']}
            return dict(nodes=len(self.nodes), edges=sum(len(n['edges']) for n in self.nodes.values()),
                        by_type=dict(Counter(n['type'] for n in self.nodes.values())),
                        by_author=dict(Counter(n['author'] for n in self.nodes.values())),
                        orphans=sorted(k for k in self.nodes if k not in linked),
                        contradictions=[dict(source=k, target=e['target']) for k, n in self.nodes.items()
                                        for e in n['edges'] if e['rel'] == 'contradicts'])

    @property
    def seq(self):
        return len(self.log)

    def snapshot(self, dest: Path):
        with self.lock:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(self.root / 'graph.jsonl', dest / 'graph.jsonl') if (self.root / 'graph.jsonl').exists() else None
        return MemoryGraph(dest)

    # ------------------------------------------------------- host helpers
    def seed(self, data_dir: Path):
        """Dataset nodes for the public files, so agent entries can be grounded."""
        for path in sorted(Path(data_dir).iterdir()):
            if path.is_file():
                slug = re.sub(r'[^a-z0-9._-]', '-', path.name.lower())
                head = path.read_text(errors='replace')[:400] if path.stat().st_size < 50_000_000 else ''
                self.add(key=f'context/data/{slug}', type='dataset', author='host', source=f'/app/data/{path.name}',
                         body=f'Public input file /app/data/{path.name} ({path.stat().st_size} bytes). Head:\n{head}')

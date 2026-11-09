# Memory and Memory + Harness² on AppWorld

`memory` adds a shared memory graph to the seven-role team. `memory_harness2` adds the same graph to
Harness² and gives Harness² a third editor that revises the team's memory policy.

## Components

| Module | Role |
|---|---|
| `mah2/memory/graph.py` | The graph: keyed notes in namespaces, typed links, backlinks, BM25 search, change feeds. It enforces a closed node ontology, provenance links, grounding (every note links to an existing node) and immutable host-logged observations. |
| `mah2/memory/policy.py` | The editable memory policy: per-role write duties, read routing (node types, namespaces, k, query hint) and subscriptions, plus full-harness scope checks. |
| `mah2/memory/team.py` | `MemoryTeam` (AppWorld team with memory fields in each prompt), `TeamMemory` (routing and note intake), and `MemoryHarness2Mixin` (proposal C, candidate C, memory-aware composer and selection). |
| `scripts/build_appworld_memory.py` | Builds the optional persistent store from scenario-disjoint tasks. |

How a run uses memory:

- The task is seeded as a dataset node. Every code execution is logged by the host as an observation.
- Each role prompt carries `memory_routed` (search under the role's read policy), `memory_feed` (new
  notes of subscribed types) and its write duty. Roles may add a `"memory"` list to their JSON reply.
  Notes that break the grounding rule are rejected and logged.
- Each execution starts from a fresh graph, or a read-only copy of the persistent store with its own layer
  on top. Nothing passes between evaluation tasks.

## Memory + Harness²

Each round runs the reference team, then three scoped editors:

- `proposal_a` revises roles.
- `proposal_b` revises one coordination route or contract.
- `proposal_c` revises the memory policy only.

Each proposal is probed in a fresh AppWorld state. The composer may revise roles. Its coordination must
equal the base or candidate B coordination, and its memory must equal the base or candidate C memory.
Adopting candidate C memory requires candidate C evidence. The selector then chooses among the round's
completed candidates using public observations only. Two rounds, eight team steps, and one $1 allowance
cover all eleven stages (shares in `config/appworld.json` → `memory_harness2`).

## Persistent store

```bash
python scripts/build_appworld_memory.py --output runs/appworld_store            # plan
python scripts/build_appworld_memory.py --output runs/appworld_store --execute  # at most $2
```

The builder works through the ten tasks under `memory_store` in `config/appworld_tasks.json`. These are
five music debug tasks and five finance train/dev tasks, scenario-disjoint from evaluation. It runs them
in sequence, imports each run's agent notes, and has a neutral curator distil up to four generic
procedures per task. No grader is used. Pass the store with `--memory-store <dir>/store` or
`MAH2_MEMORY_STORE`.

## Fixes that made Memory + Harness² work

A post-commit audit of every candidate rollout in earlier runs found three verified causes of failure.
The fixes are in the `fixes` block of `config/appworld.json` and apply to all five methods.

1. **`complete_task` misuse.** Agents passed prose summaries as `answer`. On no-answer tasks, those
   completions succeeded 0/29, against 42/57 without an answer. Sentence answers also failed question
   tasks. The team and selector prompts now restate the public API contract: answer only for questions,
   and only the value.
2. **Selection blind spots.** Evidence strings kept only their first 1400 characters, which hid the final
   `complete_task` call. The selector also chose runs that never completed, and no incomplete run ever
   succeeded. Evidence now keeps head and tail, and completed runs are selected first.
3. **Recoverable errors ended rollouts.** A syntax error, a truncated JSON reply, or a slow AppWorld
   worker start ended the rollout or the whole task. These now reach the debugger, get one paid retry, or
   fail only that rollout.

Recorded results on the 20 Music/Finance Test-N tasks (GLM `z-ai/glm-5.3-flashx`, $1 per task):

| Setting | Success |
|---|---:|
| Base multi-agent (paper) | 5/20 |
| Integrated search (paper) | 8/20 |
| Harness² (paper) | 10/20 |
| **Memory + Harness²** | **11/20** |

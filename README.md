# SciMARS: Multi-Agent Recursive Self-Improvement for Closed-Loop Scientific Discovery

[Watch video](media/SciMARS_film.mp4) · [Paper (PDF)](paper/SciMARS.pdf) · [Cite SciMARS](#citation) · [Download BibTeX](CITATION.bib)

**Improve how a team works: revise role procedures, test coordination changes, and carry the selected harness into the next round.**

SciMARS provides a seven-role execution team and a separate harness optimizer. The optimizer uses public execution traces to propose role and coordination changes, probe them, and select a current-round execution. You can connect an existing team through an execution callback, or use the included AppWorld and local scientific Python adapters. The editable surface follows the paper’s seven logical roles and existing handoffs.

![SciMARS overview: a seven-role execution team and recursive improvement of role procedures and coordination contracts](media/scimars-overview.png)

[Connect your team](docs/custom_teams.md) · [AppWorld quickstart](#appworld-quickstart) · [Memory](docs/memory.md) · [Design](docs/architecture.md) · [Configuration and evaluation](docs/reproduction.md) · [Your own Python task](docs/custom_tasks.md)

## Connect your existing team

Install from a checkout using the first installation block below through `python scripts/fetch_dependencies.py`; AppWorld installation is optional for custom teams. Then run:

```bash
python examples/existing_team.py --output runs/my-team-smoke
```

This no-credential smoke test uses scripted editors and real Python execution. Replace its executor callback with your team and supply your existing harness to `optimize_team`. The optimizer owns proposal validation, probes, composition, current-round selection, and verified execution reuse. Your callback owns environment isolation, tool limits, and applying role instructions and handoff contracts. See the [integration contract and supported surface](docs/custom_teams.md).

## AppWorld quickstart

Use Linux and Python 3.11 for the AppWorld environment. Install from a checkout:

```bash
git clone https://github.com/WeiminWu2000/SciMARS.git
cd SciMARS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python scripts/fetch_dependencies.py

python3.11 -m venv .venv-appworld
.venv-appworld/bin/pip install 'appworld==0.1.3.post1'
.venv-appworld/bin/appworld install
.venv-appworld/bin/appworld download data --root appworld_data
```

AppWorld's code and data are obtained from its own distribution and retain their own terms. See the [AppWorld project](https://github.com/StonyBrookNLP/appworld) for its setup requirements. This adapter targets package `0.1.3.post1` and data `0.1.0`.

Check a development task without credentials or model calls:

```bash
mah2 appworld --task 4ec8de5_1 --output runs/appworld-demo --dry-run
```

The check validates configuration and local paths; it does not start the AppWorld worker. To run the team:

```bash
export OPENROUTER_API_KEY_FILE=/path/to/private/openrouter.key
mah2 appworld --task 4ec8de5_1 --method harness2 \
  --output runs/appworld-demo --budget 1.0 --execute --grade
```

`--execute` permits paid calls. `--grade` runs AppWorld's official evaluator **after** final commitment. Without `--grade`, official success remains unset. Output directories must be new; the runner does not overwrite prior attempts.

The AppWorld configuration uses `z-ai/glm-5.3-flashx`, two outer rounds, and one shared API allowance per task. These are recorded settings, not a promise of current endpoint availability. Check your provider and rates before using a different configuration. `--appworld-root`, `--appworld-python`, and `--config` support other installation locations.

## Five settings, one execution team

| AppWorld `--method` | What it does |
|---|---|
| `base` | Runs the fixed seven-role team once, without outer revisions. |
| `harness2` | Proposes role and coordination changes separately, probes candidates, and composes a successor. |
| `integrate` | Integrated workflow search: the released MASS revision and selection procedure with the same team executor (alias `mass`). |
| `memory` | The base team plus a shared memory graph; optionally seeded by a persistent store built on disjoint tasks. |
| `memory_harness2` | Memory graph plus Harness² with a third, memory-policy editor; each proposal is probed and the composer combines them. |

```bash
python scripts/build_appworld_memory.py --output runs/appworld_store --execute   # optional, about $1-2
mah2 appworld --task 1150ed6_2 --method memory_harness2 --memory-store runs/appworld_store/store \
  --output runs/mh2-1150ed6_2 --budget 1.0 --execute --grade
```

All five share the team, task information, model and the $1 per-task allowance (editing, probes,
retries and selection included). The `fixes` block of `config/appworld.json` applies to every method:
the public `complete_task` contract in team and selector prompts, recoverable syntax/JSON step errors,
4096-token code replies, scoped editor patches, head-and-tail evidence and completed-run-first selection.
`--paper-settings` turns it off to reproduce the recorded configuration. See [memory](docs/memory.md).

`integrate` requires `python scripts/fetch_dependencies.py --include-mass`. Its source is obtained separately and is not redistributed here. The AppWorld adapter runs Python/API interactions through the shared team; it does not require the Qwen-code executor used by the local-file MASS adapter. This is a workflow-search adaptation, not the full trained MASS system. See [the comparison protocol](docs/reproduction.md).

## How it works

1. **Execute the current team.** Analyzer, planner, coder, verifier, router, debugger, and finalizer work in a fresh AppWorld state.
2. **Propose two scoped changes.** The role editor preserves coordination; the coordination editor preserves roles and changes one route or handoff contract.
3. **Probe and compose.** Execute the role and coordination candidates in separate fresh states. Compose a successor using their public traces. Reuse a same-round execution only when the complete harness and program hash match.
4. **Select and repeat.** Select among the current round's runnable candidates using public observations. Carry the selected harness and task-local lessons into the next round.
5. **Commit, then evaluate.** Grade only the final selected AppWorld state. Official scores never enter editor or selection prompts.

The optimizer is separate from the execution team. It adapts role instructions and existing handoffs; it does not rewrite itself. Each task starts with the initial harness. Completion claims and error-free executions are not proof that a task is solved.

## Settings and local artifacts

`config/appworld.json` contains the model, budget allocation, team step limit, and selection policy. `config/appworld_tasks.json` lists Music and Finance task identifiers for trying the adapter. These are researcher-defined task categories, not AppWorld's official app domains.

Each new run saves its configuration, source hashes, API accounting, public interaction traces, candidate decisions, and final commitment. AppWorld stores candidate world states under its own experiment directory. Keep both directories to inspect or evaluate a run.

## Try the loop without API calls

The small local-file demo uses scripted editors and real sandboxed Python execution. It requires Linux and `bubblewrap` (`bwrap`), but no AppWorld data or model credentials:

```bash
mah2 demo --output runs/demo --rounds 2
mah2 report --run runs/demo
```

For your own scientific Python tasks, use `mah2 init`, `mah2 run`, and `mah2 report`. Follow the [task guide](docs/custom_tasks.md) and [local-file setup](docs/setup.md). The AppWorld and local-file adapters share the harness schema and recursion controller, but use different execution environments.

## Development

```bash
pip install -e '.[test]'
python scripts/fetch_dependencies.py --include-mass
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Tests use synthetic fixtures and mocked model calls. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Citation

If you use SciMARS, please cite our paper:

```bibtex
@misc{wu2026scimars,
  title        = {{SciMARS}: Multi-Agent Recursive Self-Improvement for Closed-Loop Scientific Discovery},
  author       = {Wu, Weimin and Liu, Han},
  year         = {2026},
  howpublished = {Manuscript},
  url          = {https://github.com/WeiminWu2000/SciMARS/blob/main/paper/SciMARS.pdf}
}
```

## Attribution and license

The method builds on [Harness²: Recursive Agent Harnessing for an Open World](https://arxiv.org/abs/2610.14224) (Xu et al., 2026; [code](https://github.com/google-research/harness2)), with prompt components from [DS-STAR](https://github.com/google-research/ds-star), an optional [MASS](https://github.com/SakanaAI/mass) workflow-search adapter, and an execution adapter for [AppWorld](https://github.com/StonyBrookNLP/appworld).

Our code is Apache-2.0. Dependencies retain their own terms; see [NOTICE](NOTICE), [CITATION.cff](CITATION.cff), and [AppWorld source provenance](provenance/appworld.json). Benchmark data, upstream source checkouts, credentials, and generated trajectories are not included. This is an independent project, not an officially supported Google product.

## References

```bibtex
@article{xu2026harness2,
  title   = {Harness$^2$: Recursive Agent Harnessing for an Open World},
  author  = {Xu, Ruiyao and Chen, Yanfei and CuiZhu, Zhongying and Dalvi Mishra, Bhavana and Ming, Yifei and Yu, Han and Han, Rujun and Lee, Chen-Yu and Pfister, Tomas},
  journal = {arXiv preprint arXiv:2610.14224},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.14224}
}
```

# AppWorld configuration and evaluation

The original task-local AppWorld protocol is the basis for `config/appworld.json`. The public CLI adds portable paths and a dry-run mode. A fresh run is a new observation, not a guaranteed reproduction of earlier model responses.

## Recorded setting

| Setting | Value |
|---|---|
| Model | `z-ai/glm-5.3-flashx`, provider `z-ai` |
| AppWorld package / data | `0.1.3.post1` / `0.1.0` |
| Team | Analyzer, planner, coder, verifier, router, debugger, finalizer |
| Outer rounds | 2 |
| Maximum team-loop steps per candidate | 8 |
| Team response token ceiling | 2,048 |
| Editor response token ceiling | 4,096 |
| Selection pool | Current round only |
| Identical successor | Reuse verified same-round execution |
| API allowance | Shared by execution, editing, probing, selection, and reflection |

The worker limits each interaction to 45 seconds, up to 100 interactions and 100 API calls per interaction. The configuration's legacy `execution` fields describe the local-file runtime; they do not impose a total AppWorld wall-clock deadline. AppWorld uses its own worker timeouts and the team step limit.

The public task manifest contains Music and Finance examples: five Music dev tasks and ten Test-N tasks per category, one per scenario. These categories derive from public instructions; supporting apps can overlap. Development and evaluation scenarios are disjoint.

## Run a comparison

Follow the README setup, then use separate output directories and the same configuration and allowance:

```bash
for method in base harness2 integrate memory memory_harness2; do
  mah2 appworld --task 4ec8de5_1 --method "$method" \
    --output "runs/dev-$method-4ec8de5_1" --budget 1.0 --execute --grade
done
```

Use the dev tasks for integration checks before starting Test-N. To list the Test-N task IDs without running anything:

```bash
python -c 'import json; m=json.load(open("config/appworld_tasks.json")); print("\n".join(x["task_id"] for x in m["evaluation"]))'
```

Use one OS process per occurrence when running in parallel: provider bindings and the released MASS driver contain process-global state. Do not invoke concurrent experiments in Python threads. Preserve the configuration, source hashes, input instruction, failed attempts, and API records. The endpoint does not support deterministic sampling seeds.

The five methods share the team executor, task information, and total API allowance. They allocate that allowance differently and execute different numbers of candidates. MASS edits workflow text, while Harness² edits structured role procedures and coordination contracts. This comparison is not an isolated test of decomposition alone and does not reproduce the full trained MASS system.

## Official evaluation boundary

The worker loads tasks with `load_ground_truth=False`. The agent sees the instruction, public supervisor information, API documentation, and observations. It cannot request the evaluator through the adapter's interaction protocol.

`--grade` evaluates only after `commit.json` records the selected execution. Scores are written to local `grade.json` and `result.json`; they never guide a candidate choice. Without that flag, `official_success` remains unset. Generated `submission.py` records interaction blocks; it is not an independent replay script because later actions depend on earlier observations. Keep the corresponding AppWorld experiment state.

A worker error, missing submission, or incomplete evaluation must remain visible. Do not replace a committed selection after inspecting its official score. A software change requires a new output directory and provenance record.

## Dependency ownership

`python scripts/fetch_dependencies.py` obtains pinned Harness² and DS-STAR checkouts. Add `--include-mass` for the optional MASS source. The script rejects modified or differently pinned existing checkouts. Vendor source is ignored and is not part of the release.

AppWorld comes from its own package and data distribution. Preserve its applicable terms. MASS's upstream NOTICE assigns no public open-source license to its code; obtaining it separately does not make it part of this project's Apache license.

The AppWorld MASS adapter uses the team executor and needs no Qwen-code runtime. For the separate **local-file** MASS adapter, install its client:

```bash
mkdir -p .runtime/node/bin
ln -s "$(command -v node)" .runtime/node/bin/node
npm install --prefix .runtime @qwen-code/qwen-code@0.20.0
```

Choose a Node executable visible inside the local-file sandbox. Qwen-code is a client; the model is selected in the configuration.

## Checks before paid calls

Run `python -m unittest discover -s tests -v` after obtaining the optional test dependencies and pinned checkouts. The suite covers handoffs, scope restrictions, candidate reuse, current-round selection, and budget accounting using synthetic fixtures and mocks. `mah2 appworld --dry-run` makes no model calls. Neither check establishes task success or provider availability.

The local-file CLI now defaults to the revised current-round policy and exact-harness successor reuse too. Historical archive selection and four executions per round require `--selection-pool archive --no-reuse-identical-successor`. This release change does not imply that historical experiments used the revised policy.

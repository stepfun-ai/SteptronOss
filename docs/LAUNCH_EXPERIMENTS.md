# Launching Experiments

This note describes how StepTronOSS experiments are started in practice, based on the current repo tooling.

Chinese version: `docs/LAUNCH_EXPERIMENTS_ZH.md`.

## Single-task vs. Multi-task experiments

Single-task experiments have only one kind of task in `task_specs` (length 1).
Typical examples: pretrain/SFT runs without extra evaluation or controller
processes.

Multi-task experiments include multiple kinds of tasks (length > 1), such as
RL runs with rollout + training, or agentic evals with controllers, judges, and
inference workers. Tasks are distinguished by differences in resource needs,
entry commands, environment variables, or even container images.

## Quick matrix: task count vs. node count

Legend:
- Direct = `python my_exp.py` / `torchrun ...`
- MPRun = `tools/mp_run.py my_exp.py`
- BashScripts = `tools/build_scripts.py my_exp.py <out>`
- ✅ recommended, ❌ not supported

|                | Single-node | Multi-node |
|----------------|-------------|------------|
| Single-task    | ✅ Direct <br> ✅ MPRun <br> ✅ BashScripts | ✅ Direct <br> ❌ MPRun <br> ✅ BashScripts |
| Multi-task     | ❌ Direct <br> ✅ MPRun <br> ✅ BashScripts | ❌ Direct <br> ❌ MPRun <br> ✅ BashScripts |

## Launch method details

### Direct

Pros:
- Minimal tooling; easiest to debug locally.
- Works for single-task on single-node or multi-node.

Cons:
- Not suitable for multi-task (no per-task subprocess orchestration).
- You must manage `torchrun` arguments yourself for multi-node.

Commands:
```bash
# Single-node
python path/to/my_exp.py

# Multi-node
torchrun --nnodes <N> --nproc-per-node <GPUS_PER_NODE> \
  --node_rank <RANK> --master_addr <ADDR> --master_port <PORT> \
  path/to/my_exp.py
```

### MPRun

Pros:
- Recommended for multi-task on a single node.
- Spawns per-task subprocesses with their own envs.

Cons:
- Single-node only (multi-node not supported).

Commands:
```bash
export STEPTRON_MEET_DIR=/path/to/shared
tools/mp_run.py path/to/my_exp.py
```

### BashScripts

Pros:
- Most flexible; supports all combinations (single/multi task, single/multi node).
- Generates per-replica scripts for manual submission or scheduling systems.

Cons:
- Extra step: you must distribute/launch the scripts yourself.

Commands:
```bash
tools/build_scripts.py path/to/my_exp.py /mnt/entrypoints/

# On each node, run the assigned script (example):
bash /mnt/entrypoints/<exp_name>/<exp_id>/gpu/0.sh
```

## Generate multi-node launch scripts

If you want to generate scripts for later submission or manual launch, use
`tools/build_scripts.py` (it follows the same command assembly logic as `mp_run`).

```bash
tools/build_scripts.py path/to/my_exp.py /mnt/entrypoints/
```

Output layout:

```
/mnt/entrypoints/<exp_name>/<exp_id>/
  cpu/
    0.sh
  gpu/
    0.sh
    1.sh
    ...
```

Notes:
- Indices are cumulative within each node type (all GPU tasks share a single 0..N range).
- Scripts export task envs (including task-scoped `EXP_ID` and `NNODES`) and run
  the assembled command.

## Environment variables

- `STEPTRON_MEET_DIR`: shared filesystem path used by rendezvous (Redis endpoint).
- `CANNOT_BE_REDIS_SERVER=1`: set on ranks that must not start Redis; if all ranks
  set this, rendezvous will time out.

## Task specs overview (resource_cfg.task_specs)

Each `task_spec` can override:
- `replica`: number of replicas for the task
- `gpu`: number of GPUs per node
- `node_type`: `"cpu"` or `"gpu"`
- `envs`: per-task environment variables
- `command`: per-task launch template

These are combined with `resource_cfg` defaults and expanded by
`resource_cfg.extract()` into a concrete set of per-task configs.

## Platform adaptation

For custom resource managers or job schedulers (e.g., Kubernetes, Volcano,
Alibaba Cloud), we recommend writing a dedicated `tools/xx_run.py` to provide a
unified submission entrypoint.

For Docker-based task management platforms, your custom `xx_run` should:
- Parse the experiment’s `resource_cfg` / `task_specs`.
- Expand them with `resource_cfg.extract()`.
- Submit tasks in order based on resource type and replica count.

See `tools/example_submitter.py` for a runnable, platform-agnostic template
that replaces platform details with TODOs.

Example:
```bash
python tools/example_submitter.py path/to/my_exp.py
```

<div align="center">
  <h1 style="margin: 0; border-bottom: none;"> <img src="assets/stepfun.svg" width="25" style="margin-right: 10px;"/>  StepTronOSS </h1>
</div>

<p align="center">
    <strong>English</strong>&nbsp; | &nbsp;<a href="README_ZH.md">简体中文</a>
</p>


**StepTronOSS** is a lightweight **training framework** for large-scale language models with a focus on modular configs, reproducible experiments, and fast iteration across SFT, RLVR, and evaluation workflows. It can run with only PyTorch as a dependency, while also supporting operator-level replacements for acceleration.

## Key capabilities:
- Config-driven experiments with dynamic validation (`cfshow`, `sanity_check`)
- Multi-task orchestration with flexible launch tooling
- Extensible data/optimizer/model stacks for rapid research iteration

## Docs

- Launch guide (EN): `docs/LAUNCH_EXPERIMENTS.md`
- Launch guide (ZH): `docs/LAUNCH_EXPERIMENTS_ZH.md`
- SFT data prep (ZH): `docs/SFT_DATA_PREPARATION.md`
- SFT data prep (EN): `docs/SFT_DATA_PREPARATION_EN.md`
- API modules: `docs/MODULES.md`

## Installation

```bash
# from repo root
uv sync
# install redis-server
apt install -y redis-server
```

## Getting Started

uv virtual environment is recommanded. If not, prefix with `uv run`.

### Experiment Overview & Sanity Check

```bash
# Overview the experiment config and run sanity_check
uv run cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py
# Inspect a specific subtree (e.g., actor config)
uv run cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py -k actor_model_cfg
```

### Run Experiments

```bash
# Single-task experiments (e.g., SFT)
uv run torchrun playground/sft/your_exp.py

# Multi-task experiments (e.g., RL)
export STEPTRON_MEET_DIR=/path/to/shared
uv run tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py

# mp_run is also compatible with single-task experiments
uv run tools/mp_run.py playground/sft/your_exp.py

# Override experiment params (example: enable timer logging)
uv run tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py profiler_cfg.timing_log_level=1
```

See the docs section above for detailed launch guides.

### Generate Multi-Node Launch Scripts

Use `tools/build_scripts.py` to generate per-replica shell scripts based on `resource_cfg.task_specs`.

```bash
# Example: generate scripts under /mnt/entrypoints/<exp_name>/<exp_id>/
uv run tools/build_scripts.py playground/rlvr/qwen3_1p5b_rlvr_math.py /mnt/entrypoints/
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
- Script indices are cumulative within each node type (e.g., all GPU tasks share a single 0..N index range).
- Each script exports task envs (including task-scoped `EXP_ID` and `NNODES`) and invokes the command assembled by `mp_run`.
- Extra CLI args are passed through to the experiment, just like `mp_run.py`.

### Runtime Environment

Distributed rendezvous spins up a per-experiment Redis server using a shared
filesystem directory.

- `STEPTRON_MEET_DIR`: shared directory visible and writable by all nodes. It
  stores the rendezvous file that publishes the Redis server port.
- `CANNOT_BE_REDIS_SERVER=1`: set on ranks that must not start Redis (they will
  wait for another rank to start it). If every rank sets this, rendezvous will
  eventually time out.

## Zen

Core principles:
- Keep configs stateless; runtime objects carry state.
- Declare config structure at class level, fill values in `__init__`.
- Use `Ref("..path")` for cross-node linkage.
- Call `sanity_check()` and `to_dict()` for validation/serialization.

Configurize example:

```python
# Before
class A:
    def __init__(self, param_a: int, param_b: float = 1.0):
        pass

# After
from configurize import Config

class AConfig(Config):
    param_a: int
    param_b: float = 1.0

    def build(self):
        return A(cfg=self)

    def sanity_check(self):
        super().sanity_check()
        assert self.param_b > 0

class A:
    def __init__(self, cfg: AConfig):
        pass
```

## AI Native

StepTronOSS enables an AI-native workflow: its modular architecture supports easy verification and iterative development, while tools like `cfshow` provide dynamic config inspection and validation. The repo also ships with `AGENTS.md` to guide agents in contributing to code changes.

Try asking your agent:
```
write an exp of qwen3 8B sft, optimizer use muon
```

## Optimization Kernels

Only needed when you want maximum GPU performance and kernel-level speedups.

### flash-attn

Install manually:

```bash
uv pip install flash-attn --no-build-isolation
```

### grouped_gemm

Install manually:

```bash
uv pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
```

### Enable in code (set all optimizations at once)

```python
from steptronoss.utils.optimizable import set_optimization

set_optimization(
    default="torch_compile",
    AttentionCore="flash-attn",
    grouped_gemm="nv_grouped_gemm",
)
```

## Project Status

- [x] SFT exps
- [x] Reference configs: Qwen3 8B `playground/pretrain/qwen3/qwen3_8.py`, Step3.5 Flash `playground/pretrain/step3p5/step3p5_flash.py`
- [x] Eval
- [ ] RLVR implementation
- [x] Triton kernel implementation

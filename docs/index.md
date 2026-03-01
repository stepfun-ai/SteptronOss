# SteptronOss

StepTronOSS is a training framework for large-scale language models, with a strong focus on configurable experiments, reproducible launches, and practical multi-task orchestration.

## Quick Start

Install:
```bash
uv sync
```

Run a single-task experiment:
```bash
torchrun playground/sft/your_exp.py
```

Run a multi-task experiment (e.g., RL):
```bash
export STEPTRON_MEET_DIR=/path/to/shared
tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py
```

## Core Principles
- Keep configs stateless; runtime objects carry state
- Declare config structure at class level, fill values in `__init__`
- Use `Ref("..path")` for cross-node linkage
- Always run `sanity_check()` and `to_dict()` for validation/serialization

## Repository Layout
- Core package: `steptronoss/` (core, model, data, exp, optimizer, generation, tokenizer, utils, checkpointing)
- Experiments: `playground/`
- Tests: `tests/`

## Experiment Workflow
1. Inspect and validate config:
```bash
cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py
```
2. Launch with `mp_run` for multi-task or `torchrun` for single-task
3. Use `tools/build_scripts.py` to generate multi-node scripts when needed

## Runtime Environment
- `STEPTRON_MEET_DIR` is required for multi-node rendezvous
- `CANNOT_BE_REDIS_SERVER=1` prevents a rank from starting Redis (all ranks set = timeout)

## Documentation
- Start here: [LAUNCH_EXPERIMENTS](LAUNCH_EXPERIMENTS.md)
- Launch guide (EN): [LAUNCH_EXPERIMENTS](LAUNCH_EXPERIMENTS.md)
- Launch guide (ZH): [LAUNCH_EXPERIMENTS_ZH](LAUNCH_EXPERIMENTS_ZH.md)
- Memory tuning (ZH): [MEMORY_TUNE](MEMORY_TUNE.md)
- Memory tuning (EN): [MEMORY_TUNE_EN](MEMORY_TUNE_EN.md)
- API modules: [MODULES](MODULES.md)

## Optimization Kernels
Install optional kernels when needed:
```bash
uv pip install flash-attn
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
```

Enable optimizations:
```python
from steptronoss.utils.optimizable import set_optimization

set_optimization(
    default="torch_compile",
    AttentionCore="flash-attn",
    grouped_gemm="nv_grouped_gemm",
)
```

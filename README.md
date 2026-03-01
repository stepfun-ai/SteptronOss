## Installation

```bash
# from repo root
uv sync
```

### Optional: Optimization Kernels

See the **Optimization Kernels** section for installation and enablement.

## Getting Started

### Experiment Overview & Sanity Check

```bash
# Overview the experiment config and run sanity_check
cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py
# Inspect a specific subtree (e.g., actor config)
cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py -k actor_cfg
```

### Run Experiments

```bash
# Single-task experiments (e.g., SFT)
torchrun playground/sft/your_exp.py

# Multi-task experiments (e.g., RL)
export STEPTRON_MEET_DIR=/path/to/shared
tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py

# mp_run is also compatible with single-task experiments
tools/mp_run.py playground/sft/your_exp.py

# Override experiment params (example: enable timer logging)
tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py profiler_cfg.timing_log_level=1
```

See `docs/LAUNCH_EXPERIMENTS.md` for a detailed launch guide (with a Chinese version in
`docs/LAUNCH_EXPERIMENTS_ZH.md`).

### Generate Multi-Node Launch Scripts

Use `tools/build_scripts.py` to generate per-replica shell scripts based on `resource_cfg.task_specs`.

```bash
# Example: generate scripts under /mnt/entrypoints/<exp_name>/<exp_id>/
tools/build_scripts.py playground/rlvr/qwen3_1p5b_rlvr_math.py /mnt/entrypoints/
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

## Optimization Kernels

### flash-attn

Install manually:

```bash
uv pip install flash-attn
```

### grouped_gemm

Install manually:

```bash
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
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

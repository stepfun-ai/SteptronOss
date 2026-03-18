## StepTronOSS AGENTS

This file stores repo-specific priors for future agents. Keep it short, practical,
and biased toward things that save repeated exploration.

## 1. Working Rules

### Self-Improvement Loop

Do an improve pass for key tasks:

- new environment
- new task type
- new project area
- high risk / high cost work
- collaboration / handoff work

Improve pass:

1. Identify friction
2. Extract reusable priors
3. Write them down in:
   - `AGENTS.md` for repo-wide priors
   - `docs/` for process / runbook details

## 2. Repo Layout

- Core package: `steptronoss/`
  - core, model, data, exp, optimizer, generation, tokenizer, utils, checkpointing
- Experiments: `playground/`
- Tests: `tests/`
- Docs: `docs/`

### Utilities overview

- `steptronoss/utils/arguments.py`: config overrides from CLI
- `steptronoss/utils/comm_utils.py`: Redis rendezvous, queues, `LocalFuture` / `RemoteFuture`
- `steptronoss/utils/dist_utils.py`: broadcast / all-to-all helpers, packing helpers, balancing helpers
- `steptronoss/utils/general.py`: numeric helpers, list split/balance, RNG fork, retry, recursion helpers, git hash
- `steptronoss/utils/logger.py`: rank-aware logging and `StepWriter`
- `steptronoss/utils/metrics.py`: metrics system (`Metric`, `Avg`, `Percentage`, `Histogram`, `Text`, `GradNorm`, `GlobalMetrics`)
- `steptronoss/utils/optimizable.py`: `@optimizable(...)` and `set_optimization(...)`
- `steptronoss/utils/utils.py`: model unwrap, param norms, memory report, layer map, IO helpers, generic load
- `steptronoss/utils/weight_loader.py`: HF safetensors mapping / merge

## 3. Code Style

### Config style

- Config class fields should include a short triple-quoted docstring immediately after the attribute definition.
- Follow the `configurize` pattern:
  - class attrs declare sub-config types
  - instance `__init__` sets concrete values
  - use `Ref("..path")` for cross-node linkage
  - configs expose `build()` / `build_*`, `sanity_check()`, `to_dict()`
- Only `Ref(...)` the exact parameter needed, not whole config objects.

### Experiment style

- SFT experiments under `playground/sft/qwen3/*_sft_step3_data.py` typically follow:
  - `class Exp(BaseExp)`
  - `model_cfg` / `data_cfg` declared as class attrs
  - trainer / checkpoint / model fields adjusted in `__init__`
  - entrypoint is `if __name__ == "__main__": Exp().train()`

## 4. Setup Priors

- After `uv sync`, also install `redis-server`:
  - `apt install -y redis-server`

### DeepEP build

- Set:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
  - `CUDACXX=$CUDA_HOME/bin/nvcc`
- Install:
  - `pip install -e /data/DeepEP --no-build-isolation`

### nv-grouped-gemm build

- Do not rely on random prebuilt wheels; ABI mismatch is common.
- Build with CUDA 12.9:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda CUDACXX=/data/cuda/cuda-12.9/cuda/bin/nvcc <python> -m pip install -e <grouped_gemm_source> --no-build-isolation`
- Runtime constraints:
  - `batch_sizes` must be CPU-visible / `torch.int64`
  - inputs must be bf16 for `nv_grouped_gemm`

## 3. Code Style

### Config style

- Config class fields should include a short triple-quoted docstring immediately after the attribute definition.
- Follow the `configurize` pattern:
  - class attrs declare sub-config types
  - instance `__init__` sets concrete values
  - use `Ref("..path")` for cross-node linkage
  - configs expose `build()` / `build_*`, `sanity_check()`, `to_dict()`
- Only `Ref(...)` the exact parameter needed, not whole config objects.

### Experiment style

- SFT experiments under `playground/sft/qwen3/*_sft_step3_data.py` typically follow:
  - `class Exp(BaseExp)`
  - `model_cfg` / `data_cfg` declared as class attrs
  - trainer / checkpoint / model fields adjusted in `__init__`
  - entrypoint is `if __name__ == "__main__": Exp().train()`
- `playground/sft/qwen3/qwen3_sft_base.py` already provides `OneNodeResourceConfig` with `replica=1` and `gpu=8`, so derived SFT experiments default to single-node 8-GPU `torchrun` unless they override `resource_cfg`.
- Under `playground/data/sft`, keep raw source recipes and dataset configs distinct in naming:
  - `*_recipe*.py` for `DataRecipe` / source file lists only
  - `*_data_config*.py` for `CompliableDatasetsConfig`, `CompiledDatasetsConfig`, and `SFTDataConfig`
  - when adding tokenizer variants for the same source recipe, share a common base config and use thin tokenizer-specific subclasses instead of duplicating whole modules
  - for large unified SFT rebuilds, do not mutate source json/jsonl in place; materialize derived json under `/oss/...`, preserve `DataSourceFile.subsample_rate` semantics with sample seed `1234`, and do global shuffle with an external bucketized sort instead of loading everything into memory

## 4. Setup Priors

- After `uv sync`, also install `redis-server`:
  - `apt install -y redis-server`

### DeepEP build

- Set:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
  - `CUDACXX=$CUDA_HOME/bin/nvcc`
- Install:
  - `pip install -e /data/DeepEP --no-build-isolation`

### nv-grouped-gemm build

- Do not rely on random prebuilt wheels; ABI mismatch is common.
- Build with CUDA 12.9:
  - `CUDA_HOME=/data/cuda/cuda-12.9/cuda CUDACXX=/data/cuda/cuda-12.9/cuda/bin/nvcc <python> -m pip install -e <grouped_gemm_source> --no-build-isolation`
- Runtime constraints:
  - `batch_sizes` must be CPU-visible / `torch.int64`
  - inputs must be bf16 for `nv_grouped_gemm`

## 5. Experiments and Configs

### Config/module map

- `steptronoss/exp` provides abstract `*Config` interfaces (`build_*`, `get_trainer_cls`)
- Concrete configs live mainly in `steptronoss/exp/base_exp.py`
- Ready-made experiment families:
  - `PretrainExp` / `NTPTrainerConfig` in `ntp.py`
  - `SFTExp` / `SFTDataConfig` in `sft.py`
  - inference configs in `inference.py`
- Common training configs:
  - `AdamConfig`
  - constant / linear / cosine schedulers
  - checkpoint config (`SaveOptions`, `LoadOptions`, `CheckpointConfig`)

### Experiment workflow

- After creating or editing an experiment:
  - run `cfshow <exp.py>` to inspect the config tree
  - for nested playground files, use `PYTHONPATH=/data/SteptronOss cfshow <exp.py>` so imports like `from playground...` resolve
  - make sure `sanity_check()` passes
  - run `mypy <exp.py>`
- If experiment B is derived from experiment A, use `cfshow` diff to verify changes.

### Pretrain config notes

- Pretrain configs live under `playground/pretrain/`
- `playground/pretrain/step3p5/step3p5_flash.py` is the main recent Qwen3 config reference
- When translating a full `ModelConfig` into `step3p5_flash.py`, update only existing attrs
- Some keys map indirectly:
  - `disable_qk_norm` ↔ `use_qk_norm` (inverted)
  - `use_swiglu_limit` ↔ `swiglu_limit`
- If you change `num_layers`, keep all layer-wise lists in sync:
  - `qk_rope_head_dim`
  - `rope_theta`
  - `use_fused_qknorm_and_rope`
  - `use_swiglu_limit`
  - `use_swiglu_limit_shared`

## 6. Parallelism and Checkpointing

### Parallel state

- Global `PM` in `steptronoss.core.parallel_state` is the `ParallelManager`
- Typical flow:
  - `PM.initialize()`
  - `PM.set_mesh(parallel_cfg)`
  - or `with PM.use_mesh(parallel_cfg): ...`
- Common helpers:
  - `PM.define_parallel(pattern, **sizes)`
  - `PM.size_of("TP")`
  - `PM.rank_in("DP")`
  - `PM.group_of("PP")`
  - `PM.ranks_of("EP")`
- VPP uses:
  - `virtual_pipeline_model_parallel_size`
  - `get_vpp_rank()`
  - `set_vpp_rank()`
- `model_cfg.pipeline_activation_cpu_offload` applies to both `PPScheduler` and `VPPScheduler`; it is implemented with `torch.autograd.graph.save_on_cpu(...)`, currently requires `model_cfg.recompute=True`, and should be treated as graph-preserving activation offload rather than a replacement for `recompute`.

### EP / TP sizing

- `ParallelConfig.sanity_check()` requires:
  - `WORLD_SIZE` divisible by attention MP size = `PP * TP * CP`
  - `WORLD_SIZE` divisible by MoE MP size = `PP * ETP * EP`
- For an 8-GPU run with `TP=8` and `EP=8`, set:
  - `expert_tensor_parallel_size=1`
  - otherwise MoE MP size becomes 64 and the config is invalid
- In mixed dense/MoE topologies, expert params are reduced over `EDP`, not dense `DP`; the current gradient manager compensates with `TP/EP` scaling on expert grad buffers before the `EDP` reduction, so check that path before blaming an apparent extra `EP` factor.

### Checkpoint reshape

- `steptronoss/checkpointing/reshape_ops.py` contains reshape primitives:
  - `VocabPad`
  - `ColumnParallel` / `RowParallel`
  - `KeepThisTP` / `KeepThisEP`
  - `GQAMergeQKV`
  - `FFNMergeGateUp`
  - `UnbindMoE`
  - `Rename`
  - `Inverse`
- Typical usage:
  - build `Script(src=..., op=..., dst=...)`
  - return `OnlineReshaper(scripts)`
- For expert slicing from per-expert keys:
  - use `Inverse(UnbindMoE(...)) + KeepThisEP()` before TP ops
- For research-only Qwen variants that add extra parameters, subclass `steptronoss.model.qwen_dense.QwenModel` to reuse the HF reshaper, and set `checkpoint_cfg.strict_load_model = False` when loading a base Qwen checkpoint that lacks the new params.

## 7. RLVR Priors

- `TrainableItem` should not pickle / serialize tokenizer instances; drop them in `__getstate__`
- `model_name` often includes `exp_id`; persist a template like `deployed-model-{EXP_ID}` so resume survives `exp_id` changes

## 8. Optimization Guidance

### Muon

- Use `MuonConfig.mark_muon_params(model)` before grouping
- In experiments, prefer overriding `optimizer_cfg` via a `GradientManagerConfig` subclass that sets `optimizer_cfg = MuonConfig`
- Leave distributed optimizer on, but avoid byte-level sharding
- For Muon tests, prefer composing existing reshape ops instead of inventing new ones
- In `playground/sft/step3/*muon*`, `Step3p5MuonConfig.mark_muon_params` inlines the base Muon selection rules but tags trainable params with `ndim >= 2` as Muon candidates (still respecting embedding/name exclusions); Step3.5 Flash `GroupedExperts` merge ops use `UnbindMoE + Inverse(Column/RowParallel + KeepThisTP(group="ETP"))`.

### Triton workflow

- Follow:
  - `docs/TRITON_ACCELERATION_WORKFLOW.md`
  - `docs/TRITON_ACCELERATION_WORKFLOW_ZH.md`
- Rules:
  - optimized implementations belong under `steptronoss/model/optimizations/*`
  - semantic entrypoints stay in `steptronoss/model/utils/*`
  - expose alternatives through `@optimizable(...)`
  - alternatives must be strict drop-in replacements
  - add correctness tests, backward-aware benchmarks, and a short real experiment trace

## 9. Tests and Tooling

### Tooling notes

- `rg` may be unavailable; fall back to `find` / `grep`
- `python` may be missing and `python3` may not include `pytest`; prefer project tooling if available
- For compiled SFT datasets under `/oss/data/recipe_*`, prefer repo `.venv/bin/python` and `.venv/bin/torchrun`.
- `tests/conftest.py` now applies a shared skip to every `@pytest.mark.node2` test unless the run is launched under `torchrun --nproc-per-node=2`; plain `pytest` should skip them instead of hanging in distributed init.

### GPU test notes

- This environment may not have worker / GPU access; avoid running GPU-only tests when the machine does not actually have GPUs
- `@pytest.mark.node2` tests should also use `pytest.mark.xdist_group("torchrun")`
- Test layout:
  - single-node GPU tests: `tests/test_muon_optimizer.py`
  - 2-node GPU tests: `tests/test_muon_optimizer_node2.py`
- `steptronoss/model/ep_dispatcher/deepep_dispatcher.py` must keep `recv_token_probs` differentiable and pass `grad_recv_token_probs` into `buffer.combine(...)`; otherwise router main-loss gradients are cut when `TokenDispatcher="deep_ep"`.

## 10. Debugging Priors

- `steptronoss.utils.memory_tracker.CMT` only records when `MEM_DIAGNOSE=1`
- If training hangs on `Waiting for debugger... ip: ... rank: 56`, check for a stray `debug(56)` in `steptronoss/core/trainers/lm_trainer.py`
- TorchDynamo graph breaks are often triggered by `Tensor.item()` in optimizable helpers; prefer tensor-safe checks like masked `amax` + `torch._assert`
- `steptronoss/model/common/rope.py` should keep RoPE cos/sin caches and cache-generation math in `torch.float32`; module-wide `.to()/cuda()/bfloat16()` may move the cache device, but must not downcast the cache dtype.
- For scratch-init MoE research variants, do not only initialize `module.weight`; `GroupedExperts` stores expert matrices as named parameters `w1` / `w2` on the module. Missing those leaves `torch.empty(...)` garbage in expert weights and can cause immediate NaNs on the first iteration.

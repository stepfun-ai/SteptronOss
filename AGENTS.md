## Principle for AI coding
### Self-Improvement Loop

Not every interaction needs an improvement pass. But for **key tasks**, do an Improve Pass at the end.

**Key tasks** (trigger conditions—any one qualifies):

- **New environment**: First time in a workspace / new machine / new permission boundary
- **New task type**: First time running a job/tool/workflow with many unknowns
- **New project**: New repo / service / training task / deliverable
- **High risk/cost**: Could affect production, security, permissions, data, or significant cost
- **Collaboration/handoff**: Others need to reuse your conclusions, or future review needed

**Improve Pass flow**:

1. **Identify friction**: What took the most time / was most uncertain / easiest to trip on?
2. **Extract prior**: Turn friction into reusable "priors" (defaults, boundaries, naming, paths, failure modes)
3. **Write it down** (at the right level, keep it short):
   - repo `AGENTS.md`
   - Detailed procedures → `docs/`

## Memory
### Code Style
- Doc style: Config class fields should include a short triple-quote docstring immediately after the attribute definition.

- Config style: follow `configurize` pattern—class attrs declare sub-config types, instance `__init__` sets concrete values and `Ref("..path")` for cross-node links; configs expose `build()`/`build_*`, `sanity_check()`, `to_dict()`.
- SFT experiment style (`playground/sft/qwen3/*_sft_step3_data.py`): `Exp(BaseExp)` sets `model_cfg`/`data_cfg` as class attrs, tweaks trainer/checkpoint/model fields in `__init__`, and runs with `if __name__ == "__main__": Exp().train()`.

### Existing Utils

- `steptronoss/utils`: `arguments.parse_args` config overrides; `comm_utils` Redis rendezvous/queue + `LocalFuture`/`RemoteFuture`; `dist_utils` broadcast/all_to_all helpers, dict<->tensor packing, list balancing; `general` numeric helpers, list split/balance, RNG fork, retry, recur_to, git hash; `logger` rank-aware log + `StepWriter`; `metrics` Metric/Avg/Percentage/Histogram/Text/GradNorm and `GlobalMetrics`; `optimizable` decorator + `set_optimization`; `utils` model unwrap, param norm, mem report, layer map, jsonl/msgpack IO, generic load; `weight_loader` HF safetensors key mapping/merge.
- Checkpoint reshape: `steptronoss/checkpointing/reshape_ops.py` provides `ReshapeOp` primitives (e.g., `VocabPad`, `ColumnParallel`/`RowParallel`, `KeepThisTP/EP`, `GQAMergeQKV`, `FFNMergeGateUp`, `UnbindMoE`, `Rename`, `Inverse`) and `OnlineReshaper` + `Script` to map HF ↔ ST keys. Usage pattern in `steptronoss/model/qwen_dense.py`: `build_reshaper()` builds a list of `Script(src=..., op=..., dst=...)` and returns `OnlineReshaper(scripts)`. New ops: implement `forward` (HF→ST piece) + `backward` (ST→HF), compose with `+` (Sequential), and use `Script` patterns to select keys.

- Muon optimizer: use `MuonConfig.mark_muon_params(model)`; it uses config fields to set `param.is_muon_param` before grouping.
- Muon in experiments: override `optimizer_cfg` with a `GradientManagerConfig` subclass that sets `optimizer_cfg = MuonConfig` (keeps configurize pattern); leave distributed optimizer on but avoid byte-level sharding.
### Local Priors

- Repo layout: core package under `steptronoss/` (core, model, data, exp, optimizer, generation, tokenizer, utils, checkpointing); experiments live in `playground/`; tests in `tests/`.
- Parallel state: global `PM` in `steptronoss.core.parallel_state` is a `ParallelManager`. Typical flow: `PM.initialize()` → `PM.set_mesh(parallel_cfg)` (or `with PM.use_mesh(parallel_cfg): ...`) where `parallel_cfg.build_parallel()` returns `{name: ranks}`. Helpers: `PM.define_parallel(pattern, **sizes)` to build rank grids; `PM.size_of("TP")`, `PM.rank_in("DP")`, `PM.group_of("PP")`, `PM.ranks_of("EP")`. VPP uses `virtual_pipeline_model_parallel_size` + `get_vpp_rank()/set_vpp_rank()`.
- Exp/module definitions: `steptronoss/exp` provides abstract `*Config` interfaces (`build_*`/`get_trainer_cls`) and concrete configs in `base_exp.py` (GradientManagerConfig, TrainerConfig data-source gating + sync broadcast, ParallelConfig w/ TP/PP/DP/CP/EP/ETP groups, MegatronTP/PP/3D configs). Ready-made exps: `PretrainExp`/`NTPTrainerConfig` + metrics in `ntp.py`, `SFTExp`/`SFTDataConfig` in `sft.py`, inference configs in `inference.py` (VLLMInferenceConfig, InferencableModelConfig), plus `AdamConfig`, LR schedulers (Constant/Linear/Cosine), and checkpoint config (Save/LoadOptions, CheckpointConfig). New modules follow: subclass config → implement `build_*`/`get_trainer_cls` → wire into `Exp` as class attrs with `Ref(...)` links.
- Pretrain config notes: pretrain configs live under `playground/pretrain/`; `step3p5_flash.py` defines the Qwen3 config used in recent edits.
- Pretrain edit guardrails: when translating a full `ModelConfig` into `step3p5_flash.py`, update only existing attributes; some keys map indirectly (e.g., `disable_qk_norm` ↔ `use_qk_norm` inverted, `use_swiglu_limit` ↔ `swiglu_limit`). If you change `num_layers`, keep any layer-wise lists in sync (e.g., `qk_rope_head_dim`, `rope_theta`, `use_fused_qknorm_and_rope`, `use_swiglu_limit`/`use_swiglu_limit_shared`).
- Tooling note: `rg` may be unavailable in this environment; fall back to `find`/`grep` for repo-wide search.
- Test running note: `python` may be missing and `python3` may not have `pytest` installed; use project tooling if available.
- GPU note: this environment has no worker/GPU access; avoid running GPU-only tests here.
- Tests note: `@pytest.mark.node2` tests should be marked non-parallel with `pytest.mark.xdist_group("torchrun")` (see `tests/test_moe_layers.py` pattern).
- Test organization: single-node GPU tests live in `tests/test_muon_optimizer.py`; 2-node GPU tests live in `tests/test_muon_optimizer_node2.py`.
- Muon tests: prefer composing existing reshape ops (e.g., `ColumnParallel() + KeepThisTP()`) instead of adding custom ReshapeOp classes or modifying `muon.py`.

# New Model Support Notes

This document summarizes what worked well when integrating a new model into
StepTronOSS, with a focus on patterns that reduce rework. Most of the concrete
examples come from the GLM-5 integration, but the goal is to generalize them
into a reusable workflow for future models.

The main objective is not to "get the code written quickly", but to establish a
minimal closed loop that is verifiable, debuggable, and extensible.

## 1. Define the Minimum Support Milestones First

It is useful to break "new model support" into explicit stages, each with a
clear acceptance criterion:

1. Config can be instantiated:
   - `cfshow <exp.py>` expands correctly.
   - `sanity_check()` passes.
   - The model can run one forward pass on fake input.
2. Toy model can be checked against a reference:
   - Keep the full structure and only reduce layer count.
   - Load deterministic weights.
   - Reduce parallelism to `1` first and compare layer by layer.
3. Full model can train on fake data:
   - First make `train_iters=1` pass, then raise to `3`.
   - Do not chase long sequence length yet; first close the full training path.
4. Official checkpoint can be loaded:
   - Reshape is correct.
   - Logits / hidden states / router outputs can be compared to a reference.
5. Real-data training works:
   - Data compilation, tokenizer, and chat template are aligned.
   - Run at least `10` iterations and inspect loss stability.
6. Optimized implementations are switchable:
   - Python reference and optimized kernels both go through the same semantic
     entrypoints.

In practice, it is only worth moving to the next stage once the current one is
stable. Otherwise, when something breaks later, it becomes impossible to tell
whether the root cause is model code, data, weights, parallelism, or kernels.

## 2. Put Files and Responsibilities in the Right Place

Follow the existing repo layout:

- Semantic model implementation goes in `steptronoss/model/<model>.py`
- Training / parallel configs go under `playground/pretrain/<model>/` and
  `playground/sft/<model>/`
- Model-specific optimization entrypoints go in
  `steptronoss/model/utils/<model>_utils.py`
- Data compatibility logic should live on the `playground/data/...` side first,
  not in shared core logic

GLM-5 is a good reference:

- `playground/pretrain/glm5/glm5.py`
- `steptronoss/model/glm5.py`
- `steptronoss/model/utils/glm5_utils.py`
- `playground/sft/glm5/glm5_sft_base.py`

A few useful rules:

- Put the semantically correct implementation in Python first.
- Do not hardwire TileLang / Triton / Flash / custom kernel logic directly into
  the main semantic path.
- If data-template compatibility is model- or dataset-specific, solve it in the
  corresponding `data_config` instead of polluting shared `steptronoss/` code.

For reduced-size toy models, keep a separate path. For GLM-5, the toy path is:

- `playground/pretrain/glm5/glm5_toy.py`
- `playground/sft/glm5/glm5_toy_sft_step3_data.py`

The point of a toy model is not convenience. The point is to preserve the full
structure at smaller scale so that parity checks later have a stable target.

## 3. Build a Toy Model Before the Full Model

The most effective integration order is usually not "jump straight to the full
model on real data". A better sequence is:

1. Build a toy version with the full architecture and fewer layers.
2. Use deterministic weights for reference checks.
3. Reduce all parallelism to `1` so implementation error and parallelism error
   can be separated.
4. Once the toy model matches, switch to full-model fake-data training.
5. Once fake-data training is stable, move to real data.

The key rule is that the toy model must keep the complete structure. If the
structure is simplified away for convenience, many real issues never show up,
especially:

- MoE router / expert combine behavior
- Special attention paths
- RoPE / YARN details
- packed vs non-packed mask differences
- special checkpoint reshape logic

## 4. Read the Official Checkpoint Format Before Writing Reshape Logic

One of the easiest mistakes during new-model integration is assuming the weight
layout already looks like an existing model. Always inspect the official
checkpoint keys first and only then write reshape logic.

GLM-5 is a representative example:

- Official HF routed expert weights are stored per expert:
  - `model.layers.<L>.mlp.experts.<E>.gate_proj.weight`
  - `model.layers.<L>.mlp.experts.<E>.up_proj.weight`
  - `model.layers.<L>.mlp.experts.<E>.down_proj.weight`
- This is different from models that assume a pre-merged `gate_up_proj`.

In StepTronOSS, these weights need to be re-stacked from per-expert layout
before EP/TP slicing. The GLM-5 implementation in `steptronoss/model/glm5.py`
uses reshape scripts centered around:

- `Inverse(UnbindMoE(moe_key_prefix="experts."))`
- followed by `KeepThisEP()` / `FFNMergeGateUp(group="ETP")` /
  `RowParallel(group="ETP")`

Practical takeaways:

- Model the key pattern first, then write reshape logic.
- Do not start with a giant "cover everything" reshape path. First make the
  dense layers, attention, and MoE expert path load correctly.
- Once reshape is wrong, all downstream loss, parity, and training stability
  signals become contaminated.

## 5. Keep Tokenizer, Chat Template, and Data Compilation Strictly Aligned

When a newly integrated model shows an abnormal training loss, the issue is
often not the model implementation but the prompt/data formatting.

Treat the following as hard requirements:

- The tokenizer used during compilation must exactly match the tokenizer used in
  training.
- Read the official `chat_template.jinja`; do not infer the format from other
  models.
- Decode compiled samples and inspect the final prompt text.
- Explicitly verify that tool calls and tool observations are consumed
  correctly by the template.

GLM-5 hit a classic compatibility issue:

- The official template expects visible text in `item.text`
- Tool observations are expected in `tr.output`
- But `StepChatJsonDataset` commonly emits content parts like
  `{"type": "...", "value": "..."}`

So the template cannot be used directly. Otherwise:

- user / assistant visible text may be silently dropped
- tool outputs may be serialized in the wrong form

The current compatibility fix lives in
`playground/data/sft/internal260313/step_sft_data_config0313_glm5_tokenizer.py`:

- For normal text parts, remap `value -> text`
- For `role == "tool"` parts, also remap `value -> output`

The broader lesson is:

> When the template is incompatible, prefer fixing the input schema in a
> model-specific data compatibility layer instead of rewriting raw data or
> changing shared core data code first.

Also keep in mind:

- Released `transformers` versions may not recognize a new `model_type`
- GLM-5 required source `transformers` mainline to load the HF config properly

If you run HF teacher-forced loss checks, remember another common pitfall:

- Do not pass manually pre-shifted `labels` into HF `labels=...`
- HF causal LM loss shifts internally, so double-shifting can inflate loss
  badly

## 6. Numeric Parity Must Be Decomposed, Not Reduced to Final Loss

The most effective way to debug parity is not to stare only at `lm_loss`, but
to decompose the comparison:

1. Separate parallelism error from implementation error:
   - Reduce all parallelism to `1`
   - Use the same execution path as the reference
2. Then compare module by module:
   - router output
   - dispatched token / weight tensors
   - expert output
   - attention output
   - block output
   - final logits / loss
3. Then compare at operator granularity:
   - before / after matrix multiply
   - before / after norm
   - before / after RoPE
   - before / after mask / gather / scatter

At minimum, log:

- `max_abs_diff`
- `mean_abs_diff`
- token-wise cosine similarity

Cosine similarity is especially useful because it quickly tells you whether the
vector direction is aligned, which is helpful for hidden states and router
input/output tensors.

During GLM-5 parity work, the highest-priority error sources were:

- attention mask / `cu_seqlens` / packed vs non-packed path differences
- RoPE cos/sin cache precision and dtype transitions
- bf16 / fp32 conversion timing
- RMSNorm epsilon mismatch
- whether MoE router weights already include a scaling factor

Concrete takeaways:

- `cu_seqlens`-style attention arguments are one of the most common causes of
  huge discrepancies on only a few tokens, so check them early.
- GLM-5 cannot directly reuse generic `YARNRoPE`; it needs the
  `Glm5YARNRoPE` path in `steptronoss/model/glm5.py`.
- RoPE cos/sin caches must stay in `fp32`; they must not be downcast together
  with a module-wide `.bfloat16()`.
- GLM-5 MLA-internal RMSNorm should use `mla_layernorm_epsilon = 1e-6`, while
  surrounding norms remain at `1e-5`.
- For GLM-5 MoE, HF `route_tokens_to_experts()` returns top-k weights that
  already include `routed_scaling_factor`, while StepTron
  `MoEBlock.forward_router()` returns normalized but unscaled weights and
  applies scaling later during combine. This semantic gap must be handled in
  parity checks.
- Even in non-packed attention, the DSA indexer path still needs a causal mask;
  `cu_seqlens is None` does not mean "no mask".

## 7. Custom Kernels Should Always Go Through Stable Semantic Entrypoints

The recommended pattern is:

- Define semantic entrypoints in `steptronoss/model/utils/<model>_utils.py`
- Provide a Python reference by default
- Attach optimized alternatives with `@optimizable(...)`
- Switch them in experiments via `set_optimization(...)`

GLM-5 already follows this pattern:

- `steptronoss/model/utils/glm5_utils.py`
  - `lighting_indexer(...)`
  - `sparse_mla(...)`
- `playground/sft/glm5/glm5_sft_base.py`
  - `set_optimization(lighting_indexer=..., sparse_mla=...)`

Why this helps:

- Semantic entrypoints stay stable while implementations are replaceable.
- Reference and optimized paths share the same upper-level model code.
- When parity breaks, it is easy to switch back to the reference path for
  diagnosis.

Additional lessons:

- Before the root cause is known, prefer monkey-patching or switching
  implementations in experiment scripts rather than modifying the formal path
  immediately.
- Once the issue is localized, then fold the fix back into the real
  implementation.
- If the optimized path depends on extra runtime requirements, validate the
  actual worker Python and CUDA toolchain.

For TileLang, the GLM-5 integration specifically needed:

- repo `.venv/bin` first in `PATH`
- `PYTHONPATH` including both the repo and repo `.venv` site-packages
- `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
- `CUDACXX=$CUDA_HOME/bin/nvcc`

Without these checks, it is easy to waste time on fake issues where local import
works but workers are actually running a different Python/CUDA environment.

## 8. Upgrade Training in Stages: Fake Data -> Real Data -> Long Sequence

The recommended bring-up order is:

1. single-node toy / fake-data
2. full-model fake-data
3. real data at short sequence length
4. real data at long sequence length
5. optimized kernels plus long sequence

For multi-node experiments, prefer the existing launch paths in the repo rather
than inventing a new submit flow. Useful paths include:

- `platform/rlaunch_run.py <exp.py> mm_pretrain`
- `rlaunch -d ... -- tools/smartrun <exp.py>`

During debugging, keep two operational issues in mind:

- In preemptible mode, workers may exit mid-run, so first verify whether the
  failure is actually a model problem.
- Do not casually interrupt the local launcher with `Ctrl+C`; on the current
  platform this often stops the remote `rjob` as well.

To inspect or stop jobs, prefer:

- `brainctl -n shai-core get rjob <name>`
- `brainctl -n shai-core stop rjob/<name>`

The practical lesson is simple: separate platform failures from training
failures. First confirm whether workers disappeared, NCCL failed to initialize,
or a bad GPU node was involved before blaming the model implementation.

## 9. For Long Sequence Length, Distinguish Implementation Limits from Memory Limits

When tuning long sequence length, the biggest mistake is treating every OOM as
the same problem.

A better method is:

1. First make short-sequence training stable for `3` iterations.
2. Enable necessary optimizations such as FlashAttention.
3. Increase sequence length step by step with a fixed schedule.
4. For every failure, record:
   - failing iteration
   - failing rank
   - exact OOM stack
   - whether it happened in attention / indexer / optimizer / grad norm

GLM-5 exposed at least two very different OOM categories:

- optimizer OOM:
  - experiment-local Adam created a large temporary inside `_single_tensor_adam`
  - switching to `fused=True` removed that issue
- attention / indexer OOM:
  - inside `steptronoss/model/utils/glm5_utils.py`, the `sparse_mla` path
    computes
    `scores = torch.einsum("qhd,qtd->qht", q_chunk.float(), gathered_k.float()) * scaling`
    which explodes with sequence length

These need completely different fixes, so classify them first.

One useful GLM-5 result on the current branch is:

- For real-data `playground/sft/glm5/glm5_sft_zhy.py`
- on `32 x 8` H100
- the stable ceiling for `train_iters=3` is roughly `6144`
- `8192` and above OOM on the first train step in the `sparse_mla` path

This indicates that the current bottleneck is no longer just optimizer state,
but the complexity of the DSA / sparse MLA implementation itself.

General guidance:

- `pipeline_activation_cpu_offload` and `offload_optimizer_state` are useful as
  enablement tools, but should not be mistaken for the final solution.
- Prove short-sequence stability first, then push long-sequence targets.
- If the goal is `64k/96k/128k`, first determine whether quadratic growth comes
  from the mask, indexer, attention scores, or KV path.

## 10. Common Pitfalls Checklist

Before declaring support for a new model, it is worth checking at least the
following:

### Config and structure

- Did you build a toy version first instead of going straight to full-model real
  data?
- If `num_layers` changed, were all layer-wise lists updated together?
- Do model-specific constants match the official values, such as epsilon, rope
  theta, head dim, and qk norm behavior?

### Checkpoint

- Did you inspect official checkpoint keys instead of reusing reshape
  assumptions from another model?
- Are MoE expert weights merged or per-expert?
- After reshape, did you do a small-scale load plus layer-wise parity checks?

### Tokenizer and data

- Does compile-time tokenizer exactly match train-time tokenizer?
- Is the official `chat_template.jinja` actually loaded?
- Were compiled samples decoded and inspected?
- Are stepchat / toolcall / tool observation structures correctly encoded by the
  template?
- Is data compatibility handled in model-specific data config instead of shared
  core code?

### Numeric parity

- Was parallelism reduced to `1` first?
- Did you compare router outputs rather than only final loss?
- Did you log cosine similarity?
- Did you check mask logic, `cu_seqlens`, RoPE cache dtype, and RMSNorm eps?
- For HF loss comparisons, did you avoid double-shifting labels?

### Kernel and optimization path

- Is there still a Python reference path?
- Are alternative implementations switched via `@optimizable` +
  `set_optimization(...)`?
- Does the worker actually run the Python/CUDA environment you think it does?

### Training and resources

- Did you go fake-data -> real-data -> long-sequence in order?
- When a multi-node job failed, did you first rule out worker/node issues before
  blaming model code?
- Did you avoid accidentally killing the local launcher and therefore the remote
  job?

## 11. Recommended Integration Order

For the next brand-new model, the recommended sequence is:

1. Build the directory skeleton and configs.
2. Write a toy model that keeps the full structure and only reduces layer count.
3. Write a Python reference and do deterministic parity checks at parallelism
   `1`.
4. Write checkpoint reshape logic and confirm official weights can load.
5. Add tokenizer / template compatibility, compile a small dataset, and decode
   spot-check samples.
6. Run fake-data training for `1 -> 3` iterations.
7. Run real-data training for `10` iterations and inspect loss.
8. Only then move to optimized kernels, long sequence length, and throughput
   limits.

This order looks slower at first, but in practice it is the fastest. Each stage
creates a stable baseline, which makes it much easier to fall back to the last
verified state when something breaks later.

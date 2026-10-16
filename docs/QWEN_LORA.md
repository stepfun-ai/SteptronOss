# Qwen hybrid-model text LoRA

Experimental integration of the Transformers `qwen3_5` hybrid architecture into
SteptronOSS. The example names retain the Qwen3.8-27B experiment designation;
the actual model directory must contain a compatible, original BF16/FP32
`qwen3_5` checkpoint. No model weights or business datasets are included.

## Architecture and scope

- SteptronOSS owns configuration, the training loop, FW/BW scheduling, optimizer
  scheduling, data cursors and checkpoint orchestration. This does not launch
  Transformers Trainer, vLLM or another training framework.
- Transformers provides the hybrid Gated DeltaNet / full-attention model; PEFT
  injects LoRA into text decoder Linear layers. The base, vision tower and LM
  head remain frozen. This is not a native TP/PP Qwen implementation.
- The DP entrypoint keeps a complete base model on each GPU. FSDP2 instead loads
  the base on CPU rank zero, constructs other ranks on meta, and shards base
  weights and FP32 adapters over a flat all-rank mesh.
- Optional CP=2/4 splits each sample into contiguous, 64-token-aligned blocks.
  Independent data groups equal `world_size / CP`. Global batch equals micro
  batch times accumulation times independent data groups.
- Chunked loss bounds LM-head logits memory. CP full-attention backward replays
  full queries to preserve BF16 gradient accuracy; this adds compute and is not
  an optimized throughput claim.
- No full-parameter training, QLoRA, image/video training, TP/PP/EP, elastic
  checkpoint resharding or sequence-chunking backend is included.

## Dependencies

Use a dedicated environment. The model integration is tested with Transformers
5.19.0 and PEFT 0.21.2; FSDP2 requires PyTorch >=2.11. The default project
environment and lockfile remain unchanged: its vLLM dependency has separate
PyTorch constraints. This training-only environment does not install vLLM.

```bash
uv venv --python 3.12 .venv-qwen38
# Linux CUDA 12.8 example; choose the torch index appropriate for your GPU/driver.
uv pip install --python .venv-qwen38/bin/python torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen38/bin/python \
  -r requirements/qwen38-fsdp2-runtime.txt
source .venv-qwen38/bin/activate
```

For CPU reference tests, install a CPU-compatible PyTorch build instead.
FlashAttention 2 controls full attention; Gated DeltaNet separately
uses FLA and causal-conv1d when available. Those optional kernels must match the
installed PyTorch/CUDA ABI. CPU reference tests do not require these kernels.

## Data preparation

Input JSONL contains string-content messages ending in an assistant turn.
Assistant outputs, including reasoning and end markers, are supervised;
prompt and padding tokens are masked. Per-turn `loss_mask=0` preserves input
tokens while removing that assistant turn from supervision. Tools and chat
template keyword arguments are passed to the tokenizer's chat template.
Oversized samples fail explicitly; they are not silently truncated or dropped.

```json
{"messages":[{"role":"user","content":"Explain gradient accumulation."},{"role":"assistant","content":"Accumulate micro-batch gradients before one optimizer step."}]}
```

Prepare at least one complete global batch. The loader drops an incomplete
global-batch tail each epoch and reshuffles if training continues.

```bash
export PYTHONPATH="$PWD"
MODEL=/models/qwen-hybrid-27b
OUT=/shared/qwen-lora
python tools/prepare_qwen3_8_sft.py \
  --model "$MODEL" --input /data/train.jsonl \
  --output "$OUT/train.tokens.jsonl" --max-length 16384
```

Use `--enable-thinking` when appropriate for your dataset. For a Hub model, pass
the same pinned `--revision` and `model_cfg.revision` during preparation/training.
The loader verifies the tokenizer identity, token-file hash and sequence limit.

## Parameterized recipes

These are thin native `Config` experiments, not a second YAML or environment
variable configuration layer:

- DP: [experiment](../playground/sft/qwen3_8/qwen3_8_27b_lora.py).
- FSDP2/CP: [experiment](../playground/sft/qwen3_8/qwen3_8_27b_fsdp2_lora.py).

For example, after preparation:

```bash
PAD_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pad_token_id"])' \
  "$OUT/train.tokens.jsonl.meta.json")
torchrun --standalone --nproc-per-node=8 \
  playground/sft/qwen3_8/qwen3_8_27b_fsdp2_lora.py \
  model_cfg.model_path="$MODEL" \
  data_cfg.tokenized_path="$OUT/train.tokens.jsonl" \
  data_cfg.pad_token_id="$PAD_ID" \
  model_cfg.lora_rank=16 model_cfg.lora_alpha=32 \
  model_cfg.parallel_cfg.context_parallel_size=1 \
  trainer_cfg.global_seq_length=16384 \
  trainer_cfg.micro_batch_size=1 trainer_cfg.global_batch_size=8 \
  trainer_cfg.train_iters=100 \
  scheduler_cfg.total_schedule=100 scheduler_cfg.warmup_schedule=0 \
  scheduler_cfg.lr=1e-4 scheduler_cfg.min_lr=1e-5 \
  checkpoint_cfg.save_dir="$OUT/checkpoints"
```

The recipe explicitly uses the native cosine scheduler. Learning rate, warmup,
weight decay, seed, LoRA targets/dropout, resource counts, save interval and all
other declared configuration fields remain overridable. Changing a numeric
scheduler field does not change its scheduler class. Inspect with `cfshow`;
choose a different scheduler class in a derived experiment.

CP uses `model_cfg.parallel_cfg.context_parallel_size=2` or `4`; the world size
must be divisible by CP, and the sequence limit by `CP * 64`. Update preparation
and training limits together. No automatic hardware-capacity guarantee is made.

## Activation CPU offload

`model_cfg.fsdp2_activation_cpu_offload=False` is the default. This optional
memory-saving strategy stores saved activations on CPU, not model parameters
or optimizer state. It trades host RAM and CPU/GPU transfers for GPU memory;
leave it off when the chosen workload fits without it.

Enable it explicitly only with per-layer recomputation:

```text
model_cfg.gradient_checkpointing=True
model_cfg.fsdp2_activation_cpu_offload=True
model_cfg.fsdp2_activation_cpu_offload_pin_memory=False
```

CP needs nonblocking activation unpack to avoid circular waits between GPU
communication and CPU copies. The custom CP path preserves autograd edges;
CP=1 uses PyTorch `save_on_cpu`. This implementation detail must not be confused
with a fully generic or zero-overhead offload facility.

Existing 27B long-context capacity measurements used offload. Small-model
on/off equivalence does not prove that 27B/200K fits with offload disabled.

## Save, export and resume

Checkpoints contain rank-local adapter and optimizer shards, scheduler, data
cursor and RNG. The shared checkpoint directory must be visible to every rank.
Saving is synchronous; `latest_ckpt` is published only after all rank files and
standard PEFT adapter export complete.

Full resume requires the same world size and CP degree. Model-only loading may
change CP with optimizer/scheduler/data/RNG/iteration/experiment restoration
disabled. Exported PEFT adapters can be loaded without SteptronOSS or FSDP2.
This contribution does not include platform-specific CPU merge or object-store
publication.

## Validation and evidence boundaries

CPU fixtures construct tiny models locally: no weight download or private
dataset is needed.

```bash
uv pip install --python .venv-qwen38/bin/python pytest
python -m pytest -q tests/test_qwen3_8_sft.py tests/test_qwen3_8_fsdp2.py \
  tests/test_qwen38_cp_contracts.py tests/test_qwen3_8_context_parallel.py
torchrun --standalone --nproc-per-node=2 -m tests.qwen38_cp_validation \
  --cp 2 --device cpu --dtype fp32 --recompute
```

CUDA lifecycle tests are in `test_qwen3_8_sft_gpu.py` and
`test_qwen3_8_fsdp2_gpu.py`. CP CUDA validation additionally accepts
`--mode trainer --dtype bf16 --attention flash_attention_2 --recompute
--short-padding`, optionally `--offload`, with 2/4/8 workers.

Cleanup validation on macOS arm64, Python 3.12.12, PyTorch 2.14.1,
Transformers 5.19.0 and PEFT 0.21.2:

- 35 focused tests passed; 6 CUDA/multi-process-only tests skipped in plain pytest.
- 14 existing CLI/config/native-Qwen regression tests passed; 2 CUDA tests skipped.
- Real CPU/gloo CP2 with 2 workers and CP4 with 8 workers (2 independent data
  groups) passed GDN, attention, mixed-stack, frozen-prefix and empty-query tests.
- The CPU FSDP2 lifecycle test uses a single-rank fake process group; it does not
  validate CUDA kernels or multi-rank FSDP2 collectives.
- Both experiment `cfshow`/sanity checks, focused mypy, Ruff lint and formatting
  passed. CPU numerical checks used `TORCHDYNAMO_DISABLE=1` and
  `OMP_NUM_THREADS=1`; no compiled CUDA kernel performance is claimed.
- Default environments lacking the optional model dependencies skip the
  corresponding integration tests explicitly.

Prior implementation acceptance, before this cleanup, used 8 x A800 80 GiB:

| CP | Sequence limit | Result | Peak allocated / reserved GiB |
|---|---:|---|---:|
| 2 | 32,768 | 3 updates, checkpoint and adapter export | 23.99 / 31.58 |
| 4 | 49,152 | 3 updates, checkpoint and adapter export | 22.39 / 29.71 |
| 4 | 163,840 | 3 updates, checkpoint and adapter export | 43.38 / 52.55 |
| 4 | 201,728 | 3 updates, checkpoint and adapter export | 51.01 / 60.05 |

All used r16/alpha32, BF16 base, FP32 adapters, micro batch 1, global batch 8,
recomputation, activation offload without pinned staging and loss chunks of 128.
The 32K/48K runs used variable-length inputs: their configured limits are not
exact-length capacity proofs. The 163,840/201,728 inputs were synthetic capacity
fixtures, not quality evaluation. The longest run took about 34 minutes/update;
an initial time-limited attempt failed before a separately bounded rerun passed.
Those earlier GPU results are evidence for the pre-cleanup implementation,
not a claim that this cleaned PR was rerun on GPUs. No private raw logs,
datasets, deployment addresses or credentials are included.

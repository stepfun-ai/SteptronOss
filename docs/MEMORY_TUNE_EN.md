# Memory Tuning and Debugging (MEMORY_TUNE)

Quick reference for GPU memory tuning and OOM debugging in StepTron training runs.

## 1. Purpose and Scope
- For SFT / pretrain / RL training experiments
- Handle OOMs caused by GPU type or data changes; also improve throughput within safe memory limits

## 2. Main Memory Sources (Static vs Dynamic)
From `weight(params)` and `activation`:
### 2.1 Static Memory (Weights/Optimizer)
Dominated by parameters and optimizer states:
- Model parameters (including MoE experts)
- Optimizer states (Adam / distributed optimizer)
- Persistent communication buffers (in some setups)

Rough BF16 static memory estimate (N = number of parameters):
```
total = 2N(bf16 param) + 4N(fp32 grad_acc buff & allreduce buff) + 12N(fp32 param & momentum1/2)
# for zero1 setting, the 12N can be sharded across DP.
```

**Tuning knobs**:
- Adjust `parallel_cfg` (TP/PP/DP/CP/EP)
- Adjust `layermap` (PP/VPP layer distribution)

### 2.2 Dynamic Memory (Activations)
Driven by forward/backward activations and intermediate tensors:
- activation / intermediate tensors
- temp buffers from pipeline or CP/SP splitting

**Tuning knobs**:
- enable/raise `recompute`
- use `SP/CP` to reduce per-GPU activation size

### 2.3 Overlap of Static + Dynamic
- They roughly add up; peaks often occur near FW/BW and optimizer steps.
- `trainer_cfg.offload_optimizer_state=True` moves optimizer state to CPU, reducing overlap peaks.

## 3. Key Parameters (Model / Data / Trainer)

### 3.1 Model
- Parallelism defines per-GPU load (`model_cfg.parallel_cfg`)
- LayerMap imbalance creates hotspot ranks (`model_cfg.pp_vp_allocation(...)`; see `playground/sft/step3/step3_flash_sft_step3_data_muon.py`)
- MoE: usually avoid changing `model_cfg.ffn_cfg.moe_cfg`; tune **EP size** instead (`model_cfg.parallel_cfg.expert_model_parallel_size`)
- `model_cfg.recompute=True` reduces activation memory at compute cost

### 3.2 Data
- `trainer_cfg.global_seq_length`: memory grows with seq_len but **not algorithmically equivalent**, so **least recommended** (`trainer_cfg.global_seq_length`)
- Packed dataloader: `max_packing_seqlen` usually follows `global_seq_length` (`data_cfg.*.max_packing_seqlen`)
- Too large -> higher memory; too small -> dropped samples

### 3.3 Trainer
- `trainer_cfg.micro_batch_size`: strong impact on activation memory; **typically kept at 1** (`trainer_cfg.micro_batch_size`)
- `trainer_cfg.global_batch_size`: DP scaling increases gradient buffer/comm (`trainer_cfg.global_batch_size`)
- `trainer_cfg.offload_optimizer_state=True`: move optimizer state to CPU, reduce peak
- `trainer_cfg.empty_unused_memory_level`: frees cache but may hurt performance

## 8. Debugging and Triage
- After a parameter change OOM, rerun with `MEM_DIAGNOSE=1` to record memory marks
- Add marks around suspicious phases to narrow down spikes
- Example marks: `after_build_model`, `after_forward_backward`, `after_optimizer_step`
- Watch `Top1 ranks with most peak (allocated)` to find the worst rank
- **OOM at init**: check model size and parallel split; refine TP/PP/EP or layer map
- **OOM at first iter**: usually activation/optimizer state; tune `micro_batch_size` or enable `offload_optimizer_state`/`SP/CP`/`recompute`
- **Only some ranks high**: check PP/VPP layer map balance

## 10. Recommended Tuning Order
1. Estimate static memory vs GPU capacity. BF16+Adam ~ `18N` (N = #params). If static > capacity, first increase model parallel granularity or enable Zero (`distributed_optimizer`) and raise DP to shard static memory
2. If static is OK but FW/BW OOM, enable `trainer_cfg.offload_optimizer_state=True` to reduce overlap peak
3. If still OOM, use `SP/CP` to reduce per-GPU activation size (`model_cfg.parallel_cfg.context_parallel_size`, etc.)
4. If still OOM, progressively increase recompute (`model_cfg.recompute` or finer-grained scopes: `attention`/`feed_forward`/`ffn_norm`/`attn_norm`)

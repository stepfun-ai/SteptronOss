# 显存调参与排查（MEMORY_TUNE）
English version: `docs/MEMORY_TUNE_EN.md`

面向 StepTron 训练实验的显存调参与排查速查表。

## 1. 目的与适用范围
- 适用于 SFT/预训练/RL 等训练实验
- 处理换卡/换数据导致的 OOM；也可用于在安全范围内提升吞吐

## 2. 显存占用的主要来源（静态 vs 动态）
从 `weight(params)` 与 `activation` 两个角度理解：
### 2.1 静态显存（权重相关）
参数与优化器状态为主：
- 模型参数（含 MoE expert 参数）
- 优化器状态（Adam/分布式优化器等）
- 训练时常驻的通信缓冲（部分场景）

参考 BF16 训练时的静态显存粗略估算（N 为参数量）：
```
total = 2N(bf16 param) + 4N(fp32 grad_acc buff & allreduce buff) + 12N(fp32 param & momentum1/2)
# for zero1 setting, the 12N can be sharded across DP.
```

**调优手段**：
- 调整 `parallel_cfg`（TP/PP/DP/CP/EP）
- 调整 `layermap`（PP/VPP 各 rank 的层分布）

### 2.2 动态显存（激活相关）
前向/反向的激活与中间张量为主：
- activation / intermediate tensors
- pipeline 传输或 CP/SP 切分带来的临时缓存

**调优手段**：
- 开启/提高 `recompute`
- 使用 `SP/CP` 降低单卡激活峰值

### 2.3 静态与动态显存的叠加关系
- 默认近似叠加，峰值多出现在 FW/BW 与优化器步骤相邻阶段。
- `trainer_cfg.offload_optimizer_state=True` 会把优化器状态迁移到 CPU，显著降低重叠峰值。

## 3. 关键显存参数（按模型 / 数据 / Trainer 组织）

### 3.1 模型侧（Model）
- 并行规模决定单卡负载（`model_cfg.parallel_cfg`）
- LayerMap 不均会导致热点 rank（`model_cfg.pp_vp_allocation(...)`；见 `playground/sft/step3/step3_flash_sft_step3_data_muon.py`）
- MoE：一般不改 `model_cfg.ffn_cfg.moe_cfg`，主要调 **EP size**（`model_cfg.parallel_cfg.expert_model_parallel_size`）
- `model_cfg.recompute=True` 降激活显存，代价是算力

### 3.2 数据侧（Data）
- `trainer_cfg.global_seq_length`：显存随 seq_len 上升，但**算法不等效**，因此**最不推荐**（`trainer_cfg.global_seq_length`）
- Packed dataloader：`max_packing_seqlen` 通常跟随 `global_seq_length`（`data_cfg.*.max_packing_seqlen`）
- 过大显存偏高；过小样本丢弃

### 3.3 Trainer 侧（Trainer）
- `trainer_cfg.micro_batch_size`：强影响激活显存，**经验上通常为 1**（`trainer_cfg.micro_batch_size`）
- `trainer_cfg.global_batch_size`：DP 扩展 batch 会增加梯度缓存与通信（`trainer_cfg.global_batch_size`）
- `trainer_cfg.offload_optimizer_state=True`：优化器状态转移到 CPU，降低峰值
- `trainer_cfg.empty_unused_memory_level`：释放缓存但可能影响性能

## 8. 调试与监控与排查思路
- 修改参数后 OOM，可带上 `MEM_DIAGNOSE=1` 重新运行记录显存 mark
- 可在可疑位置新增 mark，缩小排查范围
- 关键 mark（例）：`after_build_model`、`after_forward_backward`、`after_optimizer_step`
- 重点关注 `Top1 ranks with most peak (allocated)`，定位“最吃显存的 rank”
- **OOM 出现在初始化**：检查模型参数与并行切分，先调细 TP/PP/EP 或 layer 分布
- **OOM 出现在第一个 iter**：多为激活/optimizer 状态过高，优先调 `micro_batch_size` 或启用 `offload_optimizer_state`/`SP/CP`/`recompute`
- **某些 rank 特别高**：检查 PP/VPP layer map 是否不均

## 10. 推荐调参策略（按优先级）
1. 先按参数量估算静态显存并对照显卡容量。BF16+Adam 粗略估算 `18N`（`N` 为参数量）。若静态显存已超过显卡容量，优先细化模型并行或启用 Zero（`distributed_optimizer`）并增大 DP 分担静态显存
2. 静态显存充足但 FW/BW OOM，先启用 `trainer_cfg.offload_optimizer_state=True` 降低重叠峰值
3. 仍不足时，使用 `SP/CP` 降低单卡激活规模（`model_cfg.parallel_cfg.context_parallel_size` 等）
4. 仍 OOM 时，逐级增加重计算：`model_cfg.recompute` 或更细粒度的 `recompute` 范围（`attention`/`feed_forward`/`ffn_norm`/`attn_norm`）

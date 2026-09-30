# 新模型接入经验总结

本文总结在 StepTronOSS 中接入新模型时，哪些做法最省时间、最不容易返工。内容主要来自本次 GLM-5 接入过程，但尽量抽象成后续模型也能复用的工作流。

重点不是“把代码先堆出来”，而是尽快形成一个可验证、可定位、可扩展的最小闭环。

## 1. 先定义“支持”的最小闭环

建议把“支持新模型”拆成下面几个阶段，每个阶段都要有明确验收条件：

1. 配置可实例化：
   - `cfshow <exp.py>` 能展开配置。
   - `sanity_check()` 能通过。
   - 模型能在 fake input 上跑通一遍 forward。
2. toy 模型可对点：
   - 保留完整结构，只缩减层数，不改语义。
   - 能加载 deterministic 权重。
   - 并行先缩到 `1`，和 reference 做逐层对点。
3. full model fake-data 可训练：
   - 至少能跑 `train_iters=1`，再拉到 `3`。
   - 先不追长序列，先保证真实结构训练链路闭环。
4. official checkpoint 可加载：
   - reshape 正确。
   - logits / hidden state / router output 能做 reference 对比。
5. real data 可训练：
   - 数据编译、tokenizer、chat template 都对齐。
   - 至少跑 `10` 个 iter 看 loss 曲线和稳定性。
6. 优化实现可切换：
   - Python reference 和优化 kernel 都能通过统一语义入口切换。

经验上，只有前一个阶段稳定了，才值得继续往后推。否则后面遇到问题时，根本分不清是模型实现、数据、权重、并行还是 kernel 的锅。

## 2. 目录和职责要从一开始就放对

推荐按现有模型组织方式接：

- 语义实现放在 `steptronoss/model/<model>.py`
- 训练/并行配置放在 `playground/pretrain/<model>/` 和 `playground/sft/<model>/`
- 模型相关的优化语义入口放在 `steptronoss/model/utils/<model>_utils.py`
- 数据兼容逻辑优先放在 `playground/data/...` 侧，而不是直接改通用核心逻辑

GLM-5 的落点可以直接当参考：

- `playground/pretrain/glm5/glm5.py`
- `steptronoss/model/glm5.py`
- `steptronoss/model/utils/glm5_utils.py`
- `playground/sft/glm5/glm5_sft_base.py`

几个组织原则：

- 新模型的“语义正确版本”先落在 Python 实现里。
- 自定义 kernel、TileLang、Triton、Flash 路径不要直接写死在主语义逻辑里。
- 数据模板兼容如果只是某个数据源/模型特有问题，优先在对应 `data_config` 做 compat，不要污染 `steptronoss/` 通用层。

如果是“缩小版 toy 模型”，也建议单独放一份，比如 GLM-5 的 toy 路径：

- `playground/pretrain/glm5/glm5_toy.py`
- `playground/sft/glm5/glm5_toy_sft_step3_data.py`

toy 的目标不是省事，而是保留完整结构、缩小规模，给后续对点留稳定靶子。

## 3. 先做 toy，再做 full model

最有效的接入顺序通常不是“直接上大模型真数据”，而是：

1. 先写 full architecture 的 toy 版本，只缩层数。
2. 用 deterministic 权重做 reference 对点。
3. 并行全部缩到 `1`，先把“实现误差”和“并行误差”拆开。
4. toy 对齐后，再切 full model fake-data。
5. fake-data 稳了，再上 real data。

这里最关键的经验是：toy 模型一定要“结构全，规模小”。如果为了偷懒把结构也简化掉，后面很多坑根本不会暴露出来，尤其是：

- MoE router / expert combine
- 特殊 attention 路径
- RoPE / YARN 细节
- packed / non-packed mask 差异
- 特殊 checkpoint reshape

## 4. checkpoint reshape 要先读官方权重格式，不要想当然

接新模型时，最容易犯的错误之一，就是默认它“长得像已有模型”。这一步一定先看官方 checkpoint key，再决定 reshape。

GLM-5 的一个典型例子：

- 官方 HF 权重里的 routed expert 是按 expert 分开的：
  - `model.layers.<L>.mlp.experts.<E>.gate_proj.weight`
  - `model.layers.<L>.mlp.experts.<E>.up_proj.weight`
  - `model.layers.<L>.mlp.experts.<E>.down_proj.weight`
- 这和一些已有模型里假设的合并后 `gate_up_proj` 不一样。

在 StepTronOSS 里，这类权重需要先按 expert 反向堆回去，再做 EP/TP 切分。GLM-5 当前实现参考 `steptronoss/model/glm5.py` 中的 reshape 脚本，核心是：

- `Inverse(UnbindMoE(moe_key_prefix="experts."))`
- 然后再 `KeepThisEP()` / `FFNMergeGateUp(group="ETP")` / `RowParallel(group="ETP")`

经验总结：

- 先对 key pattern 建模，再写 reshape。
- 不要一开始就追求“大而全”的 reshape；先让最核心的 dense 层、attention、MoE expert 路径加载正确。
- reshape 一旦错，后面的 loss、parity、训练稳定性全部会被污染。

## 5. tokenizer、chat template、数据编译要严格对齐

新模型接入时，训练 loss 不对，很多时候不是模型实现错了，而是数据模板没对齐。

强烈建议把下面几件事当成硬约束：

- compile 数据时使用的 tokenizer，必须和训练时加载的 tokenizer 完全一致。
- 必须读官方 `chat_template.jinja`，不要凭已有模板猜格式。
- 编译后一定要抽样 decode，看最终 prompt 到底长什么样。
- tool call / tool observation 这类结构化消息，要确认能被模板正确消费。

GLM-5 这里踩过一个很典型的坑：

- 官方模板期望可见文本在 `item.text`
- tool observation 在 `tr.output`
- 但 OSS 侧 `StepChatJsonDataset` 常见内容片段是 `{"type": "...", "value": "..."}`

因此不能直接套模板，否则会出现：

- user / assistant 文本被静默丢掉
- tool 输出被错误序列化

当前兼容做法在 `playground/data/sft/internal260313/step_sft_data_config0313_glm5_tokenizer.py`：

- 普通文本片段做 `value -> text`
- `role == "tool"` 的片段额外做 `value -> output`

这个经验可以抽象成一句话：

> 模板不兼容时，优先在模型专属的数据 compat 层修正输入结构，而不是改原始数据、也不是先去改通用核心数据管线。

另外还要注意：

- released `transformers` 可能还不认识新的 `model_type`
- GLM-5 需要 source `transformers` mainline 才能正确加载 HF config

如果要做 HF teacher-forced loss 对照，还要记住一个常见坑：

- 不要把已经手工 shift 过的 `labels` 再传给 HF 模型的 `labels=...`
- HF causal LM loss 会自己 shift，一旦双重 shift，loss 会被明显放大

## 6. 数值对齐要分层拆，不要只盯最终 loss

真正有效的对点方式，不是直接盯 `lm_loss`，而是逐层拆：

1. 先拆并行误差和实现误差：
   - 并行全缩到 `1`
   - 和 reference 使用同样的执行路径
2. 再拆模块：
   - router output
   - dispatch 后的 token / weight
   - expert 输出
   - attention 输出
   - block 输出
   - 最终 logits / loss
3. 再拆算子：
   - 矩阵乘前后
   - norm 前后
   - rope 前后
   - mask / gather / scatter 前后

建议至少记录这些指标：

- `max_abs_diff`
- `mean_abs_diff`
- token-wise cosine similarity

其中 cosine similarity 很有用，因为它能快速看出“整体方向对不对”，尤其适合比较 hidden state、router input/output 这类向量。

GLM-5 对点过程中，最值得优先排查的误差源有：

- attention mask / `cu_seqlens` / packed vs non-packed 路径
- RoPE cos/sin cache 的精度和 dtype 转换
- bf16 / fp32 转换时机
- RMSNorm epsilon 是否和官方一致
- MoE router weight 是否已经包含 scaling factor

几个具体经验：

- `cu_seqlens` 这类 attention 参数最容易导致“个别 token 出现巨大差异”，必须优先确认。
- GLM-5 不能直接复用通用 `YARNRoPE`，而是需要 `steptronoss/model/glm5.py` 里的 `Glm5YARNRoPE`。
- RoPE cos/sin cache 必须保持 `fp32`，不能因为模块整体 `.bfloat16()` 被一起降精度。
- GLM-5 的 MLA 内部 RMSNorm 要保持 `mla_layernorm_epsilon = 1e-6`，外围 norm 仍是 `1e-5`。
- 对于 GLM-5 MoE，HF `route_tokens_to_experts()` 返回的 top-k weights 已经乘过 `routed_scaling_factor`，而 StepTron 的 `MoEBlock.forward_router()` 输出的是归一化但未缩放的权重，真正的 scale 在后面 combine 时才生效。对 reference 时必须先对齐这个语义差异。
- 即使是 non-packed attention，DSA indexer 路径也仍然需要 causal mask；不能因为 `cu_seqlens is None` 就默认没有 mask。

## 7. 自定义 kernel 一定要走统一语义入口

推荐模式是：

- 在 `steptronoss/model/utils/<model>_utils.py` 定义语义入口
- 默认给一个 Python reference
- 用 `@optimizable(...)` 挂载优化替代实现
- 在实验里通过 `set_optimization(...)` 切换

GLM-5 当前就是这个模式：

- `steptronoss/model/utils/glm5_utils.py`
  - `lighting_indexer(...)`
  - `sparse_mla(...)`
- `playground/sft/glm5/glm5_sft_base.py`
  - `set_optimization(lighting_indexer=..., sparse_mla=...)`

这个模式的优点：

- 语义入口稳定，优化实现可替换。
- reference 和优化路径能共享同一套上层模型代码。
- 出现精度问题时，可以快速切回 reference 做定位。

额外经验：

- 在没找到误差根因前，优先在实验脚本里 monkey-patch 或切换实现，不要一上来就改正式实现。
- 真正定位到根因后，再把修复沉淀回正式路径。
- 如果优化实现依赖额外环境，务必在 worker 上验证实际解释器和 CUDA 工具链。

以 TileLang 为例，GLM-5 接入时实际需要确认：

- `PATH` 里 repo `.venv/bin` 在前面
- `PYTHONPATH` 包含 repo 和 repo `.venv` 的 site-packages
- `CUDA_HOME=/data/cuda/cuda-12.9/cuda`
- `CUDACXX=$CUDA_HOME/bin/nvcc`

否则很容易出现“本地能 import，worker 上跑的是另一套 Python/CUDA”的伪问题。

## 8. 训练启动要按“fake-data -> real-data -> 长序列”逐级升级

建议的启动顺序：

1. 单机 toy / fake-data
2. full model fake-data
3. real data 短序列
4. real data 长序列
5. 优化 kernel + 长序列

对于多机实验，优先用仓库已有启动方式，不要一上来发明新流程。当前比较实用的路径包括：

- `platform/rlaunch_run.py <exp.py> mm_pretrain`
- `rlaunch -d ... -- tools/smartrun <exp.py>`

排障时要特别注意两个运维问题：

- 可抢占模式下，worker 可能中途退出，需要看日志确认是否真的是模型问题。
- 不要随手 `Ctrl+C` 中断本地 launcher；在当前平台上，这经常会顺带把远端 `rjob` 一起停掉。

如果需要查/停任务，优先用：

- `brainctl -n shai-core get rjob <name>`
- `brainctl -n shai-core stop rjob/<name>`

这里的经验很简单：训练逻辑和平台问题要分开看。先确认是不是 worker 掉了、NCCL 没起来、GPU 节点异常，再继续怀疑模型实现。

## 9. 长序列问题先分清是“实现上限”还是“显存上限”

长序列调优时，最怕的是把所有 OOM 都归因成同一个问题。

更有效的做法是：

1. 先在短序列把 `3` 个 iter 跑稳。
2. 打开必要优化，例如 FlashAttention。
3. 按固定步长逐步拉高 sequence length。
4. 每次失败都记录：
   - 失败 iter
   - 失败 rank
   - 具体 OOM 栈
   - 发生在 attention / indexer / optimizer / grad norm 哪一段

GLM-5 这次接入里，至少出现过两类完全不同的 OOM：

- optimizer OOM：
  - experiment-local Adam 在 `_single_tensor_adam` 上出现大临时张量
  - 改成 `fused=True` 后问题消失
- attention/indexer OOM：
  - `steptronoss/model/utils/glm5_utils.py` 的 `sparse_mla` 里
    `scores = torch.einsum("qhd,qtd->qht", q_chunk.float(), gathered_k.float()) * scaling`
    会随着序列增长直接打爆显存

这两类问题的解法完全不同，所以一定要先分型。

当前 GLM-5 实测的一个有用结论是：

- 在当前分支的 real-data `playground/sft/glm5/glm5_sft_zhy.py`、`32 x 8` H100 配置下
- `train_iters=3` 的稳定上限大约是 `6144`
- `8192` 及以上会在首个 train step 的 `sparse_mla` 路径 OOM

这说明当前瓶颈已经不是单纯优化器状态，而是 DSA / sparse MLA 的实现复杂度。

一些通用建议：

- `pipeline_activation_cpu_offload`、`offload_optimizer_state` 可以作为 enablement 手段，但不要把它们当成最终解。
- 先证明“短序列稳定”，再谈长序列目标。
- 真正要冲 `64k/96k/128k`，必须先确认是 mask、indexer、attention score 还是 KV 路径在做二次方膨胀。

## 10. 常见坑 checklist

每次接新模型，建议至少过一遍下面这张表：

### 配置和结构

- 是否先做了 toy 版本，而不是直接上 full model 真数据？
- 如果改了 `num_layers`，所有 layer-wise list 是否同步更新？
- 模型专属数值常量是否和官方一致，例如 epsilon、rope theta、head dim、qk norm 行为？

### checkpoint

- 是否先核对了官方 checkpoint key，而不是复用已有模型的 reshape 假设？
- MoE expert 权重到底是 merged 还是 per-expert？
- reshape 后是否做过小规模加载和逐层对点？

### tokenizer 和数据

- compile tokenizer 和训练 tokenizer 是否完全一致？
- 官方 `chat_template.jinja` 是否已经实际读取？
- 编译后样本是否 decode 检查过？
- stepchat/toolcall/tool observation 是否被模板正确 encode？
- 数据兼容是否放在模型专属 data config，而不是改通用核心？

### 数值对齐

- 是否先把并行缩到 `1` 再对点？
- 是否对比了 router output，而不只是最终 loss？
- 是否记录了 cosine similarity？
- 是否检查了 mask、`cu_seqlens`、RoPE cache dtype、RMSNorm eps？
- HF loss 对照时，是否避免了 label 双重 shift？

### kernel 和优化路径

- 是否保留了 Python reference？
- 是否通过 `@optimizable` + `set_optimization(...)` 做切换？
- worker 上实际运行的 Python/CUDA 环境是否和本地预期一致？

### 训练和资源

- 是否先 fake-data，再 real-data，再长序列？
- 多机任务失败时，是否先确认 worker/节点问题，而不是直接怀疑模型？
- 是否避免误杀本地 launcher 导致远端实验一起停掉？

## 11. 推荐的接入顺序

如果以后再接一个全新模型，推荐直接照这个顺序推进：

1. 建立目录骨架和 config。
2. 写 toy 模型，保完整结构、缩层数。
3. 写 Python reference，单并行 deterministic 对点。
4. 写 checkpoint reshape，确认 official weights 可加载。
5. 写 tokenizer / template compat，编译一小份数据并 decode 抽检。
6. 先跑 fake-data `1 -> 3` iter。
7. 再跑 real-data `10` iter 看 loss。
8. 最后再接 optimizable kernel、长序列和极限吞吐。

这套顺序看起来慢，但实际上最省时间。因为它强迫我们在每个阶段都留下一个稳定基线，后面出问题时总能快速回退到最近的可验证状态。

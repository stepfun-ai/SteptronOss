StepTronOSS 是一个面向大规模语言模型训练的轻量化框架，强调模块化配置、可复现实验，并支持在 SFT、RLVR 与评测流程上的快速迭代。它可仅依赖 PyTorch 运行训练，同时也支持通过单个算子替换实现加速。

核心能力：
- 配置驱动的实验与动态校验（`cfshow` / `sanity_check`）
- 灵活的多任务编排与启动工具
- 可扩展的数据 / 优化器 / 模型栈，便于快速研究迭代

英文版：`README.md`

## 文档

- 启动指南（EN）：`docs/LAUNCH_EXPERIMENTS.md`
- 启动指南（ZH）：`docs/LAUNCH_EXPERIMENTS_ZH.md`
- SFT 数据准备（ZH）：`docs/SFT_DATA_PREPARATION.md`
- SFT 数据准备（EN）：`docs/SFT_DATA_PREPARATION_EN.md`
- API 模块：`docs/MODULES.md`

## 安装

```bash
# 在 repo 根目录执行
uv sync
```

## 快速开始

以下命令默认已激活 uv 虚拟环境；如未激活，请在命令前加 `uv run`。

### 实验概览与校验

```bash
# 查看实验配置并运行 sanity_check
cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py
# 查看指定子树（例如 actor 配置）
cfshow playground/rlvr/qwen3_1p5b_rlvr_math.py -k actor_cfg
```

### 启动实验

```bash
# 单任务实验（例如 SFT）
torchrun playground/sft/your_exp.py

# 多任务实验（例如 RL）
export STEPTRON_MEET_DIR=/path/to/shared
tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py

# mp_run 同样支持单任务实验
tools/mp_run.py playground/sft/your_exp.py

# 覆盖实验参数（示例：开启计时日志）
tools/mp_run.py playground/rlvr/qwen3_1p5b_rlvr_math.py profiler_cfg.timing_log_level=1
```

更详细的启动说明请参考上方文档链接。

### 生成多机启动脚本

使用 `tools/build_scripts.py` 根据 `resource_cfg.task_specs` 生成按副本拆分的脚本。

```bash
# 示例：在 /mnt/entrypoints/<exp_name>/<exp_id>/ 下生成脚本
tools/build_scripts.py playground/rlvr/qwen3_1p5b_rlvr_math.py /mnt/entrypoints/
```

输出目录结构：

```
/mnt/entrypoints/<exp_name>/<exp_id>/
  cpu/
    0.sh
  gpu/
    0.sh
    1.sh
    ...
```

说明：
- 脚本索引在同一 node_type 内是累加的（所有 GPU 任务共享 0..N 索引范围）。
- 每个脚本会 export 任务环境变量（含任务级 `EXP_ID` 与 `NNODES`），并执行由 `mp_run` 组装的命令。
- 额外的 CLI 参数会透传给实验，行为与 `mp_run.py` 一致。

### 运行时环境

分布式 rendezvous 会使用共享目录启动每个实验的 Redis 服务。

- `STEPTRON_MEET_DIR`：所有节点可见且可写的共享目录，用于发布 Redis 端口。
- `CANNOT_BE_REDIS_SERVER=1`：设置在不可启动 Redis 的 rank 上（会等待其它 rank 启动）。若所有 rank 都设置，则 rendezvous 最终会超时。

## 原则

核心原则：
- 配置无状态；运行时对象携带状态。
- 类级别声明配置结构，`__init__` 中填充值。
- 使用 `Ref("..path")` 进行跨节点引用。
- 通过 `sanity_check()` 与 `to_dict()` 进行校验与序列化。

Configurize 示例：

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

## 面向智能体

StepTronOSS 通过模块化实现了易于验证的开发方式，并提供 `cfshow` 等工具支持动态配置验证；同时提供 `AGENTS.md` 用于引导 Agents 参与代码开发。

可以这样问你的 agent：
```
write an exp of qwen3 8B sft, optimizer use muon
```

## 优化内核

仅在需要 GPU 极致性能或内核级加速时使用。

### flash-attn

手动安装：

```bash
uv pip install flash-attn
```

### grouped_gemm

手动安装：

```bash
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
```

### 代码中启用（一次性设置全部优化）

```python
from steptronoss.utils.optimizable import set_optimization

set_optimization(
    default="torch_compile",
    AttentionCore="flash-attn",
    grouped_gemm="nv_grouped_gemm",
)
```

## 项目状态

- [x] SFT exps
- [x] Reference configs: Qwen3 8B `playground/pretrain/qwen3/qwen3_8.py`, Step3.5 Flash `playground/pretrain/step3p5/step3p5_flash.py`
- [ ] Eval
- [ ] RLVR 实现
- [ ] Triton kernel 实现

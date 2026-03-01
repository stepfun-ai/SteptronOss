# 实验启动说明（中文）

本文介绍 StepTronOSS 中实验的启动方式与使用说明。

英文版见：`docs/LAUNCH_EXPERIMENTS.md`。

## 单任务 vs 多任务

单任务实验：`task_specs` 中只定义一种任务（数量为 1）。常见如无自动评测脚本的
pretrain / SFT。

多任务实验：`task_specs` 中包含多种任务（数量 > 1），例如 RL（rollout +
train）、agentic 评测（controller / judge / infer）。任务之间通常在资源需求、
入口命令、环境变量或镜像等方面不同。

## 启动方式矩阵（含兼容关系）

图例:
- Direct：`python my_exp.py` / `torchrun ...`
- MPRun：`tools/mp_run.py my_exp.py`
- BashScripts：`tools/build_scripts.py my_exp.py <out>`
- ✅ 推荐，❌ 不支持

|                | 单机 | 多机 |
|----------------|------|------|
| 单任务 | ✅ Direct <br> ✅ MPRun <br> ✅ BashScripts | ✅ Direct <br> ❌ MPRun <br> ✅ BashScripts |
| 多任务 | ❌ Direct <br> ✅ MPRun <br> ✅ BashScripts | ❌ Direct <br> ❌ MPRun <br> ✅ BashScripts |

## 启动方式说明

### Direct

优点:
- 工具最少，最适合本地调试。
- 单任务可单机或多机。

缺点:
- 不适用于多任务（无任务拆分/子进程管理）。
- 多机时需要自行配置 `torchrun` 参数。

示例:
```bash
# 单机
python path/to/my_exp.py

# 多机
torchrun --nnodes <N> --nproc-per-node <GPUS_PER_NODE> \
  --node_rank <RANK> --master_addr <ADDR> --master_port <PORT> \
  path/to/my_exp.py
```

### MPRun

优点:
- 推荐用于单机多任务。
- 按任务生成子进程，分别设置环境变量。

缺点:
- 仅支持单机（多机不支持）。

示例:
```bash
export STEPTRON_MEET_DIR=/path/to/shared
tools/mp_run.py path/to/my_exp.py
```

### BashScripts

优点:
- 覆盖所有场景（单/多任务、单/多机）。
- 生成按副本拆分的脚本，便于对接调度系统或手动分发。

缺点:
- 需要额外一步：人工分发/提交脚本。

示例:
```bash
tools/build_scripts.py path/to/my_exp.py /mnt/entrypoints/

# 每个节点执行分配到的脚本（示例）
bash /mnt/entrypoints/<exp_name>/<exp_id>/gpu/0.sh
```

## 生成多机启动脚本

`tools/build_scripts.py` 会根据 `resource_cfg.task_specs` 生成脚本，目录结构：

```
/mnt/entrypoints/<exp_name>/<exp_id>/
  cpu/
    0.sh
  gpu/
    0.sh
    1.sh
    ...
```

说明:
- 同一 node_type 的脚本索引是累加的（所有 GPU 任务共享 0..N 索引范围）。
- 脚本会 export 任务环境变量（含任务级 `EXP_ID`、`NNODES`），并执行拼装后的命令。

## 环境变量

- `STEPTRON_MEET_DIR`: 多机 rendezvous 的共享目录（Redis 地址发布点）。
- `CANNOT_BE_REDIS_SERVER=1`: 禁止某些 rank 启动 Redis；如果所有 rank 都设置，
  rendezvous 会超时。

## task_specs 概览

每个 `task_spec` 可覆盖：
- `replica`: 任务副本数
- `gpu`: 每节点 GPU 数
- `node_type`: `"cpu"` 或 `"gpu"`
- `envs`: 任务环境变量
- `command`: 任务启动模板

最终会由 `resource_cfg.extract()` 展开为具体任务配置。

## 平台适配

对接自定义资源管理/任务提交平台（如 Kubernetes、火山云、阿里云），建议实现
专用的 `tools/xx_run.py`，作为统一提交入口。

可参考 `tools/example_submitter.py` 的可运行模板，
平台细节已替换为 TODO，直接按你的调度系统改造即可。

示例:
```bash
python tools/example_submitter.py path/to/my_exp.py
```

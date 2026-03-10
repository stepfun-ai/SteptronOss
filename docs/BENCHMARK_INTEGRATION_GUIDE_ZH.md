# Benchmark 接入指南（中文）

本文说明如何在当前 OSS benchmark 栈中接入新的 benchmark。

本文内容以当前代码为准，主要对应：

- `steptronoss/generation/base_benchmark.py`
- `playground/eval/benchmarks/common.py`
- `playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py`

英文版见：

- `docs/BENCHMARK_INTEGRATION_GUIDE.md`

## 目标目录结构

新增 benchmark 时，代码放在：

```text
playground/eval/benchmarks/<BenchmarkName>/
  __init__.py
  benchmark.py
```

## 当前接线方式

当前真正的支持集，等于
`playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py` 中
`SimpleBenchmarksEvalConfig.get_benchmarks()` 返回的显式 benchmark 列表。

每个被支持的 benchmark 都在这里被直接 import 并直接构造。
`selected_datasets` 只在这个显式支持集上做过滤。
磁盘上的数据文件，只有在 `get_benchmarks()` 中被构造后，才会进入 harness。

## 先选对基类

这里有两个预期入口。

### `BaseBenchmark`

如果 benchmark 并不是天然来自导出的 jsonl，就应该直接继承
`BaseBenchmark`。

典型场景：

- synthetic benchmark
- 代码里直接生成的数据
- 来自数据库或服务的 benchmark
- 一次性的 debug benchmark

`BaseBenchmark` 只要求：

- `name`
- `get_cases()`
- `evaluate(results)`

它不假设存在 `data_path`。

### `JsonlChatBenchmark`

如果 benchmark 来自导出的 chat-style jsonl，则继承
`JsonlChatBenchmark`。

它已经处理好了：

- jsonl 行加载
- 导出 `messages` 解析
- 导出 `source_item` 解析
- chat template tokenization
- 基于 `sample_per_prompt` 的 prompt fan-out
- 默认 generation-first 聚合逻辑

它当前的构造参数是：

- `data_path`
- `tokenizer`
- `sample_per_prompt`
- `down_sample_to`
- `shuffle_prompts`
- `shuffle_seed`
- `chat_template_options`

## 核心数据模型

`steptronoss/generation/base_benchmark.py` 定义了 benchmark 侧的基础数据流。

### `Prompt`

`Prompt` 表示一条面向 client 的生成请求。

字段：

- `tokens`
- `messages`
- `prompt_token_count`
- `sampling_params`

重要约束：

- `Prompt` 是 request-ready 的对象。
- `tokens` 和 `messages` 二选一
- 具体使用哪一个，取决于 benchmark runner 调用的推理 API
- 当请求载荷使用 `messages` 时，`prompt_token_count` 用来保存 prompt 长度，供 budget 检查使用
- `sampling_params` 放在 `Prompt` 上，因为同一个逻辑 prompt 可以对应多次不同的具体生成请求

### `BenchmarkMeta`

`BenchmarkMeta` 是 benchmark 自己拥有的元数据。

字段：

- `benchmark_name`
- `item_id`
- `context`

`context` 用来承载 benchmark-specific 的评分上下文，例如：

- gold answer
- judge label
- subcategory
- references
- checklist
benchmark-specific 的评分数据就放在这里。

### `EvaluationMeta`

`EvaluationMeta` 是 evaluator 拥有的、与 benchmark 无关的运行时元数据。

字段：

- `prompt_index`
- `run_index`

这一层应保持 benchmark-agnostic。

### `EvaluationCase`

`EvaluationCase` 是送进 generation 流水线的最小单元：

```text
EvaluationCase = Prompt + BenchmarkMeta + EvaluationMeta
```

### `Generated`

`Generated` 是最终的生成结果。

字段：

- `case`
- `choice`
- `response`
- `reasoning_content`
- `error`

scorer 应通过明确的 ownership 路径访问元数据，例如：

- `result.case.benchmark.context`
- `result.case.benchmark.item_id`
- `result.case.evaluation.run_index`
`Generated` 通过 `case` 暴露元数据入口。

## 最小非 jsonl 示例

如果你要写的是最小 benchmark，且根本不需要读 jsonl，就直接继承
`BaseBenchmark`。

下面这个例子对应最简单的 case：

- prompt: “How many r are in strawberry?”
- 判定规则：回答中出现 `3` 即视为正确

```python
from steptronoss.generation.base_benchmark import (
    BaseBenchmark,
    BaseMetric,
    BenchmarkMeta,
    EvaluationCase,
    EvaluationMeta,
    Generated,
    Prompt,
    SamplingParams,
)


class StrawberryRBenchmark(BaseBenchmark):
    name = "STRAWBERRY_R"

    def get_cases(self) -> list[EvaluationCase]:
        return [
            EvaluationCase(
                prompt=Prompt(
                    messages=[{"role": "user", "content": "How many r are in strawberry?"}],
                    prompt_token_count=3,
                    sampling_params=SamplingParams(max_tokens=8, seed=0),
                ),
                benchmark=BenchmarkMeta(
                    benchmark_name=self.name,
                    item_id="strawberry_r_count_0",
                    context={"answer": "3"},
                ),
                evaluation=EvaluationMeta(prompt_index=0, run_index=0),
            )
        ]

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        sample_values = [
            1.0
            if result.error is None and "3" in result.response
            else 0.0
            for result in results
        ]
        score_avg = sum(sample_values) / len(sample_values)
        return BaseMetric(score_avg=score_avg, score_std=0.0, pass_at_k={1: score_avg})
```

这就是“没有 jsonl 时的正确起点”。
这里把 `pass_at_k` 写成 `{1: score_avg}` 只在这个最小示例里成立，
因为它对每个 prompt 只生成 1 个样本。

## 基于 jsonl 的 benchmark 模式

如果 benchmark 来自导出的 chat jsonl，继承 `JsonlChatBenchmark`。

最小形式：

```python
from playground.eval.benchmarks.common import JsonlChatBenchmark
from steptronoss.generation.base_benchmark import BaseMetric, Generated


class MyBenchmark(JsonlChatBenchmark):
    dataset_name = "MY_DATASET"

    @classmethod
    def _is_correct(cls, result: Generated) -> bool:
        answer = result.case.benchmark.context.get("answer")
        gold = answer.strip() if isinstance(answer, str) else str(answer).strip()
        return result.error is None and result.response.strip() == gold

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        sample_values = [1.0 if self._is_correct(result) else 0.0 for result in results]
        return self._build_metric(
            results=results,
            sample_values=sample_values,
            sample_per_prompt=self.sample_per_prompt,
            is_success_fn=self._is_correct,
        )
```

这里要注意：

- 导出的 `source_item` 会被 `JsonlChatBenchmark` 解析成
  `BenchmarkMeta(item_id=..., context=...)`
- scorer 读的是 `result.case.benchmark.context`
- scorer 逻辑应依赖 `BenchmarkMeta`，而不是依赖导出行的原始结构

## Metric 语义

对于会重复采样的 benchmark，`JsonlChatBenchmark._build_metric(...)`
会计算两个不同的聚合量：

- `score_avg`：把所有生成样本摊平后的逐样本平均分
- `pass_at_k`：标准的 HumanEval 风格无偏 `pass@k` 估计量，
  先按 `item_id` 分组，再根据每题的 `n` 个样本里有 `c` 个成功样本来计算

对单个题目：

- `pass@k = 1 - C(n-c, k) / C(n, k)`
- 如果 `n - c < k`，则 `pass@k = 1`

几个要点：

- `pass_at_k` 与样本顺序无关；只看成功样本个数，不看哪个 `run_index`
  先成功
- `pass_at_k` 不再是旧的“前 k 个样本里是否命中”的 prefix 指标
- 当 `sample_per_prompt == 1` 时，`pass_at_k[1] == score_avg`

## 导出符号

创建 `__init__.py`：

```python
from .benchmark import MyBenchmark

__all__ = ["MyBenchmark"]
```

## 接入 eval exp

打开：

- `playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py`

然后：

1. 在 `get_benchmarks()` 内 import 新 benchmark
2. 在 `benchmarks` 列表中显式追加构造调用

遵循当前风格：

- benchmark 构造必须显式
- 不要用 registry 隐藏选择逻辑
- `selected_datasets` 只对显式支持集做校验

## 显式处理不支持的数据集

如果某个数据集不应被支持：

- 不要把它加入 `get_benchmarks()`
- 当用户在 `selected_datasets` 里显式请求它时，保持 fail fast
- 如果磁盘上已有导出数据，最好在文档或注释里明确说明为何未接入

## 验证方式

至少做三层验证。

### 1. 静态语法检查

```bash
python3 -m py_compile \
  steptronoss/generation/base_benchmark.py \
  playground/eval/benchmarks/<BenchmarkName>/benchmark.py \
  playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py
```

### 2. Synthetic scorer 检查

手动构造一个很小的 `Generated` 样本，验证：

- 提取逻辑
- 归一化逻辑
- fallback 行为
- 明显负例

### 3. Fresh eval wiring 检查

确认 benchmark 能通过
`SimpleBenchmarksEvalConfig.get_benchmarks()` 被构造出来，并且能通过
`selected_datasets` 过滤。

## Metric 语义

分数语义必须明确属于下面之一：

- 精确 correctness
- heuristic quality
- generation-first completion

如果导出数据只能支持 heuristic，就不要把它包装成 correctness。

## 推荐检查清单

在认为 benchmark 已经接入完成之前，确认：

- 类存在于 `playground/eval/benchmarks/<BenchmarkName>/`
- `__init__.py` 正确导出符号
- eval exp 在 `get_benchmarks()` 里显式构造了它
- scorer 从 `result.case.benchmark.context` 读取 benchmark 元数据
- synthetic sample 能跑通
- 静态语法检查通过
- `selected_datasets` 能接受它的 benchmark 名称，并拒绝不支持的名称

## 常见坑

- 明明不读 jsonl，却硬继承 `JsonlChatBenchmark`
- 把导出的 `source_item` 当成稳定的 base-layer 类型
- 把 benchmark-specific 的评分字段塞进 `EvaluationMeta`
- 用 convenience property 隐藏 ownership，而不是显式写 `result.case...`
- 把 judge question 当成模型原始 prompt
- 把 reference answer 当成 gold answer
- 对信号不足的样本硬上 heuristic
- 在新类型标注里使用 `Any` 或 `dict[str, Any]`

## 经验法则

如果导出数据里有：

- gold answer：写真实 scorer
- 结构化目标：写 parser-based scorer
- 只有弱 metadata：写 heuristic scorer
- 几乎没有评分信号：保持 generation-first，或者直接排除

# Benchmark Integration Guide

This guide describes how to add a benchmark to the current OSS benchmark
stack.

It reflects the code that exists today in:

- `steptronoss/generation/base_benchmark.py`
- `playground/eval/benchmarks/common.py`
- `playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py`

## Target Layout

Add benchmark code under:

```text
playground/eval/benchmarks/<BenchmarkName>/
  __init__.py
  benchmark.py
```

## Current Wiring

Benchmark support is the explicit list returned by
`SimpleBenchmarksEvalConfig.get_benchmarks()` in
`playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py`.

Each supported benchmark is imported and constructed there directly.
`selected_datasets` filters this explicit supported set.
Dataset files on disk become active only when they are constructed in
`get_benchmarks()`.

## Choose the Base Class First

There are two intended entry points.

### `BaseBenchmark`

Use `BaseBenchmark` when the benchmark does not naturally come from an exported
jsonl file.

Typical cases:

- a synthetic benchmark
- a benchmark generated in code
- a benchmark backed by a database or service
- a one-off debug benchmark

`BaseBenchmark` only requires:

- `name`
- `get_cases()`
- `evaluate(results)`

It does not assume `data_path`.

### `JsonlChatBenchmark`

Use `JsonlChatBenchmark` when the benchmark is backed by exported chat-style
jsonl data.

It already handles:

- loading jsonl rows
- parsing exported `messages`
- parsing exported `source_item`
- chat-template tokenization
- prompt fan-out with `sample_per_prompt`
- default generation-first aggregation

Its constructor currently takes:

- `data_path`
- `tokenizer`
- `sample_per_prompt`
- `down_sample_to`
- `shuffle_prompts`
- `shuffle_seed`
- `chat_template_options`

## Core Data Model

`steptronoss/generation/base_benchmark.py` defines the benchmark-facing data
flow.

### `Prompt`

Client-side request payload for one generation.

Fields:

- `tokens`
- `messages`
- `prompt_token_count`
- `sampling_params`

Important:

- `Prompt` is the request-ready object from the benchmark/client view.
- exactly one of `tokens` or `messages` should be populated
- the choice depends on the generation API the benchmark runner calls
- `prompt_token_count` stores prompt length for budget checks when the request
  payload uses `messages`
- `sampling_params` belongs here because the same logical prompt can be issued
  as different concrete generation requests

### `BenchmarkMeta`

Benchmark-owned metadata.

Fields:

- `benchmark_name`
- `item_id`
- `context`

Use `context` for benchmark-specific scoring data such as:

- gold answers
- judge labels
- subcategories
- references
- checklist data
Benchmark-specific scoring data lives here.

### `EvaluationMeta`

Evaluator-owned runtime metadata.

Fields:

- `prompt_index`
- `run_index`

This layer should stay benchmark-agnostic.

### `EvaluationCase`

The minimal unit sent through generation:

```text
EvaluationCase = Prompt + BenchmarkMeta + EvaluationMeta
```

### `Generated`

The final generation result.

Fields:

- `case`
- `choice`
- `response`
- `reasoning_content`
- `error`

Scorers should read benchmark/evaluator metadata through explicit ownership
paths such as:

- `result.case.benchmark.context`
- `result.case.benchmark.item_id`
- `result.case.evaluation.run_index`
`Generated` exposes `case` as the metadata entry point.

## Minimal Non-jsonl Example

If you want the smallest possible benchmark, inherit `BaseBenchmark` directly.

This example asks:

- prompt: "How many r are in strawberry?"
- correctness rule: the response contains `3`

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

This is the correct starting point when there is no jsonl to read.
The `pass_at_k={1: score_avg}` shortcut is only correct here because this
example emits exactly one sample per prompt.

## Jsonl-backed Benchmark Pattern

For exported chat benchmarks, inherit `JsonlChatBenchmark`.

Minimal shape:

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

Important:

- exported `source_item` is parsed by `JsonlChatBenchmark` into
  `BenchmarkMeta(item_id=..., context=...)`
- scorers should read `result.case.benchmark.context`
- scorer logic should depend on `BenchmarkMeta`, not on raw exported row shape

## Metric Semantics

For repeated-sampling benchmarks, `JsonlChatBenchmark._build_metric(...)`
computes two different aggregates:

- `score_avg`: the mean per-sample score over all generated outputs
- `pass_at_k`: the standard unbiased HumanEval-style pass@k estimator,
  computed per `item_id` from `n` sampled outputs and `c` successful outputs

For one item:

- `pass@k = 1 - C(n-c, k) / C(n, k)`
- if `n - c < k`, then `pass@k = 1`

Important implications:

- `pass_at_k` is order-invariant: it depends on how many samples succeeded, not
  on which `run_index` succeeded first
- `pass_at_k` is not the old "did any of the first k samples succeed" prefix
  metric
- when `sample_per_prompt == 1`, `pass_at_k[1] == score_avg`

## Export the Symbol

Create `__init__.py`:

```python
from .benchmark import MyBenchmark

__all__ = ["MyBenchmark"]
```

## Wire the Benchmark into the Eval Exp

Open:

- `playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py`

Then:

1. import the benchmark inside `get_benchmarks()`
2. append an explicit constructor call to the `benchmarks` list

Follow the existing style:

- keep benchmark construction explicit
- do not hide selection behind a registry
- validate `selected_datasets` against the explicit supported set

## Handle Unsupported Datasets Explicitly

If a dataset should not be supported:

- do not add it to `get_benchmarks()`
- keep `selected_datasets` failing fast when explicitly requested
- document the omission if the exported dataset exists on disk

## Validation

Use at least three checks.

### 1. Static syntax check

```bash
python3 -m py_compile \
  steptronoss/generation/base_benchmark.py \
  playground/eval/benchmarks/<BenchmarkName>/benchmark.py \
  playground/eval/qwen3/qwen3_1p7b_eval_simple_benchmarks.py
```

### 2. Synthetic scorer check

Construct a tiny `Generated` sample directly and verify:

- extraction
- normalization
- fallback behavior
- obvious negative cases

### 3. Fresh eval wiring check

Make sure the benchmark can be constructed from
`SimpleBenchmarksEvalConfig.get_benchmarks()` and survives
`selected_datasets` filtering.

## Metric Semantics

Be explicit about what the score means.

Valid meanings:

- exact correctness
- heuristic quality
- generation-first completion

Do not claim correctness when the exported data only supports a heuristic.

## Recommended Checklist

Before considering a benchmark integrated, confirm:

- the class exists under `playground/eval/benchmarks/<BenchmarkName>/`
- `__init__.py` exports the symbol
- the eval exp constructs it explicitly in `get_benchmarks()`
- the scorer reads benchmark metadata from `result.case.benchmark.context`
- the benchmark works on a synthetic sample
- static syntax checking passes
- `selected_datasets` accepts the benchmark name and rejects unsupported names

## Common Pitfalls

- Inheriting `JsonlChatBenchmark` for a benchmark that does not read jsonl
- Treating exported `source_item` as a stable base-layer type
- Putting benchmark-specific scoring fields into `EvaluationMeta`
- Hiding data ownership with convenience properties instead of using
  `result.case...`
- Treating a judge question as if it were the original model prompt
- Treating a reference answer as a gold answer
- Forcing a heuristic onto rows with insufficient signal
- Using `Any` or `dict[str, Any]` in new benchmark-facing type hints

## Rule of Thumb

If exported data contains:

- a gold answer: write a real scorer
- a structured target: write a parser-based scorer
- only weak metadata: write a heuristic scorer
- almost no scoring signal: keep it generation-first or exclude it

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Callable
from typing import Protocol

from steptronoss.generation.base_benchmark import (
    BaseBenchmark,
    BaseMetric,
    BenchmarkMeta,
    ChatMessage,
    EvaluationCase,
    EvaluationMeta,
    Generated,
    GroundTruth,
    GroundTruthValue,
    JsonObject,
    JsonValue,
    Messages,
    Prompt,
)


class ChatTokenizer(Protocol):
    """Minimal tokenizer contract needed by jsonl-backed chat benchmarks."""

    def apply_chat_template(
        self,
        messages: Messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: JsonValue,
    ) -> list[int]: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def _estimate_pass_at_k(num_samples: int, num_successes: int, k: int) -> float:
    """Estimate pass@k with the standard unbiased HumanEval-style estimator.

    For one benchmark item with `n` sampled completions and `c` successful ones:

    - `pass@k = 1 - C(n-c, k) / C(n, k)`
    - if `n - c < k`, then `pass@k = 1`

    This estimator is order-invariant: only the number of successful samples
    matters, not which `run_index` first succeeded. When fewer than `k`
    samples are available, clip `k` to the available sample count.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_successes <= num_samples:
        raise ValueError("num_successes must be within [0, num_samples]")
    if k <= 0:
        raise ValueError("k must be positive")

    effective_k = min(k, num_samples)
    if num_successes == 0:
        return 0.0
    if num_samples - num_successes < effective_k:
        return 1.0

    return 1.0 - math.prod(
        1.0 - effective_k / denominator for denominator in range(num_samples - num_successes + 1, num_samples + 1)
    )


def _parse_json_value(raw_value: object) -> JsonValue:
    if raw_value is None or isinstance(raw_value, (bool, int, float, str)):
        return raw_value
    if isinstance(raw_value, list):
        return [_parse_json_value(value) for value in raw_value]
    if isinstance(raw_value, dict):
        parsed_object: JsonObject = {}
        for key, value in raw_value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be str, got {type(key).__name__}")
            parsed_object[key] = _parse_json_value(value)
        return parsed_object
    raise TypeError(f"Unsupported JSON value type: {type(raw_value).__name__}")


def _parse_ground_truth(raw_ground_truth: object) -> GroundTruth:
    if not isinstance(raw_ground_truth, dict):
        raise TypeError("ground_truth must be an object")
    value = raw_ground_truth.get("value")
    if not isinstance(value, dict):
        raise TypeError("ground_truth.value must be an object")
    item_id = value.get("item_id")
    dataset = value.get("dataset")
    if not isinstance(item_id, str) or not isinstance(dataset, str):
        raise TypeError("ground_truth.value.item_id and dataset must be strings")
    ground_truth_value: GroundTruthValue = {"item_id": item_id, "dataset": dataset}
    return {"value": ground_truth_value}


def _parse_message(raw_message: object) -> ChatMessage:
    if not isinstance(raw_message, dict):
        raise TypeError("message must be an object")
    role = raw_message.get("role")
    content = raw_message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise TypeError("message.role and message.content must be strings")
    message: ChatMessage = {"role": role, "content": content}
    if "ground_truth" in raw_message:
        message["ground_truth"] = _parse_ground_truth(raw_message["ground_truth"])
    return message


def _parse_messages(raw_messages: object) -> Messages:
    if not isinstance(raw_messages, list):
        raise TypeError("messages must be a list")
    return [_parse_message(raw_message) for raw_message in raw_messages]


def _parse_source_item(raw_source_item: object) -> tuple[str, JsonObject]:
    if not isinstance(raw_source_item, dict):
        raise TypeError("source_item must be an object")

    item_id = raw_source_item.get("item_id")
    if not isinstance(item_id, str):
        raise TypeError("source_item.item_id must be a string")
    context: JsonObject = {}
    for key, value in raw_source_item.items():
        if not isinstance(key, str):
            raise TypeError(f"source_item key must be str, got {type(key).__name__}")
        if key == "item_id":
            continue
        context[key] = _parse_json_value(value)
    return item_id, context


def _parse_record(raw_record: object) -> tuple[str, Messages, str, JsonObject]:
    if not isinstance(raw_record, dict):
        raise TypeError("benchmark record must be an object")
    dataset = raw_record.get("dataset")
    if not isinstance(dataset, str):
        raise TypeError("benchmark record.dataset must be a string")
    return (
        dataset,
        _parse_messages(raw_record.get("messages")),
        *_parse_source_item(raw_record.get("source_item")),
    )


class JsonlChatBenchmark(BaseBenchmark):
    dataset_name: str | None = None

    def __init__(
        self,
        data_path: str,
        tokenizer: ChatTokenizer,
        sample_per_prompt: int,
        down_sample_to: int | None = None,
        shuffle_prompts: bool = False,
        chat_template_options: JsonObject | None = None,
    ):
        if not self.dataset_name:
            raise ValueError(f"{type(self).__name__} must define dataset_name")
        self.name = self.dataset_name
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.sample_per_prompt = sample_per_prompt
        self.down_sample_to = down_sample_to
        self.shuffle_prompts = shuffle_prompts
        self.chat_template_options = chat_template_options
        self._records_cache: list[tuple[str, Messages, str, JsonObject]] | None = None

    @staticmethod
    def _normalize_messages(messages: Messages) -> Messages:
        normalized = _parse_messages(json.loads(json.dumps(messages)))
        if normalized and normalized[-1].get("role") == "assistant" and normalized[-1].get("content", "") == "":
            return normalized[:-1]
        return normalized

    def _load_records(self) -> list[tuple[str, Messages, str, JsonObject]]:
        if self._records_cache is None:
            with open(self.data_path, encoding="utf-8") as fin:
                self._records_cache = [_parse_record(json.loads(line)) for line in fin]
        records = list(self._records_cache)
        if self.shuffle_prompts:
            rng = random.Random(1234)
            rng.shuffle(records)
        if self.down_sample_to is not None:
            records = records[: self.down_sample_to]
        return records

    def get_cases(self) -> list[EvaluationCase]:
        records = self._load_records()
        cases: list[EvaluationCase] = []
        tokenization_options = {} if self.chat_template_options is None else dict(self.chat_template_options)

        for prompt_index, (_dataset, messages, item_id, context) in enumerate(records):
            normalized_messages = self._normalize_messages(messages)
            tokens = self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=True,
                add_generation_prompt=True,
                **tokenization_options,
            )
            for run_index in range(self.sample_per_prompt):
                prompt = Prompt(
                    messages=copy.deepcopy(normalized_messages),
                    prompt_token_count=len(tokens),
                )
                benchmark = BenchmarkMeta(
                    benchmark_name=self.name,
                    item_id=item_id,
                    context=copy.deepcopy(context),
                )
                evaluation = EvaluationMeta(
                    prompt_index=prompt_index,
                    run_index=run_index,
                )
                cases.append(EvaluationCase(prompt=prompt, benchmark=benchmark, evaluation=evaluation))

        return cases

    def count_response_tokens(self, response: str) -> int:
        return len(self.tokenizer.encode(response, add_special_tokens=False))

    @staticmethod
    def _is_success(result: Generated) -> bool:
        return result.error is None and result.finish_reason == "stop"

    @staticmethod
    def _build_metric(
        results: list[Generated],
        sample_values: list[float],
        sample_per_prompt: int,
        is_success_fn: Callable[[Generated], bool],
    ) -> BaseMetric:
        if not results:
            return BaseMetric(score_avg=math.nan, score_std=math.nan, pass_at_k={})

        score_avg = sum(sample_values) / len(sample_values)
        variance = sum((value - score_avg) ** 2 for value in sample_values) / len(sample_values)
        score_std = math.sqrt(variance)
        grouped: dict[str, list[Generated]] = {}
        for result in results:
            grouped.setdefault(result.case.benchmark.item_id, []).append(result)

        pass_at_k: dict[int, float] = {}
        candidate_ks = {1, sample_per_prompt}
        k = 2
        while k < sample_per_prompt:
            candidate_ks.add(k)
            k *= 2
        for candidate_k in sorted(candidate_ks):
            pass_probability_sum = 0.0
            for item_results in grouped.values():
                ordered = sorted(item_results, key=lambda item: item.case.evaluation.run_index)
                num_successes = sum(1 for item in ordered if is_success_fn(item))
                pass_probability_sum += _estimate_pass_at_k(
                    num_samples=len(ordered),
                    num_successes=num_successes,
                    k=candidate_k,
                )
            pass_at_k[candidate_k] = pass_probability_sum / max(len(grouped), 1)

        return BaseMetric(score_avg=score_avg, score_std=score_std, pass_at_k=pass_at_k)

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        sample_values = [1.0 if self._is_success(result) else 0.0 for result in results]
        return self._build_metric(
            results=results,
            sample_values=sample_values,
            sample_per_prompt=self.sample_per_prompt,
            is_success_fn=self._is_success,
        )

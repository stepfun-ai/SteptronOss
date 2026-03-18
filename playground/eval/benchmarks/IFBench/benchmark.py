from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from playground.eval.benchmarks.common import ChatTokenizer, JsonlChatBenchmark
from steptronoss.generation.base_benchmark import BaseMetric, Generated, JsonObject, SamplingParams


@dataclass
class IFBenchMetric(BaseMetric):
    evaluation_mode: str = "loose"
    official_metrics: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        payload = super().to_dict()
        payload["evaluation_mode"] = self.evaluation_mode
        payload["official_metrics"] = copy.deepcopy(self.official_metrics)
        return payload


class IFBenchBenchmark(JsonlChatBenchmark):
    """IFBench adapter backed by AllenAI's official prompt set and verifier."""

    dataset_name = "IFBENCH"
    _PROMPT_FILENAME = "IFBench_test.jsonl"
    _INLINE_REASONING_PATTERNS = (
        re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL),
        re.compile(r"^\s*<thinking>.*?</thinking>\s*", re.DOTALL),
        re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL),
        re.compile(r"^\s*<analysis>.*?</analysis>\s*", re.DOTALL),
        re.compile(r"^\s*<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*", re.DOTALL),
    )
    _OFFICIAL_SAMPLING_PARAMS = SamplingParams(temperature=0.0)

    def __init__(
        self,
        data_path: str,
        tokenizer: ChatTokenizer,
        sample_per_prompt: int,
        down_sample_to: int | None = None,
        shuffle_prompts: bool = False,
        chat_template_options: JsonObject | None = None,
        evaluation_mode: Literal["loose", "strict"] = "loose",
        strip_reasoning: bool = True,
        sampling_params: SamplingParams | None = None,
    ):
        resolved_resource_root = Path(data_path)
        if resolved_resource_root.name == self._PROMPT_FILENAME or resolved_resource_root.suffix == ".jsonl":
            raise ValueError(
                "IFBenchBenchmark.data_path must point to the IFBENCH resource directory, "
                f"not the prompt file itself: {resolved_resource_root}"
            )
        resolved_data_path = str(resolved_resource_root / self._PROMPT_FILENAME)
        self.resource_root = str(resolved_resource_root)
        self._validate_official_data_path(resolved_data_path)
        self.evaluation_mode = evaluation_mode
        self.strip_reasoning = strip_reasoning
        self.sampling_params = (
            self._OFFICIAL_SAMPLING_PARAMS if sampling_params is None else copy.deepcopy(sampling_params)
        )
        self._input_examples_cache: list[JsonObject] | None = None
        super().__init__(
            data_path=resolved_data_path,
            tokenizer=tokenizer,
            sample_per_prompt=sample_per_prompt,
            down_sample_to=down_sample_to,
            shuffle_prompts=shuffle_prompts,
            chat_template_options=chat_template_options,
        )

    @staticmethod
    def _resource_config():
        from playground.eval.benchmarks.IFBench.official import resource_config

        return resource_config

    def _evaluation_lib(self):
        self._resource_config().set_resource_root(self.resource_root)
        from playground.eval.benchmarks.IFBench.official import evaluation_lib

        return evaluation_lib

    @staticmethod
    def _validate_official_data_path(data_path: str) -> None:
        if not Path(data_path).is_file():
            raise FileNotFoundError(
                "IFBench prompt file is missing. "
                f"Expected {data_path}. "
                "Stage IFBench resources under the caller-provided IFBENCH resource root "
                "(for simple_eval this is typically <datasets_dir>/IFBENCH/)."
            )
        with open(data_path, encoding="utf-8") as fin:
            first_line = fin.readline()
        if not first_line:
            raise ValueError(f"IFBench data file is empty: {data_path}")
        first_record = json.loads(first_line)
        required_keys = {"key", "prompt", "instruction_id_list", "kwargs"}
        if not required_keys.issubset(first_record):
            raise ValueError(
                "IFBenchBenchmark requires the official IFBench_test.jsonl prompt file. "
                f"Expected keys {sorted(required_keys)} in {data_path}, got {sorted(first_record)}."
            )

    def _load_input_examples(self) -> list[JsonObject]:
        if self._input_examples_cache is None:
            input_examples: list[JsonObject] = []
            with open(self.data_path, encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    raw_record = json.loads(line)
                    prompt = raw_record.get("prompt")
                    instruction_id_list = raw_record.get("instruction_id_list")
                    kwargs = raw_record.get("kwargs")
                    key = raw_record.get("key")
                    if not isinstance(prompt, str):
                        raise TypeError("IFBench prompt record.prompt must be str")
                    if not isinstance(instruction_id_list, list) or not all(
                        isinstance(item, str) for item in instruction_id_list
                    ):
                        raise TypeError("IFBench prompt record.instruction_id_list must be list[str]")
                    if not isinstance(kwargs, list):
                        raise TypeError("IFBench prompt record.kwargs must be list")
                    normalized_kwargs: list[dict[str, object]] = []
                    for prompt_kwargs in kwargs:
                        if not isinstance(prompt_kwargs, dict):
                            raise TypeError("IFBench prompt record.kwargs entries must be dict")
                        normalized_kwargs.append({
                            field: value for field, value in prompt_kwargs.items() if value is not None
                        })
                    input_examples.append({
                        "key": str(key),
                        "prompt": prompt,
                        "instruction_id_list": list(instruction_id_list),
                        "kwargs": normalized_kwargs,
                    })
            self._input_examples_cache = input_examples
        return list(self._input_examples_cache)

    def _load_records(self) -> list[tuple[str, list[dict[str, str]], str, JsonObject]]:
        if self._records_cache is None:
            records: list[tuple[str, list[dict[str, str]], str, JsonObject]] = []
            for example in self._load_input_examples():
                context: JsonObject = {
                    "key": str(example["key"]),
                    "prompt": str(example["prompt"]),
                    "instruction_id_list": copy.deepcopy(example["instruction_id_list"]),
                    "kwargs": copy.deepcopy(example["kwargs"]),
                }
                records.append((
                    self.dataset_name,
                    [{"role": "user", "content": str(example["prompt"])}],
                    str(example["key"]),
                    context,
                ))
            self._records_cache = records
        return super()._load_records()

    def get_cases(self):
        cases = super().get_cases()
        if self.sampling_params is None:
            return cases
        return [case.with_sampling_params(copy.deepcopy(self.sampling_params)) for case in cases]

    def _build_input_example(self, context: JsonObject) -> object:
        key = context.get("key")
        prompt = context.get("prompt")
        instruction_id_list = context.get("instruction_id_list")
        kwargs = context.get("kwargs")
        if not isinstance(key, str):
            raise TypeError(f"IFBench context.key must be str, got {type(key).__name__}")
        if not isinstance(prompt, str):
            raise TypeError(f"IFBench context.prompt must be str, got {type(prompt).__name__}")
        if not isinstance(instruction_id_list, list) or not all(isinstance(item, str) for item in instruction_id_list):
            raise TypeError("IFBench context.instruction_id_list must be list[str]")
        if not isinstance(kwargs, list):
            raise TypeError(f"IFBench context.kwargs must be list, got {type(kwargs).__name__}")
        return self._evaluation_lib().InputExample(
            key=int(key),
            instruction_id_list=list(instruction_id_list),
            prompt=prompt,
            kwargs=copy.deepcopy(kwargs),
        )

    @classmethod
    def _strip_inline_reasoning(cls, response: str) -> str:
        stripped = response
        for pattern in cls._INLINE_REASONING_PATTERNS:
            stripped = pattern.sub("", stripped, count=1)
        return stripped.strip()

    def _response_for_official_eval(self, result: Generated) -> str | None:
        if result.error:
            return None
        response = result.response or ""
        if self.strip_reasoning and response:
            response = self._strip_inline_reasoning(response)
        return response

    def _input_example_for_result(self, result: Generated) -> object:
        return self._build_input_example(result.case.benchmark.context)

    def _evaluate_one(self, result: Generated, *, mode: Literal["loose", "strict"]) -> object:
        example = self._input_example_for_result(result)
        prompt_to_response = {example.prompt: self._response_for_official_eval(result)}
        evaluation_lib = self._evaluation_lib()
        if mode == "strict":
            return evaluation_lib.test_instruction_following_strict(example, prompt_to_response)
        return evaluation_lib.test_instruction_following_loose(example, prompt_to_response)

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        evaluation_lib = self._evaluation_lib()
        strict_outputs = [self._evaluate_one(result, mode="strict") for result in results]
        loose_outputs = [self._evaluate_one(result, mode="loose") for result in results]
        primary_outputs = strict_outputs if self.evaluation_mode == "strict" else loose_outputs

        sample_values = [1.0 if output.follow_all_instructions else 0.0 for output in primary_outputs]
        success_by_result = {
            id(result): output.follow_all_instructions for result, output in zip(results, primary_outputs, strict=True)
        }
        base_metric = self._build_metric(
            results=results,
            sample_values=sample_values,
            sample_per_prompt=self.sample_per_prompt,
            is_success_fn=lambda result: success_by_result[id(result)],
        )
        strict_report = evaluation_lib.build_accuracy_report(strict_outputs)
        loose_report = evaluation_lib.build_accuracy_report(loose_outputs)
        return IFBenchMetric(
            score_avg=base_metric.score_avg,
            score_std=base_metric.score_std,
            pass_at_k=dict(base_metric.pass_at_k),
            evaluation_mode=self.evaluation_mode,
            official_metrics={
                "strict": strict_report.to_dict(),
                "loose": loose_report.to_dict(),
            },
        )

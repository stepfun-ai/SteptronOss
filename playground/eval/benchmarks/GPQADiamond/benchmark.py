from __future__ import annotations

import re

from playground.eval.benchmarks.common import JsonlChatBenchmark
from steptronoss.generation.base_benchmark import BaseMetric, Generated


class GPQADiamondBenchmark(JsonlChatBenchmark):
    dataset_name = "GPQA_DIAMOND"
    _OPTION_PATTERN = re.compile(r"(?<![a-zA-Z0-9_])[A-D](?![a-zA-Z0-9_])")
    _BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

    @classmethod
    def _extract_choice(cls, response: str) -> str:
        if not response:
            return ""

        boxed_matches = cls._BOXED_PATTERN.findall(response)
        for candidate in reversed(boxed_matches):
            matches = cls._OPTION_PATTERN.findall(candidate.upper())
            if matches:
                return matches[-1]

        matches = cls._OPTION_PATTERN.findall(response.upper())
        if matches:
            return matches[-1]
        return ""

    @staticmethod
    def _gold_answer(result: Generated) -> str:
        answer = result.case.benchmark.context.get("answer")
        return answer.strip() if isinstance(answer, str) else str(answer).strip()

    @classmethod
    def _is_correct(cls, result: Generated) -> bool:
        if result.error:
            return False
        predicted = cls._extract_choice(result.response)
        return predicted == cls._gold_answer(result).upper()

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        sample_values = [1.0 if self._is_correct(result) else 0.0 for result in results]
        return self._build_metric(
            results=results,
            sample_values=sample_values,
            sample_per_prompt=self.sample_per_prompt,
            is_success_fn=self._is_correct,
        )

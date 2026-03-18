from __future__ import annotations

from functools import lru_cache

from playground.eval.benchmarks.common import JsonlChatBenchmark
from steptronoss.generation.base_benchmark import BaseMetric, Generated


class AIME25Benchmark(JsonlChatBenchmark):
    dataset_name = "AIME2025"

    @staticmethod
    def _extract_boxed_contents(response: str) -> list[str]:
        contents: list[str] = []
        marker = r"\boxed{"
        start = 0
        while True:
            boxed_index = response.find(marker, start)
            if boxed_index == -1:
                break

            index = boxed_index + len(marker)
            depth = 1
            while index < len(response) and depth > 0:
                char = response[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                index += 1

            if depth == 0:
                contents.append(response[boxed_index + len(marker) : index - 1])
                start = index
            else:
                break
        return contents

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        return (
            text
            .strip()
            .replace(r"\dfrac", r"\frac")
            .replace(r"\tfrac", r"\frac")
            .replace(r"\left", "")
            .replace(r"\right", "")
            .replace(" ", "")
            .replace("\n", "")
        )

    @staticmethod
    def _wrap_boxed(text: str) -> str:
        return rf"\boxed{{{text}}}"

    @staticmethod
    @lru_cache(maxsize=1)
    def _math_verify_ops():
        try:
            from math_verify import parse, verify
        except ImportError:
            return None, None
        return parse, verify

    @classmethod
    def _math_verify_equal(cls, predicted: str, answer: str) -> bool:
        parse, verify = cls._math_verify_ops()
        if parse is None or verify is None:
            return False

        try:
            parsed_predicted = parse(cls._wrap_boxed(predicted), parsing_timeout=30)
            parsed_answer = parse(cls._wrap_boxed(answer), parsing_timeout=30)
            return bool(verify(parsed_predicted, parsed_answer, timeout_seconds=30))
        except Exception:
            return False

    @classmethod
    def _extract_answer(cls, response: str) -> str:
        if not response:
            return ""
        boxed_matches = cls._extract_boxed_contents(response)
        if boxed_matches:
            return boxed_matches[-1].strip()
        return response.strip()

    @staticmethod
    def _gold_answer(result: Generated) -> str:
        answer = result.case.benchmark.context.get("answer")
        return answer.strip() if isinstance(answer, str) else str(answer).strip()

    @classmethod
    def _is_correct(cls, result: Generated) -> bool:
        if result.error:
            return False
        answer = cls._gold_answer(result)
        predicted_raw = cls._extract_answer(result.response)
        predicted = cls._normalize_answer_text(predicted_raw)
        normalized_answer = cls._normalize_answer_text(answer)
        if predicted == normalized_answer:
            return True
        return cls._math_verify_equal(predicted_raw, answer)

    def evaluate(self, results: list[Generated]) -> BaseMetric:
        sample_values = [1.0 if self._is_correct(result) else 0.0 for result in results]
        return self._build_metric(
            results=results,
            sample_values=sample_values,
            sample_per_prompt=self.sample_per_prompt,
            is_success_fn=self._is_correct,
        )

import re

from playground.eval.benchmarks.GPQADiamond import GPQADiamondBenchmark
from steptronoss.generation.base_benchmark import Generated


class MMLUProBenchmark(GPQADiamondBenchmark):
    dataset_name = "MMLU_PRO"

    _OPTION_PATTERN = re.compile(r"(?<![a-zA-Z0-9_])[A-I](?![a-zA-Z0-9_])")

    @classmethod
    def _is_correct(cls, result: Generated) -> bool:
        if result.error:
            return False
        predicted = cls._extract_choice(result.response)
        return predicted == cls._gold_answer(result).upper()

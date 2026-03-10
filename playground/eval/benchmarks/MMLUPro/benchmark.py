import re

from playground.eval.benchmarks.GPQADiamond import GPQADiamondBenchmark
from steptronoss.generation.base_benchmark import Generated


class MMLUProBenchmark(GPQADiamondBenchmark):
    dataset_name = "MMLU_PRO"

    _OPTION_PATTERN = re.compile(r"(?<![a-zA-Z0-9_])[A-I](?![a-zA-Z0-9_])")

    @staticmethod
    def _is_correct(result: Generated, answer: str) -> bool:
        if result.error:
            return False
        predicted = MMLUProBenchmark._extract_choice(result.response)
        return predicted == answer.strip().upper()

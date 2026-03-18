from __future__ import annotations

import pytest

from playground.eval.benchmarks.AIME25 import AIME25Benchmark
from playground.eval.benchmarks.HMMT25 import HMMT25Benchmark
from steptronoss.generation.base_benchmark import BenchmarkMeta, EvaluationCase, EvaluationMeta, Generated, Prompt


def _make_generated(response: str, answer: str = "") -> Generated:
    return Generated(
        case=EvaluationCase(
            prompt=Prompt(
                messages=[{"role": "user", "content": "Solve the problem."}],
                prompt_token_count=4,
            ),
            benchmark=BenchmarkMeta(
                benchmark_name="TEST",
                item_id="test-item",
                context={"answer": answer},
            ),
            evaluation=EvaluationMeta(prompt_index=0, run_index=0),
        ),
        response=response,
    )


def test_aime_extract_answer_handles_nested_boxed_braces():
    response = r"\boxed{1 - \frac{2}{\pi}}"
    assert AIME25Benchmark._extract_answer(response) == r"1 - \frac{2}{\pi}"


def test_hmmt_accepts_whitespace_and_dfrac_variants():
    result = _make_generated(r"\boxed{\dfrac{1}{576}}", answer=r"\frac{1}{576}")
    assert HMMT25Benchmark._is_correct(result)


def test_hmmt_accepts_boxed_answers_with_nested_braces():
    result = _make_generated(r"\boxed{1 - \frac{2}{\pi}}", answer=r"1-\frac{2}{\pi}")
    assert HMMT25Benchmark._is_correct(result)


def test_hmmt_accepts_math_verify_equivalent_forms_when_available():
    pytest.importorskip("math_verify")
    result = _make_generated(r"\boxed{\frac{9}{\sqrt{23}}}", answer=r"\frac{9 \sqrt{23}}{23}")
    assert HMMT25Benchmark._is_correct(result)

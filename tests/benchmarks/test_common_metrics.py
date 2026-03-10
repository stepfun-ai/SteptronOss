from __future__ import annotations

import pytest

from playground.eval.benchmarks.common import JsonlChatBenchmark
from steptronoss.generation.base_benchmark import BenchmarkMeta, EvaluationCase, EvaluationMeta, Generated, Prompt


def _make_generated(*, item_id: str, run_index: int, success: bool) -> Generated:
    return Generated(
        case=EvaluationCase(
            prompt=Prompt(
                messages=[{"role": "user", "content": "test prompt"}],
                prompt_token_count=2,
            ),
            benchmark=BenchmarkMeta(
                benchmark_name="TEST_BENCHMARK",
                item_id=item_id,
                context={},
            ),
            evaluation=EvaluationMeta(prompt_index=0, run_index=run_index),
        ),
        response="ok" if success else "bad",
        choice={"finish_reason": "stop", "raw": {}},
    )


def _build_metric(results: list[Generated]):
    return JsonlChatBenchmark._build_metric(
        results=results,
        sample_values=[1.0 if result.response == "ok" else 0.0 for result in results],
        sample_per_prompt=4,
        is_success_fn=lambda result: result.response == "ok",
    )


def test_pass_at_k_uses_unbiased_estimator_instead_of_prefix_hits():
    results = [
        _make_generated(item_id="item_0", run_index=0, success=False),
        _make_generated(item_id="item_0", run_index=1, success=False),
        _make_generated(item_id="item_0", run_index=2, success=False),
        _make_generated(item_id="item_0", run_index=3, success=True),
        _make_generated(item_id="item_1", run_index=0, success=False),
        _make_generated(item_id="item_1", run_index=1, success=False),
        _make_generated(item_id="item_1", run_index=2, success=False),
        _make_generated(item_id="item_1", run_index=3, success=False),
    ]

    metric = _build_metric(results)

    assert metric.score_avg == pytest.approx(0.125)
    assert metric.pass_at_k == pytest.approx({
        1: 0.125,
        2: 0.25,
        4: 0.5,
    })


def test_pass_at_k_is_order_invariant_for_successful_samples():
    late_success_metric = _build_metric([
        _make_generated(item_id="item_0", run_index=0, success=False),
        _make_generated(item_id="item_0", run_index=1, success=False),
        _make_generated(item_id="item_0", run_index=2, success=False),
        _make_generated(item_id="item_0", run_index=3, success=True),
    ])
    early_success_metric = _build_metric([
        _make_generated(item_id="item_0", run_index=0, success=True),
        _make_generated(item_id="item_0", run_index=1, success=False),
        _make_generated(item_id="item_0", run_index=2, success=False),
        _make_generated(item_id="item_0", run_index=3, success=False),
    ])

    assert late_success_metric.pass_at_k == pytest.approx(early_success_metric.pass_at_k)

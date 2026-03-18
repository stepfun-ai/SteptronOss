from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from playground.eval.benchmarks.IFBench import IFBenchBenchmark
from steptronoss.generation.base_benchmark import Generated


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, kwargs
        return list(range(sum(len(message["content"].split()) for message in messages)))

    def encode(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return text.split()


def _write_official_style_prompt_file(path: Path) -> None:
    records = [
        {
            "key": "0",
            "prompt": "Pick one option. Answer with one of the following options: Red/Blue/Green.",
            "instruction_id_list": ["format:options"],
            "kwargs": [{"options": "Red/Blue/Green"}],
        },
        {
            "key": "1",
            "prompt": "Include exactly 2 numbers in the response.",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2}],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_ifbench_directory_data_path_resolves_official_prompt_file(tmp_path):
    resource_root = tmp_path / "datasets" / "IFBENCH"
    resource_root.mkdir(parents=True)
    _write_official_style_prompt_file(resource_root / "IFBench_test.jsonl")

    benchmark = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=1,
    )

    assert benchmark.resource_root == str(resource_root)
    assert benchmark.data_path == str(resource_root / "IFBench_test.jsonl")


def test_ifbench_rejects_prompt_file_data_path(tmp_path):
    resource_root = tmp_path / "datasets" / "IFBENCH"
    resource_root.mkdir(parents=True)
    prompt_file = resource_root / "IFBench_test.jsonl"
    _write_official_style_prompt_file(prompt_file)

    with pytest.raises(ValueError, match="must point to the IFBENCH resource directory"):
        IFBenchBenchmark(
            data_path=str(prompt_file),
            tokenizer=DummyTokenizer(),
            sample_per_prompt=1,
        )


def test_ifbench_loads_official_prompt_file_and_scores_with_lazy_verifier(tmp_path):
    datasets_dir = tmp_path / "datasets"
    resource_root = datasets_dir / "IFBENCH"
    resource_root.mkdir(parents=True)
    prompt_file = resource_root / "IFBench_test.jsonl"
    _write_official_style_prompt_file(prompt_file)

    benchmark = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=2,
    )
    benchmark_without_strip = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=1,
        strip_reasoning=False,
    )

    for module_name in list(sys.modules):
        if module_name.startswith("playground.eval.benchmarks.IFBench.official."):
            sys.modules.pop(module_name)

    cases = benchmark.get_cases()
    assert [case.benchmark.item_id for case in cases] == ["0", "0", "1", "1"]
    assert [case.prompt.messages[0]["content"] for case in cases] == [
        "Pick one option. Answer with one of the following options: Red/Blue/Green.",
        "Pick one option. Answer with one of the following options: Red/Blue/Green.",
        "Include exactly 2 numbers in the response.",
        "Include exactly 2 numbers in the response.",
    ]
    assert all(case.prompt.sampling_params is not None for case in cases)
    assert all(case.prompt.sampling_params.temperature == 0.0 for case in cases)
    assert not any(
        module_name.startswith("playground.eval.benchmarks.IFBench.official.") for module_name in sys.modules
    )

    results = [
        Generated(case=cases[0], response="<think>hidden scratchpad</think> Red"),
        Generated(case=cases[1], response="wrong"),
        Generated(case=cases[2], response="1"),
        Generated(case=cases[3], response="1 2"),
    ]
    metric = benchmark.evaluate(results)

    assert metric.score_avg == 0.5
    assert metric.pass_at_k[1] == 0.5
    assert metric.pass_at_k[2] == 1.0

    no_strip_metric = benchmark_without_strip.evaluate([
        Generated(case=cases[0], response="<think>hidden scratchpad</think> Red")
    ])
    assert no_strip_metric.score_avg == 0.0


def test_ifbench_metric_includes_official_strict_and_loose_reports(tmp_path):
    resource_root = tmp_path / "datasets" / "IFBENCH"
    resource_root.mkdir(parents=True)
    _write_official_style_prompt_file(resource_root / "IFBench_test.jsonl")

    benchmark = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=2,
    )
    strict_primary_benchmark = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=2,
        evaluation_mode="strict",
    )

    cases = benchmark.get_cases()
    results = [
        Generated(case=cases[0], response="preface\nRed"),
        Generated(case=cases[1], response="wrong"),
        Generated(case=cases[2], response="1"),
        Generated(case=cases[3], response="1 2"),
    ]

    loose_metric = benchmark.evaluate(results)
    strict_metric = strict_primary_benchmark.evaluate(results)

    loose_metrics = loose_metric.to_dict()["official_metrics"]
    assert loose_metric.score_avg == 0.5
    assert loose_metrics["loose"]["prompt_level_accuracy"] == 0.5
    assert loose_metrics["strict"]["prompt_level_accuracy"] == 0.25
    assert loose_metrics["loose"]["instruction_level_accuracy"] == 0.5
    assert loose_metrics["strict"]["instruction_level_accuracy"] == 0.25
    assert loose_metrics["loose"]["tier0_accuracy"] == {"count": 0.5, "format": 0.5}
    assert loose_metrics["strict"]["tier0_accuracy"] == {"count": 0.5, "format": 0.0}
    assert loose_metrics["loose"]["tier1_accuracy"] == {"count:numbers": 0.5, "format:options": 0.5}
    assert loose_metrics["strict"]["tier1_accuracy"] == {"count:numbers": 0.5, "format:options": 0.0}

    strict_metrics = strict_metric.to_dict()["official_metrics"]
    assert strict_metric.score_avg == 0.25
    assert strict_metric.to_dict()["evaluation_mode"] == "strict"
    assert strict_metrics == loose_metrics


def test_ifbench_load_records_reuses_parent_shuffle_and_downsample(tmp_path):
    resource_root = tmp_path / "datasets" / "IFBENCH"
    resource_root.mkdir(parents=True)
    _write_official_style_prompt_file(resource_root / "IFBench_test.jsonl")

    benchmark = IFBenchBenchmark(
        data_path=str(resource_root),
        tokenizer=DummyTokenizer(),
        sample_per_prompt=1,
        shuffle_prompts=True,
        down_sample_to=1,
    )

    records = benchmark._load_records()

    assert len(records) == 1
    assert records[0][0] == "IFBENCH"
    assert records[0][1][0]["role"] == "user"

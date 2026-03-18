import asyncio
import json
from pathlib import Path

import pytest
from diskcache import Cache

from playground.eval.eval_sets import simple_eval
from playground.eval.eval_sets.simple_eval import SimpleBenchmarksEvalConfig, SimpleChatGeneratable
from steptronoss.generation.base_benchmark import (
    BenchmarkMeta,
    EvaluationCase,
    EvaluationMeta,
    Generated,
    Prompt,
    SamplingParams,
)

pytestmark = pytest.mark.cpu


class _FakeGroupedProgressBar:
    def __init__(self, totals):
        self.totals = totals
        self.total_updates = 0
        self.group_updates: list[str] = []
        self.closed = False

    def update(self, name: str):
        self.total_updates += 1
        self.group_updates.append(name)

    def close(self):
        self.closed = True


class _InlineGenerationController:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def set_tqdm(self, disabled: bool, total: int, desc: str) -> None:
        self.disabled = disabled
        self.total = total
        self.desc = desc

    def generate(self, gen_items):
        for item in gen_items:
            yield item, asyncio.run(item.generate())

    def shutdown(self):
        return None


class _CountingSimpleChatGeneratable(SimpleChatGeneratable):
    def __init__(self, case: EvaluationCase, response_text: str, call_counter: dict[str, int]):
        super().__init__(
            case=case,
            endpoint_getter=lambda: "http://unused",
            model_name_getter=lambda: "unused-model",
            max_model_len=1024,
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=32,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            ),
        )
        self.response_text = response_text
        self.call_counter = call_counter

    async def generate(self) -> Generated:
        self.call_counter[self.response_text] = self.call_counter.get(self.response_text, 0) + 1
        return Generated(case=self.case, response=self.response_text)


class _TestSimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    def get_benchmarks(self):
        return []


class _DummyTokenizer:
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


class _DummyTokenizerConfig:
    def build_tokenizer(self):
        return _DummyTokenizer()


def _build_case(*, benchmark_name: str, item_id: str, run_index: int, prompt_text: str) -> EvaluationCase:
    return EvaluationCase(
        prompt=Prompt(
            messages=[{"role": "user", "content": prompt_text}],
            prompt_token_count=8,
        ),
        benchmark=BenchmarkMeta(
            benchmark_name=benchmark_name,
            item_id=item_id,
            context={"item_id": item_id},
        ),
        evaluation=EvaluationMeta(
            prompt_index=0,
            run_index=run_index,
        ),
    )


def _write_simple_benchmark_record(path: Path, *, dataset: str, item_id: str, prompt: str, answer: str) -> None:
    path.write_text(
        json.dumps({
            "dataset": dataset,
            "messages": [{"role": "user", "content": prompt}],
            "source_item": {
                "item_id": item_id,
                "answer": answer,
            },
        })
        + "\n",
        encoding="utf-8",
    )


def _write_ifbench_prompt_file(path: Path) -> None:
    path.write_text(
        json.dumps({
            "key": "0",
            "prompt": "Pick one option. Answer with one of the following options: Red/Blue/Green.",
            "instruction_id_list": ["format:options"],
            "kwargs": [{"options": "Red/Blue/Green"}],
        })
        + "\n",
        encoding="utf-8",
    )


def test_simple_chat_generatable_fingerprint_changes_with_sampling_params():
    genable_run0 = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="item-0", run_index=0, prompt_text="What is 1+1?"),
        response_text="2",
        call_counter={},
    )
    genable_run1 = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="item-1", run_index=1, prompt_text="What is 1+1?"),
        response_text="2",
        call_counter={},
    )

    assert genable_run0.fingerprint() != genable_run1.fingerprint()


def test_get_sampling_params_merges_benchmark_overrides():
    cfg = _TestSimpleBenchmarksEvalConfig()
    cfg.max_decode_steps = 256
    cfg.chat_template_args = {"enable_thinking": True}

    sampling_params = cfg.get_sampling_params(
        SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=32,
            seed=7,
            extra_body={"guided_decoding_backend": "xgrammar"},
        )
    )

    assert sampling_params.temperature == 0.0
    assert sampling_params.top_p == 0.9
    assert sampling_params.top_k == -1
    assert sampling_params.max_tokens == 32
    assert sampling_params.seed == 7
    assert sampling_params.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True},
        "guided_decoding_backend": "xgrammar",
    }


def test_get_sampling_params_uses_defaults_when_benchmark_sampling_params_is_none():
    cfg = _TestSimpleBenchmarksEvalConfig()
    cfg.max_decode_steps = 256
    cfg.chat_template_args = {"enable_thinking": True}

    sampling_params = cfg.get_sampling_params(None)

    assert sampling_params.temperature == 1.0
    assert sampling_params.top_p == 1.0
    assert sampling_params.top_k == -1
    assert sampling_params.max_tokens == 256
    assert sampling_params.seed is None
    assert sampling_params.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_get_prompts_loads_ifbench_from_datasets_dir(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_simple_benchmark_record(
        datasets_dir / "AIME2025.jsonl",
        dataset="AIME2025",
        item_id="aime-0",
        prompt="Solve 1+1.",
        answer="2",
    )
    _write_simple_benchmark_record(
        datasets_dir / "GPQA_DIAMOND.jsonl",
        dataset="GPQA_DIAMOND",
        item_id="gpqa-0",
        prompt="Choose A, B, C, or D.",
        answer="A",
    )
    _write_simple_benchmark_record(
        datasets_dir / "HLE_TEXTONLY.jsonl",
        dataset="HLE_TEXTONLY",
        item_id="hle-0",
        prompt="State the final answer.",
        answer="42",
    )
    _write_simple_benchmark_record(
        datasets_dir / "HMMT25.jsonl",
        dataset="HMMT25",
        item_id="hmmt-0",
        prompt="Compute 2+2.",
        answer="4",
    )
    _write_simple_benchmark_record(
        datasets_dir / "MMLU_PRO.jsonl",
        dataset="MMLU_PRO",
        item_id="mmlu-0",
        prompt="Choose A through I.",
        answer="B",
    )
    ifbench_dir = datasets_dir / "IFBENCH"
    ifbench_dir.mkdir()
    _write_ifbench_prompt_file(ifbench_dir / "IFBench_test.jsonl")

    cfg = SimpleBenchmarksEvalConfig()
    cfg.datasets_dir = str(datasets_dir)
    cfg.selected_datasets = "IFBENCH"
    cfg.tokenizer_cfg = _DummyTokenizerConfig()
    cfg.router_addr_key = "unused"
    cfg.model_name_template = "unused-model"
    cfg.max_model_len = 1024
    cfg.max_decode_steps = 256
    cfg.chat_template_args = {"enable_thinking": True}

    prompts = cfg.get_prompts()

    assert len(prompts) == 1
    assert all(prompt.case.benchmark.benchmark_name == "IFBENCH" for prompt in prompts)
    assert all(prompt.case.prompt.messages[0]["content"].startswith("Pick one option.") for prompt in prompts)
    assert all(prompt.case.prompt.sampling_params is not None for prompt in prompts)
    assert all(prompt.case.prompt.sampling_params.temperature == 0.0 for prompt in prompts)
    assert all(prompt.case.prompt.sampling_params.max_tokens == 256 for prompt in prompts)
    assert all(
        prompt.case.prompt.sampling_params.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
        for prompt in prompts
    )


def test_get_prompts_can_enable_ifbench_thinking_via_chat_template_args(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_simple_benchmark_record(
        datasets_dir / "AIME2025.jsonl",
        dataset="AIME2025",
        item_id="aime-0",
        prompt="Solve 1+1.",
        answer="2",
    )
    _write_simple_benchmark_record(
        datasets_dir / "GPQA_DIAMOND.jsonl",
        dataset="GPQA_DIAMOND",
        item_id="gpqa-0",
        prompt="Choose A, B, C, or D.",
        answer="A",
    )
    _write_simple_benchmark_record(
        datasets_dir / "HLE_TEXTONLY.jsonl",
        dataset="HLE_TEXTONLY",
        item_id="hle-0",
        prompt="State the final answer.",
        answer="42",
    )
    _write_simple_benchmark_record(
        datasets_dir / "HMMT25.jsonl",
        dataset="HMMT25",
        item_id="hmmt-0",
        prompt="Compute 2+2.",
        answer="4",
    )
    _write_simple_benchmark_record(
        datasets_dir / "MMLU_PRO.jsonl",
        dataset="MMLU_PRO",
        item_id="mmlu-0",
        prompt="Choose A through I.",
        answer="B",
    )
    ifbench_dir = datasets_dir / "IFBENCH"
    ifbench_dir.mkdir()
    _write_ifbench_prompt_file(ifbench_dir / "IFBench_test.jsonl")

    cfg = SimpleBenchmarksEvalConfig()
    cfg.datasets_dir = str(datasets_dir)
    cfg.selected_datasets = "IFBENCH"
    cfg.tokenizer_cfg = _DummyTokenizerConfig()
    cfg.router_addr_key = "unused"
    cfg.model_name_template = "unused-model"
    cfg.max_model_len = 1024
    cfg.max_decode_steps = 256
    cfg.chat_template_args = {"enable_thinking": True}

    prompts = cfg.get_prompts()

    assert len(prompts) == 1
    assert prompts[0].case.prompt.sampling_params is not None
    assert prompts[0].case.prompt.sampling_params.temperature == 0.0
    assert prompts[0].case.prompt.sampling_params.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_generate_reuses_cached_generated_and_rebinds_case(monkeypatch, tmp_path):
    monkeypatch.setattr(simple_eval, "GroupedProgressBar", _FakeGroupedProgressBar)
    cfg = _TestSimpleBenchmarksEvalConfig()
    cfg.save_dir = str(tmp_path)
    cfg.run_tag = "resume-tag"
    cfg.rerun_level = None

    genable = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="current-item", run_index=0, prompt_text="cached prompt"),
        response_text="fresh",
        call_counter={},
    )
    cached_generated = Generated(
        case=_build_case(benchmark_name="bench", item_id="stale-item", run_index=0, prompt_text="cached prompt"),
        response="cached",
    )

    with Cache(directory=cfg.predictions_path) as generation_cache:
        generation_cache[genable.fingerprint()] = cached_generated

    results = cfg._generate([genable])

    assert len(results) == 1
    assert results[0].response == "cached"
    assert results[0].case == genable.case
    assert results[0].case.benchmark.item_id == "current-item"


def test_generate_rerun_level_error_reruns_cached_errors_once_per_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(simple_eval, "GenerationController", _InlineGenerationController)
    monkeypatch.setattr(simple_eval, "GroupedProgressBar", _FakeGroupedProgressBar)

    cfg = _TestSimpleBenchmarksEvalConfig()
    cfg.save_dir = str(tmp_path)
    cfg.run_tag = "resume-tag"
    cfg.rerun_level = "error"

    call_counter: dict[str, int] = {}
    genable = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="item-a", run_index=0, prompt_text="shared prompt"),
        response_text="fresh",
        call_counter=call_counter,
    )

    with Cache(directory=cfg.predictions_path) as generation_cache:
        generation_cache[genable.fingerprint()] = Generated(case=genable.case, error="cached failure")

    results = cfg._generate([genable])

    assert call_counter == {"fresh": 1}
    assert [result.response for result in results] == ["fresh"]
    assert [result.case.benchmark.item_id for result in results] == ["item-a"]

    with Cache(directory=cfg.predictions_path) as generation_cache:
        cached_generated = generation_cache[genable.fingerprint()]
    assert isinstance(cached_generated, Generated)
    assert cached_generated.error is None
    assert cached_generated.response == "fresh"


def test_generate_rejects_duplicate_fingerprints(tmp_path):
    cfg = _TestSimpleBenchmarksEvalConfig()
    cfg.save_dir = str(tmp_path)
    cfg.run_tag = "resume-tag"

    genable_a = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="item-a", run_index=0, prompt_text="shared prompt"),
        response_text="fresh-a",
        call_counter={},
    )
    genable_b = _CountingSimpleChatGeneratable(
        case=_build_case(benchmark_name="bench", item_id="item-b", run_index=0, prompt_text="shared prompt"),
        response_text="fresh-b",
        call_counter={},
    )

    with pytest.raises(ValueError, match="Duplicate generation fingerprint detected"):
        cfg._generate([genable_a, genable_b])


def test_simple_chat_generatable_warns_when_context_budget_is_clamped(monkeypatch):
    warning_messages: list[str] = []
    monkeypatch.setattr(simple_eval.logger, "warning", lambda message: warning_messages.append(message))

    case = EvaluationCase(
        prompt=Prompt(
            messages=[{"role": "user", "content": "clamped prompt"}],
            prompt_token_count=1018,
        ),
        benchmark=BenchmarkMeta(
            benchmark_name="bench",
            item_id="item-0",
            context={"item_id": "item-0"},
        ),
        evaluation=EvaluationMeta(prompt_index=0, run_index=0),
    )

    genable = _CountingSimpleChatGeneratable(
        case=case,
        response_text="ok",
        call_counter={},
    )

    assert genable.case.prompt.sampling_params is not None
    assert genable.case.prompt.sampling_params.max_tokens == 6
    assert len(warning_messages) == 1
    assert "Clamped generation budget" in warning_messages[0]

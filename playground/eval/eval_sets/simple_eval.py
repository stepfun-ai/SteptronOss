from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pprint import pformat
from typing import Literal, TypeAlias

import aiohttp
from configurize import Ref, writable_property
from diskcache import Cache
from loguru import logger

from playground.eval.benchmarks.common import JsonlChatBenchmark
from steptronoss.exp.base_exp import TokenizerConfig
from steptronoss.exp.gen_eval import GenableEvalConfig
from steptronoss.generation.async_generation import GenerationController
from steptronoss.generation.base_benchmark import (
    CompletionChoice,
    CompletionMessage,
    EvaluationCase,
    Generated,
    JsonObject,
    SamplingParams,
)
from steptronoss.generation.base_generatable import EndpointGetter, GenableItem, ModelNameGetter
from steptronoss.utils.general import GroupedProgressBar, retry_on

ChatTemplateArgValue: TypeAlias = str | bool | int
ChatTemplateArgs: TypeAlias = dict[str, ChatTemplateArgValue]


class RetriableChatCompletionError(RuntimeError):
    """Transient chat/completions failure that should be retried."""


class SimpleChatGeneratable(GenableItem):
    def __init__(
        self,
        case: EvaluationCase,
        endpoint_getter: EndpointGetter,
        model_name_getter: ModelNameGetter,
        max_model_len: int,
        sampling_params: SamplingParams,
    ):
        super().__init__()
        self.case = case
        self.endpoint_getter = endpoint_getter
        self.model_name_getter = model_name_getter

        prompt = self.case.prompt
        if prompt.messages is None:
            raise TypeError("SimpleAirChatGeneratable requires Prompt.messages for chat/completions requests.")
        if prompt.prompt_token_count is None:
            raise ValueError("SimpleAirChatGeneratable requires Prompt.prompt_token_count for context budgeting.")

        remaining_context = max_model_len - prompt.prompt_token_count
        if remaining_context < 1:
            raise ValueError(
                "Prompt for "
                f"{self.case.benchmark.benchmark_name}:{self.case.benchmark.item_id} already uses "
                f"{prompt.prompt_token_count} tokens, "
                f"which leaves no room under max_model_len={max_model_len}."
            )
        resolved_max_tokens = remaining_context
        if sampling_params.max_tokens is not None:
            resolved_max_tokens = min(sampling_params.max_tokens, remaining_context)
            if resolved_max_tokens < sampling_params.max_tokens:
                logger.warning(
                    "Clamped generation budget for "
                    f"{self.case.benchmark.benchmark_name}:{self.case.benchmark.item_id} "
                    f"from max_tokens={sampling_params.max_tokens} to {resolved_max_tokens} "
                    f"because prompt_token_count={prompt.prompt_token_count} leaves only "
                    f"{remaining_context} tokens under max_model_len={max_model_len}."
                )
        resolved_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=resolved_max_tokens,
            seed=self.case.evaluation.run_index if sampling_params.seed is None else sampling_params.seed,
            extra_body=copy.deepcopy(sampling_params.extra_body),
        )
        self.case = self.case.with_sampling_params(resolved_sampling_params)

    def fingerprint(self) -> str:
        payload = {
            "genable_type": type(self).__name__,
            "prompt": self.case.prompt.to_dict(),
        }
        serialized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
        return f"{type(self).__name__}:v1:{digest}"

    @staticmethod
    def _parse_choice(raw_choice: object) -> tuple[CompletionChoice, str, str | None]:
        if not isinstance(raw_choice, dict):
            raise TypeError(f"Unexpected vLLM choice payload: {raw_choice}")

        choice_raw: JsonObject = raw_choice
        choice: CompletionChoice = {"raw": choice_raw}

        finish_reason = choice_raw.get("finish_reason")
        if isinstance(finish_reason, str):
            choice["finish_reason"] = finish_reason

        response_text = ""
        reasoning_content: str | None = None

        message_value = choice_raw.get("message")
        if isinstance(message_value, dict):
            completion_message: CompletionMessage = {}
            message_content = message_value.get("content")
            if isinstance(message_content, str):
                completion_message["content"] = message_content
                response_text = message_content
            raw_reasoning = message_value.get("reasoning_content")
            if isinstance(raw_reasoning, str):
                completion_message["reasoning_content"] = raw_reasoning
                reasoning_content = raw_reasoning
            if completion_message:
                choice["message"] = completion_message

        text_value = choice_raw.get("text")
        if isinstance(text_value, str):
            choice["text"] = text_value
            if not response_text:
                response_text = text_value

        return choice, response_text, reasoning_content

    @retry_on(
        (aiohttp.ClientError, asyncio.TimeoutError, RetriableChatCompletionError),
        for_times=3,
        delay=1.0,
        max_delay=10.0,
        backoff=2.0,
        jitter=0.1,
    )
    async def _post_chat_completion(self, payload: JsonObject) -> str:
        timeout = aiohttp.ClientTimeout(total=7200.0)
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=False) as session,
            session.post(
                url=f"{self.endpoint_getter()}/v1/chat/completions",
                json=payload,
            ) as response,
        ):
            response_text = await response.text()
            response_status = response.status
            response_content_type = response.headers.get("content-type", "")

        if response_status != 200:
            body_preview = response_text[:4000]
            message = (
                f"vLLM chat/completions failed with HTTP {response_status} ({response_content_type}): {body_preview}"
            )
            if response_status in {408, 425, 429} or 500 <= response_status < 600:
                raise RetriableChatCompletionError(message)
            raise RuntimeError(message)
        return response_text

    async def generate(self) -> Generated:
        prompt = self.case.prompt
        if prompt.messages is None:
            raise TypeError("SimpleAirChatGeneratable requires Prompt.messages for chat/completions requests.")
        payload: JsonObject = {
            "model": self.model_name_getter(),
            "messages": prompt.messages,
        }
        if prompt.sampling_params is not None:
            payload.update(prompt.sampling_params.to_dict())

        response_text = await self._post_chat_completion(payload)

        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as exc:
            body_preview = response_text[:4000]
            raise RuntimeError(f"vLLM chat/completions returned a non-JSON body {body_preview}") from exc

        if not isinstance(response_json, dict):
            raise TypeError(f"Unexpected vLLM response: {response_json}")
        if "choices" not in response_json:
            raise ValueError(f"Unexpected vLLM response: {response_json}")
        if not isinstance(response_json["choices"], list) or not response_json["choices"]:
            raise ValueError(f"Unexpected vLLM response: {response_json}")

        choice, response_text, reasoning_content = self._parse_choice(response_json["choices"][0])
        return Generated(
            case=self.case,
            response=response_text,
            choice=choice,
            reasoning_content=reasoning_content,
        )


class SimpleBenchmarksEvalConfig(GenableEvalConfig):
    selected_datasets: str | None = None
    """Optional comma-separated dataset allowlist, e.g. "AIME2025,GPQA_DIAMOND"."""

    chat_template_args: ChatTemplateArgs = {"enable_thinking": True}
    """Chat template kwargs used for prompt rendering and request.chat_template_kwargs."""

    shuffle_prompts: bool = True
    """If True, shuffle merged prompts from all benchmarks with seed 1234 to balance engine load."""

    datasets_dir: str = "/oss/benchmarks/simple_benchmarks/datasets"
    """Directory containing the simple benchmark datasets; IFBench expects `IFBENCH/` resources under this root."""

    save_dir: str = Ref("..log_path")
    """Directory used to save prediction dumps and summaries."""

    predictions_cache_size_limit_bytes: int = 1 << 40
    """Diskcache size limit for prediction artifacts. Keep this large enough for full multi-benchmark runs."""

    predictions_cache_eviction_policy: str = "none"
    """Eviction policy for prediction cache. Use 'none' so completed generations are never silently culled."""

    max_decode_steps: int = 128 * 1024
    """Maximum generated tokens per request before the request-level context cap is applied."""

    num_concurrent_requests: int = 4096
    """Maximum number of in-flight genables allowed across GenerationController."""

    tokenizer_cfg: TokenizerConfig
    """Tokenizer config used to render prompts and count response tokens."""

    router_addr_key: str = Ref("..vllm_cfg.router_addr_key")
    """Key used to resolve the router address from Redis."""

    model_name_template: str = Ref("..vllm_cfg.model_name_template")
    """Template for the served model name in vLLM."""

    max_model_len: int = Ref("..vllm_cfg.max_seq_len")
    """Maximum total context length accepted by the backing vLLM server."""

    rerun_level: Literal["error", "all"] | None = None
    """Cache policy: None reuses all hits, 'error' reruns cached errors, 'all' reruns everything."""

    def get_sampling_params(self, benchmark_sampling_params: SamplingParams | None) -> SamplingParams:
        """Build request sampling params for one prompt, optionally merging benchmark overrides."""

        if benchmark_sampling_params is None:
            benchmark_sampling_params = SamplingParams()

        extra_body: JsonObject = {}
        if self.chat_template_args is not None:
            extra_body["chat_template_kwargs"] = copy.deepcopy(self.chat_template_args)
        extra_body.update(copy.deepcopy(benchmark_sampling_params.extra_body))

        return SamplingParams(
            temperature=1.0 if benchmark_sampling_params.temperature is None else benchmark_sampling_params.temperature,
            top_p=1.0 if benchmark_sampling_params.top_p is None else benchmark_sampling_params.top_p,
            top_k=-1 if benchmark_sampling_params.top_k is None else benchmark_sampling_params.top_k,
            max_tokens=self.max_decode_steps
            if benchmark_sampling_params.max_tokens is None
            else benchmark_sampling_params.max_tokens,
            seed=benchmark_sampling_params.seed,
            extra_body=extra_body,
        )

    @property
    def predictions_path(self) -> str:
        return os.path.join(self.save_dir, self.run_tag, "predictions")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.save_dir, self.run_tag, "summary.json")

    @writable_property
    def run_tag(self) -> str:
        """Generation-cache namespace. Defaults to a UTC timestamp unless overridden."""

        cache_tag = self.__dict__.get("_cache_tag_default")
        if cache_tag is None:
            cache_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.__dict__["_cache_tag_default"] = cache_tag
        return cache_tag

    def _select_benchmarks_by_name(self, benchmarks: list[JsonlChatBenchmark]) -> list[JsonlChatBenchmark]:
        if self.selected_datasets is None:
            return benchmarks
        selected_text = self.selected_datasets.strip()
        selected_names = [item.strip() for item in selected_text.split(",") if item.strip()]
        if len(selected_names) == 0:
            return benchmarks

        benchmark_map = {benchmark.name: benchmark for benchmark in benchmarks}
        missing = [name for name in selected_names if name not in benchmark_map]
        if missing:
            raise ValueError(f"Unsupported selected_datasets={missing}. Supported datasets: {sorted(benchmark_map)}")
        return [benchmark_map[name] for name in selected_names]

    def get_benchmarks(self) -> list[JsonlChatBenchmark]:
        from playground.eval.benchmarks.AIME25 import AIME25Benchmark
        from playground.eval.benchmarks.GPQADiamond import GPQADiamondBenchmark
        from playground.eval.benchmarks.HMMT25 import HMMT25Benchmark
        from playground.eval.benchmarks.IFBench import IFBenchBenchmark
        from playground.eval.benchmarks.MMLUPro import MMLUProBenchmark

        chat_template_options = self.chat_template_args
        tokenizer = self.tokenizer_cfg.build_tokenizer()

        benchmarks = [
            AIME25Benchmark(
                data_path=os.path.join(self.datasets_dir, "AIME2025.jsonl"),
                tokenizer=tokenizer,
                sample_per_prompt=64,
                chat_template_options=chat_template_options,
            ),
            GPQADiamondBenchmark(
                data_path=os.path.join(self.datasets_dir, "GPQA_DIAMOND.jsonl"),
                tokenizer=tokenizer,
                sample_per_prompt=16,
                chat_template_options=chat_template_options,
            ),
            HMMT25Benchmark(
                data_path=os.path.join(self.datasets_dir, "HMMT25.jsonl"),
                tokenizer=tokenizer,
                sample_per_prompt=64,
                chat_template_options=chat_template_options,
            ),
            IFBenchBenchmark(
                data_path=os.path.join(self.datasets_dir, "IFBENCH"),
                tokenizer=tokenizer,
                sample_per_prompt=1,
            ),
            MMLUProBenchmark(
                data_path=os.path.join(self.datasets_dir, "MMLU_PRO.jsonl"),
                tokenizer=tokenizer,
                sample_per_prompt=1,
                chat_template_options=chat_template_options,
            ),
        ]
        benchmarks = self._select_benchmarks_by_name(benchmarks)
        return benchmarks

    def get_prompts(self) -> list[SimpleChatGeneratable]:
        endpoint_getter = EndpointGetter(self.router_addr_key)
        model_name_getter = ModelNameGetter(self.model_name_template)

        trainables: list[SimpleChatGeneratable] = []
        for benchmark in self.get_benchmarks():
            for case in benchmark.get_cases():
                sampling_params = self.get_sampling_params(case.prompt.sampling_params)
                trainables.append(
                    SimpleChatGeneratable(
                        case=case,
                        endpoint_getter=endpoint_getter,
                        model_name_getter=model_name_getter,
                        max_model_len=self.max_model_len,
                        sampling_params=sampling_params,
                    )
                )

        if self.shuffle_prompts:
            random.Random(1234).shuffle(trainables)

        logger.info(f"Loaded {len(trainables)} generation requests from {self.datasets_dir}")
        return trainables

    @staticmethod
    def _rebind_generated(generated: Generated, genable: SimpleChatGeneratable) -> Generated:
        """Reuse cached output with the current run's case metadata."""
        return Generated(
            case=genable.case,
            choice=None if generated.choice is None else copy.deepcopy(generated.choice),
            response=generated.response,
            reasoning_content=generated.reasoning_content,
            error=generated.error,
        )

    def _summarize_results(self, results: list[Generated]) -> JsonObject:
        benchmark_map = {benchmark.name: benchmark for benchmark in self.get_benchmarks()}
        by_benchmark: dict[str, list[Generated]] = {name: [] for name in benchmark_map}
        total_errors = 0

        for result in results:
            by_benchmark.setdefault(result.case.benchmark.benchmark_name, []).append(result)
            if result.error:
                total_errors += 1

        summary: JsonObject = {
            "total_requests": len(results),
            "total_errors": total_errors,
            "by_benchmark": {},
        }

        for benchmark_name, benchmark_results in by_benchmark.items():
            benchmark = benchmark_map[benchmark_name]
            metric = benchmark.evaluate(benchmark_results)
            finish_reason_counts: dict[str, int] = {}
            total_chars = 0
            total_tokens = 0
            valid_count = 0
            error_count = 0
            for result in benchmark_results:
                finish_reason_counts[result.finish_reason] = finish_reason_counts.get(result.finish_reason, 0) + 1
                if result.error:
                    error_count += 1
                    continue
                valid_count += 1
                total_chars += len(result.response)
                total_tokens += benchmark.count_response_tokens(result.response)

            summary["by_benchmark"][benchmark_name] = {
                "count": len(benchmark_results),
                "errors": error_count,
                "finish_reason_counts": finish_reason_counts,
                "avg_response_chars": total_chars / max(valid_count, 1),
                "avg_response_tokens": total_tokens / max(valid_count, 1),
                "metric": metric.to_dict(),
            }

        return summary

    def _generate(self, genables: list[SimpleChatGeneratable]) -> list[Generated]:
        genables_by_fingerprint: dict[str, SimpleChatGeneratable] = {}
        for raw_genable in genables:
            fingerprint = raw_genable.fingerprint()
            if fingerprint in genables_by_fingerprint:
                raise ValueError(f"Duplicate generation fingerprint detected: {fingerprint}")
            genables_by_fingerprint[fingerprint] = raw_genable

        group_totals = {benchmark.name: 0 for benchmark in self.get_benchmarks()}
        for genable in genables:
            group_name = genable.case.benchmark.benchmark_name
            group_totals[group_name] = group_totals.get(group_name, 0) + 1
        progress_bar = GroupedProgressBar(group_totals)
        controller: GenerationController | None = None
        try:
            pending_genables: list[SimpleChatGeneratable] = []
            reused_count = 0
            with Cache(
                directory=self.predictions_path,
                size_limit=self.predictions_cache_size_limit_bytes,
                eviction_policy=self.predictions_cache_eviction_policy,
            ) as generation_cache:
                for fingerprint, genable in genables_by_fingerprint.items():
                    should_generate = (
                        self.rerun_level == "all"
                        or fingerprint not in generation_cache
                        or (self.rerun_level == "error" and generation_cache[fingerprint].error is not None)
                    )
                    if not should_generate:
                        generation_cache[fingerprint] = self._rebind_generated(generation_cache[fingerprint], genable)
                        reused_count += 1
                        progress_bar.update(genable.case.benchmark.benchmark_name)
                        continue
                    pending_genables.append(genable)

                logger.info(
                    f"Generation cache tag={self.run_tag} dir={self.predictions_path} "
                    f"reused={reused_count} missing={len(pending_genables)} "
                    f"total_unique={len(genables_by_fingerprint)} rerun_level={self.rerun_level}"
                )

                if pending_genables:
                    controller = GenerationController(
                        max_concurrent_genables=self.num_concurrent_requests,
                    )
                    controller.set_tqdm(disabled=True, total=len(pending_genables), desc="Evaluation Requests")
                    for raw_genable, result in controller.generate(pending_genables):
                        genable: SimpleChatGeneratable = raw_genable
                        fingerprint = genable.fingerprint()
                        if isinstance(result, Exception):
                            generated = Generated(case=genable.case, error=repr(result))
                        else:
                            generated = result
                        generation_cache[fingerprint] = generated
                        progress_bar.update(genable.case.benchmark.benchmark_name)

                results = [generation_cache[genable.fingerprint()] for genable in genables]
        finally:
            if controller is not None:
                controller.shutdown()
            progress_bar.close()

        return results

    def eval(self) -> JsonObject:
        os.makedirs(self.save_dir, exist_ok=True)
        genables = self.get_prompts()
        results = self._generate(genables)
        summary = self._summarize_results(results)
        with open(self.summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)

        logger.info(f"Saved predictions to {self.predictions_path}")
        logger.info(f"Saved summary to {self.summary_path}")
        logger.info(f"Eval summary: {pformat(summary)}")
        return summary

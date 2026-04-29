"""Step3-VL lightweight IFBench eval.

Paper target and local score scale:

    Step3-VL reports IFBench 43.28 on the official Step3-VL-10B checkpoint.
    This exp stores scores as fractions, so paper 43.28 corresponds to 0.4328.
    Defaults here use the paper-style light run: max length 65536,
    temperature=1.0, top_p=1.0, top_k=0, and IFBench repeat=4.

Official checkpoint measured on 2026-04-29 with one H200 from `step4_eval`:

    checkpoint: /oss/opensource_models/Step3-VL-10B
    run_tag: step3v_ifbench_h200_20260429_official_local_4x
    summary: /oss/logs/step3v_eval_simple_benchmarks/step3v_ifbench_h200_20260429_official_local_4x/summary.json

    launch:
        export EXP_ID=step3vifbofficialh2000429
        export STEPTRON_MEET_DIR=/tmp/steptron_meet_step3v_ifbench_official_h200_0429
        export PYTHONDONTWRITEBYTECODE=1
        python tools/mp_run.py playground/eval/step3v/step3v_eval_simple_benchmarks.py \
            eval_cfg.run_tag=step3v_ifbench_h200_20260429_official_local_4x \
            eval_cfg.rerun_level=all

    The summary-producing resume reused the same run_tag after the first
    tools/mp_run.py attempt generated 1061/1176 cached outputs but exited before
    writing summary.json.

    total_requests: 1176
    total_errors: 0
    finish_reason_counts: {"stop": 1176}
    score_avg/pass@1/loose_prompt_level_accuracy: 0.5119047619047619
    pass@2: 0.6196145124716554
    pass@4: 0.7006802721088435
    loose_instruction_level_accuracy: 0.5425373134328358
    strict_prompt_level_accuracy: 0.4608843537414966
    strict_instruction_level_accuracy: 0.49328358208955225

"""

from __future__ import annotations

import os

from configurize import Ref
from loguru import logger

from playground.eval.benchmarks.common import ChatTokenizer
from playground.eval.eval_sets.simple_eval import SimpleBenchmarksEvalConfig
from steptronoss.exp.base_exp import BaseExp, TokenizerConfig
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.resources import ResourceConfig, TaskSpec
from steptronoss.generation.base_benchmark import SamplingParams
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig


class Step3VTokenizerConfig(TokenizerConfig):
    tokenizer_path: str = Ref("...vllm_cfg.tokenizer_path")
    """Tokenizer directory for the target Step3-VL checkpoint."""

    def build_tokenizer(self) -> ChatTokenizer:  # type: ignore[override]
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path, trust_remote_code=True)


class Step3VSimpleEvalResourceConfig(ResourceConfig):
    vllm_replica: int = 1
    """Number of Step3-VL vLLM worker tasks to launch."""

    def __init__(self):
        super().__init__()
        self.command = ".venv/bin/python {COMMAND}"
        workspace_venv_bin = os.path.join(os.getcwd(), ".venv", "bin")
        current_path = os.environ.get("PATH", "")
        self.envs["PATH"] = f"{workspace_venv_bin}:{current_path}" if current_path else workspace_venv_bin
        self.replica = 1
        self.gpu = 1
        self.node_type = "gpu"
        self.vllm_replica = 1
        self.task_specs = {
            "evaluator": TaskSpec(
                gpu=0,
                node_type="cpu",
                envs={"ROLE": "evaluator"},
                is_critical=True,
            ),
            "vllm": TaskSpec(
                replica=self.vllm_replica,
                envs={"ROLE": "vllm"},
            ),
            "router": TaskSpec(
                gpu=0,
                node_type="cpu",
                envs={"ROLE": "router"},
            ),
        }


class Step3VSimpleEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/oss/opensource_models/Step3-VL-10B"
        self.tokenizer_path = "/oss/opensource_models/Step3-VL-10B"
        self.max_seq_len = 65536
        self.vllm_gpu_memory_utilization = 0.85

        self.vllm_tp = 1
        self.vllm_dp = 1

        self.reasoning_parser = "deepseek_r1"
        self.toolcall_parser = "hermes"

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.vllm_enforce_eager = True
        self.vllm_max_num_batched_tokens = 8192
        self.max_cache_size = 64


class Step3VSimpleEvalVLLMRouterConfig(VLLMRouterConfig):
    routed_methods = {
        "completions": ["POST"],
        "chat/completions": ["POST"],
    }


class Step3VSimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    tokenizer_cfg: Step3VTokenizerConfig = Step3VTokenizerConfig
    """Tokenizer config for Step3-VL prompt rendering and token counting."""

    selected_datasets = "IFBENCH"
    chat_template_args = {}
    num_concurrent_requests = 8
    max_decode_steps = 65536

    ifbench_sample_per_prompt: int = 4
    """Paper-style IFBench repeat count for Step3-VL."""

    ifbench_temperature: float = 1.0
    """Paper-style Step3-VL IFBench sampling temperature."""

    ifbench_top_p: float = 1.0
    """Paper-style Step3-VL IFBench nucleus-sampling threshold."""

    ifbench_top_k: int = 0
    """Paper-style Step3-VL IFBench top-k setting, where 0 disables top-k in vLLM."""

    def get_benchmarks(self):
        if self.selected_datasets is not None and self.selected_datasets.strip() != "IFBENCH":
            return super().get_benchmarks()

        from playground.eval.benchmarks.IFBench import IFBenchBenchmark

        tokenizer = self.tokenizer_cfg.build_tokenizer()
        benchmarks = [
            IFBenchBenchmark(
                data_path=os.path.join(self.datasets_dir, "IFBENCH"),
                tokenizer=tokenizer,
                sample_per_prompt=self.ifbench_sample_per_prompt,
                sampling_params=SamplingParams(
                    temperature=self.ifbench_temperature,
                    top_p=self.ifbench_top_p,
                    top_k=self.ifbench_top_k,
                ),
            )
        ]
        return self._select_benchmarks_by_name(benchmarks)


class Exp(BaseExp):
    vllm_cfg: Step3VSimpleEvalVLLMDeployConfig = Step3VSimpleEvalVLLMDeployConfig

    resource_cfg: Step3VSimpleEvalResourceConfig = Step3VSimpleEvalResourceConfig
    vllm_router_cfg: Step3VSimpleEvalVLLMRouterConfig = Step3VSimpleEvalVLLMRouterConfig
    eval_cfg: Step3VSimpleBenchmarksEvalConfig = Step3VSimpleBenchmarksEvalConfig

    log_dir = "/oss/logs/"

    def entrypoint(self) -> None:
        self.update_from_args()
        role = os.environ.get("ROLE", "evaluator")
        if role == "router":
            logger.info("Starting vLLM router...")
            self.vllm_router_cfg.run()
            return
        if role == "vllm":
            logger.info("Starting vLLM worker...")
            self.vllm_cfg.run_as_worker()
            return
        if role == "evaluator":
            self.sanity_check()
            logger.info("Waiting for vLLM servers to register...")
            self.vllm_cfg.build_cli().wait_for_server()
            summary = self.eval_cfg.eval()
            logger.info(f"Eval results: {summary}")
            return
        raise ValueError(f"Unknown ROLE: {role}")


if __name__ == "__main__":
    Exp().entrypoint()

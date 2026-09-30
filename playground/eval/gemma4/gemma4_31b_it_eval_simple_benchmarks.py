"""Gemma4 31B IT simple eval on the shared 19334-request benchmark subset.

Latest recorded result:
  - run: ``/oss/logs/gemma4_31b_it_eval_simple_benchmarks/gemma4_full_simple_eval_tp4_dp2_20260405T184958Z/summary.json``
  - AIME2025: 0.8589
  - GPQA_DIAMOND: 0.8539
  - HMMT25: 0.7677
  - IFBENCH loose: 0.3061
  - IFBENCH strict: 0.2755
  - MMLU_PRO: 0.8000

Step3.5 Flash reference on the same subset:
  - run: ``/oss/logs/step3p5_eval_simple_benchmarks/step3p5_it4354_0314seq_rep8/summary.json``
  - AIME2025: 0.9427
  - GPQA_DIAMOND: 0.7967
  - HMMT25: 0.9432
  - IFBENCH loose: 0.6054
  - IFBENCH strict: 0.5680
  - MMLU_PRO: 0.7663
"""

from __future__ import annotations

import os

from loguru import logger

from playground.eval.benchmarks.common import ChatTokenizer
from playground.eval.eval_sets.simple_eval import SimpleBenchmarksEvalConfig
from steptronoss.exp.base_exp import BaseExp, TokenizerConfig
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.resources import ResourceConfig, TaskSpec
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig

GEMMA4_31B_IT_MODEL_PATH = "/mnt/step2-alignment-jfs/zane/opensources_model/gemma-4-31B-it"


class Gemma4TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = GEMMA4_31B_IT_MODEL_PATH
    """Tokenizer directory for Gemma4 31B IT simple eval."""

    def build_tokenizer(self) -> ChatTokenizer:
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path)


class Gemma4SimpleEvalResourceConfig(ResourceConfig):
    vllm_replica: int = 1
    """Launch one vLLM worker task for the Gemma4 simple-eval run."""

    def __init__(self):
        super().__init__()
        self.command = "python {COMMAND}"
        self.replica = 1
        self.gpu = 8
        self.node_type = "gpu"
        self.vllm_replica = 1

    @property
    def task_specs(self):
        return {
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


class Gemma4_31BITEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = GEMMA4_31B_IT_MODEL_PATH
        self.tokenizer_path = GEMMA4_31B_IT_MODEL_PATH
        self.max_seq_len = 128 * 1024
        self.vllm_gpu_memory_utilization = 0.9

        self.vllm_tp = 4
        self.vllm_dp = 1

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.max_cache_size = 256


class Gemma4SimpleEvalVLLMRouterConfig(VLLMRouterConfig):
    routed_methods = {
        "completions": ["POST"],
        "chat/completions": ["POST"],
    }


class Gemma4SimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    tokenizer_cfg: Gemma4TokenizerConfig = Gemma4TokenizerConfig
    """Tokenizer config for Gemma4 31B IT prompt rendering and token counting."""

    num_concurrent_requests = 4096
    max_decode_steps = 128 * 1024
    # Keep the Step3.5 128k decode budget but reserve a small slack so vLLM
    # does not reject requests when prompt_token_count leaves only max_len-1
    # tokens available.
    context_budget_margin_tokens = 256


class Exp(BaseExp):
    vllm_cfg: VLLMDeployConfig = Gemma4_31BITEvalVLLMDeployConfig

    resource_cfg: Gemma4SimpleEvalResourceConfig = Gemma4SimpleEvalResourceConfig
    vllm_router_cfg: Gemma4SimpleEvalVLLMRouterConfig = Gemma4SimpleEvalVLLMRouterConfig
    eval_cfg: Gemma4SimpleBenchmarksEvalConfig = Gemma4SimpleBenchmarksEvalConfig

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

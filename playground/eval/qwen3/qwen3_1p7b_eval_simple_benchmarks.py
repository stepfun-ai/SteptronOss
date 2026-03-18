from __future__ import annotations

import os

from loguru import logger

from playground.eval.benchmarks.common import ChatTokenizer
from playground.eval.eval_sets.simple_eval import SimpleBenchmarksEvalConfig
from steptronoss.exp.base_exp import BaseExp, TokenizerConfig
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.resources import ResourceConfig, TaskSpec
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig


class Qwen3TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = "/oss/opensources_model/Qwen3-1.7B-Base/"
    """Tokenizer directory for Qwen3-1.7B."""

    def build_tokenizer(self) -> ChatTokenizer:
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path, trust_remote_code=True)


class Qwen3SimpleEvalResourceConfig(ResourceConfig):
    vllm_replica: int = 2
    """Number of vLLM worker tasks to launch for this eval."""

    def __init__(self):
        super().__init__()
        self.command = "python {COMMAND}"
        self.replica = 1
        self.gpu = 8
        self.node_type = "gpu"
        self.vllm_replica = 2
        self._sync_task_specs()

    def _sync_task_specs(self) -> None:
        self.task_specs = {
            "evaluator": TaskSpec(
                gpu=0,
                node_type="cpu",
                envs={"ROLE": "evaluator"},
                is_critical=True,
            ),
            "vllm": TaskSpec(
                replica=self.vllm_replica,
                envs={
                    "ROLE": "vllm",
                },
            ),
            "router": TaskSpec(
                gpu=0,
                node_type="cpu",
                envs={"ROLE": "router"},
            ),
        }

    def find_leaf_task_specs(self):
        self._sync_task_specs()
        return super().find_leaf_task_specs()


class Qwen3_1p7BEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/oss/opensources_model/Qwen3-1.7B-Base/"
        self.max_seq_len = 65536
        self.vllm_gpu_memory_utilization = 0.9

        self.vllm_tp = 1
        self.vllm_dp = 8

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.max_cache_size = 256


class Qwen3_8BEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/oss/opensources_model/Qwen3-8B-Base/"
        self.max_seq_len = 131072
        self.vllm_gpu_memory_utilization = 0.9

        self.vllm_tp = 1
        self.vllm_dp = 8

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.max_cache_size = 256


class Qwen3SimpleEvalVLLMRouterConfig(VLLMRouterConfig):
    routed_methods = {
        "completions": ["POST"],
        "chat/completions": ["POST"],
    }


class Qwen3SimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    tokenizer_cfg: Qwen3TokenizerConfig = Qwen3TokenizerConfig
    """Tokenizer config kept for future prompt debugging and parity checks."""
    num_concurrent_requests = 4096


class Exp(BaseExp):
    vllm_cfg: VLLMDeployConfig = Qwen3_8BEvalVLLMDeployConfig

    resource_cfg: Qwen3SimpleEvalResourceConfig = Qwen3SimpleEvalResourceConfig
    vllm_router_cfg: Qwen3SimpleEvalVLLMRouterConfig = Qwen3SimpleEvalVLLMRouterConfig
    eval_cfg: Qwen3SimpleBenchmarksEvalConfig = Qwen3SimpleBenchmarksEvalConfig

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

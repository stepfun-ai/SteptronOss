from __future__ import annotations

import os

from configurize import Ref, writable_property
from loguru import logger

from playground.eval.benchmarks.common import ChatTokenizer
from playground.eval.eval_sets.simple_eval import SimpleBenchmarksEvalConfig
from steptronoss.exp.base_exp import BaseExp, TokenizerConfig
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.resources import ResourceConfig, TaskSpec
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig


class Glm5TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = Ref("...vllm_cfg.tokenizer_path")
    """Tokenizer directory for GLM-5 prompt rendering and token counting."""

    def build_tokenizer(self) -> ChatTokenizer:
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path, trust_remote_code=True)


class Glm5SimpleEvalResourceConfig(ResourceConfig):
    vllm_replica: int = 2
    """Number of GLM-5 vLLM worker tasks to launch; GLM-5 serves as one multi-node vLLM by default."""

    def __init__(self):
        super().__init__()
        del self.task_specs
        self.command = ".venv/bin/python {COMMAND}"
        workspace_venv_bin = os.path.join(os.getcwd(), ".venv", "bin")
        current_path = os.environ.get("PATH", "")
        self.envs["PATH"] = f"{workspace_venv_bin}:{current_path}" if current_path else workspace_venv_bin
        self.replica = 1
        self.gpu = 8
        self.node_type = "gpu"
        self.vllm_replica = 2
        _ = self.task_specs

    @writable_property
    def task_specs(self) -> dict[str, TaskSpec]:
        task_specs = self.__dict__.get("task_specs")
        if task_specs is not None:
            return task_specs

        task_specs = {
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
        self.__dict__["task_specs"] = task_specs
        return task_specs


class Glm5SimpleEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/oss/opensources_model/GLM-5/"
        self.reasoning_parser = "glm45"
        self.max_seq_len = 114688
        self.vllm_gpu_memory_utilization = 0.9

        self.vllm_tp = 8
        self.vllm_pp = 2
        self.vllm_dp = 1
        self.multi_node_serving = True
        self.distributed_executor_backend = "mp"
        self.enable_expert_parallel = True
        self.all2all_backend = "deepep_high_throughput"
        self.compilation_config = {"cudagraph_mode": "NONE"}

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.vllm_max_num_batched_tokens = 4096
        self.vllm_enforce_eager = True
        self.max_cache_size = 64


class Glm5SimpleEvalVLLMRouterConfig(VLLMRouterConfig):
    routed_methods = {
        "completions": ["POST"],
        "chat/completions": ["POST"],
    }


class Glm5SimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    selected_datasets: str | None = "AIME2025"
    """Default benchmark selection for this GLM-5 eval exp."""

    max_decode_steps: int = 114688
    """Maximum generated tokens per request before prompt-length clamping."""

    tokenizer_cfg: Glm5TokenizerConfig = Glm5TokenizerConfig
    """Tokenizer config for GLM-5 simple-benchmark prompt rendering."""

    num_concurrent_requests = 4096


class Exp(BaseExp):
    vllm_cfg: VLLMDeployConfig = Glm5SimpleEvalVLLMDeployConfig

    resource_cfg: Glm5SimpleEvalResourceConfig = Glm5SimpleEvalResourceConfig
    vllm_router_cfg: Glm5SimpleEvalVLLMRouterConfig = Glm5SimpleEvalVLLMRouterConfig
    eval_cfg: Glm5SimpleBenchmarksEvalConfig = Glm5SimpleBenchmarksEvalConfig

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

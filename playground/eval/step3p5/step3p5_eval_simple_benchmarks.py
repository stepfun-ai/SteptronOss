"""
SFT model trained use `playground/sft/step3/step3p5_flash_sft_step3_data_muon.py`:
- Benchmarks
  | Benchmark    |   Requests | Trunc@128k   |   Avg chars |   Avg toks |   score_avg |   score_std |   pass@1 |
  |--------------|------------|--------------|-------------|------------|-------------|-------------|----------|
  | AIME2025     |       1920 | 0.73%        |      539.76 |     219.71 |        0.94 |        0.23 |     0.94 |
  | GPQA_DIAMOND |       3168 | 0.00%        |      671.70 |     209.06 |        0.80 |        0.40 |     0.80 |
  | HMMT25       |       1920 | 0.31%        |      323.21 |     124.54 |        0.94 |        0.23 |     0.94 |
  | IFBENCH      |        294 | 9.86%        |     1380.72 |     304.23 |        0.61 |        0.49 |     0.61 |
  | MMLU_PRO     |      12032 | 0.00%        |      323.72 |     100.02 |        0.77 |        0.42 |     0.77 |

- IFBENCH
  | IFBENCH   | per-prompt |   count |   custom |   format | ratio |   repeat |   sentence |   words |
  |-----------|------------|---------|----------|----------|-------|----------|------------|---------|
  | loose     |       0.61 |    0.76 |     0.50 |     0.77 |  0.45 |     0.33 |       0.71 |    0.46 |
  | strict    |       0.57 |    0.76 |     0.40 |     0.70 |  0.45 |     0.33 |       0.64 |    0.44 |
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
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig


class Step3p5TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = Ref("...vllm_cfg.tokenizer_path")
    """Tokenizer directory for the target Step3.5 model family."""

    def build_tokenizer(self) -> ChatTokenizer:
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path, trust_remote_code=True)


class Step3p5SimpleEvalResourceConfig(ResourceConfig):
    vllm_replica: int = 1
    """Number of Step3.5 vLLM worker tasks to launch."""

    def __init__(self):
        super().__init__()
        self.command = ".venv/bin/python {COMMAND}"
        workspace_venv_bin = os.path.join(os.getcwd(), ".venv", "bin")
        current_path = os.environ.get("PATH", "")
        self.envs["PATH"] = f"{workspace_venv_bin}:{current_path}" if current_path else workspace_venv_bin
        self.replica = 1
        self.gpu = 8
        self.node_type = "gpu"
        self.vllm_replica = 1
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
                envs={"ROLE": "vllm"},
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


class Step3p5SimpleEvalVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/oss/checkpoints/step3_flash_sft_step3_data_muon/it4716/hf_vllm/"
        self.tokenizer_path = "/oss/tokenizers/Step3.5Flash-SFT-Tokenizer/"
        self.reasoning_parser = "step3p5"
        self.max_seq_len = 128 * 1024
        self.vllm_gpu_memory_utilization = 0.9

        self.vllm_tp = 8
        self.vllm_dp = 1

        self.vllm_enable_chunked_prefill = True
        self.vllm_enable_prefix_caching = True
        self.max_cache_size = 256


class Step3p5SimpleEvalVLLMRouterConfig(VLLMRouterConfig):
    routed_methods = {
        "completions": ["POST"],
        "chat/completions": ["POST"],
    }


class Step3p5SimpleBenchmarksEvalConfig(SimpleBenchmarksEvalConfig):
    tokenizer_cfg: Step3p5TokenizerConfig = Step3p5TokenizerConfig
    """Tokenizer config for Step3.5 prompt rendering and token counting."""

    num_concurrent_requests = 4096
    max_decode_steps = 128 * 1024


class Exp(BaseExp):
    vllm_cfg: VLLMDeployConfig = Step3p5SimpleEvalVLLMDeployConfig

    resource_cfg: Step3p5SimpleEvalResourceConfig = Step3p5SimpleEvalResourceConfig
    vllm_router_cfg: Step3p5SimpleEvalVLLMRouterConfig = Step3p5SimpleEvalVLLMRouterConfig
    eval_cfg: Step3p5SimpleBenchmarksEvalConfig = Step3p5SimpleBenchmarksEvalConfig

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

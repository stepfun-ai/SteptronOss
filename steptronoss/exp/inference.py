from __future__ import annotations

import torch
from configurize import Config, Ref


class BaseInferenceConfig(Config):
    max_seq_len: int = 65536
    """Length of total seqlen in inferengine (prefill + decode)"""
    max_decode_steps: int = None
    """max decode steps (gen tokens), if None, use max_seq_len - prompt_len"""
    max_cache_size = 64
    """Inference Decode BatchSize, max_num_seqs for paged attention"""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    eos_ids: list[int] = []
    """stop if tokens[-1] == eos_id. (work for generate only)"""
    stop_strings: str | list[str] = []
    """stop if trajectory.endswith(any_of(stop_string))"""

    shuffle_all_samples = True
    """shuffle all generated samples after finish generation."""


class VLLMDeployConfig(BaseInferenceConfig):
    """Config you need to deploy a vllm server."""

    router_addr_key: str = "default"
    """Key in exp redis that stores router addr & port."""

    model_config_path: str

    load_weight_from_config_path: bool = True

    tokenizer_path: str = Ref(".model_config_path")

    vllm_tp: int = 8
    vllm_pp: int = 1

    vllm_dp: int = 1
    """Attention data parallel size; useful for deepseek v3 series of model; default to 1"""
    enable_expert_parallel: bool = False
    """If True, will do expert parallel inference (for MoE models). EP=(TPxDP) by default for vllm."""

    vllm_gpu_memory_utilization: float = 0.95
    vllm_enable_chunked_prefill: bool = True  # Better enable chunked prefill
    """If set, the prefill requests can be chunked based on the
    max_num_batched_tokens."""

    vllm_max_num_batched_tokens: int = 8192

    vllm_enable_prefix_caching: bool = True  # Better enable prefix caching
    """Enable automatic prefix caching for vLLM."""

    vllm_enforce_eager: bool = False  # Better disable eager mode
    """Enforce eager mode for vLLM. This means it won't compile the cuda graph
    to fuse operator requests."""

    vllm_trust_remote_code: bool = True
    """Trust remote code for vLLM."""

    enable_auto_tool_choice: bool = False
    toolcall_parser: str = None  # "hermes"
    quantize: str = None  # "groupwise-quant"
    block_size: int = None
    enable_log_requests: bool = False

    enable_reasoning: bool = False
    reasoning_parser: str = None  # "deepseek_r1"

    model_name_template: str = "deployed-model-{EXP_ID}"
    """Template for model name; EXP_ID is substituted at runtime."""
    hot_path: str = None

    vllm_hf_overrides: dict = {}
    """The hf_overrides to use to override the model config in huggingface for vLLM."""

    vllm_sampling_params: dict = {}
    """The sampling params to use for vLLM."""

    vllm_mtp_num_tokens: int = 0
    vllm_mtp_method: str = ""
    """The method to use for vLLM MTP."""

    def sanity_check(self):
        """Validate configuration parameters.

        Validates that resource_cfg.gpu is evenly divisible by inference_tp.
        This ensures mp_run can create equal-sized processes.

        Examples:
        - ✓ gpu=8, inference_tp=2: 8 % 2 = 0 (4 processes)
        - ✓ gpu=8, inference_tp=4: 8 % 4 = 0 (2 processes)
        - ✓ gpu=8, inference_tp=8: 8 % 8 = 0 (1 process, standard deployment)
        - ✗ gpu=8, inference_tp=3: 8 % 3 ≠ 0 (invalid)
        """
        super().sanity_check()

        if self.enable_auto_tool_choice:
            assert self.toolcall_parser

    def run_as_worker(self):
        """Run a vllm controller that deploy a vllm server and register it to exp-router."""
        from steptronoss.generation.vllm.vllm_controller import VLLMController

        controller = VLLMController(cfg=self)
        controller.start()

    @property
    def model_name(self) -> str:
        from steptronoss.utils import get_exp_id

        return self.model_name_template.format(EXP_ID=get_exp_id())

    def build_cli(self):
        """Build a VLLM client that talks to the exp router."""
        from steptronoss.generation.vllm.vllm_client import VLLMClient

        return VLLMClient(cfg=self)

    def get_entrypoint_command_and_envs(self):

        import json

        envs = {}

        cmd = [
            "vllm serve",
            f"{self.model_config_path}",
            "--port $PORT_SERVING",
            f"--served-model-name {self.model_name}",
            f"--max-model-len {self.max_seq_len}",
            f"--tensor-parallel-size {self.vllm_tp}",
            f"--gpu-memory-utilization {self.vllm_gpu_memory_utilization}",
            f"--max-num-seqs {self.max_cache_size}",
            f"--data-parallel-size {self.vllm_dp}",
            f"--max-num-batched-tokens {self.vllm_max_num_batched_tokens}",
            "--disable-cascade-attn",  # might meet cuda IMA error when using cascade attention
            "--disable-uvicorn-access-log",
        ]
        if self.vllm_hf_overrides:
            overrides = json.dumps(self.vllm_hf_overrides)
            envs["HF_OVERRIDES"] = f"'{overrides}'"
            cmd.append("--hf-overrides $HF_OVERRIDES")
        if self.tokenizer_path:
            cmd.append(f"--tokenizer {self.tokenizer_path}")
        if self.vllm_trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.toolcall_parser:
            cmd.append("--enable-auto-tool-choice")
            cmd.append(f"--tool-call-parser {self.toolcall_parser}")
        if self.reasoning_parser:
            cmd.append(f"--reasoning-parser {self.reasoning_parser}")
        if not self.load_weight_from_config_path:
            cmd.append("--load-format dummy")
        if self.vllm_enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.vllm_enforce_eager:
            cmd.append("--enforce-eager")
        if self.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        # if self.enable_reasoning:
        #     cmd.append("--enable-reasoning")
        if not self.vllm_enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if self.quantize:
            cmd.append(f"--quantization {self.quantize}")
        if self.block_size:
            cmd.append(f"--block-size {self.block_size}")
        if self.enable_log_requests:
            cmd.append("--enable-log-requests")
        if self.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")
        if self.vllm_mtp_num_tokens > 0:
            vllm_speculative_config = json.dumps({
                "num_speculative_tokens": self.vllm_mtp_num_tokens,
                "method": self.vllm_mtp_method,
            })
            envs["SPECULATIVE_CONFIG"] = f"'{vllm_speculative_config}'"
            cmd.append("--speculative-config $SPECULATIVE_CONFIG")

        cmd = " ".join(cmd)

        return cmd, envs

    def get_sampling_params(self, override_params: dict = {}):

        # Handle top_k
        self.top_k = self.top_k
        if self.top_k <= 0:
            self.top_k = -1

        # Handle stop strings
        stop_strings = self.stop_strings
        if stop_strings is not None:
            if isinstance(stop_strings, str):
                stop_strings = [stop_strings]
            if isinstance(stop_strings, list) and len(stop_strings) == 0:
                stop_strings = None

        # Handle stop token ids
        eos_ids = self.eos_ids
        if len(eos_ids) == 0:
            eos_ids = None

        sampling_params = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=stop_strings,
            stop_token_ids=eos_ids,
        )
        sampling_params.update(self.vllm_sampling_params)
        sampling_params.update(override_params)
        return sampling_params

    def deploy_training_model(self, models: list[torch.nn.Module]):

        from steptronoss.checkpointing.hf_checkpoint import dump_safetensors
        from steptronoss.core.parallel_state import PM

        cli = self.build_cli()

        dump_safetensors(
            save_path=self.hot_path,
            model_reference_path=self.model_config_path,
            tokenizer_reference_path=self.tokenizer_path,
            models=models,
        )

        if PM.world_rank == 0:
            cli.wait_for_server()
            cli.reload_weights(self.hot_path)

        cli.wait_for_server()

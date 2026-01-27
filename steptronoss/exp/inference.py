from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional, TypedDict, Union

import torch
from configurize import Config, DataClass, Ref, writable_property

from steptronoss.exp.base_exp import Megatron3DParallelModelConfig, ParallelConfig
from steptronoss.tokenizer.hf_compat_tokenizer import HFCompatTokenizer

if TYPE_CHECKING:
    from vllm import RequestOutput, TokensPrompt

    from steptronoss.core.generators.base_generator import BaseGenerator


class SamplingParams(DataClass):
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[list[str]]
    stop_token_ids: Optional[list[int]]


class MetaDict(TypedDict, total=False):
    sampling_params: SamplingParams
    multi_modal_data: dict


class GenerationInput(DataClass):
    input_ids: torch.IntTensor
    """Prompt pass to Generator"""
    prompt_id: int = None
    """ID to identify which prompt this sample belong to."""
    meta: MetaDict = MetaDict()
    """Any other info to record, will be pass to GenerationOutput.meta"""


class StopType:
    MAX_LEN = 1
    MAX_DECODE = 2
    STOP_TOKEN = 3
    STOP_STRING = 4
    STOP_PARTIAL = 5


class GenerationOutput(DataClass):
    """St: Seqlen of trajectory
    Sr: Seqlen of response
    """

    trajectory: torch.LongTensor = None
    "Prompt+Generated, Long, (St, )"
    logprobs: torch.FloatTensor = None
    "Sample Logprob for Generated, Float32, (Sr, )"
    is_gen_mask: torch.BoolTensor = None
    "Mask for trajectory, Bool, (St, )"

    stop_type: int = None
    "How this rollout stoped, one of StopType"

    prompt_id: int = None

    meta: dict = None
    """Meta info from GenInput"""


class PartialRolloutConfig(Config):
    enabled = False

    prompt_level_filter = False
    """Filter partial rollouts by prompt level"""

    stop_by_decode_len = None
    """Stop a sequence by its decode length"""

    stop_by_bs = None
    """Stop current batch when active_cache_size <= stop_by_bs * peak_active_cache_size"""

    min_decode_len = None
    """When using stop_by_bs, a sequence will be stopped only if its decode length >= min_decode_len.
    This helps limiting the maximum staleness.
    """

    max_num_partial_turns = None
    """Maximum number of turns the decode can be split into. Must be set in conjunction with
    min_decode_len. For example, when max_decode_steps=24k, set max_num_partial_turns to 2
    and min_decode_len to 12k.
    """

    verbose = False
    """Verbose logging for partial rollout"""

    def sanity_check(self):
        super().sanity_check()
        if not self.enabled:
            return
        if self.stop_by_bs is None and self.stop_by_decode_len is None:
            assert False, "No partial rollout stop condition provided"

    def record_metrics(self, metrics, full_rollouts, partial_rollouts):
        if self.enabled == False:
            return
        metrics.partial_remaining_samples.add(len(partial_rollouts))

        for genout in full_rollouts:
            partial_rollout_attr = genout.meta.get("partial_rollout_attr", {})
            metrics.partial_turns_mean.add(partial_rollout_attr.get("partial_rollout_times", 0))
            metrics.partial_turns_max.add(partial_rollout_attr.get("partial_rollout_times", 0))
            partial_rollout_tokens = partial_rollout_attr.get("partial_rollout_tokens", [])
            if len(partial_rollout_tokens) < self.max_num_partial_turns - 1:
                partial_rollout_tokens = [0] * (
                    self.max_num_partial_turns - 1 - len(partial_rollout_tokens)
                ) + partial_rollout_tokens
            for turn, token_len in enumerate(partial_rollout_tokens):
                metrics.partial_rollout_mean_tokens_in_turn.add(
                    token_len, subname=f"{self.max_num_partial_turns-1-turn}"
                )


class BaseInferenceConfig(Config):
    parallel_cfg = InferParallelConfig
    partial_rollout_cfg = PartialRolloutConfig

    max_seq_len: int = 65536
    """Length of total seqlen in inferengine (prefill + decode)"""
    max_decode_steps: int = None
    """max decode steps (gen tokens), if None, use max_seq_len - prompt_len"""
    max_cache_size = 64
    """Inference Decode BatchSize, max_num_seqs for paged attention"""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    eos_id: int | list[int] = []
    """stop if tokens[-1] == eos_id. (work for generate only)"""
    stop_strings: str | list[str] = []
    """stop if trajectory.endswith(any_of(stop_string))"""

    @writable_property
    def eos_ids(self) -> list[int]:
        if self.eos_id is None:
            return []
        elif isinstance(self.eos_id, int):
            return [self.eos_id]
        return self.eos_id

    shuffle_all_samples = True
    """shuffle all generated samples after finish generation."""

    partial_rollout_cfg = PartialRolloutConfig
    enable_partial_rollout = False


class VLLMInferenceConfig(BaseInferenceConfig):
    model_config_path: str

    build_tokenizer: Callable[[], HFCompatTokenizer] = Ref("...tokenizer_cfg.build_tokenizer")

    @writable_property
    def tokenizer_path(self):
        if isinstance(self.build_tokenizer, Callable):
            tokenizer = self.build_tokenizer()
            if isinstance(tokenizer, HFCompatTokenizer):
                return tokenizer.hf_path
            elif hasattr(tokenizer, "name_or_path"):
                return tokenizer.name_or_path

        return self.model_config_path

    vllm_gpu_memory_utilization: float = 0.8  # Tune this to avoid OOM
    """The fraction of GPU memory to use for vLLM."""

    vllm_dtype: str = "bfloat16"  # Better use this
    """The data type to use for vLLM."""

    vllm_trust_remote_code: bool = True  # If not download, can be set to False
    """Trust remote code from huggingface."""

    vllm_enable_chunked_prefill: bool = True  # Better enable chunked prefill
    """If set, the prefill requests can be chunked based on the
    max_num_batched_tokens."""

    vllm_enable_prefix_caching: bool = True  # Better enable prefix caching
    """Enable automatic prefix caching for vLLM."""

    vllm_mtp_num_tokens: int = 0
    """Enable MTP for vllm(aka draft model), trigger if > 0."""
    vllm_mtp_method: Literal["step4_mtp"] = "step4_mtp"

    vllm_enforce_eager: bool = False  # Better disable eager mode
    """Enforce eager mode for vLLM. This means it won't compile the cuda graph
    to fuse operator requests."""

    vllm_engine_init_seed: int = 42
    """The seed to use for initialize the vLLM engine."""

    vllm_hf_overrides: dict = {}
    """The hf_overrides to use to override the model config in huggingface for vLLM."""

    enable_train_infer_consistency_check: bool = True
    """If True, check the consistency between train and infer config."""

    train_infer_consistency_check_items: list[str] = ["yarn"]
    """The items to check the consistency between train and infer config."""

    vllm_sampling_params: dict = {}
    """The sampling params to use for vLLM."""

    vllm_compilation_config: dict = {}
    """The compilation config to use for vLLM."""

    vllm_perf_log_interval: int = 10
    """The interval (seconds) to log the performance of vLLM. Set to 0 to disable logging."""

    def load_weight_for_vllm_model(self, model: torch.nn.Module, state_dict: dict):
        """Customizable function to indicate how to load weight for vllm model."""
        return model.load_state_dict(state_dict, strict=True)

    def adapt_vllm_input(self, gen_input: GenerationInput) -> "TokensPrompt":
        from vllm import TokensPrompt

        input_ids = gen_input.input_ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return TokensPrompt(prompt_token_ids=input_ids)

    def adapt_vllm_output(self, vllm_output: RequestOutput, meta: dict) -> list[GenerationOutput]:
        """vLLM nest multi output in vout.outputs, we return list here"""
        outputs = []
        for single_output in vllm_output.outputs:
            trajectory = vllm_output.prompt_token_ids + single_output.token_ids
            is_gen_mask = [0] * len(vllm_output.prompt_token_ids) + [1] * len(single_output.token_ids)

            if single_output.finish_reason == "length":
                stop_type = StopType.MAX_LEN
            elif single_output.finish_reason == "partial":
                stop_type = StopType.STOP_PARTIAL
            else:
                last_token_id = trajectory[-1]
                if last_token_id in self.eos_ids:
                    stop_type = StopType.STOP_TOKEN
                else:
                    stop_type = StopType.STOP_STRING

            # Organize the logprobs
            logprobs = []
            for token_logprob in single_output.logprobs:
                logprobs.append(next(iter(token_logprob.values())).logprob)
            if "vllm_output" in meta:
                meta["vllm_output"] = single_output

            gen_output = GenerationOutput(
                prompt_id=meta.pop("prompt_id"),
                trajectory=torch.tensor(trajectory, dtype=torch.long),
                logprobs=torch.tensor(logprobs, dtype=torch.float32),
                is_gen_mask=torch.tensor(is_gen_mask, dtype=torch.bool),
                stop_type=stop_type,
                meta=meta,
            )
            outputs.append(gen_output)
        return outputs


class RolloutManagerConfig(Config):
    sample_per_prompt: int = Ref("..sample_per_prompt")

    balance_prompts_to_all_ranks: bool = True
    """If set, balance prompts to all ranks (N_gpu), otherwise keep as it is.
    - rank0 -> run([0, 1, 2, 3]) -balance-> [0, 1, 2]
    - rank1 -> run([4, 5])       -balance-> [3, 4, 5]
    """

    auto_balance_infer_dp: Union[bool, float] = 1.0
    """Automatically balance prompts when sending to inference DP. Each prompt
    will be sent to inference DP that receives the least number of prompts.
    - Set to True to force balance for all prompts.
    - Set to a float value (0.0~1.0) to randomly balance a fraction of
      prompts. Rest prompts will be sent to default queue that is shared
      by all inference DPs, just like the default behavior.
    - Set to 0 or False to disable auto balance.
    """

    build_tokenizer = Ref("...tokenizer_cfg.build_tokenizer", None)

    def build_rollout_manager(self):
        from steptron.core.rollout_managers import AsyncRolloutManager

        return AsyncRolloutManager(cfg=self)


class InferencableModelConfig(Megatron3DParallelModelConfig):
    inference_config = VLLMInferenceConfig
    """Config for InferEngine, Sub[CacheCfg, OptimusModelCfg]"""

    def build_generator(self) -> BaseGenerator:
        from steptron.core.generators.vllm_generator import VLLMGenerator

        return VLLMGenerator(self.inference_config)

    def get_infer_weight(self, models: list) -> dict[str, torch.Tensor]:
        from steptron.utils import unwrap_model

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        infer_weight = {}
        for model in models:
            infer_weight.update(unwrap_model(model).export_infer_weight(self.inference_config))
        return infer_weight

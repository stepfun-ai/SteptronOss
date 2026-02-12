from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any, ForwardRef, Literal, Optional, TypedDict

import torch
from configurize import Config, DataClass, Ref

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import (
    BaseExp,
    DataConfig,
    GradientManagerConfig,
    Megatron3DParallelModelConfig,
    MetricConfig,
    ProfilerConfig,
    TrainerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig, LoadOptions, SaveOptions
from steptronoss.exp.lr_schedulers import SchedulerConfig
from steptronoss.generation.vllm.vllm_controller import VLLMDeployConfig
from steptronoss.utils.metrics import (
    AvgMetric,
    GlobalMetrics,
    HistogramMetric,
    Metric,
    PercentageMetric,
    TextMetric,
)

TrajectoryHook = Callable[[ForwardRef("GenerationOutput")], ForwardRef("GenerationOutput")]


class StopType:
    MAX_LEN = 1
    MAX_DECODE = 2
    STOP_TOKEN = 3
    STOP_STRING = 4
    STOP_PARTIAL = 5


class EnvTrajectory(DataClass):
    """
    Trajectory from environment, with a reward from env, so it can be trained by PPO.

    St: Seqlen of trajectory
    Sr: Seqlen of response
    """

    trajectory: torch.LongTensor | list[int] = None
    "Prompt+Generated, Long, (St, )"

    logprobs: torch.FloatTensor | list[float] | None = None
    "Sample Logprob for Generated, Float32, (Sr, )"

    is_gen_mask: torch.BoolTensor | list[int | bool] = None
    "Mask for trajectory, Bool, (St, )"

    meta: dict = None
    """Meta info from GenInput"""

    stop_type: int

    raw_reward: float = None

    @property
    def can_be_trained(self):

        trainable = self.raw_reward is not None and self.trajectory is not None and self.is_gen_mask is not None
        if self.logprobs is not None:
            trainable = trainable and sum(self.is_gen_mask) == len(self.logprobs)
        return trainable


class PPOSample(EnvTrajectory):
    ground_truth = {}

    prompt_id: str

    ref_logprobs: torch.Tensor = None
    actor_logprobs: torch.Tensor = None

    advantages: torch.Tensor = None
    returns: torch.Tensor = None
    values: torch.Tensor = None

    # for model forward
    input_ids: torch.LongTensor = None
    labels: torch.LongTensor = None


class PackedPPOSamples(DataClass):
    """St: Seqlen of trajectory
    Sr: Seqlen of response
    """

    input_ids: torch.Tensor = None
    """[1, 2, 3, 1, 2, 3, 4]"""
    labels: torch.Tensor = None
    """[2, 3, 0, 2, 3, 4, 0]"""

    cu_seqlens: torch.Tensor = None
    """[0, 3, 8], with last padding"""
    cu_valid_sizes: torch.Tensor = None
    """[0, 3, 7]"""

    max_seq_len: torch.Tensor = None
    """cu_seqlens.diff().max()"""

    trajectory: torch.Tensor = None
    """same as input_ids"""

    is_gen_mask: torch.Tensor = None
    """shape [St, ], ~ arange(St) >= len(prompt)"""

    loss_mask: torch.Tensor = None  # TODO: St or Sr?
    """shape similar to logprobs, used for loss calculation"""

    advantages: torch.Tensor = None
    returns: torch.Tensor = None
    values: torch.Tensor = None

    logprobs: torch.Tensor = None  # Sr
    ref_logprobs: torch.Tensor = None  # Sr
    actor_logprobs: torch.Tensor = None  # Sr

    samples: list[PPOSample] = None

    @classmethod
    def from_samples(cls, samples: list[PPOSample]) -> PackedPPOSamples:
        from steptronoss.core.parallel_state import PM
        from steptronoss.utils.general import pad_tensor, shift_left

        packed_data = cls()
        labels, trajectories, label_is_gen_masks, logprobs = [], [], [], []
        for sample in samples:
            idxes = sample.trajectory
            # input_ids.append(idxes)
            labels.append(shift_left(idxes))
            trajectories.append(idxes)
            label_is_gen_masks.append(shift_left(sample.is_gen_mask))
            logprobs.append(sample.logprobs)

        TP = PM.size_of("TP")
        CP = PM.size_of("CP")

        packed_data.labels = torch.cat(labels, 0)

        seqlens = torch.tensor(
            [len(s.trajectory) for s in samples],
            device=labels[0].device,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat([labels[0].new_zeros((1,)), torch.cumsum(seqlens, 0)], 0).to(torch.int32)

        num_pad = cu_seqlens[-1] % (TP * CP * 2)
        if num_pad != 0:
            num_pad = TP * CP * 2 - num_pad

        packed_data.cu_valid_sizes = cu_seqlens.clone()

        cu_seqlens[-1] += num_pad
        packed_data.cu_seqlens = cu_seqlens
        packed_data.max_seq_len = packed_data.cu_seqlens.diff().max()

        def pad(x, dim=0):
            return pad_tensor(x, dim=dim, to=x.shape[dim] + num_pad, value=0)

        samples[-1].is_gen_mask = pad(samples[-1].is_gen_mask)
        samples[-1].trajectory = pad(samples[-1].trajectory)

        packed_data.trajectory = pad(torch.cat(trajectories)[None], dim=1)
        packed_data.input_ids = packed_data.trajectory
        packed_data.labels = pad(packed_data.labels[None], dim=1)
        packed_data.is_gen_mask = pad(torch.cat(label_is_gen_masks)[None], dim=1)

        packed_data.logprobs = torch.cat(logprobs)
        packed_data.samples = samples

        return packed_data


# Configs


class RoleCheckpointConfig(Config):
    load_path: str = None
    load_safetensors: bool | str | None = None
    strict_load_model: bool | None
    load_option = LoadOptions
    save_option = SaveOptions


class PPOCheckpointCfg(CheckpointConfig):
    """Multi-role support"""

    actor = RoleCheckpointConfig
    critic = RoleCheckpointConfig
    reference = RoleCheckpointConfig


class PPOMetricConfig(MetricConfig):
    """PPO Metrics! You can start your RL journey from reading the metrics below"""

    """Performance Related Metrics"""
    # reward is our objective!
    reward_mean = Metric().mean("time").mean("dp")
    """Mean of reward scores among ALL samples."""
    reward_max = Metric().max("time").max("dp")
    """Max of reward scores among ALL samples."""
    reward_min = Metric().min("time").min("dp")
    """Min of reward scores among ALL samples."""

    correctness = PercentageMetric().sum("world")
    """Percentage of all_correct/all_fail/some_correct prompts."""

    correctness_subclass = PercentageMetric().sum("world")
    """Percentage of all_correct/all_fail/some_correct prompts. Categorized by prompt tag."""

    reward_func_mean = Metric(is_group=True).mean("time").mean("world")
    """Mean of reward scores among ALL samples."""
    reward_func_max = Metric(is_group=True).mean("time").max("world")
    """Max of reward scores among ALL samples."""
    reward_func_min = Metric(is_group=True).mean("time").min("world")
    """Min of reward scores among ALL samples."""

    reward_filename_mean = Metric(is_group=True).mean("time").mean("world")
    """Mean of reward scores grouped by filename."""
    reward_filename_max = Metric(is_group=True).mean("time").max("world")
    """Max of reward scores grouped by filename."""
    reward_filename_min = Metric(is_group=True).mean("time").min("world")
    """Min of reward scores grouped by filename."""

    rollout_avg_tokens = Metric().mean("time")
    """Avg response length, important for reasoning-oriented RL."""

    rollout_avg_tokens_grouped = Metric(is_group=True).mean("time").mean("world")
    """Avg response length grouped by reward function type and filename."""

    rollout_avg_solution_tokens = Metric().mean("time")
    """Average solution token count per sample (after </think>)."""

    rollout_avg_think_tokens = Metric().mean("time")
    """Average think token count per sample (before </think>)."""

    gen_rollout_latency_s = Metric(is_group=True).mean("time").max("world")
    """Rollout-local E2E latency (s) per dp group during generation (recorded at rollout end)."""
    gen_rollout_latency_s_mean = Metric(is_group=True).mean("time").max("world").mean("dp")
    """Rollout-local E2E latency (s) per dp group during generation, averaged across dp (recorded at rollout end)."""
    gen_rollout_throughput_tps = Metric(is_group=True).mean("time").max("world")
    """Rollout throughput (decode tokens/s) per dp group during generation (recorded at rollout end)."""
    gen_rollout_throughput_tps_mean = Metric(is_group=True).mean("time").max("world").mean("dp")
    """Rollout throughput (decode tokens/s) per dp group during generation, averaged across dp (recorded at rollout end)."""
    gen_partial_rollout_count = Metric(is_group=True).mean("time").sum("world")
    """Partial rollout count per dp group during generation (recorded at rollout end)."""
    gen_partial_rollout_count_mean = Metric(is_group=True).mean("time").sum("world").mean("dp")
    """Partial rollout count per dp group during generation, averaged across dp (recorded at rollout end)."""
    gen_requests_total = Metric(is_group=True).mean("time").sum("world")
    """Total requests per dp group during generation (recorded at rollout end)."""
    gen_requests_total_mean = Metric(is_group=True).mean("time").sum("world").mean("dp")
    """Total requests per dp group during generation, averaged across dp (recorded at rollout end)."""

    truncated_rate = Metric().mean("time")
    """too high truncated rate generally leads to poor performance.
    Please review the LLM generations, as it seems to contain significant repetitions."""

    truncated_rate_grouped = Metric(is_group=True).mean("time").mean("world")
    """Truncated rate grouped by reward function type and filename."""

    rollout_logprob = Metric().mean("time")
    """Mean logprobs of responses, if high abs(logprob) should be pay attention to"""
    rollout_logprob_correct = Metric().mean("time")
    """Mean logprobs of correct response."""
    rollout_logprob_incorrect = Metric().mean("time")
    """Mean logprobs of incorrect response."""

    rollout_logprob_grouped = Metric(is_group=True).mean("time").mean("world")
    """Mean logprobs of responses grouped by reward function type and filename."""

    repeatness_score = AvgMetric().mean("world")
    """too high repeatness score generally means your training doomed"""

    repeatness_rate = AvgMetric().mean("world")
    """repeatness_score>0.2 count as repeatness_rate"""

    repeatness_rate_correct = AvgMetric().mean("world")
    """Repeatness rate of correct responses."""
    repeatness_rate_incorrect = AvgMetric().mean("world")
    """Repeatness rate of incorrect responses."""

    actor_grad_norm = Metric().mean("time")
    """> 1.0 grad norm generally means something wrong"""
    critic_grad_norm = Metric().mean("time")
    """> 1.0 grad norm generally means something wrong"""

    kl_with_ref_dist = Metric().mean("time").mean("dp")
    """kl with ref; not so important though"""

    # Off-Policy Related Metrics
    ppo_clip_count = Metric().mean("time").mean("dp")
    """this is actually clip rate; higher then 0.1 percent then you should pay attention to"""

    ppo_lower_bound_clip_count = Metric().mean("time").mean("dp")
    ppo_upper_bound_clip_count = Metric().mean("time").mean("dp")

    important_sampling_ratio = PercentageMetric().sum("world").disable()
    """importance sampling ratio metric group; search globally to see details! (disabled by default)"""

    ratio_diff = Metric().mean("time").mean("dp")
    """pi/pi_old - 1; if too high is too off-policy"""

    ratio_diff_quantile = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for pi/pi_old - 1; if too high is too off-policy.
    Sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    tis_ratio_mean = Metric().mean("time").mean("dp")
    """pi_ie/pi_old ratio mean."""

    tis_clip_fraction = Metric().mean("time").mean("dp")
    """the truncated rate of pi_ie/pi_old."""

    tis_ratio_quantile = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for exp(logp_ie - logp_old)
    Sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    sampling_ratio_diff = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for exp(logp - logp_old) - 1 (probability ratio diff); if too high is too off-policy.
    Sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    sampling_logprob_diff_quantile = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for logp - logp_old(sample) diff; if too high is too off-policy.
    For on-policy setting, this mean InferenceEngine noise. Sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    # New and Old Logprob Quantile Metrics
    new_logprob = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for new logprobs (actor_logprobs) with sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    old_logprob = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for old logprobs (sampling logprobs) with sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    # New and Old Prob Quantile Metrics
    new_prob = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for new probabilities exp(actor_logprobs) with sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    old_prob = Metric(is_group=True).mean("time").mean("dp")
    """Quantile metrics for old probabilities exp(sampling logprobs) with sub-tags: min, max, q01, q05, q10, q90, q95, q99."""

    sampling_logprob_diff = Metric().mean("time").mean("dp")
    """logp - logp_old(sample); if too high is too off-policy.
    For on-policy setting, this mean InferenceEngine noise."""

    sampling_logprob_diff_max = Metric().max("time").max("dp")
    """Max of sampling_logprob_diff."""

    ppo_value_clip_count = Metric().mean("time").mean("dp")
    """Clip value count; not so important"""

    kl_with_sample_dist = Metric().mean("time").mean("dp")
    """kl of (current policy, ie policy); if too high, means InferenceEngine is too noisy."""

    reverse_kl_with_sample_dist = Metric().mean("time").mean("dp")
    """reverse kl of (current policy, ie policy); if too high, means InferenceEngine is too noisy."""

    # General Training Metrics
    critic_loss = Metric().mean("time").mean("dp")
    """If you have critic, lower critic loss is better"""

    ppo_returns = Metric().mean("time").mean("dp")
    """For gae lambda=gamma=1, return should be close to reward"""

    ppo_values = Metric().mean("time").mean("dp")
    """PPO value is the value estimation from critic. If critic learns well,
      for gae lambda=gamma=1, value should be close to return"""

    advantage_mean = Metric().mean("time").mean("dp")
    """Adv_mean should be close to 0 when training gets stable."""
    advantage_max = Metric().max("time").max("dp")
    advantage_min = Metric().min("time").min("dp")
    advantage_std = Metric().mean("time").mean("dp")

    advantage_abs_mean = Metric().mean("time").mean("dp")
    """Mean of abs(adventage)."""

    sample_filter_rate = Metric().mean("time")
    """Ratio of filtered samples in filtering"""

    # Misc Metrics
    ppo_loss = Metric().mean("time").mean("dp")
    """ppo_loss abs value is generally meaningless"""

    rollout_tokens_hist = HistogramMetric().disable()
    """Histogram of response length; helpful for throughput and performance analysis"""

    inner_iteration = Metric().mean("time")
    """Iteration of Critic & Actor is dynamic of fix_iters not specified,
    this is the actual iteration."""

    rollout_samples = Metric().mean("time")
    """Actual rollouts, if dynamic sampling is enabled, you should pay attention to this"""

    unpacked_samples = Metric().mean("time")
    """Number of Samples(trajectory) used for training"""
    packed_samples = Metric().mean("time")
    """Number of PackedSamples used for training"""

    partial_remaining_samples = Metric().mean("time")
    """Number of remaining samples in partial rollout buffer."""
    partial_turns_mean = Metric().mean("time")
    """Mean of turns in partial rollout buffer."""
    partial_turns_max = Metric().max("time")
    """Max of turns in partial rollout buffer."""

    partial_rollout_mean_tokens_in_turn = Metric(is_group=True).mean("time")

    def __init__(self):
        super().__init__()
        # text metric! Very helpful for qualitative analysis
        self.training_generation_text = TextMetric(20).gather("world")
        self.eval_generation_text = TextMetric(10).gather("world")


GlobalMetrics: PPOMetricConfig


class ActorModelConfig(Megatron3DParallelModelConfig):
    """The Actor Model should be able to transfer to "inference mode", and contains
    a inference config that can be used for InferenceEngineGenerator

    """

    vllm_cfg: VLLMDeployConfig


class CriticModelConfig(Megatron3DParallelModelConfig):
    """Same like a plain model, with a little difference on the output embedding. (C, Vocab) -> (C, 1)"""

    def build_model(self):
        """Build a CriticModel"""
        return super().build_model()


class RewardModelConfig(Megatron3DParallelModelConfig):
    """Same like critic model"""

    def build_model(self):
        """Build a RewardModel"""
        return super().build_model()


class FlowControllerConfig(Config):
    vllm_cfg: VLLMDeployConfig = Ref("...actor_model_cfg.vllm_cfg")

    save_path: str = Ref("...checkpoint_cfg.save_path")
    """Checkpoint directory for persisting flow state."""

    async_strategy: Literal["on-policy", "one-step-off", "fully-async"]
    """Rollout scheduling strategy."""

    genable_allow_errors: bool = False
    """If set, ignore generate_for_train errors."""

    prompt_per_iter: int
    """Number of prompts scheduled per training iteration."""

    max_untrained_prompts: int | None = None
    """Max pending prompts for fully-async flow control."""

    max_staleness: int | None = None
    """Max staleness steps for fully-async flow control."""

    def sanity_check(self):
        super().sanity_check()
        assert self.vllm_cfg.hot_path is not None
        if self.async_strategy in ["fully-async"]:
            assert self.max_untrained_prompts is not None
            assert self.max_staleness is not None

    def build_flow_controller(self):
        from steptronoss.core.generators.flow_controller import SimpleFlowController

        return SimpleFlowController(flow_cfg=self)


class PPOLikeTrainerConfig(TrainerConfig):
    flow_cfg: FlowControllerConfig = FlowControllerConfig
    """Flow controller config for rollout collection."""

    def get_trainer_cls(self) -> type:
        from steptronoss.core.trainers.ppo_trainer import PPOTrainer

        return PPOTrainer

    vocab_size = Ref("..tokenizer_cfg.padded_vocab_size")
    build_tokenizer = Ref("..tokenizer_cfg.build_tokenizer")

    global_seq_length: int

    critic_epoch = 1
    actor_epoch = 1

    eval_interval = None
    """Run GenEval for each {eval_interval} iteration (trainer_iteration)"""

    ppo_actor_lambda = 0.95
    ppo_critic_lambda = 0.99
    ppo_gae_gamma = 1.0

    advantage_clip_min = -5
    advantage_clip_max = 4.5

    critic_value_clip = 0.2
    ppo_clip = 0.2

    reward_clip = 20

    sample_kl_loss_coeff = 0.0
    ref_kl_loss_coeff = 0.05
    """backward-able kl"""
    ref_kl_penalty_coeff = 0.05
    """policy gradient kl"""

    dual_clip_eps = 1.5

    critic_warmup_iters = 100
    """inner steps skiped for actor."""

    record_rollout = False

    fix_iters: int = None
    """if set, will NOT use iters = ragged_samples / global_batch_size,
    USE batch_size = ragged_samples / fix_iters instead."""

    fix_iters_critic: int = Ref(".fix_iters")
    """same as fix_iters , but for critic model."""

    policy_actor_temperature = Ref("..actor_model_cfg.inference_config.temperature", 1.0)

    onvalue_gae = False
    """If True, compute GAE iteratively on UPDATE value model, rather than once.
    Note that this flag require actor_data being subset of critic_data.
    """

    offload_data: bool = False
    """if set, offload data to CPU for memory saving."""

    offload_optimizer_state: bool = False
    """Offload optimizer when train model, one of [False, True]"""

    skip_first_eval: bool = False
    """If True, skip eval at the first iteration"""

    skip_forward_reference: bool = False
    """If True, skip forward_ref, require both ref_kl_loss_coeff and ref_kl_penalty_coeff are 0."""

    skip_forward_actor: bool = False
    """If True, skip forward_actor, require fully on-policy."""

    use_offpolicy_without_is: bool = False

    dump_sample_keys: list[str] = [
        "trajectory",
        "logprobs",
        "is_gen_mask",
        "ground_truth",
        "raw_reward",
    ]

    def sanity_check(self):
        super().sanity_check()
        if self.skip_forward_reference:
            assert self.ref_kl_loss_coeff == 0 and self.ref_kl_penalty_coeff == 0, (
                "Cannot skip get_ref_logprob when need KL regulation."
            )
        if self.skip_forward_actor:
            assert (self.fix_iters == 1 and self.actor_epoch == 1) or self.use_offpolicy_without_is, (
                "Cannot skip get_actor_logprob when fix_iters != 1 or actor_epoch != 1 (off-policy setting) except use_offpolicy_without_is."
            )

        assert self.offload_optimizer_state in [True, False, "momentum"]
        all_supported_keys = PPOSample()._defined_attributes
        if self.dump_sample_keys:
            dump_attrs = set(self.dump_sample_keys)
            if not dump_attrs.issubset(all_supported_keys):
                raise KeyError(
                    f"Unsupported Key: {dump_attrs - all_supported_keys}!! supported keys are: {all_supported_keys}"
                )
        # for wandb
        if "wandb" in self.writer_backend:
            assert os.getenv("WANDB_API_KEY") is not None, (
                "writer_backend including wandb, but WANDB_API_KEY is not set"
            )

    def is_data_source(self):
        # for PPO fw/fwbw, each TP hold same data
        return PM.i_am("PP", 0) or PM.i_am("PP", -1)

    def sync_get_data(self, data_iterator):
        from copy import deepcopy

        from steptronoss.core.parallel_state import get_vpp_rank, get_vpp_size
        from steptronoss.timers import timeit
        from steptronoss.utils.dist_utils import broadcast_tensors

        # This function Get/Broadcast/Preprocess and return data ready-to-use
        if PM.i_am("PP", 0) or PM.i_am("PP", -1):
            data = next(data_iterator)
        else:
            data = dict()

        if self.global_data_keys:
            vpp_rank = get_vpp_rank()
            vpp_size = get_vpp_size()
            with timeit("broadcast-tensors-pp", log_level=2):
                if vpp_rank == 0:
                    pp_sync_data = [data.__class__] + [data.get(k) for k in self.global_data_keys]
                    pp_sync_data = broadcast_tensors(
                        pp_sync_data,
                        src_rank=PM.ranks_of("PP")[0],
                        group=PM.group_of("PP"),
                    )
                    # cache pp_sync_data for the following model chunks
                    if not hasattr(self, "_cached_pp_sync_data"):
                        self._cached_pp_sync_data = [[] for _ in range(vpp_size - 1)]
                    for _vp in range(vpp_size - 1):
                        self._cached_pp_sync_data[_vp].append(deepcopy(pp_sync_data))
                else:
                    pp_sync_data = self._cached_pp_sync_data[vpp_rank - 1].pop(0)
        if not data:
            data_class = pp_sync_data.pop(0)
            data_items = dict(zip(self.global_data_keys, pp_sync_data))
            data = data_class(**data_items)
        return data

    def preprocess_generated(self, data: PackedPPOSamples) -> PackedPPOSamples:
        """move to cuda"""
        from steptronoss.utils import recur_to

        if data:
            data = recur_to(data, "cuda")

        return data

    def get_advantage_and_returns(
        self, ragged_samples: PackedPPOSamples, ragged_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def actor_loss_func(self, data: PackedPPOSamples, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def critic_loss_func(self, data: PackedPPOSamples, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def batch_reward_function(self, samples: list[PPOSample]) -> list[float]:
        raise NotImplementedError

    def record(self, response: str, response_token: list[int], reward: float, sample: PPOSample):
        from steptronoss.text_processing.oone_math_utils import repeatness_score

        repeatness = repeatness_score(response)
        GlobalMetrics.repeatness_score.add(repeatness)
        GlobalMetrics.repeatness_rate.add(float(repeatness > 0.2))

        if reward >= 0.5:
            GlobalMetrics.repeatness_rate_correct.add(float(repeatness > 0.2))
            GlobalMetrics.rollout_logprob_correct.add(sample.logprobs, iop=torch.mean)
        else:
            GlobalMetrics.repeatness_rate_incorrect.add(float(repeatness > 0.2))
            GlobalMetrics.rollout_logprob_incorrect.add(sample.logprobs, iop=torch.mean)

    def filter_samples(self, samples: list[PPOSample]) -> list[PPOSample]:
        GlobalMetrics.sample_filter_rate.add(0.0 / len(samples))
        return samples

    def log_solution_think_tokens(self, all_samples: list[PPOSample], tokenizer):
        """Calculate and log solution/think tokens split by </think> marker.

        Optimized: search for </think> token directly in token list using Torch ops.
        No decode/encode needed! Assumes </think> is a single token.
        """
        from steptronoss.text_processing.content_parsing import THINK_END_TOKEN

        # Get the </think> token ID (assumes it's a single token)
        marker_token_id = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)[0]

        total_solution_tokens = 0
        total_think_tokens = 0

        for sample in all_samples:
            # Get generated tokens as tensor (already on device)
            gen_tokens_tensor = sample.trajectory[sample.is_gen_mask]
            total_gen_tokens = len(gen_tokens_tensor)

            # Find first occurrence of marker token
            matches = (gen_tokens_tensor == marker_token_id).nonzero(as_tuple=True)[0]

            if len(matches) > 0:
                # Found </think> marker, split at first occurrence
                think_end_idx = matches[0].item() + 1
                think_tokens = think_end_idx
                solution_tokens = total_gen_tokens - think_end_idx
            else:
                # No </think> found, all tokens are solution
                think_tokens = 0
                solution_tokens = total_gen_tokens

            total_solution_tokens += solution_tokens
            total_think_tokens += think_tokens

        # Log averages
        if len(all_samples) > 0:
            avg_solution = float(total_solution_tokens) / len(all_samples)
            avg_think = float(total_think_tokens) / len(all_samples)
            GlobalMetrics.rollout_avg_solution_tokens.add(avg_solution)
            GlobalMetrics.rollout_avg_think_tokens.add(avg_think)

    def log_generation_metrics(self, all_samples: list[PPOSample]):
        # for generation stage
        all_tokens = float(sum(x.is_gen_mask.sum() for x in all_samples))
        if GlobalMetrics.rollout_tokens_hist.enabled:
            for sample in all_samples:
                GlobalMetrics.rollout_tokens_hist.add(sample.is_gen_mask, iop=lambda x: float(x.sum()))
        per_sample_tokens = all_tokens / len(all_samples)
        GlobalMetrics.rollout_samples.add(float(len(all_samples)))
        GlobalMetrics.rollout_avg_tokens.add(per_sample_tokens)

        # Build tokenizer for metrics logging (including solution/think token splits)
        tokenizer = self.build_tokenizer()
        self.log_solution_think_tokens(all_samples, tokenizer)

        for sample in all_samples:
            filename = sample.ground_truth.get("file_path", "unknown").split("/")[-1].split(".")[0]
            sample_tokens = float(sample.is_gen_mask.sum())
            GlobalMetrics.rollout_avg_tokens_grouped.add(
                sample_tokens,
                subname=f"{sample.ground_truth.get('reward_fn_type', 'default')}",
            )
            GlobalMetrics.rollout_avg_tokens_grouped.add(
                sample_tokens,
                subname=f"{filename}",
            )
            GlobalMetrics.rollout_logprob_grouped.add(
                sample.logprobs,
                subname=f"{sample.ground_truth.get('reward_fn_type', 'default')}",
                iop=torch.mean,
            )
            GlobalMetrics.rollout_logprob_grouped.add(
                sample.logprobs,
                subname=f"{filename}",
                iop=torch.mean,
            )
            GlobalMetrics.rollout_logprob.add(sample.logprobs, iop=torch.mean)

    def log_reward_metrics(self, all_samples: list[PPOSample]):
        prompt_rewards_cur = defaultdict(list)
        prompt_env_cur = dict()
        for sample in all_samples:
            # log reward
            GlobalMetrics.reward_mean.add(sample.raw_reward)
            GlobalMetrics.reward_min.add(sample.raw_reward)
            GlobalMetrics.reward_max.add(sample.raw_reward)

            # log reward each function
            GlobalMetrics.reward_func_mean.add(
                sample.raw_reward,
                subname=f"{sample.ground_truth.get('reward_fn_type', 'default')}",
            )
            GlobalMetrics.reward_func_max.add(
                sample.raw_reward,
                subname=f"{sample.ground_truth.get('reward_fn_type', 'default')}",
            )
            GlobalMetrics.reward_func_min.add(
                sample.raw_reward,
                subname=f"{sample.ground_truth.get('reward_fn_type', 'default')}",
            )

            # log reward by filename
            filename = sample.ground_truth.get("file_path", "unknown").split("/")[-1].split(".")[0]
            GlobalMetrics.reward_filename_mean.add(
                sample.raw_reward,
                subname=f"{filename}",
            )
            GlobalMetrics.reward_filename_max.add(
                sample.raw_reward,
                subname=f"{filename}",
            )
            GlobalMetrics.reward_filename_min.add(
                sample.raw_reward,
                subname=f"{filename}",
            )

            prompt_rewards_cur[sample.prompt_id].append(float(sample.raw_reward))
            prompt_env_cur[sample.prompt_id] = sample.ground_truth.get("reward_fn_type", "default")

        if PM.world_rank == 0:
            for p, r in prompt_rewards_cur.items():
                if all(rr >= 1.0 for rr in r):
                    GlobalMetrics.correctness.add(1.0, "all_accept")
                    GlobalMetrics.correctness_subclass.add(1.0, f"all_accept/{prompt_env_cur[p]}")
                elif all(rr <= 0.0 for rr in r):
                    GlobalMetrics.correctness.add(1.0, "all_fail")
                    GlobalMetrics.correctness_subclass.add(1.0, f"all_fail/{prompt_env_cur[p]}")
                elif any(rr >= 1.0 for rr in r):
                    ## 对每个prompt的全部response，非全对/非全错，但是有至少一个全对
                    GlobalMetrics.correctness.add(1.0, "some_accept")
                    GlobalMetrics.correctness_subclass.add(1.0, f"some_accept/{prompt_env_cur[p]}")
                else:
                    ## 对每个prompt的全部response，没有任意一个response全对，但是拿到了一部分reward
                    GlobalMetrics.correctness.add(1.0, "some_reward")
                    GlobalMetrics.correctness_subclass.add(1.0, f"some_reward/{prompt_env_cur[p]}")

    def log_sampling_logprob_metrics(self, samples: list[PackedPPOSamples]):
        """Log actor-vs-sampling logprob diagnostics for the actor forward pass."""
        from steptronoss.core import parallel_state as mpu
        from steptronoss.utils import recur_to

        def compute_logprob_diff(samples: list[PackedPPOSamples]):
            diffs = []
            for sample in samples:
                mask = sample.is_gen_mask[0]
                diffs.append((sample.actor_logprobs - sample.logprobs).abs().sum() / mask.sum())
            return torch.tensor(diffs).mean()

        def compute_logprob_diff_max(samples: list[PackedPPOSamples]):
            diffs = []
            for sample in samples:
                diffs.append((sample.actor_logprobs - sample.logprobs).abs().max())
            return torch.tensor(diffs).max()

        def compute_quantiles(tensor_data: torch.Tensor, max_elems: int = 200000):
            """Generic function to compute quantiles for any tensor data."""
            tensor_data = tensor_data.cpu().float()
            num_elements = tensor_data.numel()
            sample_elements = max(min(num_elements // 10, max_elems), 1)
            indices = torch.randperm(num_elements, device=tensor_data.device)[:sample_elements]
            target_tensor = tensor_data.flatten()[indices]
            quantiles = torch.tensor([0.01, 0.05, 0.1, 0.9, 0.95, 0.99], device=tensor_data.device)
            q_values = torch.quantile(target_tensor, quantiles)
            infos = {
                "min": tensor_data.min(),
                "max": tensor_data.max(),
                "q01": q_values[0],
                "q05": q_values[1],
                "q10": q_values[2],
                "q90": q_values[3],
                "q95": q_values[4],
                "q99": q_values[5],
            }
            return recur_to(infos, "cuda")

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            GlobalMetrics.sampling_logprob_diff.add(samples, iop=compute_logprob_diff)
            GlobalMetrics.sampling_logprob_diff_max.add(samples, iop=compute_logprob_diff_max)

            # Compute and log sampling logprob diff quantiles
            all_logprob_diffs = torch.cat([(sample.actor_logprobs - sample.logprobs).abs() for sample in samples])
            diff_quantiles = compute_quantiles(all_logprob_diffs)
            for tag, value in diff_quantiles.items():
                GlobalMetrics.sampling_logprob_diff_quantile.add(value, subname=tag)

            # Compute and log sampling ratio diff quantiles
            all_ratio_diffs = torch.cat([
                (torch.exp(sample.actor_logprobs - sample.logprobs) - 1.0).abs() for sample in samples
            ])
            ratio_diff_quantiles = compute_quantiles(all_ratio_diffs)
            for tag, value in ratio_diff_quantiles.items():
                GlobalMetrics.sampling_ratio_diff.add(value, subname=tag)

            # Compute and log new logprob quantiles (actor_logprobs)
            all_new_logprobs = torch.cat([sample.actor_logprobs for sample in samples])
            new_logprob_quantiles = compute_quantiles(all_new_logprobs)
            for tag, value in new_logprob_quantiles.items():
                GlobalMetrics.new_logprob.add(value, subname=tag)

            # Compute and log old logprob quantiles (sampling logprobs)
            all_old_logprobs = torch.cat([sample.logprobs for sample in samples])
            old_logprob_quantiles = compute_quantiles(all_old_logprobs)
            for tag, value in old_logprob_quantiles.items():
                GlobalMetrics.old_logprob.add(value, subname=tag)

            # Compute and log new probability quantiles
            all_new_probs = torch.cat([torch.exp(sample.actor_logprobs) for sample in samples])
            new_prob_quantiles = compute_quantiles(all_new_probs)
            for tag, value in new_prob_quantiles.items():
                GlobalMetrics.new_prob.add(value, subname=tag)

            # Compute and log old probability quantiles
            all_old_probs = torch.cat([torch.exp(sample.logprobs) for sample in samples])
            old_prob_quantiles = compute_quantiles(all_old_probs)
            for tag, value in old_prob_quantiles.items():
                GlobalMetrics.old_prob.add(value, subname=tag)


class PPOLikeExp(BaseExp):
    data_cfg: DataConfig
    # actor
    actor_model_cfg: ActorModelConfig
    actor_scheduler_cfg: SchedulerConfig
    actor_grad_manager_cfg: GradientManagerConfig

    # critic
    critic_model_cfg: CriticModelConfig | None = None
    critic_scheduler_cfg: SchedulerConfig | None = None
    critic_grad_manager_cfg: GradientManagerConfig | None = None

    checkpoint_cfg: PPOCheckpointCfg

    trainer_cfg: PPOLikeTrainerConfig

    metric_cfg: PPOMetricConfig

    profiler_cfg: ProfilerConfig

    def train(self):
        self.update_from_args()
        self.sanity_check()

        trainer = self.trainer_cfg.get_trainer_cls()(self)
        trainer.train()

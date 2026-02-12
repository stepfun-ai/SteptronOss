#!/usr/bin/env python3

import os
import random

import megfile
import torch
import torch.distributed
from loguru import logger

from steptronoss.checkpointing.local_checkpoint import Checkpointer
from steptronoss.core import parallel_state as mpu
from steptronoss.core.context_parallel.context_parallel import (
    gather_from_balanced_cp_region,
    scatter_to_balanced_cp_region,
)
from steptronoss.core.parallel_state import PM, get_vpp_size
from steptronoss.core.tensor_parallel import vocab_parallel_cross_entropy
from steptronoss.core.trainers.base_trainer import BaseTrainer
from steptronoss.core.trainers.packed_model import PackedModel
from steptronoss.data.nextable.nextable import Nextable
from steptronoss.exp.rl import (
    EnvTrajectory,
    PackedPPOSamples,
    PPOLikeExp,
    PPOMetricConfig,
    PPOSample,
)
from steptronoss.generation.base_generatable import GenableItem
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.timers import init_timers, timeit
from steptronoss.utils import (
    balanced_list_split,
    broadcast_tensors,
    recur_to,
    sync_point,
    try_save,
)
from steptronoss.utils.general import list_split
from steptronoss.utils.logger import setup_logger
from steptronoss.utils.memory_tracker import CMT
from steptronoss.utils.metrics import GlobalMetrics
from steptronoss.utils.rl_utils import (  # PartialRolloutUtils,; TrajManager,; gen_ckpt_consistency_check,; shift_left,
    RaggedPPOSampleDumper,
)
from steptronoss.utils.utils import get_normalizer

GlobalMetrics: PPOMetricConfig


class TrainerPromptStream(Nextable):
    def __init__(self, dataloader: Nextable, proc):
        super().__init__()
        self.dataloader = dataloader
        self.proc = proc

    def __next__(self):
        while True:
            new_genable: GenableItem = self.proc(next(self.dataloader))
            if not new_genable:
                logger.info("Meet None for processed_item, just skip for now...")
                continue
            return new_genable

    def state_dict(self):
        return self.dataloader.state_dict()

    def load_state_dict(self, sd):
        return self.dataloader.load_state_dict(sd)


class PPOTrainer(BaseTrainer):
    exp: PPOLikeExp
    """Base class for PPO trainer. Access experiment attributes via `self.exp` in trainer."""

    def __init__(self, exp: PPOLikeExp):
        self.exp = exp
        self.seed = self.exp.seed

        self.checkpointer = Checkpointer()
        self.flow_controller = None

        self.start_iteration = 0
        self.actor_iteration = 0
        self.critic_iteration = 0

        self.ppo_cfg = self.exp.trainer_cfg

        self.build_hooks(self.exp.trainer_cfg)

    @timeit()
    def initialize_dist(self):
        setup_logger(self.exp.log_path, filename="train_log", mode="a")

        PM.initialize(backend="nccl")
        PM.set_mesh(self.exp.actor_model_cfg.parallel_cfg)
        set_mpu_random_seed(self.exp.seed)

        self.timers = init_timers(self.exp.profiler_cfg)

        self.exp.metric_cfg.register()

        self.log_ranks = [PM.world_size - 1]

        self.tb_writer = None
        if PM.world_rank in self.log_ranks:
            self.tb_writer = self.exp.build_log_writer()

        logger.info(f"Printing training log to: {self.log_ranks}", at=0)

        for hook in self._after_init_hooks:
            hook(self)

    # Functions for Training:
    def train(self):
        self.before_train()
        self.train_loop()
        self.after_train()

    @timeit()
    def before_train(self):
        self.initialize_dist()
        self.build_models_and_flow_controller()

        for hook in self._before_train_hooks:
            hook(self)

    def after_train(self):
        if self.exp.checkpoint_cfg.save_path and self.iteration != 0:
            # Do not use async dump since training is done.
            with self.exp.checkpoint_cfg.modify(async_dump=False):
                self.save_checkpoint()

        for hook in self._after_train_hooks:
            hook(self)

        logger.complete()

    def train_loop(self):
        """Train the model function."""

        # Iterations.
        self.iteration = self.start_iteration

        sync_point("before the start of training step")

        # Negative log_level won't be recorded to log files
        self.timers("interval-time", log_level=-1).start(barrier=True)

        while self.iteration < self.exp.trainer_cfg.train_iters:
            for hook in self._before_step_hooks:
                hook(self)

            # Evaluate before train.
            # if self.exp.eval_cfg is not None and self.ppo_cfg.eval_interval:
            #     if self.iteration % self.ppo_cfg.eval_interval == 0:
            #         if not (self.iteration == 0 and self.ppo_cfg.skip_first_eval):
            #             self.eval()

            self.train_step()
            for hook in self._after_step_hooks:
                hook(self)
            elapsed_time = self.timers("interval-time").elapsed(barrier=False, sync_device=True)
            GlobalMetrics.add("iteration_time", elapsed_time)

            self.training_log()

            if (
                self.exp.checkpoint_cfg.save_path
                and self.exp.checkpoint_cfg.save_interval
                and self.iteration % self.exp.checkpoint_cfg.save_interval == 0
                and self.iteration > self.start_iteration
            ):
                self.save_checkpoint()
                # empty cache after saving checkpoint to release temp tensors
                torch.cuda.empty_cache()

            # empty cache after first run to clean all init buffers
            # this reduces peak memory usage
            if self.iteration == self.start_iteration:
                torch.cuda.empty_cache()

            self.iteration += 1

    def adapt_trajs_to_samples(self, rollouts: list[EnvTrajectory]) -> list[PPOSample]:
        all_samples = []
        for sample in rollouts:
            sample.trajectory = torch.tensor(sample.trajectory, dtype=torch.long)
            if sample.logprobs is not None:
                sample.logprobs = torch.tensor(sample.logprobs, dtype=torch.float32)
            else:
                sample.logprobs = torch.zeros(sum(sample.is_gen_mask), dtype=torch.float16)
            sample.is_gen_mask = torch.tensor(sample.is_gen_mask, dtype=torch.bool)
            if sample.meta is None:
                sample.meta = {}

            sample = PPOSample(**sample)
            all_samples.append(sample)
        return all_samples

    @timeit(level=1)
    def generate_trajectory(self) -> list[PPOSample]:
        """Generate rollout samples.

        Data residency: return value should be identical on all DP ranks (global rollout list).
        """
        all_samples = self.flow_controller.get_train_samples()
        if PM.world_rank == 0:
            all_samples = self.adapt_trajs_to_samples(all_samples)

        all_samples = broadcast_tensors(all_samples, src_rank=0, group=None)
        return all_samples

    def _group_samples(
        self, samples: list[PPOSample], max_seq_len=8192, divisable_by=1
    ) -> tuple[list[list[PPOSample]], list[int]]:
        """
        Group samples into chunks of max_seq_len.
        If the sample is too long, it will be skipped.
        Args:
            samples: list of PPOSample
            max_seq_len: max sequence length
            divisable_by: the number of chunks must be divisible by this number
        Returns:
            buffer: list of list of PPOSample
            sizes: list of int, the number of samples in each chunk
        """
        buffer: list[list[PPOSample]] = [[]]
        sizes: list[int] = [0]
        for s in samples:
            if sizes[-1] + len(s.trajectory) <= max_seq_len:
                buffer[-1].append(s)
                sizes[-1] += len(s.trajectory)
            elif len(s.trajectory) > max_seq_len:
                continue
            else:
                buffer.append([s])
                sizes.append(len(s.trajectory))
        extra_chunk = len(buffer) % divisable_by

        if extra_chunk != 0:
            buffer = buffer[:-extra_chunk]

        return buffer, sizes[: len(buffer)]

    @timeit(level=1)
    def data_balance_and_pack(self, all_samples: list[PPOSample]) -> list[PackedPPOSamples]:
        """
        Group samples into chunks of max_seq_len.
        If the sample is too long, it will be skipped.
        Args:
            all_samples: list of PPOSample
            max_seq_len: max sequence length for each chunk
            divisable_by: the number of chunks must be divisible by this number
        Returns:
            packed_my_samples: list of PackedPPOSamples

        Data residency: input list is identical on all DP ranks; output list is
        per-DP (each DP holds different packed samples).
        """

        # sample_lengths = [balance_key(x) for x in all_samples]
        GlobalMetrics.unpacked_samples.add(float(len(all_samples)))
        if get_vpp_size() > 1:
            d = PM.size_of("DP") // PM.size_of("CP") * PM.size_of("PP")
        else:
            d = PM.size_of("DP") // PM.size_of("CP")

        grouped_all_samples, sizes = self._group_samples(
            all_samples,
            max_seq_len=self.exp.trainer_cfg.global_seq_length,
            divisable_by=d,
        )
        logger.info(f"Global Rollout: pack({len(all_samples)}) -> {len(grouped_all_samples)}", at=-1)

        GlobalMetrics.packed_samples.add(float(len(grouped_all_samples)))

        if len(grouped_all_samples) <= mpu.get_data_world_size():
            raise Exception("No enougth rollout!")

        grouped_my_samples: list[list[PPOSample]] = balanced_list_split(
            grouped_all_samples, sizes, mpu.get_data_world_size()
        )[mpu.get_data_rank()]
        logger.info(
            f"Keeping {len(grouped_my_samples)} samples.",
            at=PM.i_am("MP", 0),
        )
        packed_my_samples = [PackedPPOSamples.from_samples(sp) for sp in grouped_my_samples]

        random.Random(1234).shuffle(packed_my_samples)

        return packed_my_samples

    @torch.no_grad()
    @timeit(level=1)
    def get_reference(self, samples: list[PackedPPOSamples]):
        """Set sample.ref_logprob.

        Data residency: samples are per-DP (each DP holds different data).
        """

        # fmt: off
        def hacked_loss_fn(samples: PackedPPOSamples, logits: torch.Tensor):
            logits: torch.Tensor  # S, B, V
            labels = samples.labels
            labels = scatter_to_balanced_cp_region(labels, dim=1)
            ref_logprobs = -vocab_parallel_cross_entropy(
                vocab_parallel_logits=logits, target=labels.T.contiguous()
            )[0].T

            ref_logprobs = gather_from_balanced_cp_region(ref_logprobs, dim=1)
            # XXX: `ref_logprobs` 2D ([B, S_global]) vs. `PackedPPOSamples.is_gen_mask` 1D ([S_global])
            # PyTorch boolean indexing behavior may be unexpected.
            # `ref_logprobs.squeeze(0)` needed?
            samples.ref_logprobs = ref_logprobs[samples.is_gen_mask].contiguous()

            return 0
        # fmt: on
        self.reference_model.forward_backward(
            data_list=samples,
            data_proc_fn=self.ppo_cfg.preprocess_generated,
            loss_fn=hacked_loss_fn,
            training=False,
        )
        return samples

    @torch.no_grad()
    @timeit(level=1)
    def get_actor_logprob(self, samples: list[PackedPPOSamples]):
        """Set sample.actor_logprobs.

        Data residency: samples are per-DP (each DP holds different data).
        """

        # fmt: off
        def hacked_loss_fn(samples: PackedPPOSamples, logits: torch.Tensor):
            logits: torch.Tensor  # S, B, V
            labels = samples.labels
            labels = scatter_to_balanced_cp_region(labels, dim=1)
            actor_logprobs = -vocab_parallel_cross_entropy(
                vocab_parallel_logits=logits / self.exp.trainer_cfg.policy_actor_temperature, target=labels.T.contiguous()
            )[0].T  # [B, S]
            actor_logprobs = gather_from_balanced_cp_region(actor_logprobs, dim=1)
            # XXX: `actor_logprobs` 2D ([B, S_global]) vs. `PackedPPOSamples.is_gen_mask` 1D ([S_global])
            # PyTorch boolean indexing behavior may be unexpected.
            # `actor_logprobs.squeeze(0)` needed?
            samples.actor_logprobs = actor_logprobs[samples.is_gen_mask].contiguous()

            return 0
        self.actor.forward_backward(
            data_list=samples,
            data_proc_fn=self.ppo_cfg.preprocess_generated,
            loss_fn=hacked_loss_fn,
            training=False,
        )

        self.ppo_cfg.log_sampling_logprob_metrics(samples)

        return samples

    @torch.no_grad()
    @timeit(level=1)
    def get_values_advantages(self, samples: list[PackedPPOSamples]):
        """Set sample.values & returns & advantages.

        NOTE:
        With (complement) CP, values from the critic model are partitioned across CP ranks,
        and each rank holds non-contiguous chunks, e.g., CP0 [0, 1, 6, 7], CP1 [2, 3, 4, 5].
        As a result, the temporal dependencies span across different CP ranks.
        GAE requires sequential access to all timesteps in strict temporal order.
        So computing GAE locally would be mathematically incorrect.

        Data residency: samples are per-DP (each DP holds different data).
        """

        def hacked_loss_fn(samples: PackedPPOSamples, ragged_values: torch.Tensor):
            ragged_values: torch.Tensor = ragged_values[:, 0, 0]  # (S_local, B, 1) -> (S_local,)

            # Gather values from balanced CP region to reconstruct full sequence for GAE to work.
            cp_size = PM.size_of("CP")
            if cp_size > 1:
                ragged_values = gather_from_balanced_cp_region(ragged_values, dim=0)
                # Shape verification
                assert ragged_values.shape[0] == samples.cu_seqlens[-1].item(), (
                    f"Shape mismatch: values={ragged_values.shape[0]}, expected={samples.cu_seqlens[-1].item()}"
                )

            self.ppo_cfg.get_advantage_and_returns(ragged_samples=samples, ragged_values=ragged_values)

            return 0

        self.critic.forward_backward(
            data_list=samples,
            data_proc_fn=self.ppo_cfg.preprocess_generated,
            loss_fn=hacked_loss_fn,
            training=False,
        )

        return samples

    def normalize_advantages(self, samples: list[PackedPPOSamples]):
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            for s in samples:  # move all adavantages to cuda, this is small
                s.advantages = s.advantages.cuda()
            advantages = torch.cat([s.advantages for s in samples])
            GlobalMetrics.advantage_max.add(advantages, iop=torch.max)
            GlobalMetrics.advantage_min.add(advantages, iop=torch.min)

            mean, std = get_normalizer(advantages)
            GlobalMetrics.advantage_std.add(std, iop=torch.clone)
            std.clip_(0.01)

            for sample in samples:
                sample.advantages = (sample.advantages - mean) / std

                GlobalMetrics.advantage_mean.add(sample.advantages, iop=torch.mean)
                GlobalMetrics.advantage_abs_mean.add(sample.advantages, iop=lambda x: x.abs().mean())

        return samples

    def make_div_pp(self, data: list):
        if get_vpp_size() > 1:
            pp = PM.size_of("PP")
            assert len(data) >= pp, "No Enough Data!"
            data = data[: len(data) // pp * pp]
        return data

    def chunk_my_samples(
        self, my_samples: list[PackedPPOSamples], fix_iters: int | None = None
    ) -> list[list[PackedPPOSamples]]:
        assert fix_iters
        min_chunk_size = PM.size_of("PP") if get_vpp_size() > 1 else 1
        chunked_data = list_split(
            my_samples,
            split=min(fix_iters, len(my_samples) // min_chunk_size),
        )
        return chunked_data

    def train_step(self):
        CMT.mark("start_of_iter")
        self.actor.offload_model()
        self.actor.offload_state()
        all_samples = self.generate_trajectory()
        CMT.mark("after_generate_trajectory")

        self.ppo_cfg.log_generation_metrics(all_samples)

        self.ppo_cfg.log_reward_metrics(all_samples)

        logger.info("Start filter sampels after reward_fn calc", at=-1)
        all_samples = self.exp.trainer_cfg.filter_samples(all_samples)

        CMT.mark("before_packing")
        # Packing samples
        logger.info(f"Start packing {len(all_samples)} samples...", at=-1)
        my_samples = self.data_balance_and_pack(all_samples)
        logger.info(f"Get {len(my_samples)} packed-samples for each DP.", at=-1)
        CMT.mark("after_packing")

        logger.info("Forward Reference and Actor...", at=-1)

        # Only skip reference if all of the following are False: KL penalty, KL loss, and log_logprobs.
        if not self.ppo_cfg.skip_forward_reference:
            self.reference_model.backload_model()
            my_samples = self.get_reference(my_samples)
            self.reference_model.offload_model()
            CMT.mark("after_get_reference")
        # Skip actor forward when onpolicy update and without KL penalty, KL loss, and log_logprobs.

        CMT.mark("before_get_actor_logprob")
        if not self.ppo_cfg.skip_forward_actor:
            self.actor.backload_model()
            my_samples = self.get_actor_logprob(my_samples)
            self.actor.offload_model()
        CMT.mark("after_get_actor_logprob")

        self.critic.backload_model()
        if not self.ppo_cfg.onvalue_gae:
            logger.info("Forward Critic...", at=-1)
            my_samples = self.get_values_advantages(my_samples)
            CMT.mark("after_get_values_advantages")

        if self.ppo_cfg.offload_data:
            with self.timers.record("offload_data", log_level=1):
                my_samples = recur_to(my_samples, "cpu")
                torch.cuda.empty_cache()

        self.critic.backload_state()
        critic_chunks = self.chunk_my_samples(my_samples, fix_iters=self.ppo_cfg.fix_iters_critic)
        GlobalMetrics.grad_norms.enabled = False  # disable for critic model
        with self.timers.record("critic_train", log_level=1):
            for critic_epoch in range(self.ppo_cfg.critic_epoch):
                GlobalMetrics.inner_iteration.add(float(len(critic_chunks)))
                logger.info(f"CriticTrain: Ep={critic_epoch} iters={len(critic_chunks)}", at=-1)
                for iter_data in critic_chunks:
                    self.critic.forward_backward(
                        data_list=self.make_div_pp(iter_data),
                        data_proc_fn=self.ppo_cfg.preprocess_generated,
                        loss_fn=self.ppo_cfg.critic_loss_func,
                        training=True,
                        offload_opt_while_forward=self.ppo_cfg.offload_optimizer_state,
                        non_blocking_offload=True,
                        offload_data=self.ppo_cfg.offload_data,
                    )
                    (
                        update_successful,
                        grad_norm,
                        num_zeros_in_grad,
                    ) = self.critic.optimizer_step()
                    loss = GlobalMetrics.critic_loss.reduce()
                    lr = GlobalMetrics.learning_rate.reduce()
                    self.step_log(
                        "Critic-inner",
                        self.critic_iteration,
                        loss=loss,
                        grad_norm=grad_norm,
                        grad_zeros=num_zeros_in_grad,
                        lr=lr,
                    )
                    GlobalMetrics.critic_grad_norm.add(grad_norm)
                    self.critic_iteration += 1

        if self.ppo_cfg.onvalue_gae:
            # actor data must be subset of critic data!
            my_samples = []
            for iter_data in critic_chunks:
                my_samples.extend(self.make_div_pp(iter_data))

        # normalize advantages
        with self.timers.record("normalize_advantages", log_level=1):
            my_samples = self.normalize_advantages(my_samples)

        # Actor
        self.critic.offload_model()
        self.critic.offload_state()
        self.actor.backload_model()
        self.actor.backload_state()

        actor_chunks = self.chunk_my_samples(my_samples, fix_iters=self.ppo_cfg.fix_iters)
        GlobalMetrics.grad_norms.enabled = self.exp.trainer_cfg.log_detailed_grad_norms
        with self.timers.record("actor_train", log_level=1):
            for actor_epoch in range(self.ppo_cfg.actor_epoch):
                logger.info(f"ActorTrain: Ep={actor_epoch} iters={len(actor_chunks)}", at=-1)
                grad_norm = 0
                for iter_data in actor_chunks:
                    self.actor.forward_backward(
                        data_list=self.make_div_pp(iter_data),
                        data_proc_fn=self.ppo_cfg.preprocess_generated,
                        loss_fn=self.ppo_cfg.actor_loss_func,
                        training=self.actor_iteration >= self.ppo_cfg.critic_warmup_iters,
                        offload_opt_while_forward=self.ppo_cfg.offload_optimizer_state,
                        non_blocking_offload=True,
                        offload_data=self.ppo_cfg.offload_data,
                    )

                    if self.actor_iteration >= self.ppo_cfg.critic_warmup_iters:
                        (
                            update_successful,
                            grad_norm,
                            num_zeros_in_grad,
                        ) = self.actor.optimizer_step()

                    loss = GlobalMetrics.ppo_loss.reduce()
                    ref_loss = GlobalMetrics.kl_with_ref_dist.reduce()
                    lr = GlobalMetrics.learning_rate.reduce()
                    self.step_log(
                        "Actor-inner",
                        self.actor_iteration,
                        ppo_loss=loss,
                        ref_loss=ref_loss,
                        grad_norm=grad_norm,
                        grad_zeros=num_zeros_in_grad,
                        lr=lr,
                    )
                    self.actor_iteration += 1
        self.actor.offload_model()
        self.actor.offload_state()

        if self.ppo_cfg.dump_sample_keys:
            with self.timers.record("dump_all_samples", log_level=1):
                self.dump_all_samples(my_samples)

    def dump_all_samples(self, my_samples: list[PackedPPOSamples]):
        """
        will run all_gather and then dump to msgpack

        Data residency: input is per-DP; all_gather makes data available across DP before dump.
        """
        logger.info(
            f"Dumping {self.ppo_cfg.dump_sample_keys}, you can use torch.load to load it",
            at=-1,
        )
        # TODO: in future, can only all_gather the keys that are needed
        all_samples_to_save: list[PackedPPOSamples] = recur_to(
            my_samples,
            to="cpu",
        )
        dumper = RaggedPPOSampleDumper()
        sample_for_dump = dumper(all_samples_to_save, self.ppo_cfg.dump_sample_keys)
        save_path = os.path.join(self.exp.checkpoint_cfg.save_path, f"ppo_samples_it{self.iteration}")
        if PM.world_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        # NOTE: only dump on last rank, due to pp issue
        torch.distributed.barrier()
        # NOTE: must read data in last pp rank!!
        if PM.i_am("PP", -1) and PM.i_am("TP", 0) and PM.i_am("CP", 0):
            dump_data_path = os.path.join(save_path, f"{PM.world_rank}.pt")

            try_save(sample_for_dump, dump_data_path)
            logger.info(
                f"Finished dumping {self.ppo_cfg.dump_sample_keys} to {dump_data_path}",
                at=0,
            )

    def step_log(self, prefix, step, **kwargs):
        if self.tb_writer:
            for k, v in kwargs.items():
                self.tb_writer.add_scalar(f"{prefix}/{k}", v, global_step=step)
        log_str = " | ".join(f"{k}: {v:.3e}" for k, v in kwargs.items())
        logger.info(f"{prefix}@it{step}: {log_str}", at=-1)

    def training_log(self):
        """Log training information such as losses, timing, ...."""
        writer = self.tb_writer

        # Collect Timing

        # let timer stat prior to globalMetrics to avoid external synchronize
        timers_prefix = f"iteration {self.iteration:8d}/{self.exp.trainer_cfg.train_iters:8d}"
        self.timers.log(
            self.iteration,
            reset=True,
            prefix=timers_prefix,
        )

        # Collect Global Metrics
        metrics = GlobalMetrics.reduce()

        if writer:
            metrics["world-size"] = PM.world_size

        if PM.world_rank in self.log_ranks:
            user_logs = self.exp.trainer_cfg.make_logs(self.iteration, metrics, writer)

            logs = {
                "iter": f"{self.iteration:8d}/{self.exp.trainer_cfg.train_iters:8d}",
            }
            if "reward_mean" in metrics:
                logs["reward_mean"] = f"{metrics['reward_mean']:.4f}"

            if user_logs:
                logs.update(user_logs)
            log_string = " | ".join(f"{k}: {v}" for k, v in logs.items())
            logger.info(log_string)
        CMT.report_over_world()

    def eval(self):
        pass

    # Builders:

    def build_models_and_flow_controller(self):
        self.set_autoresume()

        cfg = self.exp.checkpoint_cfg

        with timeit("build_actor_model"):
            with cfg.modify(**cfg.actor):
                state_dicts = self.load_checkpoint()

                self.actor_iteration = state_dicts.get("iteration", 0)

                self.actor = PackedModel(
                    model_config=self.exp.actor_model_cfg,
                    trainer_config=self.exp.trainer_cfg,
                    grad_manager_config=self.exp.actor_grad_manager_cfg,
                    scheduler_config=self.exp.actor_scheduler_cfg,
                    training=True,
                    state_dicts=state_dicts,
                    strict_load_model=cfg.strict_load_model,
                    name="actor",
                )
            self.actor.offload_model()

            self.actor.offload_state()
        CMT.mark("after_build_actor")

        with timeit("build_critic_model"):
            if self.exp.critic_model_cfg is not None:
                with cfg.modify(**cfg.critic):
                    state_dicts = self.load_checkpoint()
                    self.critic_iteration = state_dicts.get("iteration", 0)

                    self.critic = PackedModel(
                        model_config=self.exp.critic_model_cfg,
                        trainer_config=self.exp.trainer_cfg,
                        grad_manager_config=self.exp.critic_grad_manager_cfg,
                        scheduler_config=self.exp.critic_scheduler_cfg,
                        training=True,
                        state_dicts=state_dicts,
                        strict_load_model=cfg.strict_load_model,
                        name="critic",
                    )

                self.critic.offload_model()

                self.critic.offload_state()
            else:
                self.critic = None
                if self.__class__ is PPOTrainer:
                    raise ValueError("Critic model must be provided for PPO training")
        CMT.mark("after_build_critic")

        with timeit("build_reference_model"):
            with cfg.modify(**cfg.reference):
                state_dicts = self.load_checkpoint()

                self.reference_model = PackedModel(
                    model_config=self.exp.actor_model_cfg,
                    trainer_config=self.exp.trainer_cfg,
                    training=False,
                    state_dicts=state_dicts,
                    strict_load_model=cfg.strict_load_model,
                    name="reference_model",
                )
            self.reference_model.offload_model()
        CMT.mark("after_build_reference")

        # NOTE: data state stores with reference
        if PM.world_rank == 0:
            os.makedirs(self.exp.checkpoint_cfg.save_path, exist_ok=True)

        with timeit("build_flow_controller"):
            state_dicts = self.load_checkpoint()

            self.start_iteration = state_dicts.get("iteration", -1) + 1
            if PM.world_rank == 0:
                dataloader = self.exp.data_cfg.build_dataloader()
                prompt_stream = TrainerPromptStream(
                    dataloader=dataloader,
                    proc=self.exp.data_cfg.preprocess,
                )
            else:
                prompt_stream = None

            self.flow_controller = self.ppo_cfg.flow_cfg.build_flow_controller()
            self.flow_controller.start(
                dataloader=prompt_stream, model=self.actor.models, state_dict=state_dicts.get("data", None)
            )

    # Checkpointing:
    def set_autoresume(self):
        """This function set checkpoint_cfg.load_path to LAST ckpt (if exists)
        and set load_option to ALL
        """
        cfg = self.exp.checkpoint_cfg

        # Handle auto resume logic here:
        if cfg.auto_resume and cfg.save_path:
            latest_ckpt_file = os.path.join(cfg.save_path, "latest_ckpt")
            if megfile.smart_exists(latest_ckpt_file):
                with megfile.smart_open(latest_ckpt_file, "r") as f:
                    latest_ckpt = f.read().strip()
                logger.info(f"AutoResume from: {latest_ckpt_file}", at=0)
                if cfg.load_path is not None:
                    if PM.world_rank == 0:
                        logger.warning(f'AutoResume is overwriting exp.load! Raw: "{cfg.load_path}"')
                from os.path import join

                cfg.load_path = latest_ckpt
                cfg.load_option.all()
                cfg.load_safetensors = None
                cfg.actor.load_path = join(latest_ckpt, "actor")
                cfg.actor.load_option.all()
                cfg.actor.load_safetensors = None
                cfg.critic.load_path = join(latest_ckpt, "critic")
                cfg.critic.load_option.all()
                cfg.critic.load_safetensors = None
                cfg.reference.load_path = join(latest_ckpt, "reference")
                cfg.reference.load_option.all()
                cfg.reference.load_safetensors = None

                # NOTE: do not load safetensors
                cfg.load_safetensors = None

    def load_checkpoint(self):
        cfg = self.exp.checkpoint_cfg

        # if cfg.load_path is not None:
        state_dicts = self.checkpointer.load_ckpt(cfg.load_path, cfg)

        return state_dicts

    def save_checkpoint(self):

        self.checkpointer.join_dumping_thread()

        sync_point("ready for dump")
        # Now dump
        cfg = self.exp.checkpoint_cfg
        with cfg.modify(**cfg.actor):
            self.checkpointer.dump_ckpt(
                cfg=cfg,
                iter_path=os.path.join(cfg.save_path, f"it{self.iteration}"),
                sub_name="actor",
                mark_latest=False,
                iteration=self.actor_iteration,
                model=self.actor.models,
                optimizer=self.actor.grad_manager,
                opt_param_scheduler=self.actor.scheduler,
            )
        with cfg.modify(**cfg.critic):
            self.checkpointer.dump_ckpt(
                cfg=cfg,
                iter_path=os.path.join(cfg.save_path, f"it{self.iteration}"),
                sub_name="critic",
                mark_latest=False,
                iteration=self.critic_iteration,
                model=self.critic.models,
                optimizer=self.critic.grad_manager,
                opt_param_scheduler=self.critic.scheduler,
            )
        with cfg.modify(**cfg.reference):
            self.checkpointer.dump_ckpt(
                cfg=cfg,
                iter_path=os.path.join(cfg.save_path, f"it{self.iteration}"),
                sub_name="reference",
                mark_latest=False,
                iteration=self.iteration,
                model=self.reference_model.models,
            )

        # trainer state: data & iter

        if self.flow_controller is not None:
            self.flow_controller.weight_dumped()

        self.checkpointer.dump_ckpt(
            cfg=cfg,
            mark_latest=True,
            iteration=self.iteration,
            dataloader=self.flow_controller,
            extra_info={"exp": self.exp.to_dict()},
        )

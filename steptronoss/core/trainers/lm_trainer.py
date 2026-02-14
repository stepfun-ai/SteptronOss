#!/usr/bin/env python3

import os
from functools import partial

import megfile
import torch
from loguru import logger

from steptronoss.checkpointing.local_checkpoint import Checkpointer
from steptronoss.core import parallel_state as mpu
from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
from steptronoss.core.trainers.base_trainer import BaseTrainer
from steptronoss.exp.base_exp import DataConfig, Megatron3DParallelModelConfig
from steptronoss.exp.ntp import PretrainExp, PretrainMetricConfig
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.model.common.moe_block import MoEBlock

# from steptronoss.model.distributed import DistributedDataParallel as LocalDDP
from steptronoss.model.module import Float16Module
from steptronoss.model.utils import load_model_checkpoint
from steptronoss.optimizer.base_gradient_manager import GradientManager
from steptronoss.optimizer.hparam_scheduler import Scheduler
from steptronoss.timers import init_timers
from steptronoss.utils import (
    convert_num,
    print_n_params,
    setup_logger,
)
from steptronoss.utils.memory_tracker import CMT
from steptronoss.utils.metrics import GlobalMetrics

GlobalMetrics: PretrainMetricConfig


class DecoderPretrainTrainer(BaseTrainer):
    exp: PretrainExp

    def __init__(self, exp: PretrainExp):
        self.exp = exp
        self.checkpointer = Checkpointer()

        self.history = dict(
            consumed_samples=0,
            consumed_tokens=0,
            skipped_iters=0,
        )
        self._skipped_iters = 0

        self.train_iters: int = None

        self.build_hooks(self.exp.trainer_cfg)

    # Functions for Training:
    def train(self):
        self.before_train()
        self.train_loop()
        self.after_train()

    def before_train(self):
        setup_logger(self.exp.log_path, filename="train_log", mode="a")

        PM.initialize(backend="nccl")
        PM.set_mesh(self.exp.model_cfg.parallel_cfg)
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

        self.set_autoresume()

        state_dicts = self.load_checkpoint()

        if "iteration" in state_dicts:
            self.start_iteration = state_dicts["iteration"] + 1
        else:
            self.start_iteration = 0
        # Build training objects

        ## Model
        self.models = self.setup_model(self.exp.model_cfg)
        CMT.mark("after_build_model")

        if "model" in state_dicts:
            load_model_checkpoint(
                self.models,
                state_dicts,
                strict_load_model=self.exp.checkpoint_cfg.strict_load_model,
            )

        # if self.exp.trainer_cfg.eval_interval and self.exp.eval_cfg:
        #     self.evaluator = self.exp.eval_cfg.get_evaluator_cls()(self.exp)
        ## Optimizer
        self.grad_manager: GradientManager = self.exp.optimizer_cfg.build_gradient_manager(self.models)
        if "optimizer" in state_dicts:
            self.grad_manager.load_state_dict(state_dicts["optimizer"])
        CMT.mark("after_build_grad_manager")

        ## Scheduler
        self.opt_param_scheduler: Scheduler = self.exp.scheduler_cfg.build_scheduler(self.grad_manager.optimizer)
        if "scheduler" in state_dicts:
            self.opt_param_scheduler.load_state_dict(state_dicts["scheduler"])

        ## Dataloader
        self.train_data_iterators = self.build_dataloader(self.exp.data_cfg)
        if self.exp.trainer_cfg.train_iters is None:
            self.train_iters = self._compute_and_broadcast_train_iters()
        else:
            self.train_iters = self.exp.trainer_cfg.train_iters

        if self.exp.scheduler_cfg.total_schedule is None:
            self.exp.scheduler_cfg.total_schedule = self.train_iters

        if "data" in state_dicts:
            for dl in self.train_data_iterators:
                if hasattr(dl, "load_state_dict"):
                    dl.load_state_dict(state_dicts["data"])
        CMT.mark("after_build_dataloaders")

        for hook in self._before_train_hooks:
            hook(self)

    def after_train(self):
        if self.exp.checkpoint_cfg.save_path and self.iteration != 0:
            # Do not use async dump since training is done.
            with self.exp.checkpoint_cfg.modify(async_dump=False):
                self.save_checkpoint()

        for hook in self._after_train_hooks:
            hook(self)

        del self.train_data_iterators

        logger.complete()

    def train_loop(self):
        """Train the model function."""

        # Turn on training mode which enables dropout.
        for model_module in self.models:
            model_module.train()

        # Iterations.
        self.iteration = self.start_iteration

        sync_point("before the start of training step")

        # Negative log_level won't be recorded to log files
        self.timers("interval-time", log_level=-1).start(barrier=True)

        while self.iteration < self.train_iters:
            CMT.mark("start_of_iter")
            update_successful = self.train_step()
            # Logging.
            elapsed_time = self.timers("interval-time").elapsed(barrier=False, sync_device=True)
            GlobalMetrics.iteration_time.add(elapsed_time)
            self.training_log(update_successful)

            if (
                self.exp.checkpoint_cfg.save_path
                and self.exp.checkpoint_cfg.save_interval
                and self.iteration % self.exp.checkpoint_cfg.save_interval == 0
                and self.iteration > self.start_iteration
            ):
                self.save_checkpoint()
            # empty cache after first run to clean all init buffers
            # this reduces peak memory usage
            if self.iteration == self.start_iteration:
                torch.cuda.empty_cache()

            self.iteration += 1

    def train_step(self) -> bool:
        # If overlap_dp_vpp is set, we will not zero grad here,
        # Instead, we will zero grad and grad buffer after synchronizing
        # the grad of the previous step
        self.grad_manager.zero_grad()

        grad_accumulation_steps = (
            self.exp.trainer_cfg.global_batch_size // PM.size_of("DP") // self.exp.trainer_cfg.micro_batch_size
        )
        pp_scheduler = self.exp.model_cfg.get_pp_scheduler()
        pp_scheduler.configure(
            models=self.models,
            data_iterators=self.train_data_iterators,
            data_sync_fn=self.exp.trainer_cfg.sync_get_data,
            loss_fn=self.exp.trainer_cfg.loss_func,
            data_proc_fn=self.exp.data_cfg.preprocess,
            training=True,
            collect_output=False,
        )

        # Forward pass.
        with self.timers.record("forward-backward", log_level=1):
            for hook in self._before_step_hooks:
                hook(self)

            if self.exp.trainer_cfg.offload_optimizer_state:
                if self.grad_manager is not None:
                    self.grad_manager._cpu_offload(adam_only=False, zero_grad=False)

            pp_scheduler.run(grad_accumulation_steps)
        CMT.mark("after_forward_backward")

        # Empty unused memory.
        if self.exp.trainer_cfg.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        if self.exp.trainer_cfg.offload_optimizer_state:
            if self.grad_manager is not None:
                self.grad_manager._cpu_backload(adam_only=False)

        # Update parameters.
        with self.timers.record("optimizer-step", log_level=1):
            update_successful, grad_norm, num_zeros_in_grad = self.grad_manager.step()
        CMT.mark("after_optimizer_step")
        MoEBlock.update_router_balance_bias_per_gbs(self.models)

        GlobalMetrics.grad_norm.add(grad_norm)
        GlobalMetrics.grad_zeros.add(num_zeros_in_grad)

        # Update learning rate.
        if update_successful:
            if self.exp.scheduler_cfg.scheduler_unit == "iter":
                increment = 1
            elif self.exp.scheduler_cfg.scheduler_unit == "sample":
                increment = self.exp.trainer_cfg.global_batch_size
            elif self.exp.scheduler_cfg.scheduler_unit == "token":
                increment = self.exp.trainer_cfg.global_batch_size * self.exp.trainer_cfg.global_seq_length

            with self.timers.record("schedule-step", log_level=1):
                self.opt_param_scheduler.step(increment=increment)

        # Empty unused memory.
        if self.exp.trainer_cfg.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        with self.timers.record("after-step-hooks", log_level=1):
            for hook in self._after_step_hooks:
                hook(self)
        CMT.mark("after_step_hooks")

        return update_successful

    def training_log(self, update_success):
        """Log training information such as losses, timing, ...."""
        writer = self.tb_writer

        batch_size = self.exp.trainer_cfg.global_batch_size

        self.history["consumed_samples"] += batch_size
        # force update (required by bz warmup)
        self.history["consumed_tokens"] += batch_size * self.exp.trainer_cfg.global_seq_length
        if not update_success:
            self.history["skipped_iters"] += 1

        # 开销大的操作放到log interval的整数倍处执行
        if self.iteration % self.exp.trainer_cfg.log_interval == 0:
            # let timer stat prior to globalMetrics to avoid external synchronize
            timers_prefix = f"iteration {self.iteration:8d}/{self.train_iters:8d}"
            self.timers.log(
                self.iteration,
                self.exp.trainer_cfg.log_interval,
                reset=True,
                prefix=timers_prefix,
            )
            # Collect Global Metrics
            with self.timers.record("global-metrics-reduce", log_level=2):
                metrics: dict[str, torch.FloatTensor] = GlobalMetrics.reduce()

            # self.history["consumed_tokens"] += int(metrics.get("consumed_tokens", 0))
            # self.exp.history = self.history
            user_logs = {}
            if writer:
                metrics["batch-size"] = batch_size
                metrics["world-size"] = PM.world_size
                metrics["N-samples"] = self.history["consumed_samples"]
                metrics["N-tokens"] = self.history["consumed_tokens"]
                metrics["N-skipped"] = self.history["skipped_iters"]

            if PM.world_rank in self.log_ranks:
                user_logs = self.exp.trainer_cfg.make_logs(self.iteration, metrics, writer)

            tokens_per_second_per_card = (
                batch_size * self.exp.trainer_cfg.global_seq_length / metrics["iteration_time"].item() / PM.world_size
            )

            logs = {
                "iter": f"{self.iteration:8d}/{self.train_iters:8d}",
                "cum-tokens": convert_num(int(self.history["consumed_tokens"])),
                "ms/iter": f"{metrics['iteration_time'] * 1000.0:.1f}",
                "tokens-per-second-per-card": f"{tokens_per_second_per_card:.1f}",
                "lr": f"{metrics['learning_rate']:.3E}",
                "batch-size": f"{batch_size:5d}",
                "#skipped-iters": f"{self._skipped_iters}",
            }
            logs["grad-norm"] = f"{metrics['grad_norm'].item():.3f}"
            logs["#grad-zeros"] = convert_num(metrics["grad_zeros"].item())

            if user_logs:
                logs.update(user_logs)
            log_string = " | ".join(f"{k}: {v}" for k, v in logs.items())
            logger.info(log_string, at=-1)
            CMT.report_over_world()

    # Builders:
    def setup_model(self, model_config: Megatron3DParallelModelConfig) -> torch.nn.ModuleList:
        """Build the model by calling the build_model func in exp file."""

        # Build model.
        vp_size = get_vpp_size()
        model: list[torch.nn.Module] = []
        for i in range(vp_size):
            set_vpp_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            model_chunk = model_config.build_model()
            if self.exp.trainer_cfg.log_detailed_grad_norms and hasattr(model_chunk, "name_parameters"):
                model_chunk.name_parameters()
            model.append(model_chunk)

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        print_n_params(model)

        # Fp16 conversion.
        if model_config.params_dtype in [torch.float16, torch.bfloat16]:
            model = [Float16Module(model_module, model_config.params_dtype) for model_module in model]

        model = torch.nn.ModuleList(model)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # if self.exp.optimizer_cfg.optimizer == "muon":
        #     # We must setup muon tags on params before calling LocalDDP to make
        #     # comm-free zero partition work.
        #     from steptron.optimizer.muon import prepare_model_for_muon

        #     prepare_model_for_muon(model, model_config)

        return model

    def build_vpp_iterators(self, builder):
        data_iter = []
        for vp in range(get_vpp_size()):
            set_vpp_rank(vp)
            if self.exp.trainer_cfg.is_data_source():
                data_iter.append(builder())
            else:
                data_iter.append(None)
        return data_iter

    def build_dataloader(self, data_config: DataConfig):

        # get dataloader
        train_data_iterators = self.build_vpp_iterators(
            partial(
                data_config.build_dataloader,
                dp_rank=mpu.get_data_rank(),
                dp_size=mpu.get_data_world_size(),
            )
        )

        sync_point("after dataloaders are built")
        return train_data_iterators

    def _compute_and_broadcast_train_iters(self) -> int:
        """Compute train_iters from data source rank and broadcast to all ranks.

        Since only data source ranks (PP=0||PP=-1 && TP=0 && CP=0) build dataloaders,
        we need to broadcast the computed num_packed_samples to all ranks.
        """
        # Get local num_packed_samples (only data source has valid value)
        if self.exp.trainer_cfg.is_data_source():
            local_num_samples = len(self.train_data_iterators[0])
        else:
            local_num_samples = 0

        # Broadcast to all ranks using world group
        num_samples_tensor = torch.tensor([local_num_samples], dtype=torch.long, device="cuda")

        torch.distributed.broadcast(num_samples_tensor, src=0)
        num_packed_samples = num_samples_tensor.item()

        # Compute train_iters (same for all ranks)
        train_iters = num_packed_samples // self.exp.trainer_cfg.global_batch_size

        logger.info(
            f"Will train for {train_iters} iters.",
            at=0,
        )

        return train_iters

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
                cfg.load_path = latest_ckpt
                # load all
                cfg.load_option.all()

                # NOTE: do not load safetensors
                cfg.load_safetensors = None

    def load_checkpoint(self):
        cfg = self.exp.checkpoint_cfg

        state_dicts = self.checkpointer.load_ckpt(cfg.load_path, cfg)
        extra = state_dicts.get("extra_info", {})
        if cfg.load_option.exp and "exp" in extra:
            self.exp.assert_critical_attrs_expected(extra)

        self.history = extra.get("history", self.history)

        return state_dicts

    def save_checkpoint(self):

        self.checkpointer.join_dumping_thread()

        sync_point("ready for dump")
        # Now dump

        self.checkpointer.dump_ckpt(
            cfg=self.exp.checkpoint_cfg,
            iteration=self.iteration,
            model=self.models,
            optimizer=self.grad_manager,
            opt_param_scheduler=self.opt_param_scheduler,
            dataloader=self.train_data_iterators[0],
            extra_info={"exp": self.exp.to_dict(), "history": self.history},
        )

    def __repr__(self):
        # a function that generate a brief report of current training, used for debugging
        report = f"NTPTrainer @ iter={self.iteration}"
        return report


def sync_point(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    logger.info(f"Sync point [ {string} ]")

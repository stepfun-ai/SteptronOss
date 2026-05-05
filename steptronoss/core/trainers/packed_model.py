import gc
from contextlib import contextmanager, nullcontext

import torch
from loguru import logger

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
from steptronoss.exp.base_exp import Megatron3DParallelModelConfig
from steptronoss.exp.lr_schedulers import SchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig
from steptronoss.model.module import Float16Module
from steptronoss.model.utils import load_model_checkpoint
from steptronoss.optimizer.base_gradient_manager import (
    GradientManager,
    GradientManagerConfig,
)
from steptronoss.optimizer.hparam_scheduler import Scheduler
from steptronoss.timers import get_timers
from steptronoss.utils import broadcast_tensors, moving_iter, print_n_params


class PackedModel:
    """
    PackedModel combines model + optimizer + scheduler into a single wrapper,
    and provides training utilities like forward_backward() and optimizer_step().
    Pass state_dicts at __init__ so model/optimizer/scheduler are hydrated before
    any buffer allocation or hook registration, keeping internal views and shards
    consistent. A post-init load risks stale views (e.g., main_grad slices) or
    mismatched optimizer state.

    PackedModel also handle CPU offload/backload logic.
    For a typical BF16 training setting, the CUDA memory usage should be like:
    ```
    N: number-of-params
    total = 2N(bf16 param) + 4N(fp32 grad_acc buff) + 12N(fp32 param & momentum1/2)
    # for zero1 setting, the 12N can be sharded across DP.
    ```
    This class therefore support 3 individual offload control for param/buffer/optimizer

    """

    def __init__(
        self,
        trainer_config: NTPTrainerConfig,
        model_config: Megatron3DParallelModelConfig,
        grad_manager_config: GradientManagerConfig = None,
        scheduler_config: SchedulerConfig = None,
        training=False,
        state_dicts={},
        strict_load_model=False,
        name="packed_model",
    ) -> None:
        self.name = name
        self.training = training
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.scheduler_config = scheduler_config
        self.models: torch.nn.ModuleList = self.setup_model(model_config)
        if "model" in state_dicts:
            load_model_checkpoint(
                self.models,
                state_dicts,
                strict_load_model=strict_load_model,
            )

        self._offloaded = {
            "params": False,
            "grad_buffer": False,
            "optimizer_state": False,
        }
        if training:
            self.grad_manager: GradientManager = grad_manager_config.build_gradient_manager(self.models)
            if "optimizer" in state_dicts:
                self.grad_manager.load_state_dict(state_dicts["optimizer"])

            self.scheduler: Scheduler = scheduler_config.build_scheduler(self.grad_manager.optimizer)
            if "scheduler" in state_dicts:
                self.scheduler.load_state_dict(state_dicts["scheduler"])

    def setup_model(self, model_config: Megatron3DParallelModelConfig) -> torch.nn.ModuleList:
        """Build the model by calling the build_model func in exp file."""

        # Build model.
        vp_size = get_vpp_size()
        model: list[torch.nn.Module] = []
        for i in range(vp_size):
            set_vpp_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            model_chunk = model_config.build_model()
            if self.trainer_config.log_detailed_grad_norms and hasattr(model_chunk, "name_parameters"):
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
            fp32_output = getattr(model_config, "fp32_output", True)
            model = [Float16Module(model_module, model_config.params_dtype, fp32_output) for model_module in model]

        model = torch.nn.ModuleList(model)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        return model

    def _offload_param(self, non_blocking=True):
        if self._offloaded["params"]:
            return
        self._offloaded["params"] = True
        if self.training:
            self.grad_manager._release_grad_acc_hooks()
        for m in self.models:
            m.to(device="cpu", non_blocking=non_blocking)

    def _backload_param(self, non_blocking=True):
        if not self._offloaded["params"]:
            return
        self._offloaded["params"] = False
        for m in self.models:
            m.to(device=torch.cuda.current_device(), non_blocking=non_blocking)
        if self.training:
            # re-register grad-acc hooks once after params are back on GPU
            self.grad_manager._add_grad_acc_hooks(self.models)

    def _offload_grad_buffer(self):
        """WARNING: this also zero grads"""
        if self._offloaded["grad_buffer"]:
            return
        if not self.training:
            return
        self._offloaded["grad_buffer"] = True

        self.grad_manager.destroy_buffer()

    def _backload_grad_buffer(self):
        if not self._offloaded["grad_buffer"]:
            return
        if not self.training:
            return
        self._offloaded["grad_buffer"] = False

        self.grad_manager.build_buffer()

    def _offload_optimizer_state(self):
        if self._offloaded["optimizer_state"]:
            return
        self._offloaded["optimizer_state"] = True

        if not self.training:
            logger.warning("no optimizer state needs offload")
            return

        self.grad_manager._cpu_offload()

    def _backload_optimizer_state(self):
        if not self._offloaded["optimizer_state"]:
            return
        self._offloaded["optimizer_state"] = False

        if not self.training:
            logger.warning("no optimizer state needs backload")
            return

        self.grad_manager._cpu_backload()

    def offload_model(self):
        with get_timers().record(f"{self.name}_offload_model", log_level=1):
            self._offload_param(non_blocking=True)

    def offload_state(self):
        if self.training:
            with get_timers().record(f"{self.name}_offload_state", log_level=1):
                self._offload_grad_buffer()
                self._offload_optimizer_state()

    def backload_model(self):
        with get_timers().record(f"{self.name}_backload_model", log_level=1):
            self._backload_param(non_blocking=True)

    def backload_state(self):
        if self.training:
            with get_timers().record(f"{self.name}_backload_state", log_level=1):
                self._backload_grad_buffer()
                self._backload_optimizer_state()

    @contextmanager
    def on_gpu(self, model=True, optimizer=True):
        if model:
            self.backload_model()
        if optimizer:
            self.backload_state()
        yield
        if model:
            self.offload_model()
        if optimizer:
            self.offload_state()

    def optimizer_step(self):
        assert self.training

        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = self.grad_manager.step()

        # Update learning rate.
        if update_successful:
            if self.scheduler_config.scheduler_unit == "iter":
                increment = 1
            elif self.scheduler_config.scheduler_unit == "sample":
                increment = self.trainer_config.global_batch_size
            elif self.scheduler_config.scheduler_unit == "token":
                increment = self.trainer_config.global_batch_size * self.trainer_config.global_seq_length

            self.scheduler.step(increment=increment)

        # Empty unused memory.
        if self.trainer_config.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        return update_successful, grad_norm, num_zeros_in_grad

    def build_vpp_iterators(self, builder):
        data_iter = []
        for vp in range(get_vpp_size()):
            set_vpp_rank(vp)
            if self.trainer_config.is_data_source():
                data_iter.append(builder())
            else:
                data_iter.append(None)
        return data_iter

    def forward_backward(
        self,
        data_list: list,
        data_proc_fn,
        loss_fn=None,
        training=None,
        zero_grad=True,
        offload_opt_while_forward: bool | str = False,
        non_blocking_offload: bool = True,
        collect_output=False,
        offload_data=False,
    ) -> list:
        """Run forward and backward(if training) schedule for this model. The data list
        is only required on TP0PP0.

        Args:
            data_list (list): list of data, each is processed by data_proc_fn.
            data_proc_fn (Callable): Func(data) -> dict data_for_model_forward
            loss_fn (Callable, optional): Func(data, output). Defaults to None.
            training (bool, optional): Forward only?. Defaults to self.training.
            offload_opt_while_forward (bool, optional): offload opt while training. Defaults to False.
            collect_output (bool, optional): return output of model. Defaults to False.

        Returns:
            list: list of outputs (for each data)
        """
        if training is None:  # use self.training unless specified
            training = self.training
        if training:
            if zero_grad:
                self.grad_manager.zero_grad()
            if offload_opt_while_forward:
                self._offload_optimizer_state()

        if data_list:
            forward_num = len(data_list)
        else:
            forward_num = None

        forward_num = broadcast_tensors(
            forward_num,
            src_rank=PM.ranks_of("MP")[0],
            group=PM.group_of("MP"),
        )
        if offload_data:
            data_iter = self.build_vpp_iterators(lambda: moving_iter(data_list))
        else:
            data_iter = self.build_vpp_iterators(lambda: iter(data_list))
        with nullcontext() if training else torch.no_grad():
            pp_scheduler = self.model_config.get_pp_scheduler()
            pp_scheduler.configure(
                models=self.models,
                data_iterators=data_iter,
                data_sync_fn=self.trainer_config.sync_get_data,
                loss_fn=loss_fn,
                data_proc_fn=data_proc_fn,
                training=training,
                collect_output=collect_output,
            )
            outputs = pp_scheduler.run(forward_num)

        if training and offload_opt_while_forward:
            self._backload_optimizer_state()
        return outputs

# Copyright (c) 2026, STEPFUN CORPORATION. All rights reserved.

import copy
import gc
from typing import Optional

import torch

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.model.utils.comm_buffer import SteptronParameter
from steptronoss.timers import get_timers
from steptronoss.utils import recur_to

from .base_gradient_manager import GradientManager, _zero_grad_group_helper


class AccInFP32GradientManager(GradientManager):
    def __init__(
        self,
        cfg: GradientManagerConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):

        super().__init__(cfg, model, optimizer)
        self.model_comm_group = PM.group_of("MP")

        (self.fp32_params, self.fp16_params, self.fp16_params_in_fp32) = self.replace_optimizer_with_fp32(
            self.optimizer
        )

    def zero_grad(self, set_to_none=True, **kwargs):
        """We only need to zero the model related parameters, i.e.,
        fp16 parameters & fp32_from_fp32_group. We additionally zero
        the fp16 main copies as a memory optimization to reduce
        fragmentation."""
        super().zero_grad(set_to_none, **kwargs)
        _zero_grad_group_helper(self.fp16_params, set_to_none)
        _zero_grad_group_helper(self.fp16_params_in_fp32, set_to_none)
        _zero_grad_group_helper(self.fp32_params, set_to_none)

    def state_dict(self) -> dict[str]:
        state_dict = {}
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["fp32_from_fp16_params"] = self.fp16_params_in_fp32
        return state_dict

    def load_state_dict(self, state_dict: dict[str]) -> None:
        # Optimizer.
        self.optimizer.load_state_dict(state_dict["optimizer"])

        # Copy data for the main params.
        saved_params = state_dict["fp32_from_fp16_params"]
        for current_param, saved_param in zip(self.fp16_params_in_fp32, saved_params):
            current_param.data.copy_(saved_param.data)

    @torch.no_grad()
    def step(self) -> tuple[bool, float | None, int | None]:
        self._reduce_model_grads()
        # Copy gradients from model params to main params.
        with get_timers().record("optimizer-copy-to-main-grad", log_level=2):
            self._copy_model_grads_to_fp32_params()
        # Clip the main gradients.
        with get_timers().record("optimizer-clip-main-grad", log_level=2):
            grad_norm = None
            if self.cfg.clip_grad > 0.0:
                grad_norm = self.clip_grad_norm(self.cfg.clip_grad)
        # Count the zeros in the grads.
        with get_timers().record("optimizer-count-zeros", log_level=2):
            num_zeros_in_grad = self.count_zeros() if self.cfg.log_num_zeros_in_grad else None

        # Step the optimizer.
        with get_timers().record("optimizer-inner-step", log_level=2):
            self.optimizer.step()

        # Update params from main params.
        with get_timers().record("optimizer-copy-main-to-model-params", log_level=2):
            self._copy_fp32_params_to_model_params()
        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        # Only needed for the float16 params.
        for fp16_param, fp32_param in zip(self.fp16_params, self.fp16_params_in_fp32):
            fp32_param.data.copy_(fp16_param.data)

    @staticmethod
    def replace_optimizer_with_fp32(
        optimizer: torch.optim.Optimizer,
    ) -> tuple[list[SteptronParameter], list[SteptronParameter], list[SteptronParameter]]:
        fp32_params: list[SteptronParameter] = []
        fp16_params: list[SteptronParameter] = []
        fp16_params_in_fp32: list[SteptronParameter] = []

        # For all the groups in the original optimizer:
        for param_group in optimizer.param_groups:
            param_group: dict[str, list[SteptronParameter]]
            # For all the parameters in this group:
            for i, param in enumerate(param_group["params"]):
                # bfloat16 params
                if param.is_cuda and param.dtype == torch.bfloat16:
                    # Create a copy
                    main_param = param.detach().clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                    fp16_params.append(param)
                    fp16_params_in_fp32.append(main_param)

                    # Reset existing state dict key to the new main param.
                    param_group["params"][i] = main_param
                    if param in optimizer.state:
                        optimizer.state[main_param] = optimizer.state.pop(param)
                # fp32 params.
                elif param.is_cuda and param.dtype == torch.float32:
                    fp32_params.append(param)
                    param_group["params"][i] = param

                else:
                    raise TypeError(
                        "Wrapped parameters must be CUDA tensors of dtype "
                        "torch.float32 or torch.bfloat16. "
                        f"Received dtype={param.dtype}, device={param.device}"
                    )
        return fp32_params, fp16_params, fp16_params_in_fp32

    def to_device(self, device, non_blocking=False):
        for state in self.optimizer.state.values():
            state["exp_avg"] = state["exp_avg"].to(device, non_blocking=non_blocking)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(device, non_blocking=non_blocking)

        new_fp16_params_in_fp32 = []
        for param_group in self.optimizer.param_groups:
            param_group: dict[str, list[SteptronParameter]]
            # For all the parameters in this group:
            for i, param in enumerate(param_group["params"]):
                # bfloat16 params
                new_param = param.to(device, non_blocking=non_blocking)
                new_fp16_params_in_fp32.append(new_param)
                param_group["params"][i] = new_param
                if param in self.optimizer.state:
                    self.optimizer.state[new_param] = self.optimizer.state.pop(param)

        assert len(new_fp16_params_in_fp32) == len(self.fp16_params_in_fp32)
        self.fp16_params_in_fp32 = new_fp16_params_in_fp32

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def _reduce_model_grads(self):
        """Reduce gradients across data parallel ranks."""

        # All-reduce layer-norm grads (for sequence parallelism).
        with get_timers().record("layernorm-grads-all-reduce", log_level=2):
            self._custom_allreduce(
                tag="sequence_parallel",
                group=PM.group_of("TP"),
                op=torch.distributed.ReduceOp.SUM,
            )

        with get_timers().record("microdp-grads-all-reduce", log_level=2):
            self._custom_allreduce(
                tag="micro_dp",
                group=PM.group_of("TP"),
                op=torch.distributed.ReduceOp.SUM,
            )

        # All-reduce non-expert grads and properly process expert grads
        self._process_expert_parallel_grads()

        # All-reduce if needed.
        with get_timers().record("grads-all-reduce", log_level=2):
            for bucket_key, buffer_ in self._grad_buffers.items():
                reduce_group = bucket_key.allreduce_group
                world_size = PM.size_of(reduce_group)
                group = PM.group_of(reduce_group)
                buffer_ /= world_size
                torch.distributed.all_reduce(buffer_, group=group)

    def _copy_model_grads_to_fp32_params(self) -> None:
        # Copy float16/bfloat16 grads into main fp32 grads.
        for fp16_param, fp32_param in zip(self.fp16_params, self.fp16_params_in_fp32):
            fp32_param.grad = fp16_param.main_grad.float()
            fp16_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for param in self.fp32_params:
            param.grad = param.main_grad

    def _copy_fp32_params_to_model_params(self) -> None:
        # Only needed for the float16 params.
        for fp16_param, fp32_param in zip(self.fp16_params, self.fp16_params_in_fp32):
            fp16_param.data.copy_(fp32_param.data)

import copy
import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from loguru import logger
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.model.module import param_is_not_shared
from steptronoss.model.utils.comm_buffer import SteptronParameter, build_grad_buffers

from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32


def _zero_grad_group_helper(group: Iterable[Parameter], set_to_none: bool) -> None:
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


class DummyOptimizer:
    def __init__(self):
        self.param_groups = []
        self.state = None

    def zero_grad(self):
        pass

    def step(self):
        pass

    def load_state_dict(self, state_dict):
        return None

    def state_dict(self):
        pass

    def reload_model_params(self):
        pass


class GradientManager(ABC):
    """
    This class holds a model & optimizer, it controls
    - imple gradient accumulation between micro batches
    - imple gradient reduce / param prepare.
    - imple gradient clip
    - imple grad_norm & grad_zeros stats.
    """

    def __init__(
        self,
        cfg: GradientManagerConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """Input optimizer is the base optimizer for example Adam."""

        self.cfg = cfg

        self.model = model
        self.optimizer = optimizer

        self.build_buffer()
        for bucket_key, _buffer in self._grad_buffers.items():
            logger.info(f"Gradbuffer: {bucket_key}, dp_world_size: {PM.size_of(bucket_key.allreduce_group)}")

        # We need to store them so they don't go out of scope.
        self._grad_acc_hook_handles: list[tuple[RemovableHandle, object]] = []
        self._add_grad_acc_hooks(self.model)

        # NOTE: Now grad produced by torch will be automatically added to _grad_buffer

        self.model_comm_group = PM.group_of("MP")

    def get_optimizer_params(self) -> list[SteptronParameter]:
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def clip_grad_norm(self, clip_grad: float) -> float:
        return clip_grad_norm_fp32(
            parameters=self.get_optimizer_params(),
            grads_for_norm=self._get_grads_for_grad_norm(),
            max_norm=clip_grad,
            model_parallel_group=self.model_comm_group,
        )

    def count_zeros(self) -> float:
        params = self.get_optimizer_params()
        return count_zeros_fp32(params, model_parallel_group=self.model_comm_group)

    def zero_grad(self, set_to_none: bool = True, **kwargs) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero_()

    def reload_model_params(self) -> None:
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for mixed-precision
        optimizers with main parameters, the main parameters need to also be
        updated."""
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def step(self) -> tuple[bool, float | None, int | None]:
        pass

    def destroy_buffer(self):
        # NOTE: this also clear gradients
        torch.cuda.synchronize()
        self.optimizer.zero_grad()

        self._param_buffers.clear()
        self._grad_buffers.clear()
        for p in self._param_info:
            p.main_grad = None  # remove reference

        torch.cuda.empty_cache()
        gc.collect()

    def build_buffer(self):
        # create new
        self._grad_buffers, self._param_buffers, self._param_info = build_grad_buffers(self.model)
        # attach slices onto tensor.main_grad
        for param, info in self._param_info.items():
            param.main_grad = info["fp32_buffer"]()[info["range"]].view(param.shape)

    @abstractmethod
    def to_device(self, device, non_blocking=False):
        """Move optimizer state to device"""
        pass

    def _cpu_offload(self):
        self.to_device("cpu")

    def _cpu_backload(self):
        self.to_device("cuda")

    def _custom_allreduce(self, tag, group, op) -> None:
        grads = []
        for param in self.model.parameters():
            if param.requires_grad and getattr(param, tag, False):
                grad = param.main_grad
                grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=group, op=op)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def _process_expert_parallel_grads(self) -> None:
        """Expert-parallel processing including non-expert all-reduce and EP topo scaling."""
        # Apply topology-invariant scaling on expert grad buffers when present.
        # This affects distributed optimizers that operate on shared grad buffers.
        """
        data: [D0, D1, D2, D3] # Dn: 8 tokens each
        attn: [T0, T0, T0, T0] # TP1 DP4

        data: [DA, DA, DB, DB] # Dn: 16 tokens each # more token, grad need x0.5
        moe : [E0, E1, E0, E1] # EP2 EDP2

        """
        tp_size = PM.size_of("TP")
        ep_size = PM.size_of("EP")
        # Only perform scaling when expert parallelism is enabled across >1 ranks.
        if ep_size <= 1:
            return
        for bucket_key, gbuf in self._grad_buffers.items():
            if bucket_key.allreduce_group != "EDP":
                continue
            gbuf.data *= tp_size / ep_size

    def _get_grads_for_grad_norm(self) -> list[torch.Tensor]:

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        params = self.get_optimizer_params()
        grads_for_norm = []

        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            # Be careful of router
            is_moe_param = getattr(param, "expert_model_parallel", False)
            # For moe params: only check EP duplicate; for attention params: only check TP duplicate
            if is_moe_param:
                is_not_duplicate = tensor_parallel.param_is_not_expert_parallel_duplicate(param)
            else:
                is_not_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def _add_grad_acc_hooks(self, model: torch.nn.Module):
        def _make_param_hook(param):
            """Create the all-reduce hook for backprop."""

            # Hook used for back-prop.
            def param_hook(*unused):
                # Add the gradient to the buffer.
                if param.grad is not None:
                    # The gradient function of linear layers is fused with GEMMs
                    param.main_grad.add_(param.grad.data)
                    # Now we can deallocate grad memory.
                    param.grad = None

            return param_hook

        # Backward hook.
        # Accumalation function for the gradients.
        # Loop over all the parameters in the model.
        for param in model.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator functtion.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                handle = grad_acc.register_hook(_make_param_hook(param))
                self._grad_acc_hook_handles.append((handle, grad_acc))

    def _release_grad_acc_hooks(self):
        for handle, _ in self._grad_acc_hook_handles:
            handle.remove()
        self._grad_acc_hook_handles.clear()

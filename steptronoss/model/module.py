# Copyright (c) 2026, STEPFUN CORPORATION. All rights reserved.
from __future__ import annotations

"""Megatron Module"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from steptronoss.checkpointing.reshape_ops import OnlineReshaper

from steptronoss.core import parallel_state as mpu
from steptronoss.core.utils import make_viewless_tensor

if TYPE_CHECKING:
    from steptronoss.exp.inference import BaseInferenceConfig

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


class MegatronModule(torch.nn.Module):
    pp_rank: int = 0
    pp_size: int = 1

    need_mtp_sync: bool = False

    def __init__(self) -> None:
        super().__init__()
        if mpu.model_parallel_is_initialized():
            self.pp_rank = mpu.get_pipeline_model_parallel_rank(ignore_virtual=False)
            self.pp_size = mpu.get_pipeline_model_parallel_world_size(ignore_virtual=False)

    def is_pipeline_first_stage(self):
        return self.pp_rank == 0

    def is_pipeline_last_stage(self):
        return self.pp_rank == self.pp_size - 1

    def _set_input_tensor(self, input_tensor: Tensor):
        self._input_tensor = input_tensor

    @abstractmethod
    def forward_head(self, *args, **kwargs) -> Tensor:
        """Unique functions on HEAD

        Returns:
            Tensor: Must be a Tensor
        """
        pass

    @abstractmethod
    def forward_chunk(self, last_tensor: Tensor, **kwargs) -> Tensor:
        """Chunk part of the Model

        Args:
            last_tensor (Tensor): Tensor from last chunk

        Returns:
            Tensor: Must be a Tensor
        """
        pass

    @abstractmethod
    def forward_tail(self, last_tensor: Tensor, **kwargs) -> Any:
        """Unique functions on TAIL

        Args:
            last_tensor (Tensor): Tensor from last chunk

        Returns:
            Any: Anything for loss_func or other
        """
        pass

    def abstract_forward(self, *args, **kwargs):
        """The abstract of how we split pipeline model parallel"""
        assert self.pp_size == 1
        x = self.forward_head(*args, **kwargs)
        x = self.forward_chunk(x, **kwargs)
        output = self.forward_tail(x, **kwargs)
        return output

    def forward(self, *args, **kwargs) -> Any:
        if self.pp_rank == 0:
            x = self.forward_head(*args, **kwargs)
        else:
            x = self._input_tensor
        x = make_viewless_tensor(
            x,
            requires_grad=True,
            keep_graph=True,
        )

        x = self.forward_chunk(x, **kwargs)

        if self.pp_rank == self.pp_size - 1:
            x = self.forward_tail(x, **kwargs)
        return x

    def name_parameters(self):
        """Assign log_name for each module, compatiable with
        Exp.log_detailed_grad_norms
        """
        for name, child in self.named_children():
            for p in child.parameters():
                p._log_name = name

    def export_infer_weight(self, cfg: BaseInferenceConfig) -> dict[str, torch.Tensor]:
        """This function returns a state_dict of this model in inference
        model parallel setting.
        e.g. We have TP3 & PP2 while training, let sd_x be the state dict of
        a chunk.
        Training:
                TP0         TP1         TP2
        PP0     train_a     train_b     train_c
        PP1     train_d     train_e     train_f

        Given dst_tp_size=2, This function returns:
                TP0         TP1         TP2
        PP0     infer_a     infer_b     infer_a
        PP1     infer_b     infer_a     infer_b

        where infer_a is the GATHER & SLICED weight ready for inference TP0
        infer_b is the inference weight for TP1

        Args:
            dst_tp_size (int, optional): target tp size. Defaults to 1.
        """
        raise NotImplementedError

    def init_model_weight(self) -> None:
        """Initialize the model weights and mark them as initialized.

        For each parameter in the model, this method should be implemented in the corresponding module
        or its parent module to properly initialize weights and set the 'has_initialized' flag to True
        after initialization. This helps track which parameters have been properly initialized.

        Example implementation:
        ```
        def init_model_weight(self) -> None:
            # Initialize parameters using appropriate methods
            # ...
            # Mark parameters as initialized
            for param in self.parameters():
                param.has_initialized = True
        ```
        """
        pass

    reshaper: OnlineReshaper

    def load_hf_state_dict(self, state_dict, *args, **kwargs):
        assert hasattr(self, "reshaper"), f"{self.__class__}.reshaper not defined! HF load/dump disabled!"
        state_dict = self.reshaper.forward(state_dict)
        return self.load_state_dict(state_dict, *args, **kwargs)

    def hf_state_dict(self):
        assert hasattr(self, "reshaper"), f"{self.__class__}.reshaper not defined! HF load/dump disabled!"
        from steptronoss.utils import recur_to

        state_dict = recur_to(self.state_dict(), "cpu")
        return self.reshaper.backward(state_dict)


class ModelWrapperBase(torch.nn.Module):
    """Abstract class for DDP."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # Keep a pointer to the model.
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, prefix="", keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, dtype=torch.bfloat16):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = val.to(dtype)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(ModelWrapperBase):
    def __init__(self, module: MegatronModule, dtype: torch.dtype, fp32_output=True):
        super().__init__(module)

        self.module = module.to(dtype)

        self.dtype = dtype
        self.fp32_output = fp32_output

    def forward(self, *inputs, **kwargs):
        if self.module.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.dtype)
        outputs = self.module(*inputs, **kwargs)
        if self.module.is_pipeline_last_stage() and self.fp32_output:
            outputs = float16_to_fp32(outputs)
        return outputs

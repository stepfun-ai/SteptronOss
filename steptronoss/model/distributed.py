# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from loguru import logger
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from steptronoss.core.parallel_state import PM

from .module import MegatronModule
from .muon_distributed_utils import (
    check_grad_buffer_param_index_map,
    get_num_padded_elements,
    update_grad_buffer_param_index_map,
)


class MemoryBuffer:

    def __init__(
        self, numel, numel_padded, dtype, device="cuda", allreduce_grads_in_dp=False
    ):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.allreduce_grads_in_dp = allreduce_grads_in_dp
        self.data = torch.zeros(
            self.numel_padded,
            dtype=self.dtype,
            device=device,
            requires_grad=False,
        )

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module

    @abstractmethod
    def allreduce_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, prefix="", keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


@dataclass(unsafe_hash=True)
class ExtendedDtype:
    torch_dtype: torch.dtype
    is_muon_param: bool = False
    manual_prefix: str | None = None
    bucket_tag: str = "regular"  # 'regular' or 'expert'
    allreduce_grads_in_dp: bool = (
        False  # for small size of muon bucket or mtp shared embedding
    )


class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(
        self, module, accumulate_allreduce_grads_in_fp32, use_contiguous_buffers
    ):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        if self.use_contiguous_buffers:
            self._grad_buffers: dict[str, MemoryBuffer] = {}
            self._grad_buffer_param_index_map = {}
            data_parallel_world_size = PM.size_of("DP")

            # Simple function to define buffer type.
            def _get_buffer_type(param) -> ExtendedDtype:
                if hasattr(param, "extended_dtype"):
                    return param.extended_dtype

                # Determine if this is an expert parameter
                bucket_tag = (
                    "expert"
                    if getattr(param, "expert_model_parallel", False)
                    else "regular"
                )

                extended_dtype = ExtendedDtype(
                    torch_dtype=(
                        torch.float
                        if self.accumulate_allreduce_grads_in_fp32
                        else param.dtype
                    ),
                    is_muon_param=getattr(param, "is_muon_param", False),
                    manual_prefix=getattr(param, "manual_grad_bucket_prefix", None),
                    allreduce_grads_in_dp=getattr(
                        param, "allreduce_grads_in_dp", False
                    ),
                    bucket_tag=bucket_tag,
                )
                param.extended_dtype = extended_dtype
                return extended_dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            type_num_elements_lists = {}  # for muon
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = (
                        type_num_elements.get(dtype, 0) + param.data.nelement()
                    )

                    if dtype.is_muon_param:
                        from .muon_distributed_utils import (
                            update_type_num_elements_lists,
                        )

                        update_type_num_elements_lists(
                            dtype, type_num_elements_lists, param
                        )

            allocated_dp_rank_per_param = {}  # for muon
            type_num_elements_per_dp_rank = {}  # for muon
            # Allocate the buffer.
            idx = 0
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                world_size_for_dtype = (
                    PM.size_of("EDP")
                    if dtype.bucket_tag == "expert"
                    else data_parallel_world_size
                )
                num_elements_padded = world_size_for_dtype * int(
                    math.ceil(num_elements / world_size_for_dtype)
                )

                should_do_reduce = False
                if dtype.is_muon_param:
                    if num_elements_padded > 1.5 * 10**9:
                        # if True:
                        num_elements_padded = get_num_padded_elements(
                            dtype,
                            type_num_elements_lists,
                            allocated_dp_rank_per_param,
                            type_num_elements_per_dp_rank,
                            world_size_for_dtype,
                            num_elements_padded,
                        )
                    else:
                        should_do_reduce = True
                    num_elements = num_elements_padded

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(
                    num_elements,
                    num_elements_padded,
                    dtype.torch_dtype,
                    allreduce_grads_in_dp=dtype.allreduce_grads_in_dp
                    or should_do_reduce,
                )
                # Attach DP metadata directly on the grad buffer for later use
                dp_group = (
                    PM.group_of("EDP")
                    if dtype.bucket_tag == "expert"
                    else PM.group_of("DP")
                )
                dp_world_size = (
                    PM.size_of("EDP")
                    if dtype.bucket_tag == "expert"
                    else PM.size_of("DP")
                )
                dp_rank = (
                    PM.rank_in("EDP")
                    if dtype.bucket_tag == "expert"
                    else PM.rank_in("DP")
                )
                self._grad_buffers[dtype].data.dp_group = dp_group
                self._grad_buffers[dtype].data.dp_world_size = dp_world_size
                self._grad_buffers[dtype].data.dp_rank = dp_rank
                # Propagate reduction mode to the underlying tensor for later checks
                self._grad_buffers[dtype].data.allreduce_grads_in_dp = (
                    self._grad_buffers[dtype].allreduce_grads_in_dp
                )

                logger.info(
                    f"Gradbuffer {idx}: dtype: {dtype}, elements: {num_elements}, dp_world_size: {dp_world_size} "
                )
                idx += 1

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    if (
                        dtype.is_muon_param
                        and self._grad_buffers[dtype].numel_padded > 1.5 * 10**9
                    ):
                        world_size_for_dtype = (
                            PM.size_of("EDP")
                            if dtype.bucket_tag == "expert"
                            else data_parallel_world_size
                        )
                        update_grad_buffer_param_index_map(
                            dtype,
                            param,
                            type_num_elements_per_dp_rank,
                            allocated_dp_rank_per_param,
                            self._grad_buffer_param_index_map,
                            self._grad_buffers,
                            world_size_for_dtype,
                        )
                    else:
                        type_num_elements[dtype] -= param.data.nelement()
                        param.main_grad = self._grad_buffers[dtype].get(
                            param.data.shape, type_num_elements[dtype]
                        )
                        if dtype not in self._grad_buffer_param_index_map:
                            self._grad_buffer_param_index_map[dtype] = {}
                        self._grad_buffer_param_index_map[dtype][param] = (
                            type_num_elements[dtype],
                            type_num_elements[dtype] + param.data.nelement(),
                        )

            check_grad_buffer_param_index_map(self._grad_buffer_param_index_map)

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

    def _make_param_hook(self, param):
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

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, "buffers are not initialized."
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

    def broadcast_params(self):
        for param in self.module.parameters():
            # If separate expert data parallel group exists, broadcast expert params in EDP and others in DP
            is_expert = getattr(param, "expert_model_parallel", False)
            if is_expert and PM.size_of("EMP") > 1:
                torch.distributed.broadcast(
                    param.data,
                    src=PM.ranks_of("EDP")[0],
                    group=PM.group_of("EDP"),
                )
            else:
                torch.distributed.broadcast(
                    param.data,
                    src=PM.ranks_of("DP")[0],
                    group=PM.group_of("DP"),
                )

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                world = buffer_.data.dp_world_size
                group = buffer_.data.dp_group
                buffer_.data /= world
                torch.distributed.all_reduce(buffer_.data, group=group)
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= PM.size_of("DP")
                torch.distributed.all_reduce(coalesced, group=PM.group_of("DP"))
                for buf, synced in zip(
                    grads, _unflatten_dense_tensors(coalesced, grads)
                ):
                    buf.copy_(synced)

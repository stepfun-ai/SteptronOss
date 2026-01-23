# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from steptronoss.core.parallel_state import PM

from .utils import split_tensor_along_last_dim


def _reduce(input_, group: str = "TP"):
    """All-reduce the input tensor across model parallel group."""

    world_size = PM.size_of(group)
    dist_group = PM.group_of(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=dist_group)

    return input_


def _split_along_last_dim(input_, group: str = "TP"):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    world_size = PM.size_of(group)
    rank = PM.rank_in(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_, group: str = "TP"):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    world_size = PM.size_of(group)
    rank = PM.rank_in(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_, group: str = "TP"):
    """Gather tensors and concatinate along the last dimension."""

    world_size = PM.size_of(group)
    rank = PM.rank_in(group)
    dist_group = PM.group_of(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=dist_group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_, group: str = "TP"):
    """Gather tensors and concatinate along the first dimension."""
    world_size = PM.size_of(group)
    dist_group = PM.group_of(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=dist_group)

    return output


def _reduce_scatter_along_first_dim(input_, group: str = "TP"):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = PM.size_of(group)
    dist_group = PM.group_of(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=dist_group)
    return output


def split_along_first_dim_with_padding(input_, contiguous: bool = True, group: str = "TP"):
    world_size = PM.size_of(group)
    rank = PM.rank_in(group)
    if world_size == 1:
        return input_
    assert input_.size()[0] > (world_size - 1) ** 2

    dim_size = input_.size()[0]
    if dim_size % world_size == 0:
        padding_size = 0
        partition_size = dim_size // world_size
    else:
        padding_size = world_size - dim_size % world_size
        assert (dim_size + padding_size) % world_size == 0
        partition_size = (dim_size + padding_size) // world_size

    dim_offset = rank * partition_size
    if rank == world_size - 1 and padding_size > 0:
        pad_tensor_dim = list(input_.size())
        pad_tensor_dim[0] = padding_size
        pad_tensor = torch.empty(pad_tensor_dim, dtype=input_.dtype, device=input_.device)
        output = torch.cat([input_[dim_offset:], pad_tensor])
        if contiguous:
            output = output.contiguous()
    else:
        output = input_[dim_offset : dim_offset + partition_size]
        if contiguous:
            output = output.contiguous()

    return output, padding_size


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return input_

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=ctx.group), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _reduce(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        return _reduce(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _split_along_last_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return _split_along_last_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather_along_last_dim(grad_output, group=ctx.group),
            None,
        )


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _gather_along_last_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return _gather_along_last_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, group=ctx.group), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather_along_first_dim(grad_output, group=ctx.group),
            None,
        )


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True, group: str = "TP"):
        return _gather_along_first_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True, group: str = "TP"):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group

        return _gather_along_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        group = ctx.group

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(grad_output, group=group),
                None,
                None,
            )
        else:
            return (
                _split_along_first_dim(grad_output, group=group),
                None,
                None,
            )


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _reduce_scatter_along_first_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return _reduce_scatter_along_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output, group=ctx.group), None


class _SliceToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: str = "TP"):
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group: str = "TP"):
        ctx.group = group
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather_along_first_dim(grad_output, group=ctx.group),
            None,
        )


class SliceToSequenceParallelRegionRefactored(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_: torch.Tensor, group: str):
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _split_along_first_dim(input_, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather_along_first_dim(grad_output, group=ctx.group),
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_, group: str = "TP"):
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_, group: str = "TP"):
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_, group: str = "TP"):
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_, group: str = "TP"):
    return _GatherFromModelParallelRegion.apply(input_, group)


def scatter_to_sequence_parallel_region(input_, group: str = "TP"):
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True, group: str = "TP"):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad, group)


def reduce_scatter_to_sequence_parallel_region(input_, group: str = "TP"):
    return _ReduceScatterToSequenceParallelRegion.apply(input_, group)


def slice_to_sequence_parallel_region(input_, group: str = "TP"):
    return _SliceToSequenceParallelRegion.apply(input_, group)

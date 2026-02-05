# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import torch

from steptronoss.core.parallel_state import PM
from steptronoss.core.utils import divide


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> list[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False, use_expert_tp=False):
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Returns a Tensor or View with this rank's portion of the data.

    Arguments:
        tensor: The tensor to split

    Keyword Arguments:
        new_buffer (bool): If True, returns a new Tensor.
                           If False, returns a view into the existing Tensor.
                           Default is False
        use_expert_tp (bool): If True, use expert tensor parallel group.
                              If False, use regular tensor parallel group.
                              Default is False

    """
    if use_expert_tp:
        world_size = PM.size_of("ETP")
        rank = PM.rank_in("ETP")
    else:
        world_size = PM.size_of("TP")
        rank = PM.rank_in("TP")

    partition_size = torch.numel(tensor) // world_size
    start_index = partition_size * rank
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor, use_expert_tp=False):
    """Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
    model parallel ranks.

    Returns a new Tensor with the gathered data.

    Arguments:
        tensor: A Tensor or view of this rank's portion of the data.
        use_expert_tp (bool): If True, use expert tensor parallel group.
                              If False, use regular tensor parallel group.
                              Default is False
    """
    if use_expert_tp:
        world_size = PM.size_of("ETP")
        group = PM.group_of("ETP")
    else:
        world_size = PM.size_of("TP")
        group = PM.group_of("TP")

    numel_gathered = torch.numel(tensor) * world_size
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    # TODO: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed.all_gather_into_tensor(gathered, tensor, group=group)
    return gathered

"""Utility functions used throughout Megatron core"""

import operator
import random
from functools import reduce

import numpy as np
import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape {}, and new_data_tensor has shape {}.".format(
            "--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape
        ),
    )
    tensor.data = new_data_tensor


try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    multi_tensor_scale = amp_C.multi_tensor_scale
    multi_tensor_l2_norm = amp_C.multi_tensor_l2norm
except:

    def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
        """Multi tensor op applier"""
        return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)

    # computes l2 norm for a list of contiguous tensors
    # works as a drop-in replacement for amp_C.multi_tensor_l2norm
    def multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
        """
        Computes l2 norm for a list of contiguous tensors
        works as a drop-in replacement for amp_C.multi_tensor_l2norm
        """
        l2 = [[(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists]
        l2_reduced = torch.norm(torch.tensor(l2))
        l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device="cuda")
        return l2_cuda, None

    # works as a drop-in replacement for amp_C.multi_tensor_scale
    def multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
        """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
        for src, dst in zip(tensor_lists[0], tensor_lists[1]):
            dst.copy_(src * scale)


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    from torch.cuda import _lazy_call

    # newer PyTorch
    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb():
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def _get_rng_state():
    return (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
        torch.cuda.get_rng_state() if torch.cuda.device_count() else None,
    )


def _set_rng_state(state_tuple: tuple):
    torch.set_rng_state(state_tuple[0])
    random.setstate(state_tuple[1])
    np.random.set_state(state_tuple[2])
    if torch.cuda.device_count() > 0:
        _set_cuda_rng_state(state_tuple[3])


def _set_rng_seed(rng_seed: int):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)

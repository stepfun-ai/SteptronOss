"""Utility functions used throughout Megatron core"""

import operator
import os
import pickle
import time
from functools import reduce

import torch
from loguru import logger


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


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
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
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
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
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
        l2 = [
            [(torch.norm(tensor)) for tensor in tensor_list]
            for tensor_list in tensor_lists
        ]
        l2_reduced = torch.norm(torch.tensor(l2))
        l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device="cuda")
        return l2_cuda, None

    # works as a drop-in replacement for amp_C.multi_tensor_scale
    def multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
        """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
        for src, dst in zip(tensor_lists[0], tensor_lists[1]):
            dst.copy_(src * scale)


class MaxRetriesExceededError(Exception):
    """Custom exception raised when all retry attempts fail."""

    pass


def load_with_retry(file_path: str, retries: int = 3, delay: float = 0.1):
    """
    Tries to load a PyTorch file with a specified number of retries.

    This function checks for the file's existence on each attempt to handle
    filesystem delays. If all attempts fail, it raises an exception.

    Args:
        file_path (str): The path to the file.
        retries (int): The total number of attempts.
        delay (float): The delay in seconds between retries.

    Returns:
        The loaded object if successful.

    Raises:
        MaxRetriesExceededError: If all retry attempts fail due to file-not-found
                                 or loading errors.
    """
    last_exception = None
    for attempt in range(retries):
        # 1. Check for file existence *inside* the loop.
        if not os.path.isfile(file_path):
            logger.warning(
                f"Attempt {attempt + 1}/{retries}: File not found at '{file_path}'. Retrying..."
            )
            # We still wait before the next check.
            time.sleep(delay)
            continue  # Go to the next attempt

        # 2. If file exists, try to load it.
        try:
            return torch.load(file_path)
        except (IOError, EOFError, pickle.UnpicklingError, RuntimeError) as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1}/{retries} failed to load '{file_path}'. Error: {e}"
            )
            if attempt + 1 < retries:
                time.sleep(delay)

    # 3. After the loop, if we haven't returned, it means all attempts failed.
    #    Raise a comprehensive error.
    error_message = (
        f"All {retries} attempts failed for file '{file_path}'. "
        f"Last known error: {last_exception}"
    )
    raise MaxRetriesExceededError(error_message) from last_exception


def load_model_checkpoint(models, state_dict, strict_load_model=True):
    """Load model checkpoint."""
    from steptron.core.parallel_state import set_virtual_pipeline_model_parallel_rank
    from steptron.utils.utils import unwrap_model

    assert "model" in state_dict, "state_dict must contain 'model' key."

    if isinstance(state_dict["model"], str):
        load_type = "safetensors"
        from steptron.utils.weight_loader import HFWeights

        state_dict["model"] = HFWeights(state_dict["model"])
    else:
        load_type = "pt"

    for vid, model in enumerate(models):
        model = unwrap_model(model)
        msg = None
        set_virtual_pipeline_model_parallel_rank(vid)
        if load_type == "pt":
            msg = model.load_state_dict(
                state_dict["model"][vid], strict=strict_load_model
            )
        elif load_type == "safetensors":
            msg = model.load_hf_state_dict(
                state_dict["model"], strict=strict_load_model
            )
        else:
            raise NotImplementedError

        if msg is not None:
            if msg.missing_keys or msg.unexpected_keys:
                logger.warning(msg)
            else:
                # print "<All keys matched successfully>"
                logger.info(msg)
            # mark loaded. Error will be thrown if some parameter is not in ckpt when strict_load_model
            # is True. So we only need deal with missing_keys when strict_load_model is False.
            missing_keys_set = set(msg.missing_keys) if not strict_load_model else None
            for name, param in model.named_parameters():
                if strict_load_model or name not in missing_keys_set:
                    param.has_initialized = True

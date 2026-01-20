# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib
import os

import torch
from loguru import logger
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from steptronoss.core.parallel_state import PM
from steptronoss.core.utils import safely_set_viewless_tensor_data

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"
_EXPERT_MODEL_PARALLEL_RNG_TRACKER_NAME = "expert-model-parallel-rng"


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
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


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        yield
        # Update the current rng state for later use.
        self.states_[name] = torch.cuda.get_rng_state()
        # And set the state to the original state we started with.
        _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, kwargs, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.rng_states = _get_all_rng_states()
        ctx.kwargs = kwargs

        with torch.no_grad():
            outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
            )

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), " "please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )

        with _fork_rng():
            # Set the states to what it used to be before the forward pass.
            _set_all_rng_states(*ctx.rng_states)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
            del ctx.kwargs

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None) + grads


class CheckpointFunctionWithSanityCheck(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, kwargs, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_rng_state = _get_all_rng_states()
        ctx.kwargs = kwargs

        with torch.no_grad():
            outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
            )

        # Store everything.
        if isinstance(outputs, (tuple, list)):
            to_save = tuple(outputs) + tuple(args)
        else:
            to_save = (outputs,) + tuple(args)
        ctx.save_for_backward(*to_save)
        # ctx.save_for_backward(outputs, *args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), " "please use .backward() if possible")
        # fwd_outputs, *inputs = ctx.saved_tensors
        saved = ctx.saved_tensors
        fwd_outputs, inputs = saved[0], saved[1:]

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )

        # Store the current states.
        with _fork_rng():

            # Set the states to what it used to be before the forward pass.
            _set_all_rng_states(ctx.fwd_rng_state)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
            del ctx.kwargs

            if not torch.allclose(fwd_outputs, outputs, atol=1e-6):
                maxdiff = (fwd_outputs - outputs).abs().max()
                logger.warning(f"Checkpoint function with sanity check failed, diff: {maxdiff}")

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None) + grads


def checkpoint(function, distribute_saved_activations, *args, **kwargs):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            raise RuntimeError(f"Do not use keyword args for tensors that requires_grad!")

    if os.environ.get("RECOMPUTE_SANITY_CHECK", "0") == "1":
        return CheckpointFunctionWithSanityCheck.apply(function, distribute_saved_activations, kwargs, *args)
    return CheckpointFunction.apply(function, distribute_saved_activations, kwargs, *args)


def _get_all_rng_states():
    """Collect RNG states for CPU, CUDA, and tracked CUDA generators."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Restore RNG states captured via `_get_all_rng_states`."""
    torch.set_rng_state(cpu_rng_state)
    _set_cuda_rng_state(cuda_rng_state)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Context manager that restores RNG states when exiting."""
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        _set_all_rng_states(*current_states)


class CheckpointWithoutOutputFunction(torch.autograd.Function):
    """Autograd helper for output-discard checkpointing."""

    @staticmethod
    def forward(ctx, run_function, checkpoint_without_output_obj, *args):
        with torch.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*detach_variable(args))
        checkpoint_without_output_obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput:
    """Checkpoint helper that frees forward outputs and recomputes them during backward."""

    def __init__(self):
        self.run_function = None
        self.rng_states = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function, *args):
        self.run_function = run_function
        self.rng_states = _get_all_rng_states()
        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return outputs

    def _recompute(self, grad):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        with _fork_rng():
            _set_all_rng_states(*self.rng_states)
            with torch.enable_grad():
                outputs = self.run_function(*self.ctx.saved_tensors)

        self.run_function = None
        self.rng_states = None
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # restore the recomputed memory without changing the metadata
        with torch.no_grad():
            for output, recomputed in zip(self.outputs, outputs):
                output_size = recomputed.untyped_storage().size()
                output.untyped_storage().resize_(output_size)
                output.untyped_storage().copy_(recomputed.untyped_storage())

        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None
        return grad

    def discard_output_and_register_recompute(self, hook_tensor):
        if self.outputs is None:
            return
        for output in self.outputs:
            output.untyped_storage().resize_(0)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)

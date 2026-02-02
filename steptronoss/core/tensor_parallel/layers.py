import os
import warnings
from typing import Callable, Optional

import torch
from torch.nn import functional as F

from steptronoss.core.parallel_state import PM, get_global_memory_buffer
from steptronoss.utils.general import safediv

from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    split_along_first_dim_with_padding,
)
from .random import detach_variable

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
    "expert_model_parallel": False,
    "manual_grad_bucket_prefix": None,
    "is_muon_param": False,
    "merge_op": None,
    "sequence_parallel": False,
    "micro_dp": False,
    "shared": False,
    "_log_name": None,
}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel) or PM.i_am("TP", 0)


def param_is_not_expert_parallel_duplicate(param):
    return (hasattr(param, "expert_model_parallel") and param.expert_model_parallel) or PM.i_am("TP", 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


class SimpleVocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition, zero = divmod(self.num_embeddings, PM.size_of("TP"))
        assert zero == 0
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index = PM.rank_in("TP") * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize.
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

    # This is a compatibility implementation for weight initialization check. In the long term, all actual
    # initialization operations should be performed in the init_model_weight() method.
    def init_model_weight(self):
        self.weight.has_initialized = True

    def forward(self, input_: torch.IntTensor):
        if PM.size_of("TP") > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
        )
        # Mask the output embedding.
        if PM.size_of("TP") > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


def pre_function_backward(pre_func_output, grad_input, detached_inputs):
    # calculate the backward of pre-function
    if isinstance(pre_func_output, torch.Tensor):
        pre_func_output = (pre_func_output,)
    torch.autograd.backward(pre_func_output, [grad_input])
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return grads


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        is_column_parallel,
        use_moe,
    ):
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.is_column_parallel = is_column_parallel
        ctx.use_moe = use_moe
        if PM.size_of("TP") == 1:
            ctx.use_moe = False
            ctx.save_for_backward(input, weight)
            output = torch.matmul(input, weight.t())
            if bias is not None:
                output = output + bias
            return output
        if use_moe and is_column_parallel:
            # if smaller than (tp_size-1)^2, spliting may lead to outputs with different size
            # this only happens for very small inputs (e.g., seqlen <= 49 for 8 gpus)
            world_size = PM.size_of("TP")
            if input.size()[0] <= (world_size - 1) ** 2:
                ctx.moe_should_split_input = False
                ctx.save_for_backward(input, weight)
            else:
                ctx.moe_should_split_input = True
                moe_saved_input, ctx.moe_input_pad_size = split_along_first_dim_with_padding(input)
                # Note: must use clone() here, otherwise the original input
                # will be saved and thus you will see no memory usage reduction
                ctx.save_for_backward(moe_saved_input.clone(), weight)
        else:
            ctx.save_for_backward(input, weight)

        if sequence_parallel:
            world_size = PM.size_of("TP")
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed.all_gather_into_tensor(all_gather_buffer, input, group=PM.group_of("TP"))
            total_input = all_gather_buffer

            output = torch.matmul(total_input, weight.t())
        else:
            total_input = input
            output = torch.matmul(total_input, weight.t())

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        is_moe_gather_activation = ctx.use_moe and ctx.is_column_parallel and ctx.moe_should_split_input
        if not is_moe_gather_activation:
            input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        handle = None

        def get_grad_weight(total_input, grad_output, weight):
            # Doing gather + slicing during the NeMo forward pass can make this tensor
            # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
            # clones it if it's not contiguous:
            # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
            grad_output = grad_output.contiguous()
            if grad_output.dim() == 3:
                # Convert the tensor shapes to 2D for execution compatibility
                grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
                total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
            if ctx.gradient_accumulation_fusion:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)
            return grad_weight

        if PM.size_of("TP") == 1:
            grad_input = grad_output.matmul(weight)
            grad_weight = get_grad_weight(input, grad_output, weight)
            grad_bias = grad_output.sum(dim=0) if use_bias else None
            return (grad_input, grad_weight, grad_bias) + (None,) * 5

        handle = None

        if ctx.sequence_parallel:
            world_size = PM.size_of("TP")
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=PM.group_of("TP"),
                async_op=True,
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer

        else:
            if is_moe_gather_activation:
                # moe column parallel: try to overlap the re-gather activation
                # with the matmul(grad_out, weight) operation
                assert ctx.is_column_parallel
                moe_saved_input, weight = ctx.saved_tensors
                world_size = PM.size_of("TP")
                rank = PM.rank_in("TP")
                gathered_input_list = [torch.empty_like(moe_saved_input) for _ in range(world_size)]
                gathered_input_list[rank] = moe_saved_input
                gathered_input_handler = torch.distributed.all_gather(
                    gathered_input_list,
                    moe_saved_input,
                    group=PM.group_of("TP"),
                    async_op=True,
                )
            else:
                # non moe, non sequence parallel
                total_input = input

        grad_input = grad_output.matmul(weight)

        if is_moe_gather_activation:
            gathered_input_handler.wait()
            total_input = torch.cat(gathered_input_list, dim=0)
            if ctx.moe_input_pad_size > 0:
                total_input = total_input[: -ctx.moe_input_pad_size].contiguous()

        if ctx.sequence_parallel:
            if handle is not None:
                handle.wait()

        if ctx.async_grad_allreduce and not ctx.use_moe:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=PM.group_of("TP"), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            # reduce_scatter
            handle = torch.distributed.reduce_scatter_tensor(
                sub_grad_input,
                grad_input,
                group=PM.group_of("TP"),
                async_op=True,
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        grad_weight = get_grad_weight(total_input, grad_output, weight)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return (sub_grad_input, grad_weight, grad_bias) + (None,) * 5

        if ctx.async_grad_allreduce and not ctx.use_moe:
            handle.wait()

        return (grad_input, grad_weight, grad_bias) + (None,) * 5


class LinearWithGradAccumulationAndAsyncCommunicationWithPrefunction(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        is_column_parallel,
        use_moe,
        custom_pre_recompute_function,
        custom_pre_recompute_function_input,
    ):
        ctx.save_for_backward(input, weight, custom_pre_recompute_function_input)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.is_column_parallel = is_column_parallel
        ctx.use_moe = use_moe
        ctx.custom_pre_recompute_function = custom_pre_recompute_function
        assert custom_pre_recompute_function is not None

        if custom_pre_recompute_function_input is not None:
            input = custom_pre_recompute_function(input, custom_pre_recompute_function_input)
        else:
            input = custom_pre_recompute_function(input)

        if sequence_parallel:
            world_size = PM.size_of("TP")
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed.all_gather_into_tensor(all_gather_buffer, input, group=PM.group_of("TP"))
            total_input = all_gather_buffer

            output = torch.matmul(total_input, weight.t())
        else:
            total_input = input
            output = torch.matmul(total_input, weight.t())

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, custom_pre_recompute_function_input = ctx.saved_tensors
        use_bias = ctx.use_bias
        handle = None

        # Compute the forward pass.
        if custom_pre_recompute_function_input is not None:
            detached_inputs = detach_variable((input, custom_pre_recompute_function_input))
        else:
            detached_inputs = detach_variable((input,))  # must give a tuple as the input
        with torch.enable_grad():
            pre_func_output = ctx.custom_pre_recompute_function(*detached_inputs)

        input = pre_func_output

        def get_grad_weight(total_input, grad_output, weight):
            if ctx.gradient_accumulation_fusion:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)
            return grad_weight

        waiter = lambda: None

        if ctx.sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * PM.size_of("TP")

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=PM.group_of("TP"),
                async_op=True,
            )
            waiter = lambda: handle.wait()

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer

        else:
            total_input = input

        grad_input = grad_output.matmul(weight)

        waiter()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.async_grad_allreduce and not ctx.use_moe:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=PM.group_of("TP"), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation
            waiter = lambda: handle.wait()

        if ctx.sequence_parallel:
            if custom_pre_recompute_function_input is not None:
                raise NotImplementedError
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            # reduce_scatter
            handle = torch.distributed.reduce_scatter_tensor(
                sub_grad_input,
                grad_input,
                group=PM.group_of("TP"),
                async_op=True,
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation
            waiter = lambda: handle.wait()

        grad_weight = get_grad_weight(total_input, grad_output, weight)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        waiter()
        if ctx.sequence_parallel:
            return (sub_grad_input, grad_weight, grad_bias) + (None,) * 7

        grads = pre_function_backward(pre_func_output, grad_input, detached_inputs)
        if custom_pre_recompute_function_input is not None:
            custom_pre_recompute_function_input_grad = grads[1]
        else:
            custom_pre_recompute_function_input_grad = None
        return (grads[0], grad_weight, grad_bias) + (None,) * 6 + (custom_pre_recompute_function_input_grad,)


def linear_with_grad_accumulation_and_async_allreduce(
    input,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    is_column_parallel: bool,
    use_custom_tp_comm: bool = False,
    use_moe: bool = False,
    custom_pre_recompute_function: Callable = None,
    custom_pre_recompute_function_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel_enabled is True, this must be
        False, as no all reduce is performed.

    sequence_parallel_enabled (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1" and not use_custom_tp_comm:
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup (You can ignore this if using custom TP op)"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup (You can ignore this if using custom TP op)"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    if custom_pre_recompute_function is not None:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return LinearWithGradAccumulationAndAsyncCommunicationWithPrefunction.apply(
                input,
                weight,
                bias,
                gradient_accumulation_fusion,
                async_grad_allreduce,
                sequence_parallel_enabled,
                is_column_parallel,
                use_moe,
                custom_pre_recompute_function,
                custom_pre_recompute_function_input,
            )
    else:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return LinearWithGradAccumulationAndAsyncCommunication.apply(
                input,
                weight,
                bias,
                gradient_accumulation_fusion,
                async_grad_allreduce,
                sequence_parallel_enabled,
                is_column_parallel,
                use_moe,
            )


linear_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        gather_output=True,
        stride=1,
        skip_bias_add=False,
        async_tensor_model_parallel_allreduce=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        sequence_parallel_enabled: bool = False,
        use_moe: bool = False,
        weight_memory_loc=None,
        tie_word_embeddings_weight: torch.nn.Parameter = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        # Use ETP world size for MoE weights, otherwise fall back to TP
        # When expert_model_parallel_size > 1, ETP is separate from TP
        tp_world_size = PM.size_of("ETP") if use_moe else PM.size_of("TP")
        self.output_size_per_partition = safediv(output_size, tp_world_size)
        self.skip_bias_add = skip_bias_add
        self.use_moe = use_moe

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.

        if weight_memory_loc is None:
            self.weight = torch.nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
        else:
            self.weight = torch.nn.Parameter(weight_memory_loc)
        # tie weights for word embeddings
        self.weight = tie_word_embeddings_weight if tie_word_embeddings_weight is not None else self.weight

        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Allreduce for non-MoE TP gradients uses attention TP size
        tp_size_for_grads = PM.size_of("TP")
        self.async_tensor_model_parallel_allreduce = async_tensor_model_parallel_allreduce and tp_size_for_grads > 1
        if sequence_parallel_enabled:
            if tp_size_for_grads <= 1:
                warnings.warn(
                    f"`sequence_parallel_enabled` is set to `True`, but tensor model parallel size is {tp_size_for_grads}. "
                    f"Disabling sequence parallel."
                )
                sequence_parallel_enabled = False
        self.sequence_parallel_enabled = sequence_parallel_enabled

        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                raise RuntimeError(
                    "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                    "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                    "module is not found. To use gradient_accumulation_fusion you must "
                    "install APEX with --cpp_ext and --cuda_ext. For example: "
                    'pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." '
                    "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                    "gradient accumulation fusion."
                )
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` "
                "cannot be enabled at the same time."
            )

        if self.use_moe:
            assert not self.sequence_parallel_enabled
            if tp_world_size > 1:
                assert self.async_tensor_model_parallel_allreduce
            assert not bias
            setattr(self.weight, "expert_model_parallel", True)

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            is_column_parallel=True,
            use_moe=self.use_moe,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        input_is_parallel=False,
        skip_bias_add=False,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        sequence_parallel_enabled: bool = False,
        use_moe: bool = False,
        parallel_output: bool = False,
        custom_pre_recompute_function=None,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        # Use ETP world size for MoE weights, otherwise fall back to TP
        # When expert_model_parallel_size > 1, ETP is separate from TP
        tp_world_size = PM.size_of("ETP") if use_moe else PM.size_of("TP")
        self.input_size_per_partition = safediv(input_size, tp_world_size)
        self.skip_bias_add = skip_bias_add
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")
        self.use_moe = use_moe
        self.parallel_output = parallel_output
        self.custom_pre_recompute_function = custom_pre_recompute_function

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )
            setattr(self.bias, "sequence_parallel", sequence_parallel_enabled)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        if use_moe or parallel_output:
            assert not bias
            if use_moe:
                setattr(self.weight, "expert_model_parallel", True)

    def forward(self, input_, custom_pre_recompute_function_input=None):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]
            offloaded_list: a list contains all offloaded_list variables in this Transformerblock

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            is_column_parallel=False,
            use_moe=self.use_moe,
            custom_pre_recompute_function=self.custom_pre_recompute_function,
            custom_pre_recompute_function_input=custom_pre_recompute_function_input,
        )

        # if enable overlap, the reduce-scatter is already done in the previous op
        if (not self.parallel_output) and (not self.use_moe):
            # All-reduce across all the partitions.
            if self.sequence_parallel_enabled:
                output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
            else:
                output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

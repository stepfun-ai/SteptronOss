import torch

from steptronoss.utils.memory_tracker import CMT
from steptronoss.utils.optimizable import optimizable


@optimizable()
def histogram(top_k_rank: torch.Tensor, expert_num: int) -> torch.Tensor:
    """Count how many tokens route to each expert id.

    Negative indices are treated as invalid and ignored.

    Example:
        >>> top_k_rank = torch.tensor([0, 1, 1, 3, 3, 3], dtype=torch.int64)
        >>> expert_histogram(top_k_rank, expert_num=4).tolist()
        [1, 2, 0, 3]
    """
    # Count how many tokens route to each expert id.
    if not isinstance(top_k_rank, torch.Tensor):
        raise TypeError("top_k_rank must be a torch.Tensor")
    if top_k_rank.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_k_rank must be int32 or int64 tensor")
    if expert_num < 0:
        raise ValueError("expert_num must be non-negative")
    if expert_num == 0:
        return torch.zeros((0,), device=top_k_rank.device, dtype=torch.int32)

    if top_k_rank.numel() == 0:
        return torch.zeros((expert_num,), device=top_k_rank.device, dtype=torch.int32)

    flat = top_k_rank.reshape(-1)
    valid = flat >= 0
    if valid.numel() > 0:
        valid_any = valid.any()
        # Avoid scalar .item() to keep TorchDynamo graphs intact.
        max_val = flat.masked_fill(~valid, -1).amax()
        if not (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()):
            if bool(valid_any) and bool(max_val >= expert_num):
                raise ValueError("top_k_rank contains out-of-range expert index")
        torch._assert(
            torch.logical_or(~valid_any, max_val < expert_num),
            "top_k_rank contains out-of-range expert index",
        )

    # bincount returns int64; cast to int32 to match the C++ output.
    counts = torch.bincount(flat[valid].to(torch.int64), minlength=expert_num)
    return counts.to(torch.int32)


@optimizable()
def index_compute(indices: torch.Tensor, expert_histogram: torch.Tensor) -> torch.Tensor:
    """Compute stable scatter indices for expert-ordered buffers.

    This maps each (token, top_k) expert id to its position inside a
    concatenated expert buffer, preserving the original order of indices.

    Args:
        indices: Tensor of shape [token_num, top_k], int32 or int64.
        expert_histogram: Tensor of shape [num_experts], int32 or int64.

    Returns:
        scatter_index: Tensor of shape [token_num, top_k], int32.
    """
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if not isinstance(expert_histogram, torch.Tensor):
        raise TypeError("expert_histogram must be a torch.Tensor")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64 tensor")
    if expert_histogram.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_histogram must be int32 or int64 tensor")
    if indices.dim() != 2:
        raise ValueError("indices must be 2D [token_num, top_k]")
    if expert_histogram.dim() != 1:
        raise ValueError("expert_histogram must be 1D [num_experts]")
    if indices.device != expert_histogram.device:
        raise ValueError("indices and expert_histogram must be on the same device")

    num_experts = expert_histogram.numel()
    total_num = indices.numel()
    out = indices.to(torch.int32, copy=True)

    if total_num == 0:
        return out

    flat = indices.reshape(-1)
    valid = flat >= 0
    if valid.numel() > 0:
        valid_any = valid.any()
        max_val = flat.masked_fill(~valid, -1).amax()
        torch._assert(
            torch.logical_or(~valid_any, max_val < num_experts),
            "indices contains out-of-range expert index",
        )

    torch._assert(~(expert_histogram < 0).any(), "expert_histogram must be non-negative")
    torch._assert(
        expert_histogram.sum() == valid.sum(),
        "expert_histogram does not match number of valid indices",
    )
    if num_experts == 0:
        torch._assert(~valid.any(), "num_experts must be positive when indices are valid")

    hist = expert_histogram.to(torch.int64)
    base_offset = torch.cumsum(hist, dim=0) - hist

    positions = torch.arange(total_num, device=indices.device, dtype=torch.int64)
    valid_pos = positions[valid]
    valid_expert = flat[valid].to(torch.int64)

    sort_key = valid_expert * total_num + valid_pos
    order = torch.argsort(sort_key)
    sorted_pos = valid_pos[order]
    sorted_expert = valid_expert[order]

    idx = torch.arange(sorted_expert.numel(), device=indices.device, dtype=torch.int64)
    group_change = torch.ones_like(sorted_expert, dtype=torch.bool)
    group_change[1:] = sorted_expert[1:] != sorted_expert[:-1]
    group_start = torch.zeros_like(sorted_expert, dtype=torch.int64)
    group_start[group_change] = idx[group_change]
    group_start = torch.cummax(group_start, dim=0).values
    within = idx - group_start

    scatter_pos_sorted = base_offset[sorted_expert] + within

    flat_out = out.reshape(-1).to(torch.int64)
    flat_out[sorted_pos] = scatter_pos_sorted
    return flat_out.reshape_as(out).to(torch.int32)


class MoEWeightedGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        CMT.mark("moe_weighted_gather_forward_start")
        # Gather expert outputs back to token order with weights.
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("weight must be a floating tensor")
        if index.dim() != 2 or weight.dim() != 2:
            raise ValueError("index and weight must be 2D [token_num, top_k]")

        token_num, top_k = index.shape
        if weight.shape != index.shape:
            raise ValueError("index and weight must have the same shape")
        hidden_dim = input.shape[-1]

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        safe_index = flat_index.clamp_min(0).to(torch.int64)
        gathered = input.index_select(0, safe_index)
        gathered = gathered * valid.to(gathered.dtype).unsqueeze(-1)
        acc_dtype = torch.float32 if weight.dtype == torch.float32 else input.dtype
        if gathered.dtype != acc_dtype:
            gathered = gathered.to(acc_dtype)
        weight_flat = weight.reshape(-1, 1).to(acc_dtype)
        out = (gathered * weight_flat).reshape(token_num, top_k, hidden_dim).sum(dim=1)
        if out.dtype != input.dtype:
            out = out.to(input.dtype)
        CMT.mark("moe_weighted_gather_forward_end")

        ctx.save_for_backward(input, index, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        input, index, weight = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = input.shape[-1]

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        safe_index = flat_index.clamp_min(0).to(torch.int64)
        gathered = input.index_select(0, safe_index).reshape(token_num, top_k, hidden_dim)
        gathered = gathered * valid.reshape(token_num, top_k, 1).to(gathered.dtype)

        grad_weight = (grad_out.unsqueeze(1) * gathered).sum(dim=2)
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)

        grad_in = input.new_zeros(input.shape)
        grad_contrib = (grad_out.unsqueeze(1) * weight.to(grad_out.dtype).unsqueeze(-1)).reshape(-1, hidden_dim)
        grad_contrib = grad_contrib * valid.to(grad_contrib.dtype).unsqueeze(-1)
        grad_in.index_add_(0, safe_index, grad_contrib.to(grad_in.dtype))

        return grad_in, None, grad_weight


@optimizable()
def moe_weighted_gather(
    input: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Gather expert outputs back to token order with weights.

    Args:
        input: Expert outputs concatenated on dim 0, shape [num_expert_tokens, hidden_dim].
        index: Token-to-expert indices, shape [token_num, top_k]. Negative indices
            are treated as invalid and ignored.
        weight: Token-to-expert weights, shape [token_num, top_k].
    Example:
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> idx = torch.tensor([[2, 0], [1, 2]])
        >>> w = torch.tensor([[0.2, 0.8], [1.0, 0.5]])
        >>> moe_weighted_gather(x, idx, w).tolist()
        [[1.8, 2.8], [5.5, 7.0]]
    """
    return MoEWeightedGather.apply(input, index, weight)


class MoEScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # Scatter token inputs into expert-ordered buffer.
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if index.dim() != 2:
            raise ValueError("index must be 2D [token_num, top_k]")
        if input.dim() != 2:
            raise ValueError("input must be 2D [token_num, hidden_dim]")

        token_num, top_k = index.shape
        if input.shape[0] != token_num:
            raise ValueError("input and index must have the same token_num")

        hidden_dim = input.shape[-1]
        out_size = (index >= 0).sum()
        flat_index = index.reshape(-1)

        out = input.new_zeros((out_size, hidden_dim))
        if out_size == 0 or flat_index.numel() == 0:
            ctx.save_for_backward(index)
            ctx.hidden_dim = hidden_dim
            return out

        valid = flat_index >= 0
        flat_input = input.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
        if valid.all():
            out.index_copy_(0, flat_index.to(torch.int64), flat_input)
        else:
            flat_index = flat_index[valid].to(torch.int64)
            flat_input = flat_input[valid]
            if flat_index.numel() > 0:
                out.index_copy_(0, flat_index, flat_input)

        ctx.save_for_backward(index)
        ctx.hidden_dim = hidden_dim
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = ctx.hidden_dim

        if grad_out.numel() == 0:
            grad_input = grad_out.new_zeros((token_num, hidden_dim))
            return grad_input, None

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        if valid.all():
            gathered = grad_out.index_select(0, flat_index.to(torch.int64))
            grad_input = gathered.reshape(token_num, top_k, hidden_dim).sum(dim=1)
        else:
            grad_input = grad_out.new_zeros((token_num, hidden_dim))
            if valid.any():
                gathered = grad_out.index_select(0, flat_index[valid].to(torch.int64))
                tmp = grad_out.new_zeros((flat_index.numel(), hidden_dim))
                tmp[valid] = gathered
                grad_input = tmp.reshape(token_num, top_k, hidden_dim).sum(dim=1)

        return grad_input, None


@optimizable()
def moe_scatter(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Scatter token inputs into expert-ordered buffer.

    Args:
        input: Token inputs, shape [token_num, hidden_dim].
        index: Token-to-expert positions, shape [token_num, top_k]. Negative indices
            are treated as invalid and skipped.
    Example:
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> idx = torch.tensor([[0, 2], [1, 3]])
        >>> moe_scatter(x, idx).tolist()
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]
    """
    return MoEScatter.apply(input, index)


def nv_grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    from grouped_gemm import ops as grouped_gemm_ops

    xm, k = mat_a_flat.shape
    if xm == 0:
        b, m, n = mat_b.shape
        return mat_a_flat.new_zeros((xm, m if trans_b else n))
    return grouped_gemm_ops.gmm(mat_a_flat, mat_b, batch_sizes, trans_b=trans_b)


@optimizable(alternatives={"nv_grouped_gemm": nv_grouped_gemm})
def grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    batch_sizes_list = batch_sizes.tolist()
    outputs = []
    start = 0
    for i, size in enumerate(batch_sizes_list):
        rhs = mat_b[i].t() if trans_b else mat_b[i]
        outputs.append(mat_a_flat[start : start + size] @ rhs)
        start += size
    if outputs:
        return torch.cat(outputs, dim=0)
    return mat_a_flat.new_zeros((0, mat_b.shape[1] if trans_b else mat_b.shape[2]))

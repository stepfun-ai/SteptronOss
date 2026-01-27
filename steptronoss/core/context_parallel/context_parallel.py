import torch
import torch.nn.functional as F
from torch import distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.utils import lens_to_cum_len


def _gather_along_first_dim(input_):
    world_size = PM.size_of("CP")
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.zeros(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    dist.all_gather_into_tensor(output, input_.contiguous(), group=PM.group_of("CP"))
    return output


def _reduce_scatter_along_first_dim(input_):
    world_size = PM.size_of("CP")
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    dist.reduce_scatter_tensor(output, input_.contiguous(), group=PM.group_of("CP"))
    return output


class _GatherFromCPRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        res = _gather_along_first_dim(input_)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim(grad_output)


def cu_seqlens_to_balanced_cp(cu_seqlens: torch.IntTensor, cp_rank: int, cp_size: int):
    """This function parse cu_seqlens of raw sample, and generate
    [q_range, q_cu_seqlens, k_range, k_cu_seqlens] for left-part and right-part in ComplementCP.
    i.e.
    ```
    cu_seqlens = [0, 3, 5, 8] & cp_size=2
                          LEFT                                   RIGHT
          q_range q_cumlen   k_range k_cumlen     q_range q_cumlen   k_range k_cumlen
    cp0: [(0, 2), [0, 2]   , (0, 2), [0, 2]]   , [(6, 8), [0, 2],    (5, 8), [0, 3]   ]
    cp1: [(2, 4), [0, 1, 2], (0, 4), [0, 3, 4]], [(4, 6), [0, 1, 2], (3, 6), [0, 2, 3]]
    ```
    """
    seq_lens = cu_seqlens.diff().tolist()
    packed_len = cu_seqlens[-1].item()
    cu_seqlens_list = cu_seqlens.tolist()
    bcp_chunksize = cu_seqlens_list[-1] // (cp_size * 2)

    assert packed_len % cp_size == 0, "seqlen not divisible by cp size"
    assert sum(seq_lens) == packed_len, "incorrect cu_seqlens"

    def get_pieces(cum_lens: list[int], split: int) -> tuple[list[list[int]], list[list[int]]]:
        """[0, 3, 8] -> [[3, 1], [4]], [[0, 1], [1]]
        cu_seqlens -> piece_sizes, piece_ids
        """
        sample_boundries = set(cum_lens)
        cp_boundries = set([(i + 1) * cum_lens[-1] // split for i in range(split)])
        boundries = sorted(sample_boundries | cp_boundries)

        cp_lens, piece_id = [[]], [[]]
        cur_id = 0
        for i in range(1, len(boundries)):
            cp_lens[-1].append(boundries[i] - boundries[i - 1])
            piece_id[-1].append(cur_id)
            if boundries[i] in sample_boundries:
                cur_id += 1
            if boundries[i] in cp_boundries:
                cp_lens.append([])
                piece_id.append([])
        return cp_lens[:-1], piece_id[:-1]

    piece_lens, piece_ids = get_pieces(cu_seqlens_list, split=cp_size * 2)

    def get_slices_for_rank(rank: int):
        q_lens, q_ids = piece_lens[rank], piece_ids[rank]
        cp_left_boundary, cp_right_boundary = (
            rank * bcp_chunksize,
            rank * bcp_chunksize + bcp_chunksize,
        )
        first_piece_left_boundary = cu_seqlens_list[q_ids[0]]
        assert first_piece_left_boundary <= cp_left_boundary

        q_range = cp_left_boundary, cp_right_boundary
        cu_seqlens_q = lens_to_cum_len(q_lens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        max_len_q = max(q_lens)  # this is max_len of actual piece, count before make negative
        cu_seqlens_q[0] += first_piece_left_boundary - cp_left_boundary  # make it negative, for PosEmb

        k_range = first_piece_left_boundary, cp_right_boundary
        # Retrieve the full length of the first piece
        q_lens[0] += cp_left_boundary - first_piece_left_boundary
        cu_seqlens_k = lens_to_cum_len(q_lens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        max_len_k = max(q_lens)
        # cu_seqlens_k[0] += first_piece_left_boundary - first_piece_left_boundary # nothing happens for this
        return q_range, cu_seqlens_q, max_len_q, k_range, cu_seqlens_k, max_len_k

    left_rank, right_rank = cp_rank, cp_size * 2 - cp_rank - 1
    return get_slices_for_rank(left_rank), get_slices_for_rank(right_rank)


def scatter_to_balanced_cp_region(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    """Given a tensor with sequence number [0, 1, 2, 3, 4, 5, 6, 7], CP=2, TP=2
    CP partition: CP0: [0, 1, 6, 7], CP1: [2, 3, 4, 5]
    """
    cp_size = PM.size_of("CP")
    cp_rank = PM.rank_in("CP")
    tp_size = PM.size_of("TP")
    if cp_size == 1:
        return tensor

    # Split along the first dimension: the sequence dimension.
    S = tensor.shape[dim]
    assert S % (2 * cp_size * tp_size) == 0, (
        f"For the {dim}-th dimension of the tensor, size ({S}) should be divisible "
        f"by 2 * context parallel size ({cp_size}) x sequence parallel size({tp_size})"
    )
    chunk_size = S // (cp_size * 2)

    left_start = cp_rank * chunk_size
    right_start = (2 * cp_size - 1 - cp_rank) * chunk_size

    # Create indices
    left = torch.arange(left_start, left_start + chunk_size)
    right = torch.arange(right_start, right_start + chunk_size)
    indices = torch.cat((left, right), dim=0).to(tensor.device)
    # Use index_select to gather the elements
    output = torch.index_select(tensor, dim=dim, index=indices).contiguous()

    return output


def gather_from_balanced_cp_region(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    """For balanced cp, tensor's shape = [S/CP, B, H, D]
    NOTE: Assume `tensor` has been reduced from the SP region
    ```zhy:
    tensor: [0, 1, 6, 7], [2, 3, 4, 5]
    inp   : [0, 1, 6, 7, 2, 3, 4, 5]
            inp0   inp1（↑↓）
          : [0, 1] [6, 7]
          : [2, 3] [4, 5]
    res   : [0, 1, 2, 3, 4, 5, 6, 7]
    ```
    """
    if dim != 0:
        tensor = tensor.transpose(0, dim)
    inp: torch.Tensor = _GatherFromCPRegion.apply(tensor)  # shape = [S, B, H, D]

    cp_size = PM.size_of("CP")
    S_local = tensor.shape[0]
    raw_shape = tensor.shape[1:]
    # dim_size = list(tensor.size())  # dim_size = [S/CP, B, H, D]
    inp0, inp1 = inp.reshape(cp_size, S_local, *raw_shape).chunk(2, dim=1)

    inp1 = inp1.flip(0)
    res = torch.cat((inp0, inp1), dim=0).reshape(S_local * cp_size, *raw_shape)
    if dim != 0:
        res = res.transpose(0, dim)

    return res

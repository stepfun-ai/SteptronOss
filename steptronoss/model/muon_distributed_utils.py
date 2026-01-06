def update_type_num_elements_lists(dtype, type_num_elements_lists, param):
    if dtype not in type_num_elements_lists:
        type_num_elements_lists[dtype] = {}
    type_num_elements_lists[dtype][param] = param.data.nelement()


def get_num_padded_elements(
    dtype,
    type_num_elements_lists,
    allocated_dp_rank_per_param,
    type_num_elements_per_dp_rank,
    data_parallel_world_size,
    base_num_elements_padded,
):
    # Sort type_num_elements_lists[dtype] and also get the index
    from collections import OrderedDict

    sorted_type_num_elements_lists = OrderedDict(
        sorted(type_num_elements_lists[dtype].items(), key=lambda x: x[1])
    )

    allocated_num_elements_per_dp_rank = {
        dp_rank: {} for dp_rank in range(data_parallel_world_size)
    }
    if dtype not in allocated_dp_rank_per_param:
        allocated_dp_rank_per_param[dtype] = {}
    if dtype not in type_num_elements_per_dp_rank:
        type_num_elements_per_dp_rank[dtype] = []
    for idx, (param, num_elements) in enumerate(sorted_type_num_elements_lists.items()):
        # store the num_elements of the param in the corresponding dp rank, and the allocated dp rank of params
        allocated_num_elements_per_dp_rank[idx % data_parallel_world_size][
            param
        ] = num_elements
        allocated_dp_rank_per_param[dtype][param] = idx % data_parallel_world_size

    type_num_elements_per_dp_rank[dtype] = [
        sum(x.values()) for x in allocated_num_elements_per_dp_rank.values()
    ]

    num_elements_padded = data_parallel_world_size * max(
        type_num_elements_per_dp_rank[dtype]
    )
    print(
        "mem overhead of zero-resharding",
        num_elements_padded / base_num_elements_padded,
    )
    return num_elements_padded


def update_grad_buffer_param_index_map(
    dtype,
    param,
    type_num_elements_per_dp_rank,
    allocated_dp_rank_per_param,
    grad_buffer_param_index_map,
    grad_buffers,
    data_parallel_world_size,
):
    # locate the dp rank of the param
    allocated_dp_rank = allocated_dp_rank_per_param[dtype][param]
    # get the start_index and ranges of the corresponding grad buffer
    type_num_elements_per_dp_rank[dtype][allocated_dp_rank] -= param.data.nelement()
    offset_of_allocated_dp = (
        grad_buffers[dtype].numel_padded // data_parallel_world_size
    ) * allocated_dp_rank

    param.main_grad = grad_buffers[dtype].get(
        param.data.shape,
        type_num_elements_per_dp_rank[dtype][allocated_dp_rank]
        + offset_of_allocated_dp,
    )
    if dtype not in grad_buffer_param_index_map:
        grad_buffer_param_index_map[dtype] = {}

    grad_buffer_param_index_map[dtype][param] = (
        type_num_elements_per_dp_rank[dtype][allocated_dp_rank]
        + offset_of_allocated_dp,
        type_num_elements_per_dp_rank[dtype][allocated_dp_rank]
        + offset_of_allocated_dp
        + param.data.nelement(),
    )


def check_grad_buffer_param_index_map(grad_buffer_param_index_map):
    # Check that the start/end indices of all params do not overlap for all dtypes
    for dtype, param_indices in grad_buffer_param_index_map.items():
        indices = []

        for param, (start, end) in param_indices.items():
            indices.append((start, end, param))

        # Sort by start index using the start index value, not the param (which may be a tensor)
        indices.sort(key=lambda x: x[0])
        for i in range(1, len(indices)):
            prev_end = indices[i - 1][1]
            curr_start = indices[i][0]
            if curr_start < prev_end:
                raise ValueError(
                    f"Overlapping grad buffer indices for {i}/{len(indices)} params: "
                    f"[{indices[i-1][0]}, {indices[i-1][1]}) and "
                    f"[{indices[i][0]}, {indices[i][1]})"
                )
        # print(f"{dtype} is ok!")

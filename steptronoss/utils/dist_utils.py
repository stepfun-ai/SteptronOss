import io
import math
import pickle
from dataclasses import is_dataclass
from typing import List, Tuple, TypeVar

T = TypeVar("T")

import numpy as np
import torch
import torch.distributed
from configurize import DataClass

from steptronoss.core.parallel_state import PM

from .general import list_split, list_split_T, recur_to

UNSUPPORTED_DTYPE = -2
dtype_to_id = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
    None: -1,  # 特殊值表示None
}
id_to_dtype = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.int64,
    4: torch.int32,
    5: torch.int16,
    6: torch.int8,
    7: torch.uint8,
    8: torch.bool,
    -1: None,  # None类型
}
_dtype_to_id = lambda dtype: dtype_to_id.get(dtype, UNSUPPORTED_DTYPE)  # -2表示未知类型
_id_to_dtype = lambda id: id_to_dtype.get(id, UNSUPPORTED_DTYPE)  # -2表示未知类型


def object_to_tensor(obj, device):
    f = io.BytesIO()
    pickle.Pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return pickle.Unpickler(io.BytesIO(buf)).load()


def _get_tensor_info(tensors):
    if isinstance(tensors, type):
        return tensors
    if isinstance(tensors, dict):
        return {k: _get_tensor_info(v) for k, v in tensors.items()}
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)([_get_tensor_info(i) for i in tensors])
    elif isinstance(tensors, DataClass) or is_dataclass(tensors):
        return ("@@dataclass", tensors.__class__, _get_tensor_info(tensors.__dict__))
    elif isinstance(tensors, torch.Tensor):
        return ("@@tensor_info", tensors.shape, tensors.dtype, tensors.is_cuda)
    else:
        return tensors


def _convert_tensor(tensors, rank, src_rank, inp_tensor_obj: T, group, move_to_cuda=False) -> T:
    if isinstance(tensors, dict):
        new_obj = {
            k: _convert_tensor(
                v,
                rank,
                src_rank,
                inp_tensor_obj[k] if rank == src_rank else None,
                group,
                move_to_cuda,
            )
            for k, v in tensors.items()
        }
        if rank == src_rank and not move_to_cuda:
            return inp_tensor_obj
        else:
            return new_obj
    elif isinstance(tensors, tuple) and tensors[0] == "@@tensor_info":
        _, shape, dtype, is_cuda = tensors
        if rank != src_rank:
            _tensor = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        else:
            # _tensor = inp_tensor_obj.to(torch.cuda.current_device(), non_blocking=True)
            _tensor = inp_tensor_obj.to(torch.cuda.current_device(), non_blocking=True).contiguous()
        torch.distributed.broadcast(_tensor, src=src_rank, group=group)
        if not is_cuda and not move_to_cuda:
            _tensor = _tensor.to("cpu")
        return _tensor
    elif isinstance(tensors, tuple) and tensors[0] == "@@dataclass":
        new_obj = tensors[1](
            **_convert_tensor(
                tensors[2],
                rank,
                src_rank,
                inp_tensor_obj.__dict__ if rank == src_rank else None,
                group,
                move_to_cuda,
            )
        )
        if rank == src_rank and not move_to_cuda:
            return inp_tensor_obj
        else:
            return new_obj
    elif isinstance(tensors, (list, tuple)):
        new_obj = type(tensors)(
            [
                _convert_tensor(
                    value,
                    rank,
                    src_rank,
                    inp_tensor_obj[idx] if rank == src_rank else None,
                    group,
                    move_to_cuda,
                )
                for idx, value in enumerate(tensors)
            ]
        )
        if rank == src_rank and not move_to_cuda:
            return inp_tensor_obj
        else:
            return new_obj
    else:
        return tensors


def broadcast_tensors(inp_tensor_obj: T, src_rank, group, move_to_cuda=False, direct=False) -> T:
    """This function work just like broadcast_object_list, with acceleration for tensors
    and won't cause device mismatch.
    from huhp: This function broadcast any object and accelerate Tensor broadcast
    Fix by zhy: src_rank上的object都**不会**被重新创建。
    """
    # transfer tensor meta info
    rank = torch.distributed.get_rank()
    if direct:
        tensor_info = [inp_tensor_obj]
        torch.distributed.broadcast_object_list(tensor_info, src=src_rank, group=group)
        inp_tensor_obj = tensor_info[0]
        return inp_tensor_obj
    if rank == src_rank:
        info = _get_tensor_info(inp_tensor_obj)
        tensor_info = [info]
    else:
        tensor_info = [None]

    torch.distributed.broadcast_object_list(tensor_info, src=src_rank, group=group)
    tensor_info = tensor_info[0]

    ret_tensor_dict = _convert_tensor(tensor_info, rank, src_rank, inp_tensor_obj, group, move_to_cuda)

    return ret_tensor_dict


def all_to_all_tensors(tensors: list[torch.Tensor], group, dtype, device, async_op=False):
    group_size = torch.distributed.get_world_size(group)
    group_rank = torch.distributed.get_rank(group)
    assert len(tensors) == group_size
    tensor_info = [t.shape for t in tensors]
    all_tensor_infos = [None] * group_size
    torch.distributed.all_gather_object(all_tensor_infos, tensor_info, group=group)
    recv_tensors = []
    for info in all_tensor_infos:
        shape = info[group_rank]
        recv_tensors.append(torch.empty(shape, dtype=dtype, device=device))
    handle = torch.distributed.all_to_all(recv_tensors, tensors, group=group, async_op=async_op)
    if async_op:
        return handle, recv_tensors
    return recv_tensors


def all_to_all_objects(objects: list, group=None):

    device = "cuda"
    group_size = torch.distributed.get_world_size(group)
    group_rank = torch.distributed.get_rank(group)
    assert len(objects) == group_size
    tensor_info, datas = [], []
    for obj in objects:
        data = object_to_tensor(obj, device)[0]
        tensor_info.append(data.numel())
        datas.append(data)
    all_tensor_infos = [None] * group_size
    torch.distributed.all_gather_object(all_tensor_infos, tensor_info, group=group)
    recv_tensors = []
    for info in all_tensor_infos:
        shape = info[group_rank]
        recv_tensors.append(torch.empty(shape, dtype=torch.uint8, device=device))

    torch.distributed.all_to_all(recv_tensors, datas, group=group)

    objects = [tensor_to_object(data, len(data)) for data in recv_tensors]

    return objects


def dict_to_tensor(obj: dict[str, torch.Tensor], comm_device: torch.device) -> torch.Tensor:
    """
    将一个包含各种torch.Tensor的字典打包成一个单一的torch.uint8张量，优化内存使用。
    这个张量包含了所有元数据（键、原始dtype、原始shape、原始device）和实际的张量数据。
    最终张量直接在 comm_device 上构建。

    结构图:
    +----------------------------------------------+
    |   Single torch.uint8 Tensor (on comm_device) |
    +----------------------------------------------+
    | +------------------------------------------+ |
    | |         Header (16 bytes, uint8)         | |
    | |------------------------------------------| |
    | | len_pickled_meta (uint64, 8 bytes)       | |
    | |------------------------------------------| |
    | | len_all_data_bytes (uint64, 8 bytes)     | |
    | +------------------------------------------+ |
    | +------------------------------------------+ |
    | |  Pickled Overall Metadata (uint8)        | |
    | |  (length = value of len_pickled_meta)    | |
    | |------------------------------------------| |
    | | bytes( pickle.dumps([                    | |
    | | (key1, dtype_id1, shape1_list, dev_str1),| |
    | | ...                                      | |
    | | (keyN, dtype_idN, shapeN_list, dev_strN) | |
    | | ]) )                                     | |
    | +------------------------------------------+ |
    | +------------------------------------------+ |
    | |  Concatenated Data Bytes (uint8)         | |
    | |  (length = value of len_all_data_bytes)  | |
    | |------------------------------------------| |
    | | Tensor1_as_bytes (uint8)                 | |
    | | ...                                      | |
    | | TensorN_as_bytes (uint8)                 | |
    | +------------------------------------------+ |
    +----------------------------------------------+

    Args:
        obj: 待打包的字典，键为字符串，值为torch.Tensor。Tensor可能在CPU或GPU上。
        comm_device: 最终打包好的uint8张量所在的通信设备 (通常是GPU)。

    Returns:
        一个torch.uint8类型的扁平化张量 (在comm_device上)，包含了所有信息。
    """
    overall_metadata_list = []
    data_uint8_views_on_comm_device = []  # 存储已在comm_device上的uint8数据视图/副本

    # 1. 处理每个Tensor：收集元数据，准备其在comm_device上的uint8表示
    for key, tensor_val in obj.items():
        if not isinstance(tensor_val, torch.Tensor):
            raise TypeError(
                f"All values in the dictionary must be torch.Tensor. Found type {type(tensor_val)} for key '{key}'"
            )

        original_device_str = str(tensor_val.device)
        if original_device_str != "cpu":
            original_device_str = "cuda"  # ignore GPU index, e.g. "cuda:0" -> "cuda"
        original_dtype_id = _dtype_to_id(tensor_val.dtype)
        if original_dtype_id == UNSUPPORTED_DTYPE:
            raise ValueError(f"Unsupported dtype {tensor_val.dtype} for key '{key}'")

        original_shape_list = list(tensor_val.shape)

        overall_metadata_list.append((key, original_dtype_id, original_shape_list, original_device_str))

        # 确保Tensor是连续的，以进行可靠的view操作
        tensor_contig = tensor_val.contiguous()

        # 获取扁平化的uint8视图
        # view()是浅拷贝，不改变数据位置。flatten()也是浅拷贝。
        uint8_view_on_source_device = tensor_contig.view(torch.uint8).flatten()

        if uint8_view_on_source_device.device == comm_device:
            data_uint8_views_on_comm_device.append(uint8_view_on_source_device)
        else:
            # 如果不在通信设备上，则将其uint8视图复制到通信设备
            # non_blocking=True 可以在有独立复制队列的硬件上提供性能优势
            data_uint8_views_on_comm_device.append(uint8_view_on_source_device.to(comm_device, non_blocking=True))

        # 及时释放对原始tensor_val 和 tensor_contig 的引用，如果它们很大且不再需要
        del tensor_val
        del tensor_contig

    # 2. 序列化元数据 (在CPU上完成，结果很小，然后复制到comm_device)
    pickled_metadata_bytes = pickle.dumps(overall_metadata_list)
    # torch.frombuffer 会创建一个CPU tensor
    pickled_metadata_tensor_cpu = torch.frombuffer(pickled_metadata_bytes, dtype=torch.uint8)
    pickled_metadata_on_comm_device = pickled_metadata_tensor_cpu.to(comm_device, non_blocking=True)
    del pickled_metadata_tensor_cpu  # 释放CPU副本

    # 3. 创建头部 (在CPU上完成，结果很小，然后复制到comm_device)
    len_meta_bytes = len(pickled_metadata_on_comm_device)
    len_data_bytes = sum(t.numel() for t in data_uint8_views_on_comm_device)

    header_tensor_int64_cpu = torch.tensor([len_meta_bytes, len_data_bytes], dtype=torch.int64, device="cpu")
    # .numpy().tobytes() 是获取底层字节的标准方式
    header_uint8_tensor_cpu = torch.frombuffer(header_tensor_int64_cpu.numpy().tobytes(), dtype=torch.uint8)
    header_on_comm_device = header_uint8_tensor_cpu.to(comm_device, non_blocking=True)
    del header_tensor_int64_cpu, header_uint8_tensor_cpu  # 释放CPU副本

    # 4. 在comm_device上拼接所有部分
    # 注意：torch.cat的输入是一个tensor列表
    all_parts = [
        header_on_comm_device,
        pickled_metadata_on_comm_device,
    ] + data_uint8_views_on_comm_device
    final_packed_tensor = torch.cat(all_parts)

    # 考虑清理中间列表和张量，以帮助内存管理（尽管Python GC会处理）
    del overall_metadata_list, data_uint8_views_on_comm_device
    del pickled_metadata_on_comm_device, header_on_comm_device, all_parts
    # 对于非常大的操作，有时会手动调用torch.cuda.empty_cache()，但需谨慎，因为它会阻塞。
    # if comm_device.type == 'cuda':
    #     torch.cuda.synchronize(comm_device) #确保所有拷贝完成
    #     torch.cuda.empty_cache()

    return final_packed_tensor


def tensor_to_dict(flat_byte_tensor_on_comm_device: torch.Tensor, keep_on_comm_device: bool = False) -> dict:
    """
    将由 dict_to_tensor 打包的单一torch.uint8张量解包还原为原始的字典，优化内存使用。
    扁平张量在 comm_device (通常是GPU) 上。如果 keep_on_comm_device=True，则恢复的张量会留在通信设备上，否则会根据原始设备字符串恢复到CPU或GPU。

    Args:
        flat_byte_tensor_on_comm_device: 由 dict_to_tensor 生成的torch.uint8张量，位于通信设备上。
        keep_on_comm_device: 是否将恢复的张量留在通信设备上，减少显存/内存footprint的优化选项。

    Returns:
        一个字典，其结构和内容与原始打包前的字典相同。
    """
    comm_device = flat_byte_tensor_on_comm_device.device
    # 1. 解析头部 (从comm_device复制小块头部到CPU进行解析)
    header_on_comm_device = flat_byte_tensor_on_comm_device[:16]
    header_cpu = header_on_comm_device.cpu()  # 小块数据，复制成本低
    # .numpy().tobytes() 更安全，因其返回的numpy数组是可写的，frombuffer需要这个
    header_numpy_int64 = np.frombuffer(header_cpu.numpy().tobytes(), dtype=np.int64)
    len_pickled_meta = int(header_numpy_int64[0])
    # len_all_data_bytes = int(header_numpy_int64[1]) # 可用于校验或直接切片数据块
    del header_on_comm_device, header_cpu  # 释放小张量

    # 2. 提取并反序列化元数据 (从comm_device复制小块元数据到CPU进行unpickle)
    meta_start_idx = 16
    meta_end_idx = meta_start_idx + len_pickled_meta
    pickled_metadata_on_comm_device = flat_byte_tensor_on_comm_device[meta_start_idx:meta_end_idx]
    # .numpy().tobytes() 同样是为了frombuffer
    pickled_metadata_bytes_cpu = pickled_metadata_on_comm_device.cpu().numpy().tobytes()
    overall_metadata_list = pickle.loads(pickled_metadata_bytes_cpu)
    del pickled_metadata_on_comm_device, pickled_metadata_bytes_cpu  # 释放小张量

    # 3. 数据块在comm_device上的起始索引
    data_block_start_idx = meta_end_idx

    reconstructed_dict = {}
    current_data_offset_in_data_block = 0

    # 4. 逐个重建张量
    for key, dtype_id, shape, original_device_str in overall_metadata_list:
        original_dtype = _id_to_dtype(dtype_id)
        if original_dtype == UNSUPPORTED_DTYPE:  # 在_id_to_dtype中未找到
            raise ValueError(f"Unknown dtype_id {dtype_id} for key '{key}' during deserialization.")

        target_device = comm_device if keep_on_comm_device else torch.device(original_device_str)

        # 计算当前Tensor的字节数
        num_elements = np.prod(shape) if shape else 1  # 标量shape=[] prod为1
        num_bytes_for_current_tensor = 0
        if num_elements > 0:
            # 使用一个临时CPU标量来安全获取element_size
            _temp_scalar_cpu = torch.empty((), dtype=original_dtype, device="cpu")
            element_size = _temp_scalar_cpu.element_size()
            del _temp_scalar_cpu
            # 针对某些PyTorch版本或特殊dtype的element_size可能为0的情况做兼容
            if element_size == 0 and original_dtype != torch.bool:
                try:
                    element_size = torch.tensor([], dtype=original_dtype).itemsize
                except RuntimeError:
                    raise ValueError(f"Could not determine element size for dtype {original_dtype}")
            num_bytes_for_current_tensor = num_elements * element_size

        # 从comm_device上的扁平张量中获取当前Tensor的字节数据视图 (仍然在comm_device上)
        # 这是一个视图操作，非常轻量
        current_tensor_bytes_on_comm_device_view = flat_byte_tensor_on_comm_device[
            data_block_start_idx
            + current_data_offset_in_data_block : data_block_start_idx
            + current_data_offset_in_data_block
            + num_bytes_for_current_tensor
        ]

        # 根据target_device决定如何重建
        current_reconstruction_shape = shape if shape else ()  # torch.empty需要()表示标量

        if num_bytes_for_current_tensor == 0:  # 处理0字节Tensor (例如 shape=[0, N] 或特定空标量)
            reconstructed_tensor = torch.empty(current_reconstruction_shape, dtype=original_dtype, device=target_device)
        else:
            # 直接在目标设备上创建最终的空Tensor
            reconstructed_tensor = torch.empty(current_reconstruction_shape, dtype=original_dtype, device=target_device)

            if reconstructed_tensor.numel() > 0:  # 只有当目标Tensor有元素时才执行复制
                # 获取目标Tensor存储的uint8视图
                target_uint8_view = reconstructed_tensor.view(torch.uint8).view(-1)

                # 将源字节数据(在comm_device上)复制到目标uint8视图
                # 如果target_device与comm_device不同，.to(target_device)会执行一次数据拷贝
                # 否则，如果相同，则是一次设备内拷贝
                # :target_uint8_view.numel()确保只复制所需字节数
                source_bytes_for_copy = current_tensor_bytes_on_comm_device_view.to(target_device, non_blocking=True)
                target_uint8_view.copy_(source_bytes_for_copy[: target_uint8_view.numel()])
                del source_bytes_for_copy  # 如果发生了拷贝，释放这个中间拷贝

        reconstructed_dict[key] = reconstructed_tensor
        current_data_offset_in_data_block += num_bytes_for_current_tensor
        del current_tensor_bytes_on_comm_device_view  # 释放视图

    # 考虑清理对输入大张量的引用
    del flat_byte_tensor_on_comm_device
    # if torch.cuda.is_available(): # 如果使用了CUDA
    #     torch.cuda.empty_cache() # 谨慎使用

    return reconstructed_dict


def all_to_all_dicts(objs: list[dict[str, torch.Tensor]], group, device):
    comm_dtype = torch.uint8  # `dtype` from arg is ignored
    obj_tensors = [dict_to_tensor(obj, comm_device=device) for obj in objs]
    recv_tensors = all_to_all_tensors(obj_tensors, group=group, dtype=comm_dtype, device=device)
    objs = [tensor_to_dict(tensor, keep_on_comm_device=True) for tensor in recv_tensors]
    return objs


def redistributed_tp(
    splited_state_dict: list[dict[str, torch.Tensor]],
    dtype,
    device,
    dp_group=None,
) -> dict[str, torch.Tensor]:
    # ranks:
    # input:
    # 0 [[A ], [A1], [], []]
    # 1 [[B ], [B1], [], []]
    # 2 [[], [], [C ], [C1]]
    # 3 [[], [], [D ], [D1]]
    # output:
    # 0 [[A ], [B ], [], []]
    # 1 [[A1], [B1], [], []]
    # 2 [[], [], [C ], [D ]]
    # 3 [[], [], [C1], [D1]]
    group = None  # in world
    world_size = torch.distributed.get_world_size(group)

    tgt_tp_size = len(splited_state_dict)
    tgt_dp_size = world_size // tgt_tp_size

    if dp_group is None:
        src_dp_size = PM.size_of("DP")
        my_dp_rank = PM.rank_in("DP")
    else:
        src_dp_size = torch.distributed.get_world_size(dp_group)
        my_dp_rank = torch.distributed.get_rank(dp_group)

    if tgt_dp_size >= src_dp_size:
        repeat_times = tgt_dp_size // src_dp_size

        repeated_state_dicts = []  # len = world_size
        # world = 5, tgt_dp=4, src_dp=2, make last empty
        # src_dp=0: [[data], [data], [    ], [    ], [    ]]
        # src_dp=1: [[    ], [    ], [data], [data], [    ]]
        # NOTE: The inference parallel hierarchy follows a nested
        # structure: DP -> PP -> TP
        for i in range(tgt_dp_size):
            start = my_dp_rank * repeat_times
            end = (my_dp_rank + 1) * repeat_times
            if my_dp_rank == (src_dp_size - 1):
                end = tgt_dp_size
            if start <= i < end:
                repeated_state_dicts.extend(splited_state_dict)
            else:
                repeated_state_dicts.extend([{}] * tgt_tp_size)
    else:  # tgt_dp_size < src_dp_size
        # Shrinking DP: multiple source DP ranks map to the same target DP rank.
        # Since DP replicas should have identical weights, we pick a single "leader"
        # source DP rank for each target DP rank to avoid redundant sends.
        #
        repeated_state_dicts = []  # len = world_size

        # Map this source DP rank to a target DP bucket:
        my_target_dp = (my_dp_rank * tgt_dp_size) // src_dp_size
        # The leader is the first source rank that falls into this bucket:
        leader_for_bucket = (my_target_dp * src_dp_size) // tgt_dp_size
        is_leader = my_dp_rank == leader_for_bucket

        for i in range(tgt_dp_size):
            if i == my_target_dp and is_leader:
                repeated_state_dicts.extend(splited_state_dict)
            else:
                repeated_state_dicts.extend([{}] * tgt_tp_size)

    repeated_state_dicts += [{}] * (world_size - len(repeated_state_dicts))

    # repeated_state_dicts = splited_state_dict * tgt_dp_size
    all_rank_dicts = all_to_all_dicts(repeated_state_dicts, group, device=device)  # len = world
    output = {}
    for data in all_rank_dicts:
        output.update(data)
    return output


def all_gather_object(object, group=None):
    output = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(output, object, group=group)
    return output


def _get_comm_device(device):
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _all_gather_sizes(size_tensor: torch.Tensor, group=None) -> list[int]:
    world_size = torch.distributed.get_world_size(group)
    size_list = [torch.empty_like(size_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(size_list, size_tensor, group=group)
    return [int(s.item()) for s in size_list]


def all_gather_dict(
    obj: dict[str, torch.Tensor],
    group=None,
    device=None,
    keep_on_comm_device: bool = False,
):
    comm_device = _get_comm_device(device)
    packed = dict_to_tensor(obj, comm_device=comm_device)
    size_tensor = torch.tensor([packed.numel()], dtype=torch.int64, device=comm_device)
    sizes = _all_gather_sizes(size_tensor, group=group)
    max_size = max(sizes) if sizes else 0
    if packed.numel() < max_size:
        pad = torch.empty(max_size - packed.numel(), device=comm_device, dtype=packed.dtype)
        packed = torch.cat([packed, pad], dim=0)
    recv_tensors = [
        torch.empty(max_size, device=comm_device, dtype=packed.dtype)
        for _ in range(torch.distributed.get_world_size(group))
    ]
    torch.distributed.all_gather(recv_tensors, packed, group=group)
    return [
        tensor_to_dict(
            recv_tensors[i][: sizes[i]],
            keep_on_comm_device=keep_on_comm_device,
        )
        for i in range(len(recv_tensors))
    ]


def gather_dict(
    obj: dict[str, torch.Tensor],
    group=None,
    dst: int = 0,
    device=None,
    keep_on_comm_device: bool = False,
):
    comm_device = _get_comm_device(device)
    packed = dict_to_tensor(obj, comm_device=comm_device)
    size_tensor = torch.tensor([packed.numel()], dtype=torch.int64, device=comm_device)
    sizes = _all_gather_sizes(size_tensor, group=group)
    max_size = max(sizes) if sizes else 0
    if packed.numel() < max_size:
        pad = torch.empty(max_size - packed.numel(), device=comm_device, dtype=packed.dtype)
        packed = torch.cat([packed, pad], dim=0)
    global_rank = torch.distributed.get_rank()
    dst_global = torch.distributed.get_global_rank(group, dst)
    gather_list = None
    if global_rank == dst_global:
        gather_list = [
            torch.empty(max_size, device=comm_device, dtype=packed.dtype)
            for _ in range(torch.distributed.get_world_size(group))
        ]
    torch.distributed.gather(packed, gather_list=gather_list, dst=dst_global, group=group)
    if global_rank != dst_global:
        return None
    return [
        tensor_to_dict(
            gather_list[i][: sizes[i]],
            keep_on_comm_device=keep_on_comm_device,
        )
        for i in range(len(gather_list))
    ]


def concat_list_in_group(local_list: list, group=None):
    """Basically, this function should act just like:

    **sum(all_gather_object(local_list, group=group), list())**

    but optimized peak mem when data available only on single rank.
    """
    local_size = len(local_list)
    world_sizes = all_gather_object(local_size)
    have_data = [s > 0 for s in world_sizes]

    if sum(have_data) == 1:
        # broadcast
        src = have_data.index(True)
        return broadcast_tensors(local_list, src_rank=src, group=group)
    else:
        return sum(all_gather_object(local_list, group=group), list())


def gather_object(objects, group=None, dst=0):
    send = [list()] * torch.distributed.get_world_size(group)
    send[dst] = objects
    resv = all_to_all_objects(send, group=group)
    return resv


def list_balance(data: list, group=None) -> tuple[list, tuple]:
    """This function re-org data to each rank of the group.
    Rank 0: [0, 1, 2, 3, 4] -> [0, 1, 2]
    Rank 1: [5]                [3, 4, 5]
    returns: balanced_data, restore_info
    """
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    all_data_len: list[int] = all_gather_object(len(data), group=group)

    data_ids: torch.Tensor = torch.arange(sum(all_data_len))
    actual_ids = data_ids.split(all_data_len)
    balanced_ids = list_split_T(data_ids, len(all_data_len))

    send = [[data[k] for k, j in enumerate(actual_ids[rank]) if j in balanced_ids[i]] for i in range(world_size)]
    send = recur_to(send, "cpu")
    balanced_data = sum(all_to_all_objects(send, group=group), [])
    return balanced_data, (actual_ids, balanced_ids, group)


def list_unbalance(balanced_data: list, restore_info: tuple) -> list:
    """This is the inverse function of list_balance, restore data to original ranks.
    Rank 0: [0, 1, 2] -> [0, 1, 2, 3, 4]
    Rank 1: [3, 4, 5]    [5]
    """

    actual_ids, balanced_ids, group = restore_info
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)

    send = [
        [balanced_data[k] for k, j in enumerate(balanced_ids[rank]) if j in actual_ids[i]] for i in range(world_size)
    ]
    send = recur_to(send, "cpu")
    restored_data = sum(all_to_all_objects(send, group=group), [])
    return restored_data


def balanced_map(func, data: list, group=None) -> list:
    """like Map, but balance data in group"""
    balanced_data, restore_info = list_balance(data=data, group=group)

    balanced_outs = [func(d) for d in balanced_data]

    output = list_unbalance(balanced_outs, restore_info)
    return output


def send_obj(object, dst, group):
    tensor, size = object_to_tensor(object, device="cuda")
    torch.distributed.send(size, dst=dst, group=group)
    return torch.distributed.isend(tensor, dst=dst, group=group)


def isend_obj(object, dst, group):
    tensor, size = object_to_tensor(object, device="cuda")
    handles = []
    handles.append(torch.distributed.isend(size, dst=dst, group=group))
    handles.append(torch.distributed.isend(tensor, dst=dst, group=group))
    return handles


def wait_all(handles):
    for handle in handles:
        handle.wait()


def recv_obj(src, group):
    size = torch.LongTensor([0]).to("cuda")
    torch.distributed.recv(size, src=src, group=group)
    tensor = torch.empty([size.item()], dtype=torch.int8, device="cuda")
    torch.distributed.recv(tensor, src=src, group=group)
    object = tensor_to_object(tensor, size)
    return recur_to(object, "cuda")


def pack_tensors(tensors_dict: dict, names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """将不同数据类型的tensors打包为bytes tensor

    Args:
        tensors_dict: 包含不同tensor的字典
        names: tensor名称列表,如果为None则使用默认顺序

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - shapes_and_dtypes: [num_tensors, 3], 每行是[shape1, shape2, dtype_id]
            - packed_data: 打包后的bytes tensor
    """

    shapes_and_dtypes = []
    data_chunks = []

    # 统一处理int类型字段，转换为tensor
    for key, value in tensors_dict.items():
        if value is None:
            continue
        if not torch.is_tensor(value):
            assert isinstance(value, int), f"The value of key {key} must be int, but got {type(value)}:{value}"
            tensors_dict[key] = torch.tensor([value], dtype=torch.int64, device="cuda")

    for name in names:
        tensor = tensors_dict.get(name, None)
        if tensor is None:
            shapes_and_dtypes.append([-1, -1, -1])
            continue

        # 记录shape和dtype
        shape = list(tensor.shape)
        if len(shape) == 1:
            shape = [shape[0], 1]  # 1维tensor转为2维
        elif len(shape) == 0:
            shape = [1, 1]  # 标量转为2维
        elif len(shape) > 2:
            raise ValueError(f"Unsupported tensor shape: {shape}, name: {name}")
        tensor_info = shape + [_dtype_to_id(tensor.dtype)]  # [shape1, shape2, dtype_id]
        shapes_and_dtypes.append(tensor_info)

        # 直接转换为bytes tensor
        nbytes = tensor.numel() * tensor.element_size()
        data_chunks.append(tensor.contiguous().clone().cuda().view(torch.uint8).reshape(nbytes))  # 确保在GPU上

    # 打包shapes和dtypes信息
    shapes_and_dtypes = torch.tensor(shapes_and_dtypes, dtype=torch.int64, device="cuda")

    # 打包数据为bytes tensor
    packed_data = torch.cat(data_chunks) if data_chunks else torch.empty(0, dtype=torch.uint8, device="cuda")
    return shapes_and_dtypes, packed_data


def unpack_tensors(shapes_and_dtypes: torch.Tensor, packed_data: torch.Tensor, names: List[str]) -> dict:
    """从bytes解包出不同数据类型的tensors

    Args:
        shapes_and_dtypes: [num_tensors, 3], 每行是[numel, shape..., dtype]
        packed_data: 打包的bytes tensor
        names: tensor名称列表,如果为None则使用默认顺序

    Returns:
        dict: 解包后的tensor字典
    """

    tensors_dict = {}
    offset = 0

    for i, name in enumerate(names):
        tensor_info = shapes_and_dtypes[i]

        # 处理None值
        if tensor_info[0] == -1:
            tensors_dict[name] = None
            continue

        # 解析shape和dtype
        shape = tensor_info[:-1].tolist()  # [shape1, shape2]
        if shape[1] == 1 and name != "input":  # input保持2维
            shape = [shape[0]]  # 还原为1维
        dtype = _id_to_dtype(tensor_info[-1].item())
        if dtype == -2:  # 未知的dtype ID
            raise ValueError(f"Unknown dtype id: {tensor_info[-1].item()}")

        # 计算bytes大小
        numel = np.prod(shape)
        nbytes = numel * dtype.itemsize

        if nbytes == 0:
            # 处理空tensor的情况
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
        else:
            # 从packed_data中提取并转换回tensor
            tensor_bytes = packed_data[offset : offset + nbytes]
            tensor = torch.empty(numel, dtype=dtype, device="cuda")
            tensor_view = tensor.view(torch.uint8)
            tensor_view.copy_(tensor_bytes[: tensor_view.numel()])
            tensor = tensor.view(shape)

        tensors_dict[name] = tensor
        offset += nbytes

    return tensors_dict


def recv_tensors(names: List[str], src: int, group=None) -> dict:
    """接收多个不同数据类型的tensor"""
    # 更新shapes_and_dtypes的大小以容纳prefill_num
    shapes_and_dtypes = torch.empty([len(names), 3], dtype=torch.int64, device="cuda")
    # 1. 接收shapes和dtypes信息
    handle = torch.distributed.irecv(shapes_and_dtypes, src, group=group)
    handle.wait()
    # 2. 计算packed_data的总大小
    total_bytes = sum(
        np.prod(info[:-1].tolist()) * _id_to_dtype(info[-1].item()).itemsize
        for info in shapes_and_dtypes
        if info[0] != -1  # 跳过None值
    )

    # 3. 接收打包的数据
    packed_data = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    handle = torch.distributed.irecv(packed_data, src, group=group)
    handle.wait()

    # 4. 解包tensors
    tensors_dict = unpack_tensors(shapes_and_dtypes, packed_data, names)
    return tensors_dict


def send_tensors(tensors_dict: dict, dst: int, group=None) -> List[torch.distributed.Work]:
    """异步发送多个不同数据类型的tensor"""
    # 确保所有int类型数据都转换为tensor
    names = list(tensors_dict.keys())

    # 1. 打包tensors
    shapes_and_dtypes, packed_data = pack_tensors(tensors_dict, names=names)

    # 2. 发送shapes和dtypes信息
    handles = [torch.distributed.isend(shapes_and_dtypes, dst, group=group)]

    # 3. 发送打包的数据
    handles.append(torch.distributed.isend(packed_data, dst, group=group))
    return handles

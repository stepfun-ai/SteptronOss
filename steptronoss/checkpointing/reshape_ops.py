import os
import re
from typing import Any, Callable

import torch
from configurize import DataClass
from einops import rearrange
from loguru import logger

from steptronoss.core.parallel_state import PM
from steptronoss.utils.dist_utils import all_gather_dict
from steptronoss.utils.weight_loader import HFWeights, translate

UnifiedTensorDict = dict[Any, torch.Tensor]
SplitedTensorDict = dict[Any, list[torch.Tensor]]


WEIGHT_FW = Callable[[UnifiedTensorDict], SplitedTensorDict]
WEIGHT_BW = Callable[[SplitedTensorDict], UnifiedTensorDict]


def common_pattern(strs: list[str]) -> str:
    """['aan', 'abn', 'acn'] -> 'a{}n'"""
    strs = list(set(strs))
    if len(strs) == 1:
        return strs[0]

    prefix = os.path.commonprefix(strs)
    suffix = os.path.commonprefix([x[::-1] for x in strs])[::-1]
    return f"{prefix}{{}}{suffix}"


class ReshapeOp:
    def forward(self, piece: dict) -> dict:
        raise NotImplementedError

    def backward(self, piece: dict) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__qualname__ + "(" + ", ".join(f"{k}={v}" for k, v in self.__dict__.items()) + ")"

    def __add__(self, other: "ReshapeOp"):
        assert isinstance(other, ReshapeOp)
        if isinstance(self, Sequential):
            self.append(other)
            return self
        else:
            new_ops = Sequential([self, other])
            return new_ops


class Identity(ReshapeOp):
    def forward(self, piece):
        return piece

    def backward(self, piece):
        return piece


class Sequential(ReshapeOp):
    def __init__(self, ops: list[ReshapeOp]):
        self.ops = ops

    def forward(self, piece):
        for op in self.ops:
            piece = op.forward(piece)
        return piece

    def backward(self, piece):
        for op in self.ops[::-1]:
            piece = op.backward(piece)
        return piece

    def append(self, op: ReshapeOp):
        self.ops.append(op)


def _gather_reshape_piece(piece: dict, group):
    return all_gather_dict(piece, group=group)


class KeepThisTP(ReshapeOp):
    """{A: [tp0, tp1]} -> {A: tp0}"""

    def __init__(self, group="TP"):
        self.group = group

    def forward(self, piece: dict) -> dict:
        return {k: v[PM.rank_in(self.group)] for k, v in piece.items()}

    def backward(self, piece: dict) -> dict:
        # {} -> [{}, {}]
        raw_piece = _gather_reshape_piece(piece, group=PM.group_of(self.group))
        # [{A:1}, {A:2}] -> {A:[1, 2]}
        keys = set()
        for p in raw_piece:
            keys |= p.keys()
        return {k: [p.get(k, None) for p in raw_piece] for k in keys}


class KeepThisEP(ReshapeOp):
    """Filter experts based on EP rank and remap global IDs to local IDs.
    Assumes input keys are in format '...moe.{id}...' produced by UnbindMoE."""

    def __init__(self, moe_key_prefix="moe."):
        super().__init__()
        self.moe_key_prefix = moe_key_prefix
        self.pattern = re.compile(rf"{re.escape(self.moe_key_prefix)}(\d+)\.")

    def forward(self, piece: dict) -> dict:
        if PM.size_of("EP") <= 1:
            return piece

        ep_rank = PM.rank_in("EP")
        ep_size = PM.size_of("EP")

        # Identify all expert IDs to infer total count and local range
        expert_ids = []
        for k in piece.keys():
            match = self.pattern.search(k)
            if match:
                expert_ids.append(int(match.group(1)))

        if not expert_ids:
            return piece

        # Assume experts are contiguous and 0-indexed
        num_global_experts = max(expert_ids) + 1
        num_local_experts = num_global_experts // ep_size
        start_expert = ep_rank * num_local_experts
        end_expert = start_expert + num_local_experts

        new_piece = {}
        for k, v in piece.items():
            match = self.pattern.search(k)
            if match:
                eid = int(match.group(1))
                if start_expert <= eid < end_expert:
                    local_eid = eid - start_expert
                    # Replace global ID with local ID: moe.100. -> moe.36.
                    new_key = k.replace(
                        f"{self.moe_key_prefix}{eid}.",
                        f"{self.moe_key_prefix}{local_eid}.",
                    )
                    new_piece[new_key] = v
                # else: drop this key
            else:
                new_piece[k] = v
        return new_piece

    def backward(self, piece: dict) -> dict:
        if PM.size_of("EP") <= 1:
            return piece

        # Gather all pieces from EP group
        gathered_pieces = _gather_reshape_piece(piece, group=PM.group_of("EP"))

        # Infer num_local_experts to calculate offsets
        # We check all gathered pieces to find the max local ID
        max_local_id = -1
        for p in gathered_pieces:
            for k in p.keys():
                match = self.pattern.search(k)
                if match:
                    max_local_id = max(max_local_id, int(match.group(1)))

        num_local_experts = max_local_id + 1

        new_piece = {}
        for rank, rank_piece in enumerate(gathered_pieces):
            start_expert = rank * num_local_experts
            for k, v in rank_piece.items():
                match = self.pattern.search(k)
                if match:
                    local_eid = int(match.group(1))
                    global_eid = local_eid + start_expert
                    new_key = k.replace(
                        f"{self.moe_key_prefix}{local_eid}.",
                        f"{self.moe_key_prefix}{global_eid}.",
                    )
                    new_piece[new_key] = v
                else:
                    new_piece[k] = v
        return new_piece


class VocabPad(ReshapeOp):
    """{A: (V, C)} -> {A: (PV, C)}"""

    def __init__(self, target_vocab_size: int, dim: int, pad_type="last", actual_vocab_size=None):
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.dim = dim
        self.pad_type = pad_type
        self.actual_vocab_size = actual_vocab_size
        assert self.pad_type in ["last"]

    def _pad_tensor(self, tensor, to):
        orig_vocab_size = tensor.shape[self.dim]
        if orig_vocab_size > to:  # keep first N
            full_input_embed = torch.narrow(tensor, dim=self.dim, start=0, length=to)
        elif orig_vocab_size < to:  # Pad
            padding_size = to - orig_vocab_size
            if self.pad_type == "last":
                padder = torch.narrow(tensor, dim=self.dim, start=orig_vocab_size - 1, length=1).repeat_interleave(
                    padding_size, dim=self.dim
                )
            full_input_embed = torch.cat((tensor, padder), dim=self.dim)
        else:
            full_input_embed = tensor
        return full_input_embed

    def forward(self, piece: UnifiedTensorDict) -> UnifiedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = self._pad_tensor(v, self.target_vocab_size)
        return new_piece

    def backward(self, piece: UnifiedTensorDict) -> UnifiedTensorDict:
        if self.actual_vocab_size is None:
            logger.warning(
                f"{self} is a one-way conversion (safetensors->pt only), "
                "original vocab_size of saftensors cannot be restored!"
            )
        else:
            new_piece = {}
            for k, v in piece.items():
                new_piece[k] = self._pad_tensor(v, self.actual_vocab_size)
            return new_piece
        return piece


class Duplicate(ReshapeOp):
    """{A: data} -> {A: [data, data]}"""

    def __init__(self, group="TP"):
        self.group = group

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = [v for i in range(PM.size_of(self.group))]
        return new_piece

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            assert all((x == v[0]).all() for x in v), "Breaking duplicate!"
            new_piece[k] = v[0]
        return new_piece


class ColumnParallel(ReshapeOp):
    """{A: data} -> {A: data.chunk(tp_size, dim=0)}"""

    def __init__(self, group="TP"):
        self.group = group

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = v.chunk(PM.size_of(self.group), 0)
        return new_piece

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = torch.cat(v, 0)
        return new_piece


class RowParallel(ReshapeOp):
    """{A: data} -> {A: data.chunk(tp_size, dim=1)}"""

    def __init__(self, group="TP"):
        self.group = group

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = v.chunk(PM.size_of(self.group), -1)
        return new_piece

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            new_piece[k] = torch.cat(v, -1)
        return new_piece


class MHA_TP_CHUNK(ReshapeOp):
    """{Aqkv: qkv} -> {Aqkv: [qkv_tp0, qkv_tp1]}"""

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        new_pieces = {}
        for key, qkv in piece.items():
            q, k, v = torch.chunk(qkv, 3)
            q = q.chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
            k = k.chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
            v = v.chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
            qkv = torch.cat([q, k, v], dim=0)
            new_pieces[key] = [qkv if i == PM.rank_in("TP") else None for i in range(PM.size_of("TP"))]
        return new_pieces

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        new_pieces = {}
        for key, qkv_list in piece.items():
            qs, ks, vs = [], [], []
            for per_tp_qkv in qkv_list:
                q, k, v = torch.chunk(per_tp_qkv, 3)
                qs.append(q)
                ks.append(k)
                vs.append(v)

            qkv = torch.cat(qs + ks + vs, dim=0)
            new_pieces[key] = qkv
        return new_pieces


class GQAMergeQKV(ReshapeOp):
    """{Aq: q, Ak: k, Av: v} -> {Aqkv: [qkv_tp0, qkv_tp1]}"""

    def __init__(self, group_num: int, head_dim: int):
        self.group_num = group_num
        self.head_dim = head_dim

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        if not piece:
            return piece
        assert len(piece) == 3
        qk, kk, vk = sorted(piece.keys(), key=lambda x: x.count("k") + x.count("v") * 2)
        q = piece[qk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        k = piece[kk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        v = piece[vk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        kv = rearrange([k, v], "T (g h) d -> (g T h) d", T=2, g=self.group_num // PM.size_of("TP"))
        qkv = torch.cat([q, kv], dim=0)
        return {
            common_pattern(piece.keys()).format("qkv"): [
                qkv if i == PM.rank_in("TP") else None for i in range(PM.size_of("TP"))
            ]
        }

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        assert len(piece) == 1
        key_pattern: str = list(piece)[0].replace("qkv", "{}")
        if "{}" not in key_pattern:
            key_pattern += ".{}"
        tensors = piece[key_pattern.format("qkv")]

        n_kv_dims = self.group_num * self.head_dim * 2 // PM.size_of("TP")
        n_q_dims = tensors[0].shape[0] - n_kv_dims
        qs, ks, vs = [], [], []

        for tensor in tensors:
            q, kv = torch.split(
                tensor,
                [n_q_dims, n_kv_dims],
                dim=0,
            )
            k, v = rearrange(kv, "(g T h) d -> T (g h) d", T=2, g=self.group_num // PM.size_of("TP"))
            qs.append(q)
            ks.append(k)
            vs.append(v)
        new_piece = {
            key_pattern.format("q"): torch.cat(qs, 0),
            key_pattern.format("k"): torch.cat(ks, 0),
            key_pattern.format("v"): torch.cat(vs, 0),
        }
        return new_piece


class GQAMergeQKVBias(ReshapeOp):
    """{Aq: q, Ak: k, Av: v} -> {Aqkv: [qkv_tp0, qkv_tp1]}"""

    def __init__(self, group_num: int, head_dim: int):
        self.group_num = group_num
        self.head_dim = head_dim

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        if not piece:
            return piece
        assert len(piece) == 3
        qk, kk, vk = sorted(piece.keys(), key=lambda x: x.count("k") + x.count("v") * 2)
        q = piece[qk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        k = piece[kk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        v = piece[vk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        kv = rearrange([k, v], "T (g h) -> (g T h)", T=2, g=self.group_num // PM.size_of("TP"))
        qkv = torch.cat([q, kv], dim=0)
        return {
            common_pattern(piece.keys()).format("qkv"): [
                qkv if i == PM.rank_in("TP") else None for i in range(PM.size_of("TP"))
            ]
        }

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        assert len(piece) == 1
        key_pattern: str = list(piece)[0].replace("qkv", "{}")
        if "{}" not in key_pattern:
            key_pattern += ".{}"
        tensors = piece[key_pattern.format("qkv")]

        n_kv_dims = self.group_num * self.head_dim * 2 // PM.size_of("TP")
        n_q_dims = tensors[0].shape[0] - n_kv_dims
        qs, ks, vs = [], [], []

        for tensor in tensors:
            q, kv = torch.split(
                tensor,
                [n_q_dims, n_kv_dims],
                dim=0,
            )
            k, v = rearrange(kv, "(g T h) -> T (g h)", T=2, g=self.group_num // PM.size_of("TP"))
            qs.append(q)
            ks.append(k)
            vs.append(v)
        new_piece = {
            key_pattern.format("q"): torch.cat(qs, 0),
            key_pattern.format("k"): torch.cat(ks, 0),
            key_pattern.format("v"): torch.cat(vs, 0),
        }
        return new_piece


class GQAMergeQKVG(ReshapeOp):
    """{Aq: q, Ak: k, Av: v, Ag: g} -> {Aqkv: [qkv_tp0, qkv_tp1]}"""

    def __init__(self, group_num: int, head_dim: int, gate_dims: int):
        self.group_num = group_num
        self.head_dim = head_dim
        self.gate_dims = gate_dims

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        if not piece:
            return piece
        assert len(piece) == 4
        qk, kk, vk, gk = sorted(
            piece.keys(),
            key=lambda x: x.count("k") + x.count("v") * 2 + x.count("g") * 3,
        )
        q = piece[qk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        k = piece[kk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        v = piece[vk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        g = piece[gk].chunk(PM.size_of("TP"), 0)[PM.rank_in("TP")]
        kv = rearrange([k, v], "T (g h) d -> (g T h) d", T=2, g=self.group_num // PM.size_of("TP"))
        qkv = torch.cat([q, kv, g], dim=0)
        return {
            common_pattern(piece.keys()).format("qkv"): [
                qkv if i == PM.rank_in("TP") else None for i in range(PM.size_of("TP"))
            ]
        }

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        assert len(piece) == 1
        key_pattern: str = list(piece)[0].replace("qkv", "{}")
        if "{}" not in key_pattern:
            key_pattern += ".{}"
        tensors = piece[key_pattern.format("qkv")]

        n_kv_dims = self.group_num * self.head_dim * 2 // PM.size_of("TP")
        n_g_dims = self.gate_dims // PM.size_of("TP")
        n_q_dims = tensors[0].shape[0] - n_kv_dims - n_g_dims
        qs, ks, vs, gs = [], [], [], []

        for tensor in tensors:
            q, kv, g = torch.split(
                tensor,
                [n_q_dims, n_kv_dims, n_g_dims],
                dim=0,
            )
            k, v = rearrange(kv, "(g T h) d -> T (g h) d", T=2, g=self.group_num // PM.size_of("TP"))
            qs.append(q)
            ks.append(k)
            vs.append(v)
            gs.append(g)
        new_piece = {
            key_pattern.format("q"): torch.cat(qs, 0),
            key_pattern.format("k"): torch.cat(ks, 0),
            key_pattern.format("v"): torch.cat(vs, 0),
            key_pattern.format("g"): torch.cat(gs, 0),
        }
        return new_piece


class FFNMergeGateUp(ReshapeOp):
    def __init__(self, group="TP"):
        self.group = group

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        if not piece:
            return piece
        assert len(piece) == 2
        kgate = kup = None
        for k in piece:
            if "gate" in k:
                kgate = k
            if "up" in k:
                kup = k
        assert all([kgate, kup])
        gate = piece[kgate].chunk(PM.size_of(self.group), -2)[PM.rank_in(self.group)]
        up = piece[kup].chunk(PM.size_of(self.group), -2)[PM.rank_in(self.group)]
        w1 = torch.cat([gate, up], dim=-2)
        return {
            common_pattern(piece.keys()).format("gate_up"): [
                w1 if i == PM.rank_in(self.group) else None for i in range(PM.size_of(self.group))
            ]
        }

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        assert len(piece) == 1
        key_pattern: str = list(piece)[0].replace("gate_up", "{}")
        if "{}" not in key_pattern:
            key_pattern += ".{}"
        tensors = piece[key_pattern.format("gate_up")]

        gates, ups = [], []

        for tensor in tensors:
            gate, up = torch.chunk(tensor, 2, dim=-2)
            gates.append(gate)
            ups.append(up)
        new_piece = {
            key_pattern.format("gate"): torch.cat(gates, -2),
            key_pattern.format("up"): torch.cat(ups, -2),
        }
        return new_piece


class UnbindMoE(ReshapeOp):
    """{A: (E, O, I)} -> {A.0: (O, I), A.1: (O, I)}"""

    def __init__(self, moe_key_prefix="moe."):
        super().__init__()
        self.moe_key_prefix = moe_key_prefix
        self.pattern = re.compile(rf"{re.escape(self.moe_key_prefix)}(\d+)\.")

    def forward(self, piece: UnifiedTensorDict) -> SplitedTensorDict:
        new_piece = {}
        for k, v in piece.items():
            for i, vv in enumerate(v.unbind()):
                new_piece[k.replace(self.moe_key_prefix, f"{self.moe_key_prefix}{i}.")] = vv
        return new_piece

    def backward(self, piece: SplitedTensorDict) -> UnifiedTensorDict:
        # 创建分组字典：{base_key: {index: tensor}}
        from collections import defaultdict

        groups = defaultdict(dict)

        for key, tensor in piece.items():
            # 匹配 moe_key_prefix 后的数字索引
            match = self.pattern.search(key)
            if not match:
                continue

            index = int(match.group(1))
            # 提取基础键名（移除索引部分）
            base_key = key[: match.start()] + self.moe_key_prefix + key[match.end() :]

            # 确保索引唯一性
            if index in groups[base_key]:
                raise ValueError(f"Duplicate index {index} for key {base_key}")
            groups[base_key][index] = tensor

        # 合并各组张量
        new_piece = {}
        for base_key, tensor_dict in groups.items():
            # 按索引排序后堆叠张量
            sorted_tensors = [tensor_dict[i] for i in sorted(tensor_dict)]
            new_piece[base_key] = torch.stack(sorted_tensors)

        return new_piece


class Inverse(ReshapeOp):
    def __init__(self, op: ReshapeOp):
        super().__init__()
        self.op = op

    def forward(self, piece: dict) -> dict:
        return self.op.backward(piece)

    def backward(self, piece: dict) -> dict:
        return self.op.forward(piece)


class Rename(ReshapeOp):
    def __init__(self, name_mapping: dict[str, str] | str):
        if isinstance(name_mapping, str):
            dst, src = name_mapping.split(":")
            name_mapping = {dst.strip(): src.strip()}
        self.naming_indicator = [f"{k}:{v}" for k, v in name_mapping.items()]
        self.reversed_naming_indicator = [f"{v}:{k}" for k, v in name_mapping.items()]

    def forward(self, piece: dict) -> dict:
        return HFWeights.export(piece, self.naming_indicator)

    def backward(self, piece: dict) -> dict:
        return HFWeights.export(piece, self.reversed_naming_indicator)


class Script(DataClass):
    dst: str | list[str] = None
    op: ReshapeOp
    src: str | list[str] = None


class OnlineReshaper(ReshapeOp):
    """NOTE:
    - OP_FW: HF -> ST
    - OP_BW: ST -> HF
    """

    def __init__(self, scripts: list[Script] | dict[Script]):
        if isinstance(scripts, dict):
            _scripts = []
            for dst, script in scripts.items():
                script.dst = dst
                _scripts.append(script)
            scripts = _scripts
        self.scripts = scripts

    def check_forwardable(self):
        forward_fail = [s for s in self.scripts if s.src is None]
        if forward_fail:
            raise RuntimeError(f".src not defined for {forward_fail}, call backward to auto-record.")

    def check_backwardable(self):
        backward_fail = [s for s in self.scripts if s.dst is None]
        if backward_fail:
            raise RuntimeError(f".dst not defined for {backward_fail}, call forward to auto-record.")

    @staticmethod
    def get_piece_by_pattern(weights: dict, patterns: str | list[str]) -> dict:
        if not isinstance(patterns, list):
            patterns = [patterns]
        matched = set()
        for pattern in patterns:
            pattern = translate(pattern)
            for k in weights.keys():
                if re.match(pattern, k):
                    matched.add(k)

        return {k: weights[k] for k in matched}

    def forward(self, weights: dict) -> dict:
        self.check_forwardable()
        output = {}
        for script_piece in self.scripts:
            raw_piece = self.get_piece_by_pattern(weights, script_piece.src)

            new_piece = script_piece.op.forward(raw_piece)

            if script_piece.dst is None:
                script_piece.dst = list(new_piece.keys())

            output.update(new_piece)
        return output

    def backward(self, weights: dict) -> dict:
        self.check_backwardable()
        output = {}
        for script_piece in self.scripts:
            raw_piece = self.get_piece_by_pattern(weights, script_piece.dst)

            new_piece = script_piece.op.backward(raw_piece)

            if script_piece.src is None:
                script_piece.src = list(new_piece.keys())

            output.update(new_piece)
        return output

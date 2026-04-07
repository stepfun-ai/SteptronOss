from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.utils.dist_utils import all_to_all_objects


@dataclass(frozen=True)
class _RankInfo:
    replica_id: int
    pp_rank: int
    cp_rank: int
    tp_rank: int


@dataclass(frozen=True)
class _NodeKey:
    replica_id: int
    pp_rank: int
    cp_rank: int


@dataclass(frozen=True)
class _ShardMeta:
    batch_size: int
    shard_count: int
    shard_id: int


class MeshConnector:
    """Static connector between a source mesh and a destination mesh.

    TP ranks are treated as duplicates: only the TP leader participates in cross-mesh
    transfer, and payloads are fanned out to TP followers locally.
    """

    def __init__(self, src_mesh: ParallelConfig, dst_mesh: ParallelConfig, *, is_data_source: bool):
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh
        self.is_data_source = is_data_source
        self.world_size = PM.world_size

        self.src_rank_infos = self._describe_mesh(src_mesh)
        self.dst_rank_infos = self._describe_mesh(dst_mesh)
        self.local_src_info = self.src_rank_infos[PM.world_rank]
        self.local_dst_info = self.dst_rank_infos[PM.world_rank]

        self.src_tp_members = self._build_tp_members(self.src_rank_infos)
        self.dst_tp_members = self._build_tp_members(self.dst_rank_infos)

        gathered_source_flags = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_source_flags, is_data_source)

        self.source_nodes = self._compile_source_nodes(gathered_source_flags)
        self.canonical_source_node_by_replica = {
            replica_id: sorted(nodes, key=lambda node: (node.pp_rank, node.cp_rank))[0]
            for replica_id, nodes in self._group_nodes_by_replica(self.source_nodes).items()
        }

        self.forward_targets = self._compile_forward_targets()
        self.backward_targets = self._compile_backward_targets()

        self._last_forward_meta: _ShardMeta | None = None

    def _describe_mesh(self, mesh: ParallelConfig) -> dict[int, _RankInfo]:
        with PM.use_mesh(mesh):
            dp_groups = PM.world_ranks_of("DP")
            pp_groups = PM.world_ranks_of("PP")
            cp_groups = PM.world_ranks_of("CP")
            tp_groups = PM.world_ranks_of("TP")

        def _find(groups: list[list[int]], rank: int) -> tuple[int, int]:
            for gid, ranks in enumerate(groups):
                if rank in ranks:
                    return gid, ranks.index(rank)
            raise RuntimeError(f"Rank {rank} not found in mesh groups")

        infos = {}
        for rank in range(self.world_size):
            dp_replica, _ = _find(dp_groups, rank)
            _, pp_rank = _find(pp_groups, rank)
            _, cp_rank = _find(cp_groups, rank)
            _, tp_rank = _find(tp_groups, rank)
            infos[rank] = _RankInfo(
                replica_id=dp_replica,
                pp_rank=pp_rank,
                cp_rank=cp_rank,
                tp_rank=tp_rank,
            )
        return infos

    def _build_tp_members(self, rank_infos: dict[int, _RankInfo]) -> dict[_NodeKey, list[int]]:
        members: dict[_NodeKey, list[int]] = {}
        for rank, info in rank_infos.items():
            key = _NodeKey(info.replica_id, info.pp_rank, info.cp_rank)
            members.setdefault(key, []).append(rank)
        for ranks in members.values():
            ranks.sort(key=lambda rank: rank_infos[rank].tp_rank)
        return members

    def _group_nodes_by_replica(self, nodes: set[_NodeKey]) -> dict[int, list[_NodeKey]]:
        grouped: dict[int, list[_NodeKey]] = {}
        for node in nodes:
            grouped.setdefault(node.replica_id, []).append(node)
        return grouped

    def _compile_source_nodes(self, gathered_source_flags: list[bool]) -> set[_NodeKey]:
        source_nodes = set()
        for rank, has_data in enumerate(gathered_source_flags):
            if not has_data:
                continue
            info = self.src_rank_infos[rank]
            source_nodes.add(_NodeKey(info.replica_id, info.pp_rank, info.cp_rank))
        return source_nodes

    def _src_leader(self, node: _NodeKey) -> int:
        return self.src_tp_members[node][0]

    def _dst_leader(self, node: _NodeKey) -> int:
        return self.dst_tp_members[node][0]

    def _dst_node(self, rank: int) -> _NodeKey:
        info = self.dst_rank_infos[rank]
        return _NodeKey(info.replica_id, info.pp_rank, info.cp_rank)

    def _src_node(self, rank: int) -> _NodeKey:
        info = self.src_rank_infos[rank]
        return _NodeKey(info.replica_id, info.pp_rank, info.cp_rank)

    def _compile_forward_targets(self) -> dict[int, list[int]]:
        targets = {rank: [] for rank in range(self.world_size)}
        for node_key, _leader_ranks in self.dst_tp_members.items():
            replica_id = node_key.replica_id
            if replica_id not in self.canonical_source_node_by_replica:
                continue
            src_leader = self._src_leader(self.canonical_source_node_by_replica[replica_id])
            targets[src_leader].append(self._dst_leader(node_key))
        return targets

    def _compile_backward_targets(self) -> dict[int, list[int]]:
        targets = {rank: [] for rank in range(self.world_size)}
        replica_sources = self._group_nodes_by_replica(self.source_nodes)
        for node_key, _leader_ranks in self.dst_tp_members.items():
            dst_leader = self._dst_leader(node_key)
            for src_node in sorted(
                replica_sources.get(node_key.replica_id, []), key=lambda node: (node.pp_rank, node.cp_rank)
            ):
                targets[dst_leader].append(self._src_leader(src_node))
        return targets

    def _send(self, payload_builder) -> list[Any]:
        send = [payload_builder(dst_rank) for dst_rank in range(self.world_size)]
        if self.world_size == 1:
            return send
        return all_to_all_objects(send, group=None)

    def _dst_shard_info(self, rank: int, batch_size: int) -> _ShardMeta:
        node = self._dst_node(rank)
        replica_id = node.replica_id
        replica_nodes = [key for key in self.dst_tp_members if key.replica_id == replica_id]
        replica_nodes.sort(key=lambda key: (key.pp_rank, key.cp_rank))
        shard_count = len(replica_nodes)
        shard_id = replica_nodes.index(node)
        return _ShardMeta(batch_size=batch_size, shard_count=shard_count, shard_id=shard_id)

    def _slice_shard(self, data: torch.Tensor, meta: _ShardMeta) -> torch.Tensor:
        split_size = math.ceil(meta.batch_size / meta.shard_count)
        start = meta.shard_id * split_size
        end = min(start + split_size, meta.batch_size)
        if start >= meta.batch_size:
            return data.new_empty((0, *data.shape[1:]))
        return data[start:end].contiguous()

    def _tp_fanout(self, payload, members: list[int], leader: int):
        if len(members) == 1:
            return payload
        group = dist.new_group(members)
        object_list = [payload if PM.world_rank == leader else None]
        dist.broadcast_object_list(object_list, src=leader, group=group)
        return object_list[0]

    def broadcast(self, data):
        local_node = self._src_node(PM.world_rank)
        local_members = self.src_tp_members[local_node]
        local_leader = self._src_leader(local_node)
        local_receives = local_node in self.source_nodes

        def _payload_builder(dst_rank: int):
            if PM.world_rank != local_leader or local_node not in self.source_nodes:
                return None
            dst_node = self._src_node(dst_rank)
            if dst_node not in self.source_nodes:
                return None
            if self._src_leader(dst_node) != dst_rank:
                return None
            return data

        recv = self._send(_payload_builder)
        leader_payloads = [item for item in recv if item is not None]
        leader_payload = leader_payloads[0] if leader_payloads else None
        if not local_receives:
            return None
        return self._tp_fanout(leader_payload, local_members, local_leader)

    def forward(self, data: torch.Tensor | None) -> torch.Tensor | None:
        local_src_node = self._src_node(PM.world_rank)
        src_leader = self._src_leader(local_src_node)
        local_is_sender = PM.world_rank == src_leader and local_src_node == self.canonical_source_node_by_replica.get(
            local_src_node.replica_id
        )

        def _payload_builder(dst_rank: int):
            if not local_is_sender or data is None:
                return None
            if dst_rank not in self.forward_targets[PM.world_rank]:
                return None
            meta = self._dst_shard_info(dst_rank, data.shape[0])
            return {
                "tensor": self._slice_shard(data, meta),
                "meta": meta,
            }

        recv = self._send(_payload_builder)
        leader_payloads = [item for item in recv if item is not None]

        local_dst_node = self._dst_node(PM.world_rank)
        local_members = self.dst_tp_members[local_dst_node]
        local_leader = self._dst_leader(local_dst_node)

        leader_payload = leader_payloads[0] if leader_payloads else None
        payload = self._tp_fanout(leader_payload, local_members, local_leader)
        if payload is None:
            self._last_forward_meta = None
            return None

        self._last_forward_meta = payload["meta"]
        return payload["tensor"]

    def backward(self, data: torch.Tensor | None) -> torch.Tensor | None:
        if data is None:
            return None
        if self._last_forward_meta is None:
            raise RuntimeError("MeshConnector.backward() called before forward().")

        local_dst_node = self._dst_node(PM.world_rank)
        dst_leader = self._dst_leader(local_dst_node)
        local_is_sender = PM.world_rank == dst_leader

        def _payload_builder(dst_rank: int):
            if not local_is_sender:
                return None
            if dst_rank not in self.backward_targets[PM.world_rank]:
                return None
            return {
                "tensor": data,
                "meta": self._last_forward_meta,
            }

        recv = self._send(_payload_builder)

        local_src_node = self._src_node(PM.world_rank)
        local_members = self.src_tp_members[local_src_node]
        local_leader = self._src_leader(local_src_node)

        leader_payloads = [item for item in recv if item is not None]
        if PM.world_rank == local_leader:
            shard_tensors = [payload["tensor"] for payload in leader_payloads]
            shard_metas = [payload["meta"] for payload in leader_payloads]
            shard_pairs = sorted(zip(shard_metas, shard_tensors, strict=False), key=lambda item: item[0].shard_id)
            if shard_pairs:
                restored = torch.cat([tensor for _meta, tensor in shard_pairs], dim=0)[
                    : shard_pairs[0][0].batch_size
                ].contiguous()
            else:
                restored = None
        else:
            restored = None
        return self._tp_fanout(restored, local_members, local_leader)

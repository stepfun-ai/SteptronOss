from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.utils.dist_utils import all_to_all_objects, broadcast_tensors


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
        self.src_tp_groups = self._build_tp_groups(self.src_tp_members)
        self.dst_tp_groups = self._build_tp_groups(self.dst_tp_members)

        gathered_source_flags = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_source_flags, is_data_source)

        self.source_nodes = self._compile_source_nodes(gathered_source_flags)
        self.sorted_source_nodes = sorted(
            self.source_nodes, key=lambda node: (node.replica_id, node.pp_rank, node.cp_rank)
        )
        self.canonical_source_node_by_replica = {
            replica_id: sorted(nodes, key=lambda node: (node.pp_rank, node.cp_rank))[0]
            for replica_id, nodes in self._group_nodes_by_replica(self.source_nodes).items()
        }
        self.broadcast_source_rank = self._src_leader(self.sorted_source_nodes[0]) if self.sorted_source_nodes else None

        self.forward_targets = self._compile_forward_targets()
        self.backward_targets = self._compile_backward_targets()

        self._last_forward_meta: _ShardMeta | None = None

    def _describe_mesh(self, mesh: ParallelConfig) -> dict[int, _RankInfo]:
        with PM.use_mesh(mesh):
            mp_groups = PM.world_ranks_of("MP")
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
            mp_replica, _ = _find(mp_groups, rank)
            _, pp_rank = _find(pp_groups, rank)
            _, cp_rank = _find(cp_groups, rank)
            _, tp_rank = _find(tp_groups, rank)
            infos[rank] = _RankInfo(
                replica_id=mp_replica,
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

    def _build_tp_groups(self, members_by_node: dict[_NodeKey, list[int]]) -> dict[_NodeKey, dist.ProcessGroup | None]:
        groups: dict[_NodeKey, dist.ProcessGroup | None] = {}
        for node in sorted(members_by_node, key=lambda key: (key.replica_id, key.pp_rank, key.cp_rank)):
            members = members_by_node[node]
            groups[node] = None if len(members) == 1 else dist.new_group(members)
        return groups

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
        base_size = meta.batch_size // meta.shard_count
        extra = meta.batch_size % meta.shard_count
        start = meta.shard_id * base_size + min(meta.shard_id, extra)
        shard_size = base_size + (1 if meta.shard_id < extra else 0)
        end = min(start + shard_size, meta.batch_size)
        if start >= meta.batch_size:
            return data.new_empty((0, *data.shape[1:]))
        return data[start:end].contiguous()

    def _tp_fanout(self, payload, members: list[int], leader: int, group: dist.ProcessGroup | None):
        if len(members) == 1:
            return payload
        object_list = [payload if PM.world_rank == leader else None]
        dist.broadcast_object_list(object_list, src=leader, group=group)
        return object_list[0]

    def broadcast(self, data):
        if self.broadcast_source_rank is None:
            return None
        local_data = data if PM.world_rank == self.broadcast_source_rank else None
        return broadcast_tensors(local_data, src_rank=self.broadcast_source_rank, group=None, move_to_cuda=False)

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
        payload = self._tp_fanout(leader_payload, local_members, local_leader, self.dst_tp_groups[local_dst_node])
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
            target_device = data.device
            shard_tensors = [payload["tensor"].to(target_device) for payload in leader_payloads]
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
        return self._tp_fanout(restored, local_members, local_leader, self.src_tp_groups[local_src_node])

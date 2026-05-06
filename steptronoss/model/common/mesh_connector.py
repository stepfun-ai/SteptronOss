from __future__ import annotations

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

    def rank_in(self, dim: str) -> int:
        return {
            "PP": self.pp_rank,
            "CP": self.cp_rank,
            "TP": self.tp_rank,
        }[dim]


@dataclass(frozen=True)
class _NodeKey:
    replica_id: int
    pp_rank: int | None = None
    cp_rank: int | None = None
    tp_rank: int | None = None


@dataclass(frozen=True)
class _ShardMeta:
    batch_size: int
    shard_count: int
    shard_id: int


class MeshConnector:
    """Move batch-shaped tensors between two static parallel meshes.

    This is used when an auxiliary module runs under a different mesh from the
    caller, for example text-model ranks sending raw images to a vision encoder
    mesh and receiving encoded image features back.

    A connector views each mesh as replicas made of logical nodes:

    - ``replica_id`` is the data-replica coordinate after removing PP/CP/TP
      from the world-rank layout. Transfers never mix replicas.
    - ``dup_dim`` names dimensions where ranks are expected to hold identical
      payloads for this transfer. Step3V uses ``dup_dim=["TP"]`` because each TP
      rank sees the same raw image batch.
    - A node is the rank coordinate after dropping the duplicate dimensions.
      Ranks in the same node form a duplicate group; only the group leader
      participates in cross-mesh all-to-all, then the leader fans the payload
      out to the other duplicate ranks.

    Public method roles:

    - ``forward(tensor)`` sends batch-dimension shards from each replica's
      canonical source node to destination node leaders, then fans out within
      destination duplicate groups.
    - ``backward(tensor)`` sends destination shards back to source node leaders,
      restores the original batch order, and fans out within source duplicate
      groups.

    The routing table is compiled at construction time from ``src_mesh``,
    ``dst_mesh``, ``is_data_source``, and ``dup_dim``. It assumes the first
    tensor dimension is the shardable batch dimension. Callers are responsible
    for keeping all ranks on the same transfer sequence; ranks with an empty
    local batch should still call ``forward`` / ``backward`` with empty tensors
    rather than skipping the connector collective.

    Compact example for ``dup_dim=["TP"]``:

    ``forward`` sends one duplicate copy and shards by destination nodes::

        src: TP0 DP0  TP1 DP0        dst: DP0  DP1
             [0, 1]   [0, 1]   -->       [0]   [1]

    ``backward`` restores shard order and fans out to source duplicates::

        dst: DP0  DP1                 src: TP0 DP0  TP1 DP0
             f[0] f[1]        -->          f[0,1]    f[0,1]
    """

    _SUPPORTED_DUP_DIMS = ("PP", "CP", "TP")

    def __init__(
        self,
        src_mesh: ParallelConfig,
        dst_mesh: ParallelConfig,
        *,
        is_data_source: bool,
        dup_dim: list[str] | tuple[str, ...] = ("TP",),
    ):
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh
        self.is_data_source = is_data_source
        self.dup_dim = tuple(dim.upper() for dim in dup_dim)
        if len(set(self.dup_dim)) != len(self.dup_dim):
            raise ValueError(f"MeshConnector dup_dim contains duplicates: {self.dup_dim}")
        unsupported = sorted(set(self.dup_dim) - set(self._SUPPORTED_DUP_DIMS))
        if unsupported:
            raise ValueError(
                f"MeshConnector dup_dim only supports {self._SUPPORTED_DUP_DIMS}, got unsupported dims {unsupported}"
            )
        self.world_size = PM.world_size

        self.src_rank_infos = self._describe_mesh(src_mesh)
        self.dst_rank_infos = self._describe_mesh(dst_mesh)
        self.local_src_info = self.src_rank_infos[PM.world_rank]
        self.local_dst_info = self.dst_rank_infos[PM.world_rank]

        self.src_dup_members = self._build_duplicate_members(self.src_rank_infos)
        self.dst_dup_members = self._build_duplicate_members(self.dst_rank_infos)
        self.src_dup_groups = self._build_duplicate_groups(self.src_dup_members)
        self.dst_dup_groups = self._build_duplicate_groups(self.dst_dup_members)

        gathered_source_flags = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_source_flags, is_data_source)

        self.source_nodes = self._compile_source_nodes(gathered_source_flags)
        self.sorted_source_nodes = sorted(self.source_nodes, key=self._node_sort_key)
        self.canonical_source_node_by_replica = {
            replica_id: sorted(nodes, key=self._node_sort_key)[0]
            for replica_id, nodes in self._group_nodes_by_replica(self.source_nodes).items()
        }

        self.forward_targets = self._compile_forward_targets()
        self.backward_targets = self._compile_backward_targets()

        self._last_forward_meta: _ShardMeta | None = None

    def _describe_mesh(self, mesh: ParallelConfig) -> dict[int, _RankInfo]:
        with PM.use_mesh(mesh):
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
            _, pp_rank = _find(pp_groups, rank)
            _, cp_rank = _find(cp_groups, rank)
            _, tp_rank = _find(tp_groups, rank)
            infos[rank] = _RankInfo(
                replica_id=self._mesh_replica_id(mesh, rank),
                pp_rank=pp_rank,
                cp_rank=cp_rank,
                tp_rank=tp_rank,
            )
        return infos

    def _mesh_replica_id(self, mesh: ParallelConfig, rank: int) -> int:
        pp_size = int(mesh.pipeline_model_parallel_size)
        cp_size = int(mesh.context_parallel_size)
        tp_size = int(mesh.tensor_model_parallel_size)
        model_parallel_size = pp_size * cp_size * tp_size
        if model_parallel_size <= 0 or self.world_size % model_parallel_size != 0:
            raise ValueError(
                "MeshConnector requires WORLD_SIZE to be divisible by "
                f"PP*CP*TP, got world_size={self.world_size}, PP={pp_size}, CP={cp_size}, TP={tp_size}"
            )
        data_parallel_size = self.world_size // model_parallel_size
        return (rank // (cp_size * tp_size)) % data_parallel_size

    def _node_key(self, info: _RankInfo) -> _NodeKey:
        return _NodeKey(
            replica_id=info.replica_id,
            pp_rank=None if "PP" in self.dup_dim else info.pp_rank,
            cp_rank=None if "CP" in self.dup_dim else info.cp_rank,
            tp_rank=None if "TP" in self.dup_dim else info.tp_rank,
        )

    @staticmethod
    def _node_sort_key(node: _NodeKey) -> tuple[int, int, int, int]:
        return (
            node.replica_id,
            -1 if node.pp_rank is None else node.pp_rank,
            -1 if node.cp_rank is None else node.cp_rank,
            -1 if node.tp_rank is None else node.tp_rank,
        )

    def _duplicate_sort_key(self, rank_infos: dict[int, _RankInfo], rank: int) -> tuple[int, ...]:
        return tuple(rank_infos[rank].rank_in(dim) for dim in self.dup_dim)

    def _build_duplicate_members(self, rank_infos: dict[int, _RankInfo]) -> dict[_NodeKey, list[int]]:
        members: dict[_NodeKey, list[int]] = {}
        for rank, info in rank_infos.items():
            key = self._node_key(info)
            members.setdefault(key, []).append(rank)
        for ranks in members.values():
            ranks.sort(key=lambda rank: self._duplicate_sort_key(rank_infos, rank))
        return members

    def _group_nodes_by_replica(self, nodes: set[_NodeKey]) -> dict[int, list[_NodeKey]]:
        grouped: dict[int, list[_NodeKey]] = {}
        for node in nodes:
            grouped.setdefault(node.replica_id, []).append(node)
        return grouped

    def _build_duplicate_groups(
        self, members_by_node: dict[_NodeKey, list[int]]
    ) -> dict[_NodeKey, dist.ProcessGroup | None]:
        groups: dict[_NodeKey, dist.ProcessGroup | None] = {}
        for node in sorted(members_by_node, key=self._node_sort_key):
            members = members_by_node[node]
            groups[node] = None if len(members) == 1 else dist.new_group(members)
        return groups

    def _compile_source_nodes(self, gathered_source_flags: list[bool]) -> set[_NodeKey]:
        source_nodes = set()
        for rank, has_data in enumerate(gathered_source_flags):
            if not has_data:
                continue
            info = self.src_rank_infos[rank]
            source_nodes.add(self._node_key(info))
        return source_nodes

    def _src_leader(self, node: _NodeKey) -> int:
        return self.src_dup_members[node][0]

    def _dst_leader(self, node: _NodeKey) -> int:
        return self.dst_dup_members[node][0]

    def _dst_node(self, rank: int) -> _NodeKey:
        return self._node_key(self.dst_rank_infos[rank])

    def _src_node(self, rank: int) -> _NodeKey:
        return self._node_key(self.src_rank_infos[rank])

    def _compile_forward_targets(self) -> dict[int, list[int]]:
        targets = {rank: [] for rank in range(self.world_size)}
        for node_key, _leader_ranks in self.dst_dup_members.items():
            replica_id = node_key.replica_id
            if replica_id not in self.canonical_source_node_by_replica:
                continue
            src_leader = self._src_leader(self.canonical_source_node_by_replica[replica_id])
            targets[src_leader].append(self._dst_leader(node_key))
        return targets

    def _compile_backward_targets(self) -> dict[int, list[int]]:
        targets = {rank: [] for rank in range(self.world_size)}
        replica_sources = self._group_nodes_by_replica(self.source_nodes)
        for node_key, _leader_ranks in self.dst_dup_members.items():
            dst_leader = self._dst_leader(node_key)
            for src_node in sorted(replica_sources.get(node_key.replica_id, []), key=self._node_sort_key):
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
        replica_nodes = [key for key in self.dst_dup_members if key.replica_id == replica_id]
        replica_nodes.sort(key=self._node_sort_key)
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

    def _duplicate_fanout(self, payload, members: list[int], leader: int, group: dist.ProcessGroup | None):
        if len(members) == 1:
            return payload
        object_list = [payload if PM.world_rank == leader else None]
        dist.broadcast_object_list(object_list, src=leader, group=group)
        return object_list[0]

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
        local_members = self.dst_dup_members[local_dst_node]
        local_leader = self._dst_leader(local_dst_node)

        leader_payload = leader_payloads[0] if leader_payloads else None
        payload = self._duplicate_fanout(
            leader_payload, local_members, local_leader, self.dst_dup_groups[local_dst_node]
        )
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
        local_members = self.src_dup_members[local_src_node]
        local_leader = self._src_leader(local_src_node)

        leader_payloads = [item for item in recv if item is not None]
        if PM.world_rank == local_leader:
            target_device = data.device
            shard_tensors = [payload["tensor"].to(target_device) for payload in leader_payloads]
            shard_metas = [payload["meta"] for payload in leader_payloads]
            shard_pairs = sorted(zip(shard_metas, shard_tensors, strict=False), key=lambda item: item[0].shard_id)
            non_empty_shards = [tensor for _meta, tensor in shard_pairs if tensor.shape[0] > 0]
            if non_empty_shards:
                restored = torch.cat(non_empty_shards, dim=0)[: shard_pairs[0][0].batch_size].contiguous()
            elif shard_pairs:
                restored = data.new_empty((0, *data.shape[1:]))
            else:
                restored = None
        else:
            restored = None
        return self._duplicate_fanout(restored, local_members, local_leader, self.src_dup_groups[local_src_node])

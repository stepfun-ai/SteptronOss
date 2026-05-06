from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

import torch

from steptronoss.model.common import mesh_connector as mesh_connector_module
from steptronoss.model.common.mesh_connector import MeshConnector, _NodeKey


def _build_connectors(monkeypatch, src_infos, dst_infos, source_flags, *, dup_dim=("TP",)):
    src_mesh = SimpleNamespace(name="src")
    dst_mesh = SimpleNamespace(name="dst")

    def fake_describe_mesh(self, mesh):
        if mesh is src_mesh:
            return src_infos
        if mesh is dst_mesh:
            return dst_infos
        raise AssertionError(f"unexpected mesh: {mesh}")

    def fake_all_gather_object(output, local_flag):
        del local_flag
        output[:] = list(source_flags)

    monkeypatch.setattr(MeshConnector, "_describe_mesh", fake_describe_mesh)
    monkeypatch.setattr(mesh_connector_module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(mesh_connector_module.dist, "new_group", lambda members: tuple(members))
    monkeypatch.setattr(mesh_connector_module.PM, "world_size", len(src_infos), raising=False)

    connectors = []
    for rank in range(len(src_infos)):
        monkeypatch.setattr(mesh_connector_module.PM, "world_rank", rank, raising=False)
        connectors.append(
            MeshConnector(
                src_mesh=src_mesh,
                dst_mesh=dst_mesh,
                is_data_source=source_flags[rank],
                dup_dim=dup_dim,
            )
        )
    return connectors


def _normalize_payload(payload):
    if payload is None:
        return None
    if isinstance(payload, tuple) and payload and payload[0] in {"tensor", "dataclass"}:
        return payload
    if isinstance(payload, torch.Tensor):
        return ("tensor", payload.clone())
    if is_dataclass(payload):
        return ("dataclass", payload.__class__.__name__, asdict(payload))
    if isinstance(payload, dict):
        return {key: _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_normalize_payload(item) for item in payload)
    return payload


def _assert_payload_equal(actual, expected):
    actual = _normalize_payload(actual)
    expected = _normalize_payload(expected)
    if isinstance(actual, tuple) and actual and actual[0] == "tensor":
        assert expected[0] == "tensor"
        torch.testing.assert_close(actual[1], expected[1])
        return
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_payload_equal(actual[key], expected[key])
        return
    if isinstance(actual, list):
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected, strict=True):
            _assert_payload_equal(a_item, e_item)
        return
    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for a_item, e_item in zip(actual, expected, strict=True):
            _assert_payload_equal(a_item, e_item)
        return
    assert actual == expected


def test_mesh_connector_replica_id_uses_data_coordinate_not_cp_coordinate():
    connector = object.__new__(MeshConnector)
    connector.world_size = 8
    mesh = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        tensor_model_parallel_size=2,
    )

    replica_ids = [connector._mesh_replica_id(mesh, rank) for rank in range(8)]

    assert replica_ids == [0, 0, 0, 0, 1, 1, 1, 1]


def test_mesh_connector_compiles_tp_duplicate_routes(monkeypatch):
    src_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        2: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=1),
    }
    dst_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=1, tp_rank=0),
        2: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=1, tp_rank=0),
    }
    source_flags = [True, False, True, False]

    connectors = _build_connectors(monkeypatch, src_infos, dst_infos, source_flags)
    connector = connectors[0]

    assert connector.source_nodes == {
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=0),
        _NodeKey(replica_id=1, pp_rank=0, cp_rank=0),
    }
    assert connector.src_dup_members == {
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=0): [0, 1],
        _NodeKey(replica_id=1, pp_rank=0, cp_rank=0): [2, 3],
    }
    assert connector.forward_targets == {
        0: [0, 1],
        1: [],
        2: [2, 3],
        3: [],
    }
    assert connector.backward_targets == {
        0: [0],
        1: [0],
        2: [2],
        3: [2],
    }


def test_mesh_connector_can_duplicate_over_cp_instead_of_tp(monkeypatch):
    src_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=1, tp_rank=0),
        2: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        3: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=1, tp_rank=1),
    }
    dst_infos = src_infos.copy()
    source_flags = [True, False, True, False]

    connectors = _build_connectors(monkeypatch, src_infos, dst_infos, source_flags, dup_dim=["CP"])
    connector = connectors[0]

    assert connector.source_nodes == {
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=None, tp_rank=0),
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=None, tp_rank=1),
    }
    assert connector.src_dup_members == {
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=None, tp_rank=0): [0, 1],
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=None, tp_rank=1): [2, 3],
    }
    assert connector.forward_targets == {
        0: [0, 2],
        1: [],
        2: [],
        3: [],
    }
    assert connector.backward_targets == {
        0: [0, 2],
        1: [],
        2: [0, 2],
        3: [],
    }


def test_mesh_connector_collapses_duplicate_source_nodes_but_fans_back_out(monkeypatch):
    src_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=1, tp_rank=0),
        2: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=1, tp_rank=0),
    }
    dst_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        2: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=1, pp_rank=0, cp_rank=0, tp_rank=1),
    }
    source_flags = [True, True, True, True]

    connectors = _build_connectors(monkeypatch, src_infos, dst_infos, source_flags)
    connector = connectors[0]

    assert connector.source_nodes == {
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=0),
        _NodeKey(replica_id=0, pp_rank=0, cp_rank=1),
        _NodeKey(replica_id=1, pp_rank=0, cp_rank=0),
        _NodeKey(replica_id=1, pp_rank=0, cp_rank=1),
    }
    assert connector.canonical_source_node_by_replica == {
        0: _NodeKey(replica_id=0, pp_rank=0, cp_rank=0),
        1: _NodeKey(replica_id=1, pp_rank=0, cp_rank=0),
    }
    assert connector.forward_targets == {
        0: [0],
        1: [],
        2: [2],
        3: [],
    }
    assert connector.backward_targets == {
        0: [0, 1],
        1: [],
        2: [2, 3],
        3: [],
    }


def test_mesh_connector_e2e_forward_backward_with_patched_collectives(monkeypatch):
    src_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        2: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=1),
    }
    dst_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        2: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=1),
    }
    source_flags = [True, False, False, False]
    connectors = _build_connectors(monkeypatch, src_infos, dst_infos, source_flags)

    world_size = len(connectors)
    dup_broadcast_store = {}

    monkeypatch.setattr(
        mesh_connector_module.dist,
        "new_group",
        lambda members: tuple(members),
    )

    def fake_broadcast_object_list(object_list, src, group):
        key = tuple(group)
        rank = mesh_connector_module.PM.world_rank
        if rank == src:
            dup_broadcast_store[key] = copy.deepcopy(object_list[0])
        else:
            object_list[0] = copy.deepcopy(dup_broadcast_store[key])

    monkeypatch.setattr(mesh_connector_module.dist, "broadcast_object_list", fake_broadcast_object_list)

    phases = {}
    current_phase = {"name": None}

    def fake_all_to_all_objects(objects, group=None):
        del group
        phase = phases[current_phase["name"]]
        rank = mesh_connector_module.PM.world_rank
        expected_send = phase["send"][rank]
        assert len(objects) == len(expected_send)
        for actual, expected in zip(objects, expected_send, strict=True):
            _assert_payload_equal(actual, expected)
        return phase["recv"][rank]

    monkeypatch.setattr(mesh_connector_module, "all_to_all_objects", fake_all_to_all_objects)

    source_batch = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    phases["forward"] = {"send": {}, "recv": {}}
    for rank, connector in enumerate(connectors):
        mesh_connector_module.PM.world_rank = rank
        if rank == 0:
            send = []
            for dst_rank in range(world_size):
                if dst_rank in connector.forward_targets[rank]:
                    meta = connector._dst_shard_info(dst_rank, source_batch.shape[0])
                    send.append({"tensor": connector._slice_shard(source_batch, meta), "meta": meta})
                else:
                    send.append(None)
        else:
            send = [None] * world_size
        phases["forward"]["send"][rank] = send
    for rank in range(world_size):
        recv = [phases["forward"]["send"][src_rank][rank] for src_rank in range(world_size)]
        phases["forward"]["recv"][rank] = recv

    current_phase["name"] = "forward"
    forward_outputs = {}
    for rank in [0, 1, 2, 3]:
        mesh_connector_module.PM.world_rank = rank
        forward_outputs[rank] = connectors[rank].forward(source_batch if rank == 0 else None)

    torch.testing.assert_close(forward_outputs[0], torch.tensor([[0.0], [1.0], [2.0]]))
    torch.testing.assert_close(forward_outputs[1], torch.tensor([[0.0], [1.0], [2.0]]))
    torch.testing.assert_close(forward_outputs[2], torch.tensor([[3.0], [4.0]]))
    torch.testing.assert_close(forward_outputs[3], torch.tensor([[3.0], [4.0]]))

    backward_inputs = {
        0: torch.tensor([[10.0], [11.0], [12.0]]),
        1: torch.tensor([[999.0], [999.0], [999.0]]),
        2: torch.tensor([[13.0], [14.0]]),
        3: torch.tensor([[888.0], [888.0]]),
    }
    phases["backward"] = {"send": {}, "recv": {}}
    for rank, connector in enumerate(connectors):
        mesh_connector_module.PM.world_rank = rank
        if connector._dst_leader(connector._dst_node(rank)) == rank:
            send = []
            for dst_rank in range(world_size):
                if dst_rank in connector.backward_targets[rank]:
                    send.append({"tensor": backward_inputs[rank], "meta": connector._last_forward_meta})
                else:
                    send.append(None)
        else:
            send = [None] * world_size
        phases["backward"]["send"][rank] = send
    for rank in range(world_size):
        recv = [phases["backward"]["send"][src_rank][rank] for src_rank in range(world_size)]
        phases["backward"]["recv"][rank] = recv

    current_phase["name"] = "backward"
    backward_outputs = {}
    dup_broadcast_store.clear()
    for rank in [0, 1, 2, 3]:
        mesh_connector_module.PM.world_rank = rank
        backward_outputs[rank] = connectors[rank].backward(backward_inputs[rank])

    expected_full = torch.tensor([[10.0], [11.0], [12.0], [13.0], [14.0]])
    torch.testing.assert_close(backward_outputs[0], expected_full)
    torch.testing.assert_close(backward_outputs[1], expected_full)
    assert backward_outputs[2] is None
    assert backward_outputs[3] is None


def test_mesh_connector_backward_skips_empty_destination_shards(monkeypatch):
    src_infos = {
        0: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=0),
        1: mesh_connector_module._RankInfo(replica_id=0, pp_rank=0, cp_rank=0, tp_rank=1),
        2: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=0),
        3: mesh_connector_module._RankInfo(replica_id=0, pp_rank=1, cp_rank=0, tp_rank=1),
    }
    dst_infos = src_infos.copy()
    source_flags = [True, False, False, False]
    connectors = _build_connectors(monkeypatch, src_infos, dst_infos, source_flags)

    world_size = len(connectors)
    dup_broadcast_store = {}

    def fake_broadcast_object_list(object_list, src, group):
        key = tuple(group)
        rank = mesh_connector_module.PM.world_rank
        if rank == src:
            dup_broadcast_store[key] = copy.deepcopy(object_list[0])
        else:
            object_list[0] = copy.deepcopy(dup_broadcast_store[key])

    monkeypatch.setattr(mesh_connector_module.dist, "broadcast_object_list", fake_broadcast_object_list)

    phases = {}
    current_phase = {"name": None}

    def fake_all_to_all_objects(objects, group=None):
        del group
        phase = phases[current_phase["name"]]
        rank = mesh_connector_module.PM.world_rank
        return phase["recv"][rank]

    monkeypatch.setattr(mesh_connector_module, "all_to_all_objects", fake_all_to_all_objects)

    source_batch = torch.tensor([[0.0]])
    phases["forward"] = {"recv": {}}
    forward_sends = {}
    for rank, connector in enumerate(connectors):
        mesh_connector_module.PM.world_rank = rank
        send = []
        for dst_rank in range(world_size):
            if rank == 0 and dst_rank in connector.forward_targets[rank]:
                meta = connector._dst_shard_info(dst_rank, source_batch.shape[0])
                send.append({"tensor": connector._slice_shard(source_batch, meta), "meta": meta})
            else:
                send.append(None)
        forward_sends[rank] = send
    for rank in range(world_size):
        phases["forward"]["recv"][rank] = [forward_sends[src_rank][rank] for src_rank in range(world_size)]

    current_phase["name"] = "forward"
    forward_outputs = {}
    for rank in range(world_size):
        mesh_connector_module.PM.world_rank = rank
        forward_outputs[rank] = connectors[rank].forward(source_batch if rank == 0 else None)

    torch.testing.assert_close(forward_outputs[0], torch.tensor([[0.0]]))
    torch.testing.assert_close(forward_outputs[1], torch.tensor([[0.0]]))
    torch.testing.assert_close(forward_outputs[2], torch.empty(0, 1))
    torch.testing.assert_close(forward_outputs[3], torch.empty(0, 1))

    backward_inputs = {
        0: torch.tensor([[10.0, 11.0]]),
        1: torch.tensor([[999.0, 999.0]]),
        2: torch.empty(0, 0),
        3: torch.empty(0, 0),
    }
    phases["backward"] = {"recv": {}}
    backward_sends = {}
    for rank, connector in enumerate(connectors):
        mesh_connector_module.PM.world_rank = rank
        send = []
        if connector._dst_leader(connector._dst_node(rank)) == rank:
            for dst_rank in range(world_size):
                if dst_rank in connector.backward_targets[rank]:
                    send.append({"tensor": backward_inputs[rank], "meta": connector._last_forward_meta})
                else:
                    send.append(None)
        else:
            send = [None] * world_size
        backward_sends[rank] = send
    for rank in range(world_size):
        phases["backward"]["recv"][rank] = [backward_sends[src_rank][rank] for src_rank in range(world_size)]

    current_phase["name"] = "backward"
    dup_broadcast_store.clear()
    backward_outputs = {}
    for rank in range(world_size):
        mesh_connector_module.PM.world_rank = rank
        backward_outputs[rank] = connectors[rank].backward(backward_inputs[rank])

    expected = torch.tensor([[10.0, 11.0]])
    torch.testing.assert_close(backward_outputs[0], expected)
    torch.testing.assert_close(backward_outputs[1], expected)
    assert backward_outputs[2] is None
    assert backward_outputs[3] is None

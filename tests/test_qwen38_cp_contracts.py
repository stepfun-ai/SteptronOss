"""CPU checks for CP label boundaries, empty ranks, padding and resume policy."""

import json
from types import SimpleNamespace

import pytest
import torch

from steptronoss.data.qwen3_8_sft import ResumableTextBatches
from steptronoss.model.optimizations import context_parallel_qwen38 as cp
from steptronoss.model.optimizations.chunked_causal_loss import chunked_causal_loss, cp_chunked_causal_loss


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("sparse", [False, True])
def test_cp_loss_preserves_boundaries_and_empty_supervision(monkeypatch, size, sparse):
    torch.manual_seed(9)
    hidden = torch.randn(2, size * 64, 8, requires_grad=True)
    weight = torch.randn(17, 8)
    labels = torch.randint(0, 17, (2, size * 64))
    if sparse:
        labels[:, :] = -100
        labels[:, 64] = 3  # last hidden of rank 0 predicts first target of rank 1
    reference = chunked_causal_loss(hidden, weight, labels, 8)
    expected = torch.autograd.grad(reference, hidden)[0]
    actual = hidden.detach().clone().requires_grad_(True)
    parts = []
    for rank in range(size):
        monkeypatch.setattr(cp, "cp_info", lambda r=rank: (size, r, None))
        local = actual[:, rank * 64 : (rank + 1) * 64]
        parts.append(cp_chunked_causal_loss(local, weight, labels, 8) / size)
    result = sum(parts)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(torch.autograd.grad(result, actual)[0], expected)


def test_padding_preserves_all_tokens_and_masks(tmp_path):
    path = tmp_path / "tokens.jsonl"
    ids = list(range(131))
    path.write_text(json.dumps(dict(input_ids=ids, labels=[-100] + ids[1:])) + "\n")
    data = ResumableTextBatches(path, 0, 1, 1, 1, 256, 0, cp_size=4)
    batch = next(data)
    assert batch["input_ids"].shape == (1, 256)
    assert batch["input_ids"][0, :131].tolist() == ids
    assert batch["labels"][0, 131:].eq(-100).all()
    assert int(batch["attention_mask"].sum()) == 131


@pytest.mark.parametrize("old_cp", [None, 2])
def test_changed_topology_rejects_resume_but_allows_model_only(tmp_path, monkeypatch, old_cp):
    from steptronoss.checkpointing import fsdp2_checkpoint as ck

    manifest = dict(format=ck.FORMAT, world_size=1, iteration=3)
    state = dict(format=ck.FORMAT, world_size=1, rank=0, extra_info={}, model=[{}])
    torch.save(state, tmp_path / "rank0.pt")
    manifest["rank_file_sizes"] = [(tmp_path / "rank0.pt").stat().st_size]
    if old_cp is not None:
        manifest["context_parallel_size"] = old_cp
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(ck.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(ck.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ck.PM, "parallels", {"CP": None})
    monkeypatch.setattr(ck.PM, "size_of", lambda name: 4)
    options = SimpleNamespace(model=True, optimizer=True, scheduler=True, data=True, rng_state=True, iter=True)
    cfg = SimpleNamespace(load_option=options)
    with pytest.raises(ValueError, match="same context_parallel_size"):
        ck.FSDP2Checkpointer().load_ckpt(tmp_path, cfg)
    for k in ("optimizer", "scheduler", "data", "rng_state", "iter"):
        setattr(options, k, False)
    assert ck.FSDP2Checkpointer().load_ckpt(tmp_path, cfg)["model"] == [{}]


def test_empty_bf16_rank_returns_fp32_collective_loss(monkeypatch):
    monkeypatch.setattr(cp, "cp_info", lambda: (2, 0, None))
    hidden = torch.randn(1, 64, 8, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.full((1, 128), -100, dtype=torch.long)
    labels[:, 100] = 3
    loss = cp_chunked_causal_loss(hidden, torch.randn(17, 8), labels, 8)
    assert loss.dtype == torch.float32
    loss.backward()
    assert hidden.grad is not None and hidden.grad.eq(0).all()


def test_gdn_cache_preserves_carried_state_and_gradients():
    pytest.importorskip("transformers", minversion="5.19", reason="Hybrid cache requires Transformers >=5.19")
    recurrent = torch.randn(1, 2, 3, 4, requires_grad=True)
    conv = torch.randn(1, 4, 4, requires_grad=True)
    before = conv.detach().clone()
    next_conv = torch.randn(1, 4, 2, requires_grad=True)
    cache = cp._OneLayerCache(0, recurrent, conv)
    combined = cache.update_conv_state(next_conv, 0)
    cache.update_recurrent_state(recurrent * 2, 0)
    state, halo = cache.harvest()
    torch.testing.assert_close(conv, before)
    torch.testing.assert_close(halo, combined[..., -4:])
    (combined.sum() + state.sum()).backward()
    torch.testing.assert_close(recurrent.grad, torch.full_like(recurrent, 2))
    torch.testing.assert_close(conv.grad, torch.ones_like(conv))
    torch.testing.assert_close(next_conv.grad, torch.ones_like(next_conv))

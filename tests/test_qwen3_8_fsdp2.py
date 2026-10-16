"""CPU math + single-rank FSDP2 lifecycle. Multi-GPU validation is separate.

The single-rank fake process group does not validate distributed communication.
It permits real FSDP2 hooks/DTensor/optimizer execution in a socket-free runner.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from steptronoss.model.optimizations.chunked_causal_loss import chunked_causal_loss
from steptronoss.model.qwen3_8_hf import Qwen38HFConfig, Qwen38HFModel
from tests.test_qwen3_8_sft import tiny_base

pytest.importorskip("torch", minversion="2.11", reason="FSDP2 lifecycle tests require PyTorch >=2.11")


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_chunked_loss_matches_full_projection_and_hidden_grad(dtype, chunk_size):
    torch.manual_seed(41)
    hidden = torch.randn(2, 9, 16, dtype=dtype, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_(True)
    weight = torch.randn(39, 16, dtype=dtype)
    labels = torch.randint(0, 39, (2, 9))
    labels[0, :5] = -100
    labels[1, 3:7] = -100
    loss = chunked_causal_loss(hidden, weight, labels, chunk_size)
    expected = F.cross_entropy(F.linear(reference[:, :-1], weight).float().reshape(-1, 39), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
    (loss * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(
        hidden.grad,
        reference.grad,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=0.002 if dtype == torch.bfloat16 else 1e-6,
    )


@pytest.mark.cpu
def test_loss_guards_and_chunk_bound(monkeypatch):
    hidden = torch.randn(1, 11, 8, requires_grad=True)
    weight = torch.randn(31, 8)
    labels = torch.full((1, 11), -100)
    with pytest.raises(ValueError, match="no supervised"):
        chunked_causal_loss(hidden, weight, labels, 2)
    with pytest.raises(ValueError, match="frozen LM head"):
        chunked_causal_loss(hidden, weight.requires_grad_(True), labels, 2)
    weight.requires_grad_(False)
    labels[:, 3:] = 5
    original, rows = F.linear, []

    def track(x, w, *args, **kwargs):
        rows.append(x.shape[0])
        return original(x, w, *args, **kwargs)

    monkeypatch.setattr(F, "linear", track)
    loss = chunked_causal_loss(hidden, weight, labels, 2)
    loss.backward()
    assert rows and max(rows) <= 2  # Covers both forward and backward allocations.


@pytest.mark.cpu
def test_actual_hybrid_lora_chunked_matches_hf_gradients():
    torch.manual_seed(123)
    cfg = Qwen38HFConfig()
    cfg.params_dtype = torch.float32
    cfg.gradient_checkpointing = True
    cfg.lora_rank = 2
    original = Qwen38HFModel(tiny_base(), cfg)
    cfg2 = Qwen38HFConfig()
    cfg2.params_dtype = torch.float32
    cfg2.lora_rank = 2
    cfg2.loss_backend = "chunked"
    cfg2.loss_chunk_size = 2
    chunked = Qwen38HFModel(tiny_base(), cfg2)
    chunked.hf_model.load_state_dict(original.hf_model.state_dict())
    data = {"input_ids": torch.tensor([[3, 4, 5, 6, 7]]), "labels": torch.tensor([[-100, -100, 5, 6, 7]])}
    loss, reference = chunked(**data), original(**data)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    expected = dict(original.named_parameters())
    for name, param in chunked.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, expected[name].grad, rtol=1e-4, atol=1e-6)


@pytest.fixture
def single_rank_fsdp_group():
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("Single-rank fake PG test must run outside torchrun")
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.mark.cpu
def test_fsdp2_actual_hybrid_update_export_and_local_resume(tmp_path, single_rank_fsdp_group, monkeypatch):
    from steptronoss.checkpointing.fsdp2_checkpoint import FSDP2Checkpointer
    from steptronoss.core.parallel_state import PM
    from steptronoss.core.pipeline_parallel import schedules
    from steptronoss.exp.checkpointing import CheckpointConfig
    from steptronoss.model.qwen3_8_fsdp2 import build_fsdp2_qwen
    from steptronoss.optimizer.fsdp2_gradient_manager import FSDP2GradientManager, pack_local_state

    tiny_base().save_pretrained(tmp_path / "base")
    cfg = Qwen38HFConfig()
    cfg.model_path = str(tmp_path / "base")
    cfg.params_dtype = torch.bfloat16  # Actual mixed frozen-bf16 / adapter-fp32 grouping.
    cfg.lora_rank = 2
    cfg.lora_alpha = 4
    cfg.loss_backend = "chunked"
    cfg.loss_chunk_size = 2
    model = build_fsdp2_qwen(cfg, device="cpu")
    options = SimpleNamespace(clip_grad=1.0, log_num_zeros_in_grad=False)
    params = [p for p in model.parameters() if p.requires_grad]
    manager = FSDP2GradientManager(options, torch.nn.ModuleList([model]), torch.optim.Adam(params, lr=0.01))
    data = {"input_ids": torch.tensor([[3, 4, 5, 6, 7]]), "labels": torch.tensor([[-100, -100, 5, 6, 7]])}
    # Exercise native micro-batch accumulation and its C++ autograd entrypoint
    # with real FSDP2 hooks, rather than only Tensor.backward().
    monkeypatch.setattr(schedules, "is_pipeline_last_stage", lambda: True)
    cfg.check_nan = False
    scheduler = schedules.FWBWScheduler(cfg)
    scheduler.configure(
        models=[model],
        data_iterators=[iter([data, data])],
        data_sync_fn=next,
        data_proc_fn=lambda batch: batch,
        loss_fn=lambda batch, loss: loss,
    )
    scheduler.run(2)
    assert all(p.dtype == torch.float32 for p in params)
    success, norm, _ = manager.step()
    assert success and norm > 0
    saved_model, saved_optim = pack_local_state(model.state_dict()), copy.deepcopy(manager.state_dict())
    model.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert all("lora_" in n for n in saved_model if n != "_qwen38_identity")
    manager.zero_grad()
    expected = model(**data).detach()
    with torch.no_grad():
        for p in params:
            p.zero_()
    model.load_state_dict(saved_model)
    manager.load_state_dict(saved_optim)
    torch.testing.assert_close(model(**data), expected)
    # Validate checkpoint completion and same-world-size guard independently.
    monkeypatch.setattr(PM, "get_all_rng", lambda: {"torch": torch.get_rng_state()})
    monkeypatch.setattr(PM, "set_all_rng", lambda state: torch.set_rng_state(state["torch"]))
    cp_cfg = CheckpointConfig()
    cp_cfg.save_dir = str(tmp_path / "ckpt")
    cp_cfg.exp_name = "test"
    checkpoint = FSDP2Checkpointer()
    state_obj = SimpleNamespace(state_dict=lambda: {"cursor": 2})
    checkpoint.dump_ckpt(cp_cfg, 1, [model], manager, state_obj, state_obj, {"history": {}})
    directory = tmp_path / "ckpt" / "test" / "it1"
    checkpoint.publish_latest(cp_cfg, 1)
    loaded = checkpoint.load_ckpt(directory, cp_cfg)
    assert loaded["iteration"] == 1 and loaded["data"]["cursor"] == 2
    model.load_state_dict(loaded["model"][0])
    manager.load_state_dict(loaded["optimizer"])
    (directory / "rank0.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint.load_ckpt(directory, cp_cfg)


@pytest.mark.cpu
def test_fsdp_experiment_rejects_native_zero_and_resume_changes(monkeypatch):
    from playground.sft.qwen3_8.qwen3_8_27b_fsdp2_lora import Exp

    monkeypatch.setenv("WORLD_SIZE", "8")
    exp = Exp()
    exp.sanity_check()
    old = {"exp": exp.to_dict()}
    exp.model_cfg.loss_chunk_size = 64
    with pytest.raises(ValueError, match="loss_chunk_size"):
        exp.assert_critical_attrs_expected(old)
    exp = Exp()
    exp.optimizer_cfg.use_distributed_optimizer = True
    with pytest.raises(ValueError, match="ZeRO-1"):
        exp.sanity_check()


@pytest.mark.cpu
def test_meta_builder_has_same_parameter_shapes_and_dtypes(tmp_path):
    tiny_base().save_pretrained(tmp_path / "base")
    cfg = Qwen38HFConfig()
    cfg.model_path = str(tmp_path / "base")
    cfg.params_dtype = torch.bfloat16
    normal, meta = cfg.build_model(), cfg.build_model(meta_init=True)
    expected = {n: (p.shape, p.dtype) for n, p in normal.named_parameters()}
    assert expected == {n: (p.shape, p.dtype) for n, p in meta.named_parameters()}
    assert all(p.is_meta for p in meta.parameters())

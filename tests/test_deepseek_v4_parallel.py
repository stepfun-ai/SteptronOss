import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.xdist_group("torchrun"),
    pytest.mark.gpu,
    pytest.mark.node2,
]


@pytest.fixture(scope="session", autouse=True)
def init_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek V4 distributed tests")
        return
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported")
        return

    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    except Exception as e:
        print(f"Failed to initialize torch.distributed: {e}")
        pytest.skip("Failed to initialize torch.distributed")
        return

    if dist.get_world_size() != 2:
        pytest.skip(
            "Need 2 processes in dist group. "
            "You can run with `torchrun --nproc-per-node=2 -m pytest tests/test_deepseek_v4_parallel.py`."
        )

    from steptronoss.core.parallel_state import PM

    PM.initialize(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_parallel(tp: int, pp: int = 1, ep: int = 1):
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import ParallelConfig

    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = tp
    parallel_cfg.pipeline_model_parallel_size = pp
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = ep
    parallel_cfg.expert_tensor_parallel_size = 1
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)
    return parallel_cfg


def _make_toy_cfg(num_layers: int, tp: int, pp: int = 1, ep: int = 1, *, hash_moe: bool = False):
    from playground.pretrain.deepseek_v4.deepseek_v4_toy import DeepseekV4ToyModelConfig

    cfg = DeepseekV4ToyModelConfig()
    cfg.num_layers = num_layers
    cfg.global_seq_length = 4
    cfg.micro_batch_size = 1
    cfg.attn_cfg.layer_types = ["sliding_attention"] * num_layers
    cfg.moe_cfg.mlp_layer_types = ["hash_moe" if hash_moe else "moe"] * num_layers
    cfg.parallel_cfg.tensor_model_parallel_size = tp
    cfg.parallel_cfg.pipeline_model_parallel_size = pp
    cfg.parallel_cfg.context_parallel_size = 1
    cfg.parallel_cfg.expert_model_parallel_size = ep
    cfg.parallel_cfg.expert_tensor_parallel_size = 1
    cfg.parallel_cfg.virtual_pipeline_model_parallel_size = 1
    cfg.tp_cfg.sequence_parallel = False
    cfg.tp_cfg.async_tensor_model_parallel_allreduce = False
    cfg.tp_cfg.gradient_accumulation_fusion = False
    cfg.sanity_check()
    return cfg


def _force_hash_routes_to_both_ep_ranks(model):
    for layer in model.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        tid2eid = getattr(gate, "tid2eid", None)
        if tid2eid is None:
            continue
        with torch.no_grad():
            tid2eid[:, 0].fill_(0)
            tid2eid[:, 1].fill_(model.cfg.moe_cfg.n_routed_experts // 2)


def _slice_for_local_parallel(key: str, tensor: torch.Tensor, cfg):
    from steptronoss.core.parallel_state import PM

    tp_rank = PM.rank_in("TP")
    ep_rank = PM.rank_in("EP")
    tp_size = PM.size_of("TP")
    ep_size = PM.size_of("EP")
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return tensor.chunk(tp_size, dim=0)[tp_rank].contiguous()
    if key.endswith(".self_attn.q_b_proj.weight"):
        return tensor.chunk(tp_size, dim=0)[tp_rank].contiguous()
    if key.endswith(".self_attn.o_b_proj.weight"):
        return tensor.chunk(tp_size, dim=1)[tp_rank].contiguous()
    if key.endswith(".self_attn.o_a_proj.weight"):
        local_o_groups = cfg.attn_cfg.o_groups // tp_size
        start = tp_rank * local_o_groups * cfg.attn_cfg.o_lora_rank
        end = start + local_o_groups * cfg.attn_cfg.o_lora_rank
        return tensor[start:end].contiguous()
    if key.endswith(".self_attn.sinks"):
        return tensor.chunk(tp_size, dim=0)[tp_rank].contiguous()
    if key.endswith(".mlp.experts.gate_up_proj") or key.endswith(".mlp.experts.down_proj"):
        return tensor.chunk(ep_size, dim=0)[ep_rank].contiguous()
    return tensor


def _load_parallel_from_hf(local_model, hf_state_dict, cfg):
    translated = {}
    for local_key, hf_key in local_model._hf_key_map.items():
        if hf_key in hf_state_dict:
            translated[local_key] = _slice_for_local_parallel(hf_key, hf_state_dict[hf_key], cfg)
    local_model.load_state_dict(translated, strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSeek V4 parallel test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
@pytest.mark.parametrize("tp,ep,hash_moe", [(2, 1, False), (1, 2, True), (2, 2, True)])
def test_deepseek_v4_toy_tp_ep_fwbw_smoke(tp, ep, hash_moe):
    from steptronoss.core.parallel_state import PM

    _set_parallel(tp=tp, ep=ep)
    torch.manual_seed(1234)
    cfg = _make_toy_cfg(num_layers=1, tp=tp, ep=ep, hash_moe=hash_moe)
    model = cfg.build_model().cuda().bfloat16()
    if hash_moe:
        _force_hash_routes_to_both_ep_ranks(model)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
    logits = model.abstract_forward(input_ids=input_ids)
    assert logits.shape == (4, 1, cfg.vocab_size // PM.size_of("TP"))
    assert torch.isfinite(logits).all().item()

    loss = logits.float().square().mean()
    loss.backward()

    attn = model.layers[0].self_attn
    assert attn.num_local_heads == cfg.attn_cfg.num_attention_heads // tp
    assert attn.sinks.shape == (attn.num_local_heads,)
    assert attn.q_b_proj.weight.grad is not None
    assert attn.o_b_proj.weight.grad is not None

    experts = model.layers[0].mlp.experts
    assert experts.gate_up_proj.shape[0] == cfg.moe_cfg.n_routed_experts // ep
    if ep > 1:
        assert experts.gate_up_proj.grad is not None
        assert experts.gate_up_proj.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSeek V4 PP test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_deepseek_v4_toy_pp_stage_build_and_chunk_shapes():
    from steptronoss.core.parallel_state import PM
    from steptronoss.core.pipeline_parallel import p2p_communication as p2p

    _set_parallel(tp=1, pp=2, ep=1)
    torch.manual_seed(1234)
    cfg = _make_toy_cfg(num_layers=2, tp=1, pp=2, ep=1)
    model = cfg.build_model().cuda().bfloat16()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")

    layer_ids = [layer.layer_id for layer in model.layers if hasattr(layer, "layer_id")]
    if PM.rank_in("PP") == 0:
        assert layer_ids == [0]
        assert hasattr(model, "tok_embeddings")
        assert not hasattr(model, "out_embeddings")
        hidden_states = model.forward_head(input_ids=input_ids)
        streams = model.forward_chunk(hidden_states, input_ids=input_ids)
        assert streams.shape == (4, 1, cfg.hc_cfg.hc_mult, cfg.hidden_size)
        p2p.send_forward(cfg, streams)
    else:
        assert layer_ids == [1]
        assert not hasattr(model, "tok_embeddings")
        assert hasattr(model, "out_embeddings")
        hidden_states = p2p.recv_forward(cfg)
        assert hidden_states.shape == (4, 1, cfg.hc_cfg.hc_mult, cfg.hidden_size)
        streams = model.forward_chunk(hidden_states, input_ids=input_ids)
        logits = model.forward_tail(streams)
        assert logits.shape == (4, 1, cfg.vocab_size)
        assert torch.isfinite(logits).all().item()
    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSeek V4 parallel parity requires CUDA")
def test_deepseek_v4_toy_tp2_ep2_fwbw_matches_huggingface():
    from test_deepseek_v4_hf_parity import _head_values, _load_hf_deepseek_v4, _make_hf_config, _make_local_config

    from steptronoss.core import tensor_parallel
    from steptronoss.core.parallel_state import PM

    hf_config_cls, hf_model_cls = _load_hf_deepseek_v4()
    _set_parallel(tp=2, ep=2)
    torch.manual_seed(1234)

    hf_model = hf_model_cls(_make_hf_config(hf_config_cls)).cuda().float()
    local_cfg = _make_local_config()
    local_cfg.parallel_cfg.tensor_model_parallel_size = 2
    local_cfg.parallel_cfg.expert_model_parallel_size = 2
    local_cfg.parallel_cfg.expert_tensor_parallel_size = 1
    local_cfg.tp_cfg.async_tensor_model_parallel_allreduce = False
    local_cfg.global_seq_length = input_seq_len = 5
    local_cfg.micro_batch_size = 1
    local_cfg.sanity_check()
    local_model = local_cfg.build_model().cuda().float()
    _load_parallel_from_hf(local_model, hf_model.state_dict(), local_cfg)

    input_ids = torch.tensor([[3, 7, 11, 5, 13]], dtype=torch.long, device="cuda")
    assert input_ids.shape[1] == input_seq_len
    hf_model.train()
    local_model.train()

    hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
    local_sharded_logits = local_model.abstract_forward(input_ids=input_ids)
    local_logits = tensor_parallel.gather_from_tensor_model_parallel_region(local_sharded_logits).transpose(0, 1)
    torch.testing.assert_close(local_logits, hf_logits, rtol=2e-4, atol=2e-4)

    hf_loss = hf_logits[..., [1, 3, 5]].sum()
    local_loss = local_logits[..., [1, 3, 5]].sum()
    hf_loss.backward()
    local_loss.backward()

    hf_params = dict(hf_model.named_parameters())
    local_params = dict(local_model.named_parameters())
    checked_keys = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_proj.weight",
        "model.layers.0.self_attn.o_a_proj.weight",
        "model.layers.0.self_attn.o_b_proj.weight",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "lm_head.weight",
    }
    max_grad_diff = 0.0
    max_grad_diff_key = None
    grad_samples = {}
    for local_key, hf_key in local_model._hf_key_map.items():
        if hf_key not in checked_keys:
            continue
        local_grad = local_params[local_key].grad
        hf_grad = _slice_for_local_parallel(hf_key, hf_params[hf_key].grad, local_cfg)
        torch.testing.assert_close(local_grad, hf_grad, rtol=5e-4, atol=5e-4, msg=hf_key)
        grad_diff = (local_grad - hf_grad).abs().max().item()
        if grad_diff > max_grad_diff:
            max_grad_diff = grad_diff
            max_grad_diff_key = hf_key
        grad_samples[hf_key] = {
            "local_shape": tuple(local_grad.shape),
            "local": _head_values(local_grad),
            "hf": _head_values(hf_grad),
            "max_abs_diff": grad_diff,
        }

    snapshot = {
        "rank": PM.world_rank,
        "logits_slice": local_logits[0, 0, :6].detach().cpu().tolist(),
        "hf_loss": hf_loss.detach().cpu().item(),
        "local_loss": local_loss.detach().cpu().item(),
        "max_logits_abs_diff": (local_logits - hf_logits).abs().max().detach().cpu().item(),
        "max_checked_grad_abs_diff": max_grad_diff,
        "max_checked_grad_key": max_grad_diff_key,
        "grad_samples": grad_samples,
    }
    print(f"DEEPSEEK_V4_TP2_EP2_FWBW_SNAPSHOT={snapshot}")
    assert snapshot["max_logits_abs_diff"] < 1e-7
    assert snapshot["max_checked_grad_abs_diff"] < 1e-6

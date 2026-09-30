import importlib

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _patch_transformers_main_imports():
    """Patch transient Transformers-main lazy-import metadata gaps in test only."""

    try:
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return
    mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if isinstance(mapping, dict):
        mapping.setdefault("flash_attn_interface", ["flash-attn"])


def _load_hf_deepseek_v4():
    _patch_transformers_main_imports()
    try:
        transformers = importlib.import_module("transformers")
        return transformers.DeepseekV4Config, transformers.DeepseekV4ForCausalLM
    except (AttributeError, ImportError):
        pass

    try:
        module = importlib.import_module("transformers.models.deepseek_v4")
        return module.DeepseekV4Config, module.DeepseekV4ForCausalLM
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"installed transformers does not provide DeepSeek V4 yet: {exc}")


def _make_local_config():
    from steptronoss.model.deepseek_v4 import DeepseekV4ModelConfig

    cfg = DeepseekV4ModelConfig()
    cfg.vocab_size = 32
    cfg.hidden_size = 16
    cfg.num_layers = 1
    cfg.layernorm_epsilon = 1e-5
    cfg.params_dtype = torch.float32

    cfg.attn_cfg.num_attention_heads = 2
    cfg.attn_cfg.num_key_value_heads = 1
    cfg.attn_cfg.head_dim = 8
    cfg.attn_cfg.q_lora_rank = 8
    cfg.attn_cfg.qk_rope_head_dim = 4
    cfg.attn_cfg.rope_theta = 10000.0
    cfg.attn_cfg.compress_rope_theta = 160000.0
    cfg.attn_cfg.attention_dropout = 0.0
    cfg.attn_cfg.sliding_window = 8
    cfg.attn_cfg.layer_types = ["sliding_attention"]
    cfg.attn_cfg.o_groups = 2
    cfg.attn_cfg.o_lora_rank = 4
    cfg.attn_cfg.index_n_heads = 2
    cfg.attn_cfg.index_head_dim = 4
    cfg.attn_cfg.index_topk = 2
    cfg.attn_cfg.layernorm_epsilon = cfg.layernorm_epsilon

    cfg.moe_cfg.moe_intermediate_size = 12
    cfg.moe_cfg.num_experts_per_tok = 2
    cfg.moe_cfg.n_routed_experts = 4
    cfg.moe_cfg.n_shared_experts = 1
    cfg.moe_cfg.scoring_func = "sigmoid"
    cfg.moe_cfg.routed_scaling_factor = 1.5
    cfg.moe_cfg.swiglu_limit = 10.0
    cfg.moe_cfg.mlp_layer_types = ["moe"]
    cfg.moe_cfg.layernorm_epsilon = cfg.layernorm_epsilon

    cfg.hc_cfg.hc_mult = 2
    cfg.hc_cfg.hc_sinkhorn_iters = 2
    cfg.hc_cfg.hc_eps = 1e-6
    cfg.hc_cfg.layernorm_epsilon = cfg.layernorm_epsilon

    cfg.tok_embed_cfg.vocab_size = cfg.vocab_size
    cfg.out_embed_cfg.vocab_size = cfg.vocab_size
    cfg.out_embed_cfg.layernorm_epsilon = cfg.layernorm_epsilon
    cfg.out_embed_cfg.fp32_lm_head_out = True
    return cfg


def _make_hf_config(hf_config_cls):
    return hf_config_cls(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        partial_rotary_factor=0.5,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        max_position_embeddings=64,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        layer_types=["sliding_attention"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        mlp_layer_types=["moe"],
        swiglu_limit=10.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
    )


SNAPSHOT_GRAD_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.o_b_proj.weight",
    "lm_head.weight",
)

EXPECTED_SNAPSHOT = {
    "hf_logits_slice": [
        -0.03734314814209938,
        0.07475943863391876,
        0.029178069904446602,
        0.04551984369754791,
        -0.10299525409936905,
        -0.009581279940903187,
    ],
    "local_logits_slice": [
        -0.03734314814209938,
        0.07475943863391876,
        0.029178069904446602,
        0.04551984369754791,
        -0.10299525409936905,
        -0.009581279940903187,
    ],
    "hf_loss": 0.2017396092414856,
    "local_loss": 0.2017396092414856,
    "max_logits_abs_diff": 0.0,
    "grad_samples": {
        "model.embed_tokens.weight": {
            "hf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "local": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "max_abs_diff": 5.960464477539063e-08,
        },
        "model.layers.0.self_attn.q_a_proj.weight": {
            "hf": [
                0.06729774922132492,
                -0.050311435014009476,
                0.12994006276130676,
                0.019431836903095245,
                0.029320677742362022,
                -0.1035708412528038,
            ],
            "local": [
                0.06729774922132492,
                -0.050311435014009476,
                0.12994006276130676,
                0.019431836903095245,
                0.029320677742362022,
                -0.1035708412528038,
            ],
            "max_abs_diff": 0.0,
        },
        "model.layers.0.self_attn.o_b_proj.weight": {
            "hf": [
                -0.15786071121692657,
                -0.08581886440515518,
                0.044399797916412354,
                0.10218603163957596,
                -0.0053260428830981255,
                0.023058559745550156,
            ],
            "local": [
                -0.15786071121692657,
                -0.08581886440515518,
                0.044399797916412354,
                0.10218603163957596,
                -0.0053260428830981255,
                0.023058559745550156,
            ],
            "max_abs_diff": 3.725290298461914e-09,
        },
        "lm_head.weight": {
            "hf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "local": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "max_abs_diff": 0.0,
        },
    },
}


def _head_values(tensor: torch.Tensor, count: int = 6) -> list[float]:
    return tensor.detach().flatten()[:count].cpu().tolist()


def _assert_values_close(actual: list[float], expected: list[float], *, atol: float = 1e-7):
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=0.0, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSeek V4 parity needs CUDA for repo modules")
def test_deepseek_v4_toy_fwbw_matches_huggingface():
    hf_config_cls, hf_model_cls = _load_hf_deepseek_v4()
    torch.manual_seed(1234)

    hf_model = hf_model_cls(_make_hf_config(hf_config_cls)).cuda().float()
    local_cfg = _make_local_config()
    from steptronoss.core.parallel_state import PM

    PM.initialize()
    PM.set_mesh(local_cfg.parallel_cfg)
    local_model = local_cfg.build_model().cuda().float()
    local_model.load_hf_state_dict(hf_model.state_dict(), strict=True)

    input_ids = torch.tensor([[3, 7, 11, 5, 13]], dtype=torch.long, device="cuda")
    hf_model.train()
    local_model.train()

    hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
    local_logits = local_model.abstract_forward(input_ids=input_ids).transpose(0, 1)

    torch.testing.assert_close(local_logits, hf_logits, rtol=2e-4, atol=2e-4)

    hf_loss = hf_logits[..., [1, 3, 5]].sum()
    local_loss = local_logits[..., [1, 3, 5]].sum()
    hf_loss.backward()
    local_loss.backward()

    snapshot = {
        "hf_logits_slice": _head_values(hf_logits[0, 0, :6]),
        "local_logits_slice": _head_values(local_logits[0, 0, :6]),
        "hf_loss": hf_loss.detach().cpu().item(),
        "local_loss": local_loss.detach().cpu().item(),
        "max_logits_abs_diff": (local_logits - hf_logits).abs().max().detach().cpu().item(),
        "grad_samples": {},
    }

    hf_params = dict(hf_model.named_parameters())
    local_params = dict(local_model.named_parameters())
    for local_key, hf_key in local_model._hf_key_map.items():
        if hf_key not in hf_params:
            continue
        local_grad = local_params[local_key].grad
        hf_grad = hf_params[hf_key].grad
        if local_grad is None and hf_grad is None:
            continue
        assert local_grad is not None, local_key
        assert hf_grad is not None, hf_key
        torch.testing.assert_close(local_grad, hf_grad, rtol=5e-4, atol=5e-4, msg=hf_key)
        if hf_key in SNAPSHOT_GRAD_KEYS:
            snapshot["grad_samples"][hf_key] = {
                "hf": _head_values(hf_grad),
                "local": _head_values(local_grad),
                "max_abs_diff": (local_grad - hf_grad).abs().max().detach().cpu().item(),
            }

    print(f"DEEPSEEK_V4_FWBW_SNAPSHOT={snapshot}")
    _assert_values_close(snapshot["local_logits_slice"], snapshot["hf_logits_slice"])
    assert snapshot["local_loss"] == pytest.approx(snapshot["hf_loss"], abs=1e-7)
    assert snapshot["max_logits_abs_diff"] < 1e-7
    assert snapshot["grad_samples"].keys() == set(SNAPSHOT_GRAD_KEYS)
    for actual in snapshot["grad_samples"].values():
        _assert_values_close(actual["local"], actual["hf"])
        assert actual["max_abs_diff"] < 1e-7

    _assert_values_close(snapshot["hf_logits_slice"], EXPECTED_SNAPSHOT["hf_logits_slice"])
    _assert_values_close(snapshot["local_logits_slice"], EXPECTED_SNAPSHOT["local_logits_slice"])
    assert snapshot["hf_loss"] == pytest.approx(EXPECTED_SNAPSHOT["hf_loss"], abs=1e-7)
    assert snapshot["local_loss"] == pytest.approx(EXPECTED_SNAPSHOT["local_loss"], abs=1e-7)
    assert snapshot["max_logits_abs_diff"] == pytest.approx(EXPECTED_SNAPSHOT["max_logits_abs_diff"], abs=1e-10)
    assert snapshot["grad_samples"].keys() == EXPECTED_SNAPSHOT["grad_samples"].keys()
    for hf_key, expected in EXPECTED_SNAPSHOT["grad_samples"].items():
        actual = snapshot["grad_samples"][hf_key]
        _assert_values_close(actual["hf"], expected["hf"])
        _assert_values_close(actual["local"], expected["local"])
        assert actual["max_abs_diff"] == pytest.approx(expected["max_abs_diff"], abs=1e-10)

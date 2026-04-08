import pytest
import torch

from playground.pretrain.glm5.glm5 import GLM5Config
from playground.pretrain.glm5.glm5_toy import GLM5ToyConfig
from playground.sft.glm5.glm5_sft_step3_data import Exp
from playground.sft.glm5.glm5_toy_sft_step3_data import Exp as ToyExp
from steptronoss.model.glm5 import Glm5Attention

pytestmark = pytest.mark.cpu


def test_glm5_model_config_matches_expected_core_shape():
    cfg = GLM5Config()

    assert cfg.num_layers == 78
    assert cfg.hidden_size == 6144
    assert cfg.attn_cfg.num_attention_heads == 64
    assert cfg.attn_cfg.q_lora_rank == 2048
    assert cfg.attn_cfg.mla_layernorm_epsilon == 1e-6
    assert cfg.ffn_cfg.moe_cfg.moe_num_experts == 256
    assert cfg.ffn_cfg.moe_cfg.enable_auxiliary_loss_free_load_balance is True
    assert cfg.ffn_cfg.moe_cfg.moe_layer_list[0] == 3
    assert cfg.ffn_cfg.moe_cfg.moe_layer_list[-1] == 77


def test_glm5_sft_exp_has_consistent_parallel_plan():
    exp = Exp()
    exp.sanity_check()

    assert exp.resource_cfg.replica == 13
    assert exp.resource_cfg.gpu == 8
    assert exp.model_cfg.parallel_cfg.tensor_model_parallel_size == 8
    assert exp.model_cfg.parallel_cfg.pipeline_model_parallel_size == 13
    assert exp.model_cfg.parallel_cfg.expert_model_parallel_size == 8
    assert exp.trainer_cfg.global_seq_length == 8192


def test_glm5_toy_config_keeps_full_structure_with_fewer_layers():
    cfg = GLM5ToyConfig()

    assert cfg.num_layers == 6
    assert cfg.hidden_size == 6144
    assert cfg.attn_cfg.num_attention_heads == 64
    assert cfg.attn_cfg.mla_layernorm_epsilon == 1e-6
    assert cfg.ffn_cfg.moe_cfg.moe_num_experts == 256
    assert cfg.ffn_cfg.moe_cfg.moe_layer_list == [3, 4, 5]
    assert cfg.parallel_cfg.pipeline_model_parallel_size == 1


def test_glm5_toy_sft_exp_is_single_node():
    exp = ToyExp()
    exp.sanity_check()

    assert exp.resource_cfg.replica == 1
    assert exp.resource_cfg.gpu == 8
    assert exp.model_cfg.parallel_cfg.tensor_model_parallel_size == 8
    assert exp.model_cfg.parallel_cfg.pipeline_model_parallel_size == 1
    assert exp.trainer_cfg.global_seq_length == 1024


def test_glm5_indexer_mask_is_causal_for_plain_sequences():
    mask = Glm5Attention._build_indexer_mask(
        batch_size=2,
        seq_len=4,
        cu_seqlens=None,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [0.0, float("-inf"), float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mask[0], expected)
    assert torch.equal(mask[1], expected)

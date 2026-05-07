"""
Toy DeepSeek V4 model config for fast training and HF parity checks.

The toy keeps the architectural features that are specific to DeepSeek V4
(mHC, shared-KV attention with sinks, grouped output projection, and routed
MoE) while shrinking dimensions enough for smoke runs. It intentionally uses
`sliding_attention` and standard `moe` on every layer so the first training path
does not depend on long-context compressor or hash-router resources.
"""

import torch

from steptronoss.model.deepseek_v4 import DeepseekV4ModelConfig


class DeepseekV4ToyModelConfig(DeepseekV4ModelConfig):
    def __init__(self):
        super().__init__()
        self.vocab_size = 2048
        self.hidden_size = 256
        self.num_layers = 2
        self.layernorm_epsilon = 1e-5
        self.params_dtype = torch.bfloat16

        self.attn_cfg.num_attention_heads = 4
        self.attn_cfg.num_key_value_heads = 1
        self.attn_cfg.head_dim = 64
        self.attn_cfg.q_lora_rank = 128
        self.attn_cfg.qk_rope_head_dim = 32
        self.attn_cfg.rope_theta = 10000.0
        self.attn_cfg.compress_rope_theta = 160000.0
        self.attn_cfg.sliding_window = 128
        self.attn_cfg.layer_types = ["sliding_attention"] * self.num_layers
        self.attn_cfg.o_groups = 4
        self.attn_cfg.o_lora_rank = 64
        self.attn_cfg.index_n_heads = 4
        self.attn_cfg.index_head_dim = 32
        self.attn_cfg.index_topk = 16
        self.attn_cfg.layernorm_epsilon = self.layernorm_epsilon

        self.moe_cfg.moe_intermediate_size = 384
        self.moe_cfg.num_experts_per_tok = 2
        self.moe_cfg.n_routed_experts = 8
        self.moe_cfg.n_shared_experts = 1
        self.moe_cfg.scoring_func = "sqrtsoftplus"
        self.moe_cfg.routed_scaling_factor = 1.5
        self.moe_cfg.swiglu_limit = 10.0
        self.moe_cfg.mlp_layer_types = ["moe"] * self.num_layers
        self.moe_cfg.layernorm_epsilon = self.layernorm_epsilon

        self.hc_cfg.hc_mult = 2
        self.hc_cfg.hc_sinkhorn_iters = 4
        self.hc_cfg.hc_eps = 1e-6
        self.hc_cfg.layernorm_epsilon = self.layernorm_epsilon

        self.tok_embed_cfg.vocab_size = self.vocab_size
        self.out_embed_cfg.vocab_size = self.vocab_size
        self.out_embed_cfg.layernorm_epsilon = self.layernorm_epsilon

    def build_model(self):
        model = super().build_model()
        self._init_toy_weights(model)
        return model

    @staticmethod
    def _init_toy_weights(model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("norm.weight") or name.endswith("q_a_norm.weight") or name.endswith("kv_norm.weight"):
                    param.fill_(1.0)
                elif name.endswith(".base") or name.endswith("hc_base") or name.endswith("sinks"):
                    param.zero_()
                elif name.endswith(".scale") or name.endswith("hc_scale"):
                    param.fill_(1.0)
                elif "position_bias" in name:
                    param.zero_()
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            for buffer_name, buffer in model.named_buffers():
                if buffer_name.endswith("e_score_correction_bias") or buffer_name.endswith("tid2eid"):
                    buffer.zero_()

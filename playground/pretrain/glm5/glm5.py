import torch
from configurize import Ref

from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.moe_block import MoEConfig
from steptronoss.model.common.moe_share_expert_ffn import MoEFeedForwardConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
)
from steptronoss.model.decoder_model import DecoderLLMConfig


class Glm5AttentionConfig(AttentionConfig):
    """GLM-5 attention configuration."""

    q_lora_rank: int
    kv_lora_rank: int
    mla_layernorm_epsilon: float
    qk_nope_head_dim: int
    v_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    index_layernorm_epsilon: float
    dsa_indexer_query_chunk_size: int
    dsa_attention_query_chunk_size: int

    def __init__(self):
        super().__init__()
        self.causal = True
        self.attention_dropout = 0.0
        self.use_sliding_window = False

        self.num_attention_heads = 64
        self.num_attention_groups = 64

        self.head_dim = 256
        self.hidden_size = Ref("..hidden_size")

        self.use_headwise_attn_gate = False
        self.use_qkv_bias = False
        self.sliding_window_size = None

        self.use_qk_norm = False
        self.layernorm_epsilon = 1e-5
        self.mla_layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute_qknorm_rope = False

        self.q_lora_rank = 2048
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.qk_nope_head_dim = 192
        self.v_head_dim = 256

        self.rope_theta = 1_000_000.0
        self.yarn_beta_slow = 1.0
        self.yarn_beta_fast = 32.0
        self.ntk_interp_ratio = 1.0
        self.max_position_embeddings = 202752

        self.index_head_dim = 128
        self.index_n_heads = 32
        self.index_topk = 2048
        self.index_layernorm_epsilon = 1e-6
        self.dsa_indexer_query_chunk_size = 32
        self.dsa_attention_query_chunk_size = 32

    def build_model(self, layer_id: int):
        from steptronoss.model.glm5 import Glm5Attention

        return Glm5Attention(cfg=self, layer_id=layer_id)


class Glm5MoEConfig(MoEConfig):
    """GLM-5 routed-expert configuration."""

    def __init__(self):
        super().__init__()
        self.tp_cfg = Ref("...tp_cfg")
        self.hidden_size = Ref("...hidden_size")
        self.activation = Ref("..activation")

        self.moe_num_experts = 256
        self.moe_top_k = 8
        self.moe_aux_loss_coef = 0.0
        self.moe_hidden_size = 2048
        self.fp32_gate_output = True

        self.routed_scaling_factor = 2.5
        self.enable_sigmoid_router = True
        self.router_bias_update_rate = 0.0
        self.enable_auxiliary_loss_free_load_balance = True
        self.norm_expert_weight = True
        self.force_balance = False

        self.moe_layer_list = []
        self.share_expert_dim = 2048


class Glm5MoEFeedForwardConfig(MoEFeedForwardConfig):
    """GLM-5 dense+MoE feed-forward configuration."""

    moe_cfg = Glm5MoEConfig

    def __init__(self):
        super().__init__()
        self.hidden_size = Ref("..hidden_size")
        self.ffn_hidden_size = 12288
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = False
        self.swiglu_recompute_silu_out_proj = True


class Glm5InputEmbeddingConfig(InputEmbeddingConfig):
    """GLM-5 input embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 154880
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Glm5OutputEmbeddingConfig(OutputEmbeddingConfig):
    """GLM-5 output embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 154880
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True
        self.fp32_lm_head_out = True

        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-5

        self.gather_output = False


class Glm5ParallelConfig(ParallelConfig):
    """GLM-5 parallelism configuration."""

    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 8
        self.pipeline_model_parallel_size = 13
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 8
        self.expert_tensor_parallel_size = 1


class GLM5Config(DecoderLLMConfig):
    """GLM-5 model configuration.

    Model architecture parameters follow the official `zai-org/GLM-5` release:
    - 78 layers
    - 6144 hidden size
    - 64 MLA heads
    - 12288 dense FFN hidden size
    - 256 routed experts with 1 shared expert
    """

    ffn_cfg: Glm5MoEFeedForwardConfig = Glm5MoEFeedForwardConfig
    attn_cfg: Glm5AttentionConfig = Glm5AttentionConfig
    tok_embed_cfg: Glm5InputEmbeddingConfig = Glm5InputEmbeddingConfig
    out_embed_cfg: Glm5OutputEmbeddingConfig = Glm5OutputEmbeddingConfig
    parallel_cfg: Glm5ParallelConfig = Glm5ParallelConfig

    def __init__(self):
        super().__init__()
        self.num_layers = 78
        self.hidden_size = 6144
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = False
        self.recompute = False
        self.tie_embedding = False

        self.params_dtype = torch.bfloat16
        self.variable_seq_lengths = True
        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False

        self.ffn_cfg.moe_cfg.moe_layer_list = list(range(3, self.num_layers))

    def build_model(self):
        from steptronoss.model.glm5 import Glm5Model

        return Glm5Model(cfg=self, layer_map=self.build_layer_map())

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


class Qwen3AttentionConfig(AttentionConfig):
    """Qwen3 attention configuration."""

    def __init__(self):
        super().__init__()
        self.causal = True
        self.attention_dropout = 0.0

        self.use_sliding_window = False

        self.num_attention_heads = 32
        self.num_attention_groups = 4  # KV heads for GQA

        self.head_dim = 128
        self.hidden_size = Ref("..hidden_size")

        self.use_headwise_attn_gate = False
        self.use_qkv_bias = False

        self.sliding_window_size = -1

        self.use_qk_norm = False  # Inverted in module: False here enables QK norm
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False

        self.recompute_qknorm_rope = False

        self.qk_rope_head_dim = None
        self.rope_theta = 1_000_000.0
        self.yarn_beta_slow = 1.0
        self.yarn_beta_fast = 32.0
        self.ntk_interp_ratio = 1.0
        self.max_position_embeddings = 32768


class Qwen3MoEConfig(MoEConfig):
    def __init__(self):
        super().__init__()
        self.tp_cfg = Ref("...tp_cfg")
        self.hidden_size = Ref("...hidden_size")
        self.activation = Ref("..activation")

        self.moe_num_experts = 128
        self.moe_top_k = 8
        self.moe_aux_loss_coef = 0.0

        self.enable_auxiliary_loss_free_load_balance = False
        self.router_bias_update_rate = 0.0

        self.moe_hidden_size = 768
        self.routed_scaling_factor = 1.0
        self.enable_sigmoid_router = False
        self.moe_enable_deepep = False
        self.norm_expert_weight = True
        self.moe_layer_list = list(range(48))
        self.share_expert_dim = 0


class Qwen3MoEFeedForwardConfig(MoEFeedForwardConfig):
    """Qwen3 feed-forward configuration."""

    moe_cfg = Qwen3MoEConfig

    def __init__(self):
        super().__init__()

        self.hidden_size = Ref("..hidden_size")
        self.ffn_hidden_size = 6144

        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False

        self.swiglu_limit = None
        self.swiglu_recompute_silu_out_proj = True


class Qwen3InputEmbeddingConfig(InputEmbeddingConfig):
    """Qwen3 input embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Qwen3OutputEmbeddingConfig(OutputEmbeddingConfig):
    """Qwen3 output embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True

        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-6

        self.gather_output = False


class Qwen3ParallelConfig(ParallelConfig):
    """Qwen3 parallelism configuration."""

    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 4
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 8
        self.expert_tensor_parallel_size = 1


class Qwen3_30A3BConfig(DecoderLLMConfig):
    """Qwen3 8B model configuration.

    Model architecture parameters based on Qwen3-8B:
    - 48 layers
    - 2048 hidden size
    - 32 attention heads with 4 KV heads (GQA)
    - 6144 FFN hidden size
    - 128 head dimension
    - 151936 vocab size
    """

    ffn_cfg = Qwen3MoEFeedForwardConfig
    attn_cfg = Qwen3AttentionConfig
    tok_embed_cfg = Qwen3InputEmbeddingConfig
    out_embed_cfg = Qwen3OutputEmbeddingConfig
    parallel_cfg = Qwen3ParallelConfig

    def __init__(self):
        super().__init__()
        # Model architecture
        self.num_layers = 48
        self.hidden_size = 2048
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute = False
        self.tie_embedding = False

        # Precision
        self.params_dtype = torch.bfloat16

        # Other settings
        self.variable_seq_lengths = True
        self.tp_cfg.sequence_parallel = True
        self.tp_cfg.async_tensor_model_parallel_allreduce = False

    def build_model(self):
        from steptronoss.model.qwen_dense import QwenModel

        return QwenModel(cfg=self, layer_map=self.build_layer_map())

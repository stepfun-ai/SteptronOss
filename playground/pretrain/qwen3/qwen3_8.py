import torch

from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.common.feed_forward import FeedForwardConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
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
        self.num_attention_groups = 8  # KV heads for GQA

        self.head_dim = 128
        self.hidden_size = 4096

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
        self.max_position_embeddings = 327680


class Qwen3FeedForwardConfig(FeedForwardConfig):
    """Qwen3 feed-forward configuration."""

    def __init__(self):
        super().__init__()

        self.hidden_size = 4096
        self.ffn_hidden_size = 12288

        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False

        self.swiglu_limit = None
        self.swiglu_recompute_silu_out_proj = True


class Qwen3InputEmbeddingConfig(InputEmbeddingConfig):
    """Qwen3 input embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = 4096
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Qwen3OutputEmbeddingConfig(OutputEmbeddingConfig):
    """Qwen3 output embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = 4096
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
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1


class Qwen3_8BConfig(DecoderLLMConfig):
    """Qwen3 8B model configuration.

    Model architecture parameters based on Qwen3-8B:
    - 36 layers
    - 4096 hidden size
    - 32 attention heads with 8 KV heads (GQA)
    - 12288 FFN hidden size
    - 128 head dimension
    - 151936 vocab size
    """

    ffn_cfg = Qwen3FeedForwardConfig
    attn_cfg = Qwen3AttentionConfig
    tok_embed_cfg = Qwen3InputEmbeddingConfig
    out_embed_cfg = Qwen3OutputEmbeddingConfig
    parallel_cfg = Qwen3ParallelConfig

    def __init__(self):
        super().__init__()
        # Model architecture
        self.num_layers = 36
        self.hidden_size = 4096
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


class Qwen3_8BConfig_128K_80G(Qwen3_8BConfig):
    def __init__(self):
        super().__init__()
        self.parallel_cfg.tensor_model_parallel_size = 8

    def pp_vp_allocation(self, abs_pp_rank):
        assert self.parallel_cfg.pipeline_model_parallel_size == 1
        return [{"recompute": True}] * self.num_layers

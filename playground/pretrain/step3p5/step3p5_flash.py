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
from steptronoss.model.step3p5 import Step3p5ModelConfig


class Step3p5FlashAttentionConfig(AttentionConfig):
    """Step3p5Flash attention configuration."""

    def __init__(self):
        super().__init__()
        self.causal = True
        self.attention_dropout = 0.0

        self.use_sliding_window = False

        self.num_attention_heads = 64
        self.num_attention_groups = 8  # KV heads for GQA

        self.head_dim = 128
        self.hidden_size = Ref("..hidden_size")

        self.use_headwise_attn_gate = True
        self.use_qkv_bias = False

        self.sliding_window_size = 512

        self.use_qk_norm = False  # Inverted in module: False here enables QK norm
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = True

        self.recompute_qknorm_rope = True

        self.qk_rope_head_dim = 64  # partial rope
        self.rope_theta = 5000_000
        self.yarn_beta_slow = 1.0
        self.yarn_beta_fast = 32.0
        self.ntk_interp_ratio = 1.0
        self.max_position_embeddings = 4096


class Step3p5FlashSWAConfig(Step3p5FlashAttentionConfig):
    def __init__(self):
        super().__init__()
        self.use_sliding_window = True
        self.num_attention_heads = 96
        self.qk_rope_head_dim = 128
        self.rope_theta = 10_000


class Step3p5FlashMoEConfig(MoEConfig):
    def __init__(self):
        super().__init__()
        self.tp_cfg = Ref("...tp_cfg")
        self.hidden_size = Ref("...hidden_size")
        self.activation = Ref("..activation")

        self.moe_num_experts = 288
        self.moe_top_k = 8
        self.moe_aux_loss_coef = 0.0

        self.enable_auxiliary_loss_free_load_balance = True
        self.router_bias_update_rate = 0

        self.moe_hidden_size = 1280
        self.routed_scaling_factor = 3.0
        self.enable_sigmoid_router = True
        self.moe_enable_deepep = True
        self.norm_expert_weight = True
        self.moe_layer_list = list(range(3, 45))
        self.share_expert_dim = 1280


class Step3p5FlashMoEFeedForwardConfig(MoEFeedForwardConfig):
    """Step3p5Flash feed-forward configuration."""

    moe_cfg = Step3p5FlashMoEConfig

    def __init__(self):
        super().__init__()

        self.hidden_size = Ref("..hidden_size")
        self.ffn_hidden_size = 11264

        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = True

        self.swiglu_limit = None
        self.swiglu_recompute_silu_out_proj = True


class Step3p5FlashInputEmbeddingConfig(InputEmbeddingConfig):
    """Step3p5Flash input embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 128896
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Step3p5FlashOutputEmbeddingConfig(OutputEmbeddingConfig):
    """Step3p5Flash output embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 128896
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True

        self.rms_norm_zero_gamma = True
        self.layernorm_epsilon = 1e-5

        self.gather_output = False


class Step3p5FlashParallelConfig(ParallelConfig):
    """Step3p5Flash parallelism configuration."""

    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 8
        self.virtual_pipeline_model_parallel_size = 3
        self.context_parallel_size = 8
        self.expert_model_parallel_size = 8
        self.expert_tensor_parallel_size = 1


class Step3p5FlashModelConfig(Step3p5ModelConfig):
    """Step3p5Flash model configuration.

    Model architecture parameters based on Step3p5Flash:
    - 45 layers
    - 4096 hidden size
    - 64 attention heads with 8 KV heads (GQA)
    - 11264 FFN hidden size
    - 128 head dimension
    - 128896 vocab size
    """

    ffn_cfg = Step3p5FlashMoEFeedForwardConfig
    attn_cfg = Step3p5FlashAttentionConfig
    swa_cfg = Step3p5FlashSWAConfig
    tok_embed_cfg = Step3p5FlashInputEmbeddingConfig
    out_embed_cfg = Step3p5FlashOutputEmbeddingConfig
    parallel_cfg = Step3p5FlashParallelConfig

    swa_layer_list: list[bool]

    def __init__(self):
        super().__init__()
        # Model architecture
        self.num_layers = 45
        self.swa_layer_list = [False] + [True, True, True, False] * 11
        self.hidden_size = 4096
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = True
        self.recompute = False
        self.tie_embedding = False

        # Precision
        self.params_dtype = torch.bfloat16

        # Other settings
        self.variable_seq_lengths = True
        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False

    def build_model(self):
        from steptronoss.model.step3p5 import Step3p5Model

        return Step3p5Model(cfg=self, layer_map=self.build_layer_map())

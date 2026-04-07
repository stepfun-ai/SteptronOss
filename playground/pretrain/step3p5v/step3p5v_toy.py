"""
Toy setting for Step3p5v multimodal smoke tests.
"""

import torch

from playground.pretrain.step3p5v.step3p5v_flash import Step3p5vFlashModelConfig


class Step3p5vToyModelConfig(Step3p5vFlashModelConfig):
    def __init__(self):
        super().__init__()

        self.params_dtype = torch.float32

        self.num_layers = 0
        self.swa_layer_list = []
        self.hidden_size = 16

        self.attn_cfg.num_attention_heads = 4
        self.attn_cfg.num_attention_groups = 1
        self.attn_cfg.head_dim = 4
        self.swa_cfg.num_attention_heads = 4
        self.swa_cfg.num_attention_groups = 1
        self.swa_cfg.head_dim = 4

        self.ffn_cfg.ffn_hidden_size = 32
        self.ffn_cfg.moe_cfg.moe_layer_list = []
        self.ffn_cfg.moe_cfg.share_expert_dim = 0

        self.tok_embed_cfg.vocab_size = 256
        self.out_embed_cfg.vocab_size = 256

        self.parallel_cfg.tensor_model_parallel_size = 1
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 1
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.variable_seq_lengths = False

        self.tok_embed_cfg.encoder_cfg.image_size = 28
        self.tok_embed_cfg.encoder_cfg.patch_size = 14
        self.tok_embed_cfg.encoder_cfg.hidden_size = 8
        self.tok_embed_cfg.encoder_cfg.ffn_hidden_size = 16
        self.tok_embed_cfg.encoder_cfg.num_layers = 1
        self.tok_embed_cfg.encoder_cfg.num_attention_heads = 2
        self.tok_embed_cfg.encoder_cfg.vit_downsampler_hidden_dim = 12
        self.tok_embed_cfg.encoder_cfg.output_dim = self.hidden_size

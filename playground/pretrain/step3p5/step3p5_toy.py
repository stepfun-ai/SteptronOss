"""
Toy setting for step3.5 flash, same feature, toy size.
"""

from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig


class Step3p5ToyModelConfig(Step3p5FlashModelConfig):
    def __init__(self):
        super().__init__()

        self.hidden_size = 2048
        self.num_layers = 48
        self.swa_layer_list = [True, True, True, False] * 12

        self.attn_cfg.num_attention_heads = 32
        self.swa_cfg.num_attention_heads = 48

        self.ffn_cfg.moe_cfg.moe_num_experts = 128
        self.ffn_cfg.moe_cfg.moe_hidden_size = 768
        self.ffn_cfg.moe_cfg.share_expert_dim = 768
        self.ffn_cfg.moe_cfg.moe_layer_list = list(range(1, 48))
        self.ffn_cfg.moe_cfg.enable_auxiliary_loss_free_load_balance = True
        self.ffn_cfg.moe_cfg.router_bias_update_rate = 0.1

        self.ffn_cfg.ffn_hidden_size = 7168

        self.parallel_cfg.tensor_model_parallel_size = 8
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 8
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = True

    def build_model(self):
        model = super().build_model()
        for p in model.parameters():
            from torch.nn.init import trunc_normal_

            trunc_normal_(p)
        return model

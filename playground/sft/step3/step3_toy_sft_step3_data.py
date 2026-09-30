"""
Single-node SFT debugging experiment for the Step3.5 Flash runtime path.

Launch from the repo root with 8 GPUs on one machine:

    torchrun --standalone --nproc-per-node=8 playground/sft/step3/step3_toy_sft_step3_data.py

This experiment is intended to stay aligned with the single-node Step3.5
Flash setup in this repo while swapping in the toy Step3.5 model for faster
debugging. Aside from the toy model definition and checkpoint/weight handling,
the SFT data pipeline and the key runtime settings are kept consistent with
the Step3.5 Flash single-node configuration.
"""

from playground.data.sft.oss260312.step_sft_data_config0311_step3p5_tokenizer import Recipe0311CompiledSFTDataConfig
from playground.pretrain.step3p5.step3p5_toy import Step3p5ToyModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.exp.ntp import MoePretrainMetricConfig


class Exp(BaseExp):
    """Toy-model SFT debug config that mirrors the Step3.5 Flash 1-node path."""

    model_cfg = Step3p5ToyModelConfig

    data_cfg = Recipe0311CompiledSFTDataConfig
    metric_cfg = MoePretrainMetricConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 128 * 1024

        self.model_cfg.num_layers = 6
        self.model_cfg.hidden_size = 4096
        self.model_cfg.swa_layer_list = [False, True, True, True, False, True]
        self.model_cfg.attn_cfg.num_attention_heads = 64
        self.model_cfg.attn_cfg.num_attention_groups = 8
        self.model_cfg.attn_cfg.head_dim = 128
        self.model_cfg.swa_cfg.num_attention_heads = 96
        self.model_cfg.ffn_cfg.ffn_hidden_size = 11264
        self.model_cfg.ffn_cfg.moe_cfg.moe_num_experts = 288
        self.model_cfg.ffn_cfg.moe_cfg.moe_hidden_size = 1280
        self.model_cfg.ffn_cfg.moe_cfg.share_expert_dim = 1280
        self.model_cfg.ffn_cfg.moe_cfg.moe_layer_list = list(range(6))

        self.trainer_cfg.train_iters = None
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        # self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.none()
        self.checkpoint_cfg.save_interval = 100
        self.checkpoint_cfg.auto_resume = False
        self.model_cfg.recompute = True
        self.model_cfg.parallel_cfg.context_parallel_size = 8
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 1
        self.model_cfg.tp_cfg.sequence_parallel = False
        self.model_cfg.ffn_cfg.moe_cfg.force_balance = True
        self.profiler_cfg.timing_log_level = 2

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            routed_grouped_ffn="fused",
            moe_weighted_gather="triton",
            TokenDispatcher="deep_ep",
        )

            # grouped_gemm="nv_grouped_gemm",
            # AttentionCore="flash-attn-3",

if __name__ == "__main__":
    Exp().train()

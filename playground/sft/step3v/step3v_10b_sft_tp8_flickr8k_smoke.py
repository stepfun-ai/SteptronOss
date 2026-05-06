from __future__ import annotations

from playground.pretrain.step3v.step3v_10b import Step3V10BConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3v.step3v_flickr8k_data import Step3VFlickr8kDataConfig


class Exp(BaseExp):
    """Run with: python tools/mp_run.py playground/sft/step3v/step3v_10b_sft_tp8_flickr8k_smoke.py"""

    log_dir = "/oss/logs/"

    model_cfg = Step3V10BConfig
    data_cfg = Step3VFlickr8kDataConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 1
        self.trainer_cfg.global_seq_length = self.data_cfg.max_seq_len
        self.trainer_cfg.train_iters = 2
        self.trainer_cfg.log_interval = 1

        self.scheduler_cfg.total_schedule = 2
        self.scheduler_cfg.warmup_schedule = 0
        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.weight_decay = 0.0

        self.model_cfg.recompute = True
        # Image insertion currently runs before sequence-parallel scattering.
        self.model_cfg.tp_cfg.sequence_parallel = False
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 8
        self.model_cfg.tok_embed_cfg.encoder_cfg.parallel_cfg.tensor_model_parallel_size = 8

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_option.none()
        self.checkpoint_cfg.load_path = None
        self.checkpoint_cfg.load_safetensors = "/oss/opensource_models/Step3-VL-10B"
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_interval = 0
        self.checkpoint_cfg.save_path = None
        self.checkpoint_cfg.auto_resume = False

        self.profiler_cfg.timing_log_level = 1

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization()


if __name__ == "__main__":
    Exp().train()

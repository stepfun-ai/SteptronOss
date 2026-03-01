from playground.data.sft.step_recipe0226 import (
    Step3SFTDataStep3TokenizedConfig,
)
from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig
from steptronoss.core.parallel_state import PM, get_vpp_size
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.exp.resources import TorchrunResourceConfig


# import sys, time
# sys.excepthook = lambda a, b, c: [print("HoldOneError"), time.sleep(3600)]
class Step3F128kSFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 8
        self.gpu = 8
        self.mounts.extend([
            "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:/mnt/step2-alignment-jfs",
            "juicefs+s3://oss.i.shaipower.com/tenant:/mnt/shared-storage/tenant",
            "gpfs://inspurfs/step2_alignment:/mnt/shared-storage/groups/step2_alignment_2",
        ])
        self.envs |= {
            # "CUDA_LAUNCH_BLOCKING": "1",
            # "MEM_DIAGNOSE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        }


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p5MuonConfig
    """Use Muon for 2D params with AdamW fallback."""


class Step3p5FlashModelConfigBalanced(Step3p5FlashModelConfig):
    """Adjust layermap to reduce PP7 memory by moving layers to PP1/PP2."""

    def __init__(self):
        super().__init__()
        # Disable context parallel to reduce communication/overheads.
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.tensor_model_parallel_size = 8
        self.tp_cfg.sequence_parallel = True

    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        # PP=8, VPP=3 -> 24 slots. Start from 2 layers/slot and drop 1 layer on
        # a few slots to get 45 layers total, while keeping PP7 off the floor.
        lengths = [2] * (PM.size_of("PP") * get_vpp_size())
        lengths[22] = 1  # PP6/vp2
        lengths[23] = 0  # PP7/vp2

        expected = PM.size_of("PP") * get_vpp_size()
        if len(lengths) != expected:
            raise ValueError(f"layermap lengths={len(lengths)} != PP*VPP={expected}")
        return [{}] * lengths[abs_pp_rank]


class Exp(BaseExp):
    log_dir = "/mnt/shared-storage/groups/step2_alignment_2/hujingcheng/tensorboard/steptron_logs_gpfs2/sft_baseline/step4air_midtrain0111_selfdistill_d0116_g32_0225"

    resource_cfg = Step3F128kSFTResourceConfig

    model_cfg = Step3p5FlashModelConfigBalanced

    metric_cfg = MoePretrainMetricConfig

    data_cfg = Step3SFTDataStep3TokenizedConfig

    optimizer_cfg = MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 1024 * 128
        self.trainer_cfg.train_iters = None  # auto get

        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 1e-6
        self.scheduler_cfg.warmup_schedule = 140

        self.trainer_cfg.log_interval = 1
        self.profiler_cfg.timing_log_level = 2

        self.checkpoint_cfg.load_safetensors = (
            "/mnt/step2-alignment-jfs/quansun/midtrain/checkpoints/step4air_stage2_128k_0108/it21708_hf_mtp/"
        )
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100

        self.model_cfg.recompute = True
        self.trainer_cfg.offload_optimizer_state = False

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            # grouped_gemm="nv_grouped_gemm",
            AttentionCore="flash-attn",
            default="torch_compile",
            # default=None,
        )


if __name__ == "__main__":
    Exp().train()

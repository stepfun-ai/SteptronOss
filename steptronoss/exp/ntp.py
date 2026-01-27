import os

import torch

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import (
    BaseExp,
    DataConfig,
    GradientManagerConfig,
    Megatron3DParallelModelConfig,
    MetricConfig,
    ProfilerConfig,
    TrainerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import SchedulerConfig
from steptronoss.utils import GlobalMetrics, Metric


class PretrainMetricConfig(MetricConfig):

    def __init__(self):
        super().__init__()

        self.lm_loss = Metric().mean("time").mean("dp").mean("ep")

        self.grad_norm = Metric().mean("time")

        self.grad_zeros = Metric().mean("time")

        self.acc = Metric().mean("time").sum("tp").mean("dp").mean("ep")

        self.logits_n_avg = Metric().mean("time").mean("tp").mean("dp").mean("ep")

        self.logits_max = Metric().max("time").max("tp").max("dp").max("ep")

        self.logits_n_max = Metric().max("time").max("tp").max("dp").max("ep")


class MoePretrainMetricConfig(PretrainMetricConfig):
    def __init__(self):
        super().__init__()
        from steptronoss.utils.metrics import ModelLayerMetric

        # MoE Metrics
        self.moe_aux_loss = ModelLayerMetric(is_group=True).mean("time").mean("ep").mean("dp").sum("pp")

        self.moe_minp = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")

        self.moe_sumk = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")

        self.moe_std = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")

        self.moe_max = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")

        self.moe_avg = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")

        self.expert_distri = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp").disable()
        self.expert_distri.disable()  # disable expert_distri to prevent reduce() from stacking heterogeneous data
        self.expert_coef = ModelLayerMetric(is_group=True).mean("time").mean("dp").sum("pp")
        self.peak_to_avg_ratio = ModelLayerMetric(is_group=True).max("time").max("dp").max("ep").sum("pp")
        """
        Expert distribution and expert coef
        """

        self.router_logits_max = Metric(is_group=True).max("time").max("tp").max("dp").max("ep").sum("pp")

        self.router_logits_std = Metric(is_group=True).mean("time").mean("tp").mean("dp").mean("ep").sum("pp")

        self.zloss = Metric(is_group=True).mean("time").mean("tp").mean("ep").mean("dp").sum("pp")


GlobalMetrics: PretrainMetricConfig


class NTPTrainerConfig(TrainerConfig):
    """Next Token Prediction trainer. Use as Exp.trainer_cfg"""

    micro_batch_size: int
    global_batch_size: int
    global_seq_length: int

    def get_trainer_cls(self):
        from steptronoss.core.trainers.lm_trainer import DecoderPretrainTrainer

        return DecoderPretrainTrainer

    def apply_loss_mask(self, losses: torch.FloatTensor, loss_mask: torch.Tensor):
        from steptronoss.core.context_parallel import scatter_to_balanced_cp_region

        cp_size = PM.size_of("CP")
        if cp_size > 1:
            local_loss_mask = scatter_to_balanced_cp_region(loss_mask, dim=1)
            local_loss_mask = local_loss_mask.view(-1).float()
            losses = losses.reshape(-1)
            losses[local_loss_mask == 0] = 0  # filter masked nan
            if torch.count_nonzero(loss_mask) == 0:
                loss = torch.sum(losses * local_loss_mask) / (loss_mask.sum() + 1e-8) * cp_size
            else:
                loss = torch.sum(losses * local_loss_mask) / (loss_mask.sum()) * cp_size
        else:
            loss_mask = loss_mask.reshape(-1).float()
            losses = losses.reshape(-1)
            losses[loss_mask == 0] = 0
            if torch.count_nonzero(loss_mask) == 0:
                loss = torch.sum(losses * loss_mask) / (loss_mask.sum() + 1e-8)
            else:
                loss = torch.sum(losses * loss_mask) / (loss_mask.sum())
        return loss

    def loss_func(self, data: dict, logits: torch.Tensor):
        """This function take as input the model outputs and
        returns ONE backwardable scalar

        Args:
            data (dict): the data dict same as for model inputs
            outputs (torch.Tensor): the outputs from model

        Returns:.backward()
        """
        if logits.numel() == 1:  # Some model handle loss calc inside model, just return if so.
            return logits
        from steptronoss.core.context_parallel import scatter_to_balanced_cp_region
        from steptronoss.core.tensor_parallel import vocab_parallel_cross_entropy

        # labels B, S, logits should be [S, B, V / tp]
        labels = data["labels"].transpose(1, 0).contiguous()
        labels = scatter_to_balanced_cp_region(labels, dim=0)
        losses, acc = vocab_parallel_cross_entropy(logits.float(), labels)

        losses: torch.Tensor = losses.transpose(1, 0).contiguous()  # make B, S

        loss_masks = data.get("loss_masks", None)
        if loss_masks is None and "loss_mask" in data:
            raise RuntimeError("Critical Typo: please use 'loss_masks' instead of 'loss_mask' in data!")
        if loss_masks is not None:
            loss = self.apply_loss_mask(losses, loss_masks)
        else:
            loss = losses.mean()

        GlobalMetrics.losses.add(losses, iop=lambda x: x.detach())
        GlobalMetrics.lm_loss.add(loss, iop=lambda x: x.detach())
        GlobalMetrics.acc.add(acc, iop=lambda x: x.detach())
        GlobalMetrics.consumed_tokens.add(labels, iop=lambda x: x.numel())

        return loss

    def make_logs(self, iteration, metrics, tb_writer=None):
        logs = super().make_logs(iteration, metrics, tb_writer)
        logs["lm_loss"] = f"{metrics['lm_loss']:.6E}"
        return logs

    def sanity_check(self):
        super().sanity_check()
        assert self.global_batch_size > 0


class PretrainExp(BaseExp):
    trainer_cfg: NTPTrainerConfig
    model_cfg: Megatron3DParallelModelConfig
    optimizer_cfg: GradientManagerConfig
    scheduler_cfg: SchedulerConfig
    data_cfg: DataConfig
    checkpoint_cfg: CheckpointConfig

    metric_cfg: PretrainMetricConfig
    profiler_cfg: ProfilerConfig

    def sanity_check(self):
        super().sanity_check()
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        model_parallel_size = (
            self.model_cfg.parallel_cfg.pipeline_model_parallel_size
            * self.model_cfg.parallel_cfg.tensor_model_parallel_size
            * self.model_cfg.parallel_cfg.expert_model_parallel_size
        )
        assert world_size >= model_parallel_size
        assert world_size % model_parallel_size == 0

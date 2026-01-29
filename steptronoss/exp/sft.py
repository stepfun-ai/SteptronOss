from __future__ import annotations

from typing import TYPE_CHECKING

from configurize import Config

from steptronoss.exp.base_exp import (
    BaseExp,
    DataConfig,
    GradientManagerConfig,
    Megatron3DParallelModelConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import SchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig, PretrainMetricConfig

if TYPE_CHECKING:
    from steptronoss.data.datasets.base_language_dataset import Dataset
    from steptronoss.model.decoder_model import DecoderLLMConfig


class SFTDatasetsConfig(Config):
    def build_datasets(self) -> dict[str, tuple[Dataset, float]]:
        """Build Datasets and related epoches."""
        pass


class SFTDataConfig(DataConfig):
    """For SFT dataloader, we extract an individual dataset_cfg"""

    dataset_cfg: SFTDatasetsConfig = SFTDatasetsConfig

    def build_dataloader(self, dp_rank=0, dp_size=1):
        raise NotImplementedError


class SFTExp(BaseExp):
    trainer_cfg: NTPTrainerConfig
    model_cfg: DecoderLLMConfig
    optimizer_cfg: GradientManagerConfig
    scheduler_cfg: SchedulerConfig
    data_cfg: SFTDataConfig
    checkpoint_cfg: CheckpointConfig

    metric_cfg: PretrainMetricConfig
    profiler_cfg: ProfilerConfig

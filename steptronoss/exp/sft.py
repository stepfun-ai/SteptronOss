from __future__ import annotations

from typing import TYPE_CHECKING

from configurize import Config

from steptronoss.exp.base_exp import DataConfig

if TYPE_CHECKING:
    from steptronoss.data.datasets.base_language_dataset import Dataset


class SFTDatasetsConfig(Config):
    def build_datasets(self) -> dict[str, tuple[Dataset, float]]:
        """Build Datasets and related epoches."""
        pass


class SFTDataConfig(DataConfig):
    dataset_cfg: SFTDatasetsConfig = SFTDatasetsConfig

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader import MixedDataloader

        dataset_epochs = self.dataset_cfg.build_datasets()
        dataloader = MixedDataloader(
            datasets=[x[0] for x in dataset_epochs.values()],
            epochs=[x[1] for x in dataset_epochs.values()],
        )
        return dataloader

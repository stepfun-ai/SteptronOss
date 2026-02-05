import abc
from collections.abc import Callable, Iterator

import torch

from steptronoss.exp.inference import GenerationInput, GenerationOutput
from steptronoss.utils.comm_utils import ReusableDistributor


class BaseGenerator(abc.ABC):
    @property
    def stats(self):
        if not hasattr(self, "_stats"):
            from steptron.core.generators.generator_stats import GeneratorStats

            self._stats = GeneratorStats(enabled=False)
        return self._stats

    @stats.setter
    def stats(self, value):
        self._stats = value

    @abc.abstractmethod
    def reload(self, *args, **kwargs):
        """Reload the generator back to GPU."""

    @abc.abstractmethod
    def offload(self):
        """Release the generator from GPU."""

    @abc.abstractmethod
    @torch.no_grad()
    def generate_stream(
        self,
        prompts: ReusableDistributor,
    ) -> Iterator[GenerationOutput]:
        """Take inputs from a queue and generate outputs as a generator."""

    @abc.abstractmethod
    @torch.no_grad()
    def list_generate(
        self,
        prompts: list[GenerationInput],
        max_rollout: int | None = None,
        offload: bool = True,
    ) -> list[GenerationOutput]:
        """Take inputs from a list and generate outputs as a list."""

    @abc.abstractmethod
    def close_when(self, condition: Callable[[], bool]):
        """Close input stream on some condition."""

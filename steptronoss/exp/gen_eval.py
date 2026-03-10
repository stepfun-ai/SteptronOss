from __future__ import annotations

from typing import TYPE_CHECKING

from configurize import Config

if TYPE_CHECKING:
    from steptronoss.generation.base_generatable import GenableItem


class GenableEvalConfig(Config):
    """使用Genable Eval流程非常简单，无需和trainer对接，直接使用GenerationController生成并计算metric即可。"""

    record_eval_rollout: bool = False
    """If True, trainer may pass a prefix to dump eval rollouts."""

    def get_prompts(self) -> list[GenableItem]:
        raise NotImplementedError

    def eval(self) -> dict:
        from steptronoss.generation.async_generation import GenerationController

        controller = GenerationController()

        prompts = self.get_prompts()
        results = []
        for _genable, result in controller.generate(prompts):
            results.append(result)
        return results

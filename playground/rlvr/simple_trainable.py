import re
from typing import Any

from loguru import logger

from steptronoss.exp.rl import EnvTrajectory, StopType
from steptronoss.generation.base_generatable import TrainableItem
from steptronoss.utils import get_exp_id


class SimpleTrainable(TrainableItem):
    """Simple trainable that calls a vLLM endpoint and rewards boxed answers.

    It sends a prompt string to the OpenAI-compatible vLLM endpoint, expects
    the model to emit an answer formatted as ``\\boxed{...}``, and compares the
    extracted contents to the provided ground truth string.

    Example:
        trainable = SimpleTrainable(
            prompt_text="Compute 1+2. Answer in \\boxed{...}.",
            gt="3",
            endpoint="http://127.0.0.1:8000",
            model_name_template="deployed-model-{EXP_ID}",
            sampling_params={"max_tokens": 32},
            max_tokens=32,
        )
    """

    def __init__(
        self,
        prompt_text: str,
        gt: str,
        endpoint: str,
        model_name_template: str,
        sampling_params: dict[str, Any],
        max_tokens: int,
    ):
        super().__init__()
        self.prompt_text = prompt_text
        self.gt = gt
        self.endpoint = endpoint
        self.model_name_template = model_name_template
        self.sampling_params = sampling_params
        self.max_tokens = max_tokens

    @staticmethod
    def _extract_boxed(answer: str) -> str:
        matches = re.findall(r"\\boxed\{([^}]*)\}", answer)
        if not matches:
            return ""
        return matches[-1].strip()

    async def generate(self) -> dict[str, Any]:
        import aiohttp

        payload = {
            "model": self.model_name,
            "prompt": self.prompt_text,
            "return_token_ids": True,
        }
        payload.update(self.sampling_params)

        async with aiohttp.request(
            method="POST",
            url=f"{self.endpoint}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=7200.0),
        ) as response:
            response = await response.json()

        choice = response["choices"][0]
        return {
            "choice": choice,
            "prompt": self.prompt_text,
        }

    async def generate_for_train(self):
        generated = await self.generate()
        choice = generated["choice"]

        prompt_ids = choice["prompt_token_ids"]
        decode_ids = choice["token_ids"]
        finish_reason = choice.get("finish_reason", "")
        decoded_text = choice.get("text") or ""

        predicted = self._extract_boxed(decoded_text)
        is_correct = predicted == self.gt.strip()
        raw_reward = 1.0 if is_correct else 0.0

        trajectory = prompt_ids + decode_ids
        is_gen_mask = [0] * len(prompt_ids) + [1] * len(decode_ids)
        stop_type = StopType.MAX_LEN if finish_reason == "length" else StopType.STOP_STRING

        # logger.info(f"New Traj Generated! boxed={predicted!r} gt={self.gt!r} correct={is_correct}")
        return [
            EnvTrajectory(
                trajectory=trajectory,
                is_gen_mask=is_gen_mask,
                raw_reward=raw_reward,
                stop_type=stop_type,
            )
        ]

    @property
    def model_name(self) -> str:
        return self.model_name_template.format(EXP_ID=get_exp_id())

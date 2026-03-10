import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from steptronoss.exp.rl import EnvTrajectory, StopType
from steptronoss.generation.base_generatable import TrainableItem


class SimpleTrainable(TrainableItem):
    """Simple trainable that calls a vLLM endpoint and rewards boxed answers.

    It sends a prompt string to the OpenAI-compatible vLLM endpoint, expects
    the model to emit an answer formatted as ``\\boxed{...}``, and compares the
    extracted contents to the provided ground truth string.

    Note:
        ``endpoint_getter`` / ``model_name_getter`` are used (instead of concrete
        values) so resumed runs can resolve updated router endpoints and EXP_ID.

    Example:
        trainable = SimpleTrainable(
            prompt_text="Compute 1+2. Answer in \\boxed{...}.",
            gt="3",
            endpoint_getter=my_endpoint_getter,
            model_name_getter=my_model_name_getter,
            sampling_params={"max_tokens": 32},
            max_tokens=32,
        )
    """

    def __init__(
        self,
        prompt_text: str,
        gt: str,
        endpoint_getter: Callable[[], str],
        model_name_getter: Callable[[], str],
        sampling_params: dict[str, Any],
        max_tokens: int,
    ):
        super().__init__()
        self.prompt_text = prompt_text
        self.gt = gt
        self.endpoint_getter = endpoint_getter
        self.model_name_getter = model_name_getter
        self.max_tokens = max_tokens
        self.sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": max_tokens,
        }
        self.sampling_params.update(sampling_params)

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
        if "choices" not in response:
            raise ValueError(f"Unexpected vLLM response: {response}")
        choice = response["choices"][0]
        return {
            "choice": choice,
            "prompt": self.prompt_text,
        }

    def fingerprint(self) -> str:
        payload = {
            "genable_type": type(self).__name__,
            "prompt": self.prompt_text,
            "sampling_params": self.sampling_params,
        }
        serialized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
        return f"{type(self).__name__}:v1:{digest}"

    async def generate_for_train(self):
        generated = await self.generate()
        choice = generated["choice"]

        prompt_ids = choice["prompt_token_ids"]
        decode_ids = choice["token_ids"]
        finish_reason = choice.get("finish_reason", "")
        decoded_text = choice.get("text") or ""

        predicted = self._extract_boxed(decoded_text)
        is_correct = predicted == self.gt.strip()
        if finish_reason == "length":
            raw_reward = 0.0
        else:
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
        return self.model_name_getter()

    @property
    def endpoint(self) -> str:
        return self.endpoint_getter()

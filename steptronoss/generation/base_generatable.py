from abc import abstractmethod
from typing import Any
from configurize import DataClass

from steptronoss.exp.inference import StopType
from steptronoss.exp.rl import EnvTrajectory
from steptronoss.utils import get_exp_id
import aiohttp
from os.path import join

# Abstract


class GenableItem(DataClass):

    sampling_params: dict = {}
    # Prompt specific sampling params, if empty, use inference_config.

    meta: dict = {}

    @abstractmethod
    async def generate(self) -> Any:
        pass


class TrainableItem(GenableItem):

    @abstractmethod
    async def generate_for_train(self) -> list[EnvTrajectory]:
        pass


# Imple
class OpenAITrainableItem(TrainableItem):
    base_url: str = "http://stepcast-router:9200/v1"
    """BaseUrl should contains '/v1/', completion api can be get by f'{base_url}/chat/completion'"""

    api_key: str = ""
    """Not required for internal endpoint"""

    model_name_template: str = "deployed-model-{EXP_ID}"
    """Required when using router"""

    @property
    def model_name(self):
        return self.model_name_template.format(EXP_ID=get_exp_id())


class SingleTurnPrompt(OpenAITrainableItem):
    prompt: list[int]

    async def generate(self) -> dict:
        async with aiohttp.request(
            method="POST",
            url=join(self.base_url, "completion"),
            json={
                "model": self.model_name,
                "prompt": self.prompt,
                "return_token_ids": True,
                **self.sampling_params,
            },
            timeout=aiohttp.ClientTimeout(total=7200.0),
        ) as response:
            response = await response.json()
            decode_ids: list[int] = response['choices'][0]['model_extra']["token_ids"]
            finish_reason = response['choices'][0]['finish_reason']

        return dict(
            prompt=self.prompt,
            response=decode_ids,
            finish_reason=finish_reason,
        )

    async def generate_for_train(self) -> list[EnvTrajectory]:
        generated = await self.generate()
        prompt_ids = generated["prompt"]
        decoded_ids = generated["response"]
        trajectory = prompt_ids + decoded_ids
        is_gen_mask = [0] * len(prompt_ids) + [1] * len(decoded_ids)
        if generated["finish_reason"] == "length":
            stop_type = StopType.MAX_LEN
        else:
            stop_type = StopType.STOP_STRING

        # TODO: we have to support get logprobs from vllm, code below do not support TIS.
        sample = EnvTrajectory(
            trajectory=trajectory,
            is_gen_mask=is_gen_mask,
            stop_type=stop_type,
        )
        sample.raw_reward = 1.0  # just for example, call your own reward_fn
        assert sample.can_be_trained
        return [sample]

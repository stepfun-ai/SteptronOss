from abc import ABC, abstractmethod
from typing import Any

from steptronoss.exp.rl import EnvTrajectory
from steptronoss.utils.comm_utils import block_get_redis, get_exp_redis
from steptronoss.utils.utils import get_exp_id


# Abstract
class GenableItem(ABC):
    def __init__(self, meta: dict | None = None) -> None:
        self.meta: dict = meta or {}

    @abstractmethod
    async def generate(self) -> Any:
        pass


class TrainableItem(GenableItem):
    @abstractmethod
    async def generate_for_train(self) -> list[EnvTrajectory]:
        pass


class EndpointGetter:
    """Picklable endpoint resolver.

    We pass a getter instead of a concrete endpoint string because the router
    address can change between runs/resume; resolving at call time ensures we
    always hit the current router endpoint after restore.
    """

    def __init__(self, router_addr_key: str):
        self.router_addr_key = router_addr_key

    def __call__(self) -> str:
        exp_redis = get_exp_redis()
        redis_key = f"VLLM_ROUTER_ADDR_PORT_{self.router_addr_key}"
        return f"http://{block_get_redis(exp_redis, redis_key).decode()}"


class ModelNameGetter:
    """Picklable model-name resolver.

    We pass a getter instead of a concrete model name because model names are
    derived from EXP_ID at runtime; resolving at call time keeps resumes valid
    even when EXP_ID changes.
    """

    def __init__(self, model_name_template: str):
        self.model_name_template = model_name_template

    def __call__(self) -> str:
        return self.model_name_template.format(EXP_ID=get_exp_id())

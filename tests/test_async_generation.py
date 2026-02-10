import asyncio
from queue import Queue

import pytest

from steptronoss.exp.rl import EnvTrajectory, StopType
from steptronoss.generation.async_generation import GenerationController
from steptronoss.generation.base_generatable import TrainableItem

pytestmark = pytest.mark.cpu


class FakeGenable(TrainableItem):
    async def generate(self):
        await asyncio.sleep(0)
        return {"ok": True}

    async def generate_for_train(self):
        await asyncio.sleep(0)
        return [
            EnvTrajectory(
                trajectory=[1, 2, 3],
                is_gen_mask=[0, 1, 1],
                raw_reward=1.0,
                stop_type=StopType.STOP_TOKEN,
            )
        ]


def test_generation_controller_with_fake_genable():
    controller = GenerationController(num_workers=1)
    try:
        result_queue: Queue = Queue()
        controller.submit_with_callback(
            FakeGenable(),
            for_train=True,
            callback=lambda item, result, q=result_queue: q.put((item, result)),
        )

        item, result = result_queue.get(timeout=5)
        assert isinstance(item, FakeGenable)
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, EnvTrajectory)
        assert traj.can_be_trained is True
        assert traj.raw_reward == 1.0
        assert traj.trajectory == [1, 2, 3]
        assert traj.is_gen_mask == [0, 1, 1]
    finally:
        controller.shutdown()

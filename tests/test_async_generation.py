import asyncio
import multiprocessing as mp
import time
from queue import Queue

import pytest

import steptronoss.generation.async_generation as async_generation
from steptronoss.exp.rl import EnvTrajectory, StopType
from steptronoss.generation.async_generation import GenerationController
from steptronoss.generation.base_generatable import GenableItem, TrainableItem

pytestmark = pytest.mark.cpu


class FakeTrainable(TrainableItem):
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


class FakeGeneratable(GenableItem):
    async def generate(self):
        await asyncio.sleep(0)
        return {"ok": True}


class BlockingGenable(GenableItem):
    def __init__(self, state, lock, release_event):
        super().__init__()
        self.state = state
        self.lock = lock
        self.release_event = release_event

    async def generate(self):
        with self.lock:
            active = self.state["active"] + 1
            self.state["active"] = active
            self.state["entered"] += 1
            self.state["max_active"] = max(self.state["max_active"], active)

        try:
            while not self.release_event.is_set():
                await asyncio.sleep(0.01)
            return {"ok": True}
        finally:
            with self.lock:
                self.state["active"] -= 1


class _FakeTqdm:
    def __init__(self, *, total=None, desc="", initial=0, disable=False):
        self.total = total
        self.desc = desc
        self.n = initial
        self.disable = disable
        self.closed = False

    def update(self, n=1):
        self.n += n

    def close(self):
        self.closed = True


def test_generation_controller_with_fake_trainable():
    controller = GenerationController(num_workers=1)
    try:
        result_queue: Queue = Queue()
        controller.submit_with_callback(
            FakeTrainable(),
            for_train=True,
            callback=lambda item, result, q=result_queue: q.put((item, result)),
        )

        item, result = result_queue.get(timeout=5)
        assert isinstance(item, FakeTrainable)
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


def test_generation_controller_set_tqdm(monkeypatch):
    progress_bars = []

    def _fake_tqdm(*args, **kwargs):
        bar = _FakeTqdm(**kwargs)
        progress_bars.append(bar)
        return bar

    monkeypatch.setattr(async_generation, "tqdm", _fake_tqdm)

    controller = GenerationController(num_workers=1)
    try:
        controller.set_tqdm(disabled=False, total=1, desc="Friendly Progress")
        result_queue: Queue = Queue()
        controller.submit_with_callback(
            FakeGeneratable(),
            callback=lambda item, result, q=result_queue: q.put((item, result)),
        )

        item, result = result_queue.get(timeout=5)
        assert isinstance(item, FakeGeneratable)
        assert result == {"ok": True}

        deadline = time.time() + 5
        while time.time() < deadline and (not progress_bars or progress_bars[0].n < 1):
            time.sleep(0.05)

        assert len(progress_bars) == 1
        assert progress_bars[0].desc == "Friendly Progress"
        assert progress_bars[0].total == 1
        assert progress_bars[0].n == 1
        assert progress_bars[0].disable is False
    finally:
        controller.shutdown()

    assert progress_bars[0].closed is True


def test_generation_controller_with_fake_generatable():
    controller = GenerationController(num_workers=1)
    try:
        result_queue: Queue = Queue()
        controller.submit_with_callback(
            FakeGeneratable(),
            callback=lambda item, result, q=result_queue: q.put((item, result)),
        )

        item, result = result_queue.get(timeout=5)
        assert isinstance(item, FakeGeneratable)
        assert result == {"ok": True}
    finally:
        controller.shutdown()


def test_generation_controller_max_concurrent_genables_limits_global_inflight():
    manager = mp.Manager()
    state = manager.dict(active=0, entered=0, max_active=0)
    lock = manager.Lock()
    release_event = manager.Event()

    controller = GenerationController(num_workers=2, max_concurrent_genables=1)
    try:
        result_queue: Queue = Queue()
        for _ in range(2):
            controller.submit_with_callback(
                BlockingGenable(state=state, lock=lock, release_event=release_event),
                callback=lambda item, result, q=result_queue: q.put((item, result)),
            )

        deadline = time.time() + 5
        while time.time() < deadline and state["entered"] < 1:
            time.sleep(0.05)

        assert state["entered"] == 1

        time.sleep(0.3)
        assert state["entered"] == 1
        assert state["max_active"] == 1

        release_event.set()

        first_item, first_result = result_queue.get(timeout=5)
        second_item, second_result = result_queue.get(timeout=5)
        assert isinstance(first_item, BlockingGenable)
        assert isinstance(second_item, BlockingGenable)
        assert first_result == {"ok": True}
        assert second_result == {"ok": True}
        assert state["entered"] == 2
        assert state["max_active"] == 1
    finally:
        controller.shutdown()
        manager.shutdown()

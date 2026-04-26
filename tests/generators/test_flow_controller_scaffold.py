import threading
import time
from queue import Queue

import pytest

from steptronoss.core.generators.flow_controller import (
    FullyAsyncFlowController,
    SimpleFlowController,
)
from steptronoss.core.generators.flow_controller_simulator import SimulatedFlowDataloader, simulate_flow_controller
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.rl import EnvTrajectory, FlowControllerConfig, FullyAsyncFlowControllerConfig
from steptronoss.generation.base_generatable import TrainableItem
from steptronoss.utils.rl_utils import PersistentFlow, PersistentQueue, PersistentSource


def build_cfg(strategy: str) -> FlowControllerConfig | FullyAsyncFlowControllerConfig:
    if strategy == "fully-async":
        cfg = FullyAsyncFlowControllerConfig()
        cfg.prompt_per_iter = 2
        cfg.max_untrained_prompts = 4
        cfg.max_staleness = 2
        cfg.vllm_cfg = VLLMDeployConfig()
        return cfg
    cfg = FlowControllerConfig()
    cfg.async_strategy = strategy
    cfg.prompt_per_iter = 2
    cfg.vllm_cfg = VLLMDeployConfig()
    return cfg


def test_build_flow_controller_dispatches_to_scaffold_for_fully_async():
    cfg = build_cfg("fully-async")

    controller = cfg.build_flow_controller()

    assert isinstance(controller, FullyAsyncFlowController)
    assert controller.cfg is cfg
    assert controller.vllm_cfg is cfg.vllm_cfg


def test_build_flow_controller_keeps_simple_controller_for_on_policy():
    cfg = build_cfg("on-policy")

    controller = cfg.build_flow_controller()

    assert isinstance(controller, SimpleFlowController)


def test_fully_async_requires_start_before_get_train_samples():
    cfg = build_cfg("fully-async")
    controller = FullyAsyncFlowController(flow_cfg=cfg)

    with pytest.raises(RuntimeError, match="start"):
        controller.get_train_samples()


def test_fully_async_staleness_helper_matches_simulator_rule():
    cfg = build_cfg("fully-async")
    cfg.max_staleness = 1
    controller = FullyAsyncFlowController(flow_cfg=cfg)

    controller.train_weight_version = 0
    controller.running_genables = {"a": {"scheduled_version": 0}}
    assert controller._can_advance_weight_locked() is True

    controller.train_weight_version = 1
    assert controller._can_advance_weight_locked() is False


def test_fully_async_scheduling_helper_allows_restored_source_buffer():
    cfg = build_cfg("fully-async")
    controller = FullyAsyncFlowController(flow_cfg=cfg)
    controller.flow = PersistentFlow(
        source=PersistentSource(nextable=SimulatedFlowDataloader([1])),
        pre_gen=PersistentQueue(),
        pre_train=PersistentQueue(),
    )
    controller.source_exhausted = True

    with controller.flow.lock:
        controller.flow["source"].data.appendleft(next(controller.flow["source"]._source))
        assert controller._can_schedule_prompt_locked() is True


class FakeTrainableItem(TrainableItem):
    def __init__(self, item_id: int, delay_s: float):
        super().__init__()
        self.item_id = item_id
        self.delay_s = delay_s

    async def generate(self):
        raise RuntimeError("not used in test")

    async def generate_for_train(self):
        raise RuntimeError("not used in test")

    def fingerprint(self) -> str:
        return f"fake:{self.item_id}"


class FakeNextable:
    def __init__(self, items):
        self.items = list(items)
        self.index = 0

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state_dict):
        self.index = state_dict["index"]


class FakeGenerationController:
    def __init__(self, max_concurrent_genables=None):
        self.max_concurrent_genables = max_concurrent_genables
        worker_count = max_concurrent_genables or 1
        self._queue: Queue = Queue()
        self._workers = [threading.Thread(target=self._worker, daemon=True) for _ in range(worker_count)]
        for worker in self._workers:
            worker.start()

    def _worker(self):
        while True:
            genable, callback = self._queue.get()
            time.sleep(genable.delay_s)
            callback(
                genable,
                [
                    EnvTrajectory(
                        trajectory=[genable.item_id],
                        logprobs=[0.0],
                        is_gen_mask=[True],
                        meta={"item_id": genable.item_id},
                        stop_type=0,
                        raw_reward=1.0,
                    )
                ],
            )

    def submit_with_callback(self, genable, for_train=False, callback=None, task_id=None):
        assert callback is not None
        self._queue.put((genable, callback))


class ErrorGenerationController(FakeGenerationController):
    """Generation controller that returns errors for specific item IDs."""

    def __init__(self, error_ids: set[int], **kwargs):
        super().__init__(**kwargs)
        self.error_ids = error_ids

    def _worker(self):
        while True:
            genable, callback = self._queue.get()
            time.sleep(genable.delay_s)
            if genable.item_id in self.error_ids:
                callback(genable, RuntimeError(f"generation failed for {genable.item_id}"))
            else:
                callback(
                    genable,
                    [
                        EnvTrajectory(
                            trajectory=[genable.item_id],
                            logprobs=[0.0],
                            is_gen_mask=[True],
                            meta={"item_id": genable.item_id},
                            stop_type=0,
                            raw_reward=1.0,
                        )
                    ],
                )


class DummyVLLMClient:
    def wait_for_server(self):
        return None


def test_fully_async_controller_smoke(monkeypatch):
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 2
    cfg.max_untrained_prompts = 2
    cfg.max_staleness = 2
    cfg.max_concurrent_genables = 1
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        FakeGenerationController,
    )

    controller = FullyAsyncFlowController(flow_cfg=cfg)
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01 + i * 0.01) for i in range(4)]),
        model=[],
    )
    assert controller.generator.max_concurrent_genables == 1

    batch0 = controller.get_train_samples()
    controller.weight_dumped()
    batch1 = controller.get_train_samples()

    assert [traj.meta["item_id"] for traj in batch0] == [0, 1]
    assert [traj.meta["item_id"] for traj in batch1] == [2, 3]
    assert controller.train_weight_version == 1


def test_simple_controller_forwards_max_concurrent_genables(monkeypatch):
    cfg = build_cfg("one-step-off")
    cfg.max_concurrent_genables = 1
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        FakeGenerationController,
    )

    controller = cfg.build_flow_controller()
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01 + i * 0.01) for i in range(2)]),
        model=[],
    )

    assert controller.generator.max_concurrent_genables == 1


def test_fully_async_controller_rejects_deadlocking_config():
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 4
    cfg.max_untrained_prompts = 2

    with pytest.raises(ValueError, match="deadlocks"):
        FullyAsyncFlowController(flow_cfg=cfg)


def test_fully_async_controller_allows_one_extra_untrained_prompt_buffer():
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 4
    cfg.max_untrained_prompts = 3

    controller = FullyAsyncFlowController(flow_cfg=cfg)

    assert isinstance(controller, FullyAsyncFlowController)


def _collect_runtime_batches(controller, num_batches: int) -> list[list[int]]:
    batches = []
    for _ in range(num_batches):
        batch = controller.get_train_samples()
        batches.append([traj.meta["item_id"] for traj in batch])
        controller.weight_dumped()
    return batches


def test_one_step_off_runtime_matches_simulator(monkeypatch):
    cfg = build_cfg("one-step-off")
    cfg.prompt_per_iter = 2
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        FakeGenerationController,
    )

    controller = cfg.build_flow_controller()
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01 + i * 0.01) for i in range(4)]),
        model=[],
    )

    runtime_batches = _collect_runtime_batches(controller, num_batches=2)
    simulated = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 4],
        train_cost=1,
    )

    assert runtime_batches == [list(snapshot.yielded_ids) for snapshot in simulated.yield_snapshots]


def test_fully_async_error_with_allow_errors_does_not_hang(monkeypatch):
    """When genable_allow_errors=True and a generation fails, the pipeline
    should skip the failed item and continue without hanging."""
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 2
    cfg.max_untrained_prompts = 4
    cfg.max_staleness = 2
    cfg.max_concurrent_genables = 2
    cfg.genable_allow_errors = True
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None

    error_ids = {1}
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        lambda **kwargs: ErrorGenerationController(error_ids=error_ids, **kwargs),
    )

    controller = FullyAsyncFlowController(flow_cfg=cfg)
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01) for i in range(4)]),
        model=[],
    )

    # Should complete within timeout — no hang
    batch0 = controller.get_train_samples()
    controller.weight_dumped()
    batch1 = controller.get_train_samples()
    controller.weight_dumped()

    all_ids = [traj.meta["item_id"] for traj in batch0 + batch1]
    # Item 1 failed, so its trajectories are empty — remaining items should appear
    assert 1 not in all_ids
    assert 0 in all_ids


def test_fully_async_error_cleans_up_running_genables(monkeypatch):
    """When a generation error occurs, running_genables should be cleaned up
    so staleness checks don't block forever."""
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 1
    cfg.max_untrained_prompts = 4
    cfg.max_staleness = 1
    cfg.max_concurrent_genables = 2
    cfg.genable_allow_errors = True
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None

    error_ids = {0}
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        lambda **kwargs: ErrorGenerationController(error_ids=error_ids, **kwargs),
    )

    controller = FullyAsyncFlowController(flow_cfg=cfg)
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01) for i in range(3)]),
        model=[],
    )

    # Collect multiple batches — should not hang due to stale running_genables
    for _ in range(3):
        controller.get_train_samples()
        controller.weight_dumped()

    # All running_genables should have been cleaned up
    assert len(controller.running_genables) == 0


def test_simple_controller_error_with_allow_errors_does_not_hang(monkeypatch):
    """SimpleFlowController: when genable_allow_errors=True and a generation
    fails, active_counter should still decrement so the pipeline doesn't hang."""
    cfg = build_cfg("one-step-off")
    cfg.prompt_per_iter = 2
    cfg.max_concurrent_genables = 2
    cfg.genable_allow_errors = True
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None

    error_ids = {1}
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        lambda **kwargs: ErrorGenerationController(error_ids=error_ids, **kwargs),
    )

    controller = cfg.build_flow_controller()
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01) for i in range(4)]),
        model=[],
    )

    # Should complete within timeout — no hang from active_counter leak
    batch0 = controller.get_train_samples()
    controller.weight_dumped()

    all_ids = [traj.meta["item_id"] for traj in batch0]
    assert 1 not in all_ids
    assert 0 in all_ids


def test_fully_async_runtime_matches_simulator(monkeypatch):
    cfg = build_cfg("fully-async")
    cfg.prompt_per_iter = 2
    cfg.max_untrained_prompts = 2
    cfg.max_staleness = 2
    cfg.vllm_cfg.build_cli = lambda: DummyVLLMClient()
    cfg.vllm_cfg.deploy_training_model = lambda model: None
    monkeypatch.setattr(
        "steptronoss.core.generators.flow_controller.GenerationController",
        FakeGenerationController,
    )

    controller = FullyAsyncFlowController(flow_cfg=cfg)
    controller.start(
        dataloader=FakeNextable([FakeTrainableItem(i, 0.01 + i * 0.01) for i in range(4)]),
        model=[],
    )

    runtime_batches = _collect_runtime_batches(controller, num_batches=2)
    simulated = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 4],
        train_cost=1,
        max_concurrent=cfg.max_untrained_prompts,
    )

    assert runtime_batches == [list(snapshot.yielded_ids) for snapshot in simulated.yield_snapshots]

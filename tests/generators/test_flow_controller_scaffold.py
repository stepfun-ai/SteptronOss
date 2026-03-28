import threading
import time
from contextlib import nullcontext

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
        self._sema = threading.Semaphore(max_concurrent_genables) if max_concurrent_genables is not None else None

    def submit_with_callback(self, genable, for_train=False, callback=None, task_id=None):
        assert callback is not None

        def worker():
            cm = self._sema if self._sema is not None else nullcontext()
            with cm:
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

        threading.Thread(target=worker, daemon=True).start()


class DummyVLLMClient:
    def wait_for_server(self):
        return None


def test_fully_async_controller_smoke(monkeypatch):
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

    batch0 = controller.get_train_samples()
    controller.weight_dumped()
    batch1 = controller.get_train_samples()

    assert [traj.meta["item_id"] for traj in batch0] == [0, 1]
    assert [traj.meta["item_id"] for traj in batch1] == [2, 3]
    assert controller.train_weight_version == 1


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

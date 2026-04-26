from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from threading import Condition, Thread
from uuid import uuid4

import torch
from loguru import logger

from steptronoss.checkpointing.hf_checkpoint import dump_safetensors
from steptronoss.core.parallel_state import PM
from steptronoss.data.nextable.nextable import Nextable
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.rl import EnvTrajectory, FlowControllerConfig, FullyAsyncFlowControllerConfig
from steptronoss.generation.async_generation import GenerationController
from steptronoss.generation.base_generatable import GenableItem
from steptronoss.timers import timeit
from steptronoss.utils.rl_utils import PersistentFlow, PersistentQueue, PersistentSource


class FlowController(metaclass=ABCMeta):
    """How to use FlowController in trainer:
    self.controller = FlowController()
    self.controller.start()

    for i in range(train_iters):
        sample = self.controller.get_train_samples()
        models.train_on(sample)

    """

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_train_samples(self) -> list[EnvTrajectory]:
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def _load_state_dict(self):
        pass


class SimpleFlowController(FlowController):
    def __init__(self, flow_cfg: FlowControllerConfig):
        self.cfg = flow_cfg
        assert self.cfg.async_strategy in ["on-policy", "one-step-off"]

        self.infer_weight_version = -1
        self.train_weight_version = -1

        self.yielded_but_not_acked_train_samples = []
        self.source_exhausted = False

    def start(
        self,
        dataloader: Nextable,
        model: torch.nn.Module,
        state_dict: dict | None = None,
    ):
        self.model = model
        if PM.world_rank == 0:
            self.vllm_client = self.cfg.vllm_cfg.build_cli()

            self.generator = GenerationController(
                max_concurrent_genables=self.cfg.max_concurrent_genables,
            )

            self.flow = PersistentFlow(
                source=PersistentSource(nextable=dataloader),
                # pre_gen Queue: --> [... P3, P2, Require(weight1), P1, P0, Require(weight0)]
                pre_gen=PersistentQueue(),
                # pre_train Queue: --> [..., CanTrain(), R3, R2, CanTrain(), R1, R0]
                pre_train=PersistentQueue(),
            )
            if state_dict:
                self._load_state_dict(state_dict)
            Thread(target=self._generation_worker, daemon=True).start()
            Thread(target=self._control_worker, daemon=True).start()

    def state_dict(self):
        if PM.world_rank == 0:
            return {
                "infer_weight_version": self.infer_weight_version,
                "train_weight_version": self.train_weight_version,
                "flow": self.flow.state_dict(),
            }
        return {}

    def _load_state_dict(self, state_dict):
        if PM.world_rank != 0 or not state_dict:
            return
        self.infer_weight_version = -1
        self.train_weight_version = state_dict.get("train_weight_version", -1)
        if "flow" in state_dict:
            self.flow.load_state_dict(state_dict["flow"])
            logger.warning(f"FlowCkpt loaded, flow:\n{self.flow}")

    @timeit(level=1)
    def sync_weight(self):
        """Synchronize weights from train to infer, if already synced, do nothing."""
        if self.infer_weight_version != self.train_weight_version:
            with timeit("model_to_generator", level=1):
                self.cfg.vllm_cfg.deploy_training_model(self.model)

            self.infer_weight_version = self.train_weight_version

    def _control_worker(self):
        """
        transfer num_prompts samples from source to pre_gen stream:
        ```
        - for on-policy  : req(0), p0, p1, train(), req(1), p2, p3, train(), req(2), ...
        - for OSOP       : req(0), p0, p1, train(), req(0), p2, p3, train(), req(1), ...
        ```"""
        if self.cfg.async_strategy == "on-policy":
            self.flow.meta.setdefault("scheduled-weight-version", 0)
        elif self.cfg.async_strategy == "one-step-off":
            self.flow.meta.setdefault("scheduled-weight-version", -1)

        while 1:
            if self.source_exhausted:
                return
            if self.flow.meta["scheduled-weight-version"] <= self.train_weight_version + 1:
                version = max(self.flow.meta["scheduled-weight-version"], 0)

                with self.flow.lock:
                    self.flow["pre_gen"].put(PersistentFlow.Signal("need_version", version=version))

                    for _i in range(self.cfg.prompt_per_iter):
                        try:
                            self.flow["pre_gen"].put(self.flow["source"].pop())
                        except StopIteration:
                            self.source_exhausted = True
                            return

                    self.flow["pre_gen"].put(PersistentFlow.Signal("train"))

                    self.flow.meta["scheduled-weight-version"] += 1

            time.sleep(1)

    def _generation_worker(self):
        """
        transfer all Trainable from pre_gen to post-gen:
        ```
        pre_gen: req(0), p0, p1, train(), req(1), p2, p3, train(), ...

        req(n): wait till infer_weight_version >= n
        train(): wait till prev prompts done. (ack req)
        ```"""
        active_counter = 0
        ack_id_gen_req_version = None

        def flow_callback(genable: GenableItem, generated: dict):
            nonlocal active_counter
            try:
                if isinstance(generated, Exception):
                    if not self.cfg.genable_allow_errors:
                        raise generated
                    else:  # ignore this
                        generated = []

                with self.flow.lock:
                    self.flow["pre_train"].put(generated)
                    self.flow["pre_gen"].ack(genable.meta.pop("ack_pre_id"))
            finally:
                active_counter -= 1

        while 1:
            ack_id, data = self.flow["pre_gen"].get()
            if isinstance(data, PersistentFlow.Signal):
                if data.name == "need_version":
                    while self.infer_weight_version < data["version"]:
                        time.sleep(1)  # wait for new weight

                    ack_id_gen_req_version = ack_id

                elif data.name == "train":
                    while active_counter:
                        time.sleep(1)  # wait for all pending

                    with self.flow.lock:
                        if ack_id_gen_req_version:
                            self.flow["pre_gen"].ack(ack_id_gen_req_version)
                        self.flow["pre_train"].put(data)
                        self.flow["pre_gen"].ack(ack_id)

            else:
                data.meta["ack_pre_id"] = ack_id
                self.generator.submit_with_callback(data, for_train=True, callback=flow_callback)
                active_counter += 1

    def get_train_samples(self) -> list[EnvTrajectory]:
        """Get samples for single train iter, return of this function is also a train trigger."""
        self.train_weight_version += 1
        self.sync_weight()
        trajectories = []
        if PM.world_rank == 0:
            while True:
                ack_id, trajs = self.flow["pre_train"].get()
                self.yielded_but_not_acked_train_samples.append(ack_id)

                if trajs == PersistentFlow.Signal("train"):
                    break
                else:
                    prompt_id = str(uuid4())
                    for t in trajs:
                        t.prompt_id = prompt_id
                    trajectories.extend(trajs)
        return trajectories

    def weight_dumped(self):
        if PM.world_rank == 0:
            for ack_id in self.yielded_but_not_acked_train_samples:
                self.flow["pre_train"].ack(ack_id)
            self.yielded_but_not_acked_train_samples.clear()


class FullyAsyncFlowController(FlowController):
    """Fully-async decouples infer and train with a buffer of untrained prompts.

    Think of the knobs as "where do we insert idle":

    - `prompt_per_iter`: train-side start threshold. If `pre_train` has fewer
      ready prompts than this, train idles and waits for infer to fill the
      batch.
    - `max_untrained_prompts`: infer-side backpressure cap. If
      `len(pre_train) + len(running)` grows too large, infer idles and stops
      pulling new prompts from the source.
    - `max_staleness`: train-side version-switch guard. After each train step,
      if the still-running prompts are too old relative to the next weight
      version, train idles until the old tail shrinks.

    Legend below:
    - `T`: training
    - digits on the infer line: concurrent running genables
    - `Y`: a yield point where `prompt_per_iter` prompts move to training
    - `+`: idle newly introduced by changing one knob relative to the baseline

    Baseline (`prompt_per_iter=2`, `max_untrained_prompts=8`,
    `max_staleness=2`):
    ```
    train:        2 T T T        2 T T T
    infer: 2 2 2 2 Y 2 2 2 2 Y
    ```

    If `prompt_per_iter` gets larger, train waits longer before each yield:
    ```
    train:        + + + + 4 T T T + + + + 4 T T T
    infer: 2 2 2 2 2 2 2 2 Y 2 2 2 2 2 2 2 2 Y
    ```

    If `prompt_per_iter` gets smaller, train starts more often with smaller
    batches:
    ```
    train:    1 T T T    1 T T T    1 T T T
    infer: 2 2 Y 2 2 Y 2 2 Y
    ```

    If `max_untrained_prompts` gets smaller, infer hits backpressure earlier
    and idles more:
    ```
    train:        2 T T T          2 T T T
    infer: 2 2 Y + + + + 2 2 Y
    ```

    If `max_untrained_prompts` gets larger, infer can stay ahead of train for
    longer and idles less:
    ```
    train:        2 T T T        2 T T T
    infer: 2 2 2 2 Y 2 2 2 2 Y
    ```

    If `max_staleness` gets smaller, train waits longer at version boundaries
    (especially with long-tail prompts):
    ```
    train:     1 T + + + 1 T
    infer: 2 2 Y 2 2 2 Y
    ```

    If `max_staleness` gets larger, train is more willing to update weights
    while old-version infer work is still in flight:
    ```
    train:     1 T   1 T
    infer: 2 2 Y 2 Y
    ```
    """

    def __init__(self, flow_cfg: FullyAsyncFlowControllerConfig):
        self.cfg = flow_cfg
        self.vllm_cfg = self.cfg.vllm_cfg
        assert self.cfg.async_strategy in ["fully-async"]
        if self.cfg.max_untrained_prompts is None or self.cfg.max_staleness is None:
            raise ValueError("fully-async requires max_untrained_prompts and max_staleness")
        if self.cfg.max_untrained_prompts + 1 < self.cfg.prompt_per_iter:
            raise ValueError(
                "fully-async deadlocks when max_untrained_prompts + 1 < prompt_per_iter: "
                f"{self.cfg.max_untrained_prompts} + 1 < {self.cfg.prompt_per_iter}"
            )

        self.infer_weight_version = -1
        self.train_weight_version = -1

        self.yielded_but_not_acked_train_samples = []
        self.running_genables: dict[str, dict] = {}
        self.source_exhausted = False
        self.started = False

    def start(
        self,
        dataloader: Nextable,
        model: torch.nn.Module,
        state_dict: dict | None = None,
    ):
        self.model = model
        if PM.world_rank == 0:
            self.vllm_client = self.vllm_cfg.build_cli()
            self.generator = GenerationController(
                max_concurrent_genables=(
                    self.cfg.max_concurrent_genables
                    if self.cfg.max_concurrent_genables is not None
                    else self.cfg.max_untrained_prompts
                ),
            )
            self.flow = PersistentFlow(
                source=PersistentSource(nextable=dataloader),
                pre_gen=PersistentQueue(),
                pre_train=PersistentQueue(),
            )
            self._cv = Condition(self.flow.lock)
            self.flow.meta.setdefault("scheduled-weight-version", 0)
            self.flow.meta.setdefault("max-untrained-prompts", self.cfg.max_untrained_prompts)
            self.flow.meta.setdefault("max-staleness", self.cfg.max_staleness)
            if state_dict:
                self._load_state_dict(state_dict)
            Thread(target=self._generation_worker, daemon=True).start()
            Thread(target=self._control_worker, daemon=True).start()

        self.started = True

    def state_dict(self):
        if PM.world_rank == 0:
            return {
                "controller_type": "fully-async",
                "infer_weight_version": self.infer_weight_version,
                "train_weight_version": self.train_weight_version,
                "running_genables_snapshot": self.running_genables,
                "source_exhausted": self.source_exhausted,
                "flow": self.flow.state_dict() if hasattr(self, "flow") else None,
            }
        return {}

    def _load_state_dict(self, state_dict):
        if PM.world_rank != 0 or not state_dict:
            return
        self.infer_weight_version = -1
        self.train_weight_version = state_dict.get("train_weight_version", -1)
        # In-flight prompts are reconstructed from pre_gen pending items after load.
        self.running_genables = {}
        self.source_exhausted = state_dict.get("source_exhausted", False)
        if "flow" in state_dict and state_dict["flow"] is not None:
            self.flow.load_state_dict(state_dict["flow"])
            logger.warning(f"FullyAsyncFlowController state loaded, flow:\n{self.flow}")

    @timeit(level=1)
    def sync_weight(self):
        """Synchronize weights from train to infer, if already synced, do nothing."""
        if self.infer_weight_version != self.train_weight_version:
            with timeit("model_to_generator", level=1):
                self.vllm_cfg.deploy_training_model(self.model)
            self.infer_weight_version = self.train_weight_version
            if PM.world_rank == 0 and hasattr(self, "_cv"):
                with self._cv:
                    self._cv.notify_all()

    def _can_schedule_prompt_locked(self) -> bool:
        if self.source_exhausted and len(self.flow["source"]) == 0:
            return False
        if len(self.flow["pre_gen"]) > 0:
            return False
        outstanding = len(self.flow["pre_train"]) + len(self.running_genables)
        return outstanding <= self.cfg.max_untrained_prompts

    def _can_advance_weight_locked(self) -> bool:
        if not self.running_genables:
            return True
        next_version = self.train_weight_version + 1
        max_staleness_after_update = max(
            next_version - meta["scheduled_version"] for meta in self.running_genables.values()
        )
        return max_staleness_after_update <= self.cfg.max_staleness

    def _wait_for_next_weight(self):
        if PM.world_rank != 0:
            self.train_weight_version += 1
            self.sync_weight()
            return

        while True:
            with self._cv:
                if self._can_advance_weight_locked():
                    self.train_weight_version += 1
                    self.flow.meta["scheduled-weight-version"] = self.train_weight_version
                    break
                self._cv.wait(timeout=0.5)
        self.sync_weight()

    def _control_worker(self):
        while True:
            with self._cv:
                scheduled_any = False
                while self._can_schedule_prompt_locked():
                    try:
                        genable = self.flow["source"].pop()
                    except StopIteration:
                        self.source_exhausted = True
                        break

                    genable.meta["scheduled_version"] = max(self.train_weight_version, 0)
                    self.flow["pre_gen"].put(genable)
                    scheduled_any = True

                if scheduled_any:
                    self._cv.notify_all()
                self._cv.wait(timeout=0.5)

    def _generation_worker(self):
        def flow_callback(genable: GenableItem, generated: dict):
            try:
                if isinstance(generated, Exception):
                    if not self.cfg.genable_allow_errors:
                        raise generated
                    generated = []

                with self._cv:
                    self.flow["pre_train"].put(generated)
                    self.flow["pre_gen"].ack(genable.meta.pop("ack_pre_id"))
                    self.running_genables.pop(genable.meta.pop("running_ack_id"), None)
                    self._cv.notify_all()
            except Exception:
                with self._cv:
                    self.running_genables.pop(genable.meta.get("running_ack_id"), None)
                    self._cv.notify_all()
                raise

        while True:
            with self._cv:
                while len(self.flow["pre_gen"]) == 0:
                    self._cv.wait(timeout=0.5)
                ack_id, data = self.flow["pre_gen"].get()
                scheduled_version = data.meta.get("scheduled_version", max(self.train_weight_version, 0))
                self.running_genables[ack_id] = {
                    "scheduled_version": scheduled_version,
                    "submitted_at": time.time(),
                }
                self._cv.notify_all()

            while self.infer_weight_version < scheduled_version:
                with self._cv:
                    if self.infer_weight_version >= scheduled_version:
                        break
                    self._cv.wait(timeout=0.5)

            data.meta["ack_pre_id"] = ack_id
            data.meta["running_ack_id"] = ack_id
            self.generator.submit_with_callback(data, for_train=True, callback=flow_callback)

    def get_train_samples(self) -> list[EnvTrajectory]:
        if not self.started:
            raise RuntimeError("FullyAsyncFlowController.start() must be called before get_train_samples()")

        self._wait_for_next_weight()

        trajectories = []
        if PM.world_rank == 0:
            with self._cv:
                while len(self.flow["pre_train"]) < self.cfg.prompt_per_iter:
                    self._cv.wait(timeout=0.5)

                for _ in range(self.cfg.prompt_per_iter):
                    ack_id, trajs = self.flow["pre_train"].get()
                    self.yielded_but_not_acked_train_samples.append(ack_id)

                    prompt_id = str(uuid4())
                    for t in trajs:
                        t.prompt_id = prompt_id
                    trajectories.extend(trajs)

                self._cv.notify_all()
        return trajectories

    def weight_dumped(self):
        if PM.world_rank == 0 and hasattr(self, "flow"):
            for ack_id in self.yielded_but_not_acked_train_samples:
                self.flow["pre_train"].ack(ack_id)
            self.yielded_but_not_acked_train_samples.clear()
            if hasattr(self, "_cv"):
                with self._cv:
                    self._cv.notify_all()

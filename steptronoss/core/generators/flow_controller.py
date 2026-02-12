from __future__ import annotations

import os
import time
from abc import ABCMeta, abstractmethod
from threading import Thread
from typing import Literal, Optional
from uuid import uuid4

import torch
from configurize import Config
from loguru import logger

from steptronoss.checkpointing.hf_checkpoint import dump_safetensors
from steptronoss.core.parallel_state import PM
from steptronoss.data.nextable.nextable import Nextable
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.rl import EnvTrajectory, FlowControllerConfig
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

    def start(self, dataloader: Nextable, model: torch.nn.Module, state_dict: dict | None = None):
        self.model = model
        if PM.world_rank == 0:
            self.vllm_client = self.cfg.vllm_cfg.build_cli()

            self.generator = GenerationController()

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
            if self.flow.meta["scheduled-weight-version"] <= self.train_weight_version + 1:
                version = max(self.flow.meta["scheduled-weight-version"], 0)

                with self.flow.lock:
                    self.flow["pre_gen"].put(PersistentFlow.Signal("need_version", version=version))

                    for _i in range(self.cfg.prompt_per_iter):
                        self.flow["pre_gen"].put(self.flow["source"].pop())

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
            if isinstance(generated, Exception):
                if not self.cfg.genable_allow_errors:
                    raise generated
                else:  # ignore this
                    generated = []

            with self.flow.lock:
                self.flow["pre_train"].put(generated)
                self.flow["pre_gen"].ack(genable.meta.pop("ack_pre_id"))
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
    """```
    Un/Wn: update weight n
    T: wait for train samples

    # max-concurrent = 1

    # infer-so-fast (infer-sleep when len(pre_train) + len(running) > max_untrained_prompts)
    train: <U0>   <T>00000<U1><T>11111<U2>
    infer: <W0>00011122233<W1>34445556<W2>66777

    # train-so-fast (train-sleep when len(pre_train) < prompt_per_iter)
    train: <U0>      <T>0<U1>  <T>1<U2>  <T>2<U3>  <T>3
    infer: <W0>0000001111<W1>112222<W2>223333<W3>3344444

    # max-concurrent = 2
    # too-long-tail: P0(100000 tokens), P1~P10(1 tokens)

    # infer-so-fast (train-sleep when max(running_staleness) >= max_staleness)
    train: <U0>   <T>11111<U1><T>22222<U2>
    infer: <W0>010<W1>20304050<W2>6070<W2>

    # train-so-fast (train-sleep when max(running_staleness) >= max_staleness)
    train: <U0>        <T>1<U1>    <T>2<U2>    <T>3<U3>    <T>4
    infer: <W0>010101010202<W1>02020303<W2>03030404<W3>04040505

    ```"""

    def __init__(self, flow_cfg: FlowControllerConfig, vllm_cfg: VLLMDeployConfig):
        assert self.cfg.async_strategy in ["fully-async"]

        self.cfg = flow_cfg
        self.vllm_cfg = vllm_cfg

        self.infer_weight_version = -1
        self.train_weight_version = -1

        self.yielded_but_not_acked_train_samples = []

    def start(self, dataloader: Nextable, model: torch.nn.Module):
        self.vllm_client = self.vllm_cfg.build_cli()
        self.model = model

        self.generator = GenerationController()

        self.flow = PersistentFlow(
            dump_path=os.path.join(self.cfg.save_path, "flow_ckpt.pt"),
            **{
                "source": PersistentSource(nextable=dataloader),
                # pre_gen Queue: --> [... P3, P2, Require(weight1), P1, P0, Require(weight0)]
                "pre_train": PersistentQueue(),
                # trained samples move to dumper. NOTE: this ack by self.dump_checkpoint()
            },
        )

        Thread(target=self._generation_worker, daemon=True).start()

    def state_dict(self):
        return {
            "infer_weight_version": self.infer_weight_version,
            "train_weight_version": self.train_weight_version,
            "flow": self.flow.state_dict(),
        }

    def _load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.infer_weight_version = state_dict.get("infer_weight_version", -1)
        self.train_weight_version = state_dict.get("train_weight_version", -1)
        if "flow" in state_dict:
            self.flow.load_state_dict(state_dict["flow"])

    def sync_weight(self):
        """Synchronize weights from train to infer, if already synced, do nothing."""
        if self.infer_weight_version != self.train_weight_version:
            with timeit("model_to_generator", level=1):
                dump_safetensors(
                    save_path=self.vllm_cfg.hot_path,
                    model_reference_path=self.vllm_cfg.model_config_path,
                    tokenizer_reference_path=self.vllm_cfg.tokenizer_path,
                    models=self.model,
                )

                if PM.world_rank == 0:
                    self.vllm_client.wait_for_server()
                    self.vllm_client.reload_weights(self.vllm_cfg.hot_path)

                self.vllm_client.wait_for_server()

            self.infer_weight_version = self.train_weight_version

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
            if isinstance(generated, Exception):
                if not self.cfg.genable_allow_errors:
                    raise generated
                else:  # ignore this
                    generated = []

            with self.flow.lock:
                self.flow["pre_train"].put(generated)
                self.flow["pre_gen"].ack(genable.meta.pop("ack_pre_id"))
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
        while True:
            ack_id, trajs = self.flow["pre_train"].get()
            self.yielded_but_not_acked_train_samples.append(ack_id)

            if trajs == PersistentFlow.Signal("train"):
                break
            else:
                trajectories.extend(trajs)
        return trajectories

    def weight_dumped(self):
        for ack_id in self.yielded_but_not_acked_train_samples:
            self.flow["pre_train"].ack(ack_id)
        self.yielded_but_not_acked_train_samples.clear()

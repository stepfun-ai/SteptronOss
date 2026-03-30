"""Utilities for reasoning about flow-controller scheduling without real models.

Usage
-----
This module exposes a Fire CLI with one public command:

```bash
uv run python steptronoss/core/generators/flow_controller_simulator.py \
  render --strategy <strategy> --prompt_per_iter <prompt_per_iter> \
  --train_cost <train_cost> --genables <genables> [flags...]
```

Recommended `genables` format is a comma-separated string such as
`1,2,3,4,1,2,3,4`, where each integer is the simulated inference cost for one
genable in dataloader order.

Examples
--------
On-policy / one-step-off:

```bash
uv run python steptronoss/core/generators/flow_controller_simulator.py \
  render \
  --strategy one-step-off \
  --prompt_per_iter 3 \
  --train_cost 4 \
  --genables 1,2,3,1,2,3
```

Fully-async:

```bash
uv run python steptronoss/core/generators/flow_controller_simulator.py \
  render \
  --strategy fully-async \
  --prompt_per_iter 2 \
  --train_cost 3 \
  --genables 1,2,3,4,1,2,3,4 \
  --max_untrained_prompts 8 \
  --max_staleness 1 \
  --max_concurrent 4
```

The optional `max_concurrent` flag applies to every strategy:

- omitted / `None`: treat infer as having enough slots for all currently
  dispatchable prompts
- positive integer: infer may advance at most that many prompts concurrently

For `fully-async`, the simulator additionally assumes:

- `prompt_per_iter`: how many ready prompts one train step consumes
- `max_untrained_prompts`: how many prompts infer may keep in
  `running + pre_train` before backpressure inserts infer idle
- `max_staleness`: how far old running prompts may lag the next train version
  before version-switch idle is inserted
- `max_concurrent`: how many genables infer may run at once in the simulator

Also note that the current fully-async model rejects obviously deadlocking
configs where `max_untrained_prompts + 1 < prompt_per_iter`.

Balanced Base-Cost Regime
-------------------------
When you want a clean baseline where train and infer tie in steady state before
adding long-tail noise, choose a constant per-prompt infer cost that satisfies:

```text
ceil(prompt_per_iter / max_concurrent) * infer_cost == train_cost
```

Example: with `prompt_per_iter=10`, `train_cost=500`, and `max_concurrent=10`,
setting every genable cost to `500` makes the infer side produce one full batch
every `500` ticks in steady state. That gives a useful baseline for comparing:

- no jitter: `one-step-off` and `fully-async` should tie
- centered signed jitter: per-prompt costs become `base_cost + jitter_i`, where
  positive and negative jitter cancel in expectation
- long-tail comparison: if `fully-async` is useful, it should keep more steps
  near the train-cost floor while `one-step-off` is pulled up by block maxima

How to read `render`
--------------------
The output is intentionally compact:

- `Strategy ... Versions ...`: the chosen flow-control mode and the train-side
  version used for each yielded batch
- `Genables ... TrainCost ...`: the input cost sequence and the fixed per-batch
  train cost
- `Train:` timeline:
  - space = train idle
  - `T` = train is running
  - digit = a yield event; the digit is how many prompts were yielded
- `Infer:` timeline:
  - space = infer idle
  - digit = number of concurrently running genables
  - `Y` = a yield event aligned with the train line
- `Yield k: trunc[...] yield[...]`:
  - `trunc[...]` lists prompts still running at that yield as
    `genable_id(generated/total)`
  - `yield[...]` lists the genable ids handed to training at that yield
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import fire

from steptronoss.data.nextable.nextable import Nextable
from steptronoss.exp.rl import EnvTrajectory, FlowControllerConfig, FullyAsyncFlowControllerConfig
from steptronoss.generation.base_generatable import TrainableItem


class SimulatedGenableItem(TrainableItem):
    """Synthetic genable used only for flow-controller timeline simulation."""

    def __init__(self, infer_cost: int, item_id: int, meta: dict | None = None):
        if infer_cost < 0:
            raise ValueError(f"infer_cost must be >= 0, got {infer_cost}")
        super().__init__(meta=meta)
        self.infer_cost = infer_cost
        self.item_id = item_id

    async def generate(self) -> Any:
        raise RuntimeError("SimulatedGenableItem is for timeline simulation only")

    async def generate_for_train(self) -> list[EnvTrajectory]:
        raise RuntimeError("SimulatedGenableItem is for timeline simulation only")

    def fingerprint(self) -> str:
        return f"simulated-genable:{self.item_id}:{self.infer_cost}"


class SimulatedFlowDataloader(Nextable):
    """Finite Nextable that yields simulated genables with fixed inference costs."""

    def __init__(self, infer_costs: Sequence[int]):
        self.infer_costs = list(infer_costs)
        self.index = 0

    def __next__(self) -> SimulatedGenableItem:
        if self.index >= len(self.infer_costs):
            raise StopIteration
        item = SimulatedGenableItem(infer_cost=self.infer_costs[self.index], item_id=self.index)
        self.index += 1
        return item

    def state_dict(self) -> dict[str, int]:
        return {"index": self.index}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.index = state_dict["index"]


@dataclass(frozen=True)
class FlowSimulationBlock:
    index: int
    required_version: int
    infer_costs: tuple[int, ...]
    infer_concurrency: tuple[int, ...]
    request_time: int
    sync_time: int
    gen_start: int
    ready_time: int
    yield_time: int
    train_start: int
    train_end: int

    @property
    def prompt_count(self) -> int:
        return len(self.infer_costs)


@dataclass(frozen=True)
class FullyAsyncPromptTrace:
    item_id: int
    infer_cost: int
    launched_version: int
    launch_time: int
    completion_time: int


@dataclass(frozen=True)
class FullyAsyncTrainBatch:
    batch_index: int
    train_version: int
    prompt_ids: tuple[int, ...]
    prompt_versions: tuple[int, ...]
    start_time: int
    end_time: int


@dataclass(frozen=True)
class YieldSnapshot:
    yield_index: int
    entries: tuple[tuple[int, int, int], ...]
    truncated_ids: tuple[int, ...] = ()
    yielded_ids: tuple[int, ...] = ()


@dataclass
class _RunningGenable:
    item_id: int
    remaining: int
    launched_version: int
    infer_cost: int


@dataclass
class FlowSimulationResult:
    strategy: str
    prompt_per_iter: int
    train_cost: int
    max_concurrent: int | None
    genable_costs: list[int]
    blocks: list[FlowSimulationBlock]
    train_timeline: list[str]
    infer_timeline: list[str]
    train_markers: dict[int, list[str]] = field(default_factory=dict)
    infer_markers: dict[int, list[str]] = field(default_factory=dict)
    prompt_traces: list[FullyAsyncPromptTrace] = field(default_factory=list)
    train_batches: list[FullyAsyncTrainBatch] = field(default_factory=list)
    yield_snapshots: list[YieldSnapshot] = field(default_factory=list)

    def render(self) -> str:
        versions = (
            [block.required_version for block in self.blocks]
            if self.blocks
            else [batch.train_version for batch in self.train_batches]
        )
        lines = [
            (
                f"Strategy: {self.strategy} PromptPerIter: {self.prompt_per_iter} "
                f"MaxConcurrent: {self.max_concurrent} Versions: {versions}"
                if self.strategy == "fully-async"
                else f"Strategy: {self.strategy} PromptPerIter: {self.prompt_per_iter} Versions: {versions}"
            ),
            f"Genables: {self.genable_costs} TrainCost: {self.train_cost}",
            "Legend: T: Training Y: Yield InferDigits: Concurrent Gen",
            f"Train: {self._render_timeline(self.train_timeline)}",
            f"Infer: {self._render_timeline(self.infer_timeline)}",
        ]
        for snapshot in self.yield_snapshots:
            trunc_state = ", ".join(f"{item_id}({gened}/{total})" for item_id, gened, total in snapshot.entries)
            trunc_ids = ", ".join(str(item_id) for item_id in snapshot.truncated_ids)
            yield_ids = ", ".join(str(item_id) for item_id in snapshot.yielded_ids)
            if trunc_ids:
                if trunc_state:
                    trunc_state = f"{trunc_state}, {trunc_ids}"
                else:
                    trunc_state = trunc_ids
            lines.append(f"Yield {snapshot.yield_index}: trunc[{trunc_state}] yield[{yield_ids}]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    @staticmethod
    def _render_timeline(tokens: Sequence[str], markers: dict[int, list[str]] | None = None) -> str:
        rendered: list[str] = []
        markers = markers or {}
        for index, token in enumerate(tokens):
            rendered.extend(markers.get(index, []))
            rendered.append(" " if token == "N" else token)
        rendered.extend(markers.get(len(tokens), []))
        return " ".join(rendered)


class FlowControllerSimulator:
    """Deterministic simulator for flow-controller scheduling semantics."""

    def __init__(self, flow_cfg: FlowControllerConfig | FullyAsyncFlowControllerConfig):
        self.flow_cfg = flow_cfg
        strategy = self.flow_cfg.async_strategy
        if strategy not in {"on-policy", "one-step-off", "fully-async"}:
            raise NotImplementedError(f"Unsupported async_strategy for simulation: {strategy}")

        prompt_per_iter = getattr(self.flow_cfg, "prompt_per_iter", None)
        if prompt_per_iter is None or prompt_per_iter <= 0:
            raise ValueError(f"prompt_per_iter must be a positive int, got {prompt_per_iter}")

    @staticmethod
    def build_dataloader(infer_costs: Sequence[int]) -> SimulatedFlowDataloader:
        return SimulatedFlowDataloader(infer_costs=infer_costs)

    def simulate(
        self,
        infer_costs: Sequence[int] | None = None,
        train_cost: int = 0,
        dataloader: Nextable | None = None,
        max_concurrent: int | None = None,
    ) -> FlowSimulationResult:
        if train_cost < 0:
            raise ValueError(f"train_cost must be >= 0, got {train_cost}")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        items = self._materialize_items(infer_costs=infer_costs, dataloader=dataloader)
        genable_costs = [item.infer_cost for item in items]
        if self.flow_cfg.async_strategy == "fully-async":
            if self.flow_cfg.max_untrained_prompts is None or self.flow_cfg.max_staleness is None:
                raise ValueError("fully-async simulation requires max_untrained_prompts and max_staleness")
            if self.flow_cfg.max_untrained_prompts + 1 < self.flow_cfg.prompt_per_iter:
                raise ValueError(
                    "fully-async deadlocks when max_untrained_prompts + 1 < prompt_per_iter: "
                    f"{self.flow_cfg.max_untrained_prompts} + 1 < {self.flow_cfg.prompt_per_iter}"
                )
            return self._simulate_fully_async(
                items=items,
                genable_costs=genable_costs,
                train_cost=train_cost,
                max_concurrent=max_concurrent,
            )

        blocks = self._build_blocks(
            genable_costs=genable_costs,
            train_cost=train_cost,
            max_concurrent=max_concurrent,
        )
        train_timeline, infer_timeline = self._render_timelines(blocks=blocks)
        return FlowSimulationResult(
            strategy=self.flow_cfg.async_strategy,
            prompt_per_iter=self.flow_cfg.prompt_per_iter,
            train_cost=train_cost,
            max_concurrent=max_concurrent,
            genable_costs=genable_costs,
            blocks=blocks,
            train_timeline=train_timeline,
            infer_timeline=infer_timeline,
            yield_snapshots=[
                YieldSnapshot(
                    yield_index=index,
                    entries=(),
                    yielded_ids=tuple(
                        range(
                            index * self.flow_cfg.prompt_per_iter,
                            min((index + 1) * self.flow_cfg.prompt_per_iter, len(genable_costs)),
                        )
                    ),
                )
                for index, _block in enumerate(blocks)
            ],
        )

    def _simulate_fully_async(
        self,
        items: list[SimulatedGenableItem],
        genable_costs: list[int],
        train_cost: int,
        max_concurrent: int | None,
    ) -> FlowSimulationResult:
        if self.flow_cfg.max_untrained_prompts is None:
            raise ValueError("fully-async simulation requires flow_cfg.max_untrained_prompts")
        if self.flow_cfg.max_staleness is None:
            raise ValueError("fully-async simulation requires flow_cfg.max_staleness")

        source = deque(items)
        running: deque[_RunningGenable] = deque()
        pre_train: deque[FullyAsyncPromptTrace] = deque()
        completed_traces: list[FullyAsyncPromptTrace] = []
        trace_meta: dict[int, dict[str, int]] = {}
        train_batches: list[FullyAsyncTrainBatch] = []
        train_markers: dict[int, list[str]] = defaultdict(list)
        infer_markers: dict[int, list[str]] = defaultdict(list)
        train_tokens: list[str] = []
        infer_tokens: list[str] = []
        yield_events: dict[int, list[str]] = defaultdict(list)
        yield_snapshots: list[YieldSnapshot] = []

        current_version = 0
        next_batch_index = 0
        training_remaining = 0
        blocked_on_staleness = False

        while True:
            current_time = len(train_tokens)

            if training_remaining == 0:
                if blocked_on_staleness and self._can_advance_fully_async_version(current_version, running):
                    current_version += 1
                    blocked_on_staleness = False

                if len(pre_train) >= self.flow_cfg.prompt_per_iter and not blocked_on_staleness:
                    current_batch = [pre_train.popleft() for _ in range(self.flow_cfg.prompt_per_iter)]
                    train_batches.append(
                        FullyAsyncTrainBatch(
                            batch_index=next_batch_index,
                            train_version=current_version,
                            prompt_ids=tuple(sample.item_id for sample in current_batch),
                            prompt_versions=tuple(sample.launched_version for sample in current_batch),
                            start_time=current_time,
                            end_time=current_time + train_cost,
                        )
                    )
                    yield_events[current_time].append(str(len(current_batch)))
                    yield_snapshots.append(
                        YieldSnapshot(
                            yield_index=len(yield_snapshots),
                            entries=self._snapshot_running_state(running),
                            yielded_ids=tuple(sample.item_id for sample in current_batch),
                        )
                    )
                    training_remaining = train_cost
                    next_batch_index += 1

                    if train_cost == 0:
                        if self._has_future_fully_async_work(source, running, pre_train, self.flow_cfg.prompt_per_iter):
                            if self._can_advance_fully_async_version(current_version, running):
                                current_version += 1
                            else:
                                blocked_on_staleness = True
                        continue

            self._launch_fully_async_genables(
                source=source,
                running=running,
                pre_train=pre_train,
                completed_traces=completed_traces,
                trace_meta=trace_meta,
                current_version=current_version,
                current_time=current_time,
                max_concurrent=max_concurrent,
            )

            if (
                training_remaining == 0
                and not source
                and not running
                and len(pre_train) < self.flow_cfg.prompt_per_iter
                and not blocked_on_staleness
            ):
                break

            train_token = "T" if training_remaining > 0 else "N"
            infer_token = str(len(running)) if running else "N"

            if running:
                next_running: deque[_RunningGenable] = deque()
                while running:
                    active = running.popleft()
                    active.remaining -= 1
                    if active.remaining == 0:
                        trace = FullyAsyncPromptTrace(
                            item_id=active.item_id,
                            infer_cost=active.infer_cost,
                            launched_version=active.launched_version,
                            launch_time=trace_meta[active.item_id]["launch_time"],
                            completion_time=current_time + 1,
                        )
                        completed_traces.append(trace)
                        pre_train.append(trace)
                    else:
                        next_running.append(active)
                running.extend(next_running)

            train_tokens.append(train_token)
            infer_tokens.append(infer_token)

            if training_remaining > 0:
                training_remaining -= 1
                if training_remaining == 0:
                    if self._has_future_fully_async_work(source, running, pre_train, self.flow_cfg.prompt_per_iter):
                        if self._can_advance_fully_async_version(current_version, running):
                            current_version += 1
                        else:
                            blocked_on_staleness = True

        train_tokens, infer_tokens = self._insert_yield_events(
            train_tokens=train_tokens,
            infer_tokens=infer_tokens,
            yield_events=yield_events,
        )

        return FlowSimulationResult(
            strategy=self.flow_cfg.async_strategy,
            prompt_per_iter=self.flow_cfg.prompt_per_iter,
            train_cost=train_cost,
            max_concurrent=max_concurrent,
            genable_costs=genable_costs,
            blocks=[],
            train_timeline=train_tokens,
            infer_timeline=infer_tokens,
            train_markers=dict(train_markers),
            infer_markers=dict(infer_markers),
            prompt_traces=sorted(completed_traces, key=lambda trace: trace.item_id),
            train_batches=train_batches,
            yield_snapshots=yield_snapshots,
        )

    def _materialize_items(
        self,
        infer_costs: Sequence[int] | None,
        dataloader: Nextable | None,
    ) -> list[SimulatedGenableItem]:
        if infer_costs is None and dataloader is None:
            raise ValueError("Provide either infer_costs or dataloader")
        if infer_costs is not None and dataloader is not None:
            raise ValueError("Provide only one of infer_costs or dataloader")

        if infer_costs is not None:
            return [SimulatedGenableItem(infer_cost=cost, item_id=index) for index, cost in enumerate(infer_costs)]

        assert dataloader is not None
        snapshot = dataloader.state_dict()
        items = []
        try:
            while True:
                item = next(dataloader)
                if not isinstance(item, SimulatedGenableItem):
                    raise TypeError(f"Simulation dataloader must yield SimulatedGenableItem, got {type(item).__name__}")
                items.append(item)
        except StopIteration:
            pass
        finally:
            dataloader.load_state_dict(snapshot)
        return items

    def _build_blocks(
        self,
        genable_costs: list[int],
        train_cost: int,
        max_concurrent: int | None,
    ) -> list[FlowSimulationBlock]:
        prompt_per_iter = self.flow_cfg.prompt_per_iter
        blocks: list[FlowSimulationBlock] = []
        sync_times: dict[int, int] = {}

        for block_index, start in enumerate(range(0, len(genable_costs), prompt_per_iter)):
            infer_costs = tuple(genable_costs[start : start + prompt_per_iter])
            required_version = self._required_version(block_index)
            request_time = 0 if block_index == 0 else blocks[-1].train_end
            sync_times[block_index] = request_time
            sync_time = sync_times[required_version]
            infer_duration, infer_concurrency = self._simulate_parallel_window(
                infer_costs=infer_costs,
                max_concurrent=max_concurrent,
            )
            gen_start = max(
                blocks[-1].ready_time if blocks else 0,
                sync_time,
            )
            ready_time = gen_start + infer_duration
            yield_time = max(request_time, ready_time)
            train_start = yield_time
            train_end = train_start + train_cost
            blocks.append(
                FlowSimulationBlock(
                    index=block_index,
                    required_version=required_version,
                    infer_costs=infer_costs,
                    infer_concurrency=infer_concurrency,
                    request_time=request_time,
                    sync_time=sync_time,
                    gen_start=gen_start,
                    ready_time=ready_time,
                    yield_time=yield_time,
                    train_start=train_start,
                    train_end=train_end,
                )
            )

        return blocks

    @staticmethod
    def _simulate_parallel_window(
        infer_costs: Sequence[int],
        max_concurrent: int | None,
    ) -> tuple[int, tuple[int, ...]]:
        if not infer_costs:
            return 0, ()

        waiting = deque(int(cost) for cost in infer_costs)
        running: deque[int] = deque()
        concurrency: list[int] = []
        limit = len(infer_costs) if max_concurrent is None else max_concurrent

        while waiting and len(running) < limit:
            cost = waiting.popleft()
            if cost > 0:
                running.append(cost)

        while running:
            concurrency.append(len(running))
            next_running: deque[int] = deque()
            while running:
                remaining = running.popleft() - 1
                if remaining > 0:
                    next_running.append(remaining)
            running = next_running

            while waiting and len(running) < limit:
                cost = waiting.popleft()
                if cost > 0:
                    running.append(cost)

        return len(concurrency), tuple(concurrency)

    def _required_version(self, block_index: int) -> int:
        strategy = self.flow_cfg.async_strategy
        if strategy == "on-policy":
            return block_index
        if strategy == "one-step-off":
            return max(block_index - 1, 0)
        raise NotImplementedError(f"Unsupported async_strategy for simulation: {strategy}")

    def _launch_fully_async_genables(
        self,
        source: deque[SimulatedGenableItem],
        running: deque[_RunningGenable],
        pre_train: deque[FullyAsyncPromptTrace],
        completed_traces: list[FullyAsyncPromptTrace],
        trace_meta: dict[int, dict[str, int]],
        current_version: int,
        current_time: int,
        max_concurrent: int | None,
    ) -> None:
        while (
            source
            and (max_concurrent is None or len(running) < max_concurrent)
            and self._can_dispatch_fully_async(pre_train, running)
        ):
            item = source.popleft()
            trace_meta[item.item_id] = {
                "launch_time": current_time,
                "launched_version": current_version,
                "infer_cost": item.infer_cost,
            }
            if item.infer_cost == 0:
                trace = FullyAsyncPromptTrace(
                    item_id=item.item_id,
                    infer_cost=item.infer_cost,
                    launched_version=current_version,
                    launch_time=current_time,
                    completion_time=current_time,
                )
                completed_traces.append(trace)
                pre_train.append(trace)
                continue
            running.append(
                _RunningGenable(
                    item_id=item.item_id,
                    remaining=item.infer_cost,
                    launched_version=current_version,
                    infer_cost=item.infer_cost,
                )
            )

    def _can_dispatch_fully_async(
        self,
        pre_train: deque[FullyAsyncPromptTrace],
        running: deque[_RunningGenable],
    ) -> bool:
        outstanding = len(pre_train) + len(running)
        return outstanding <= self.flow_cfg.max_untrained_prompts

    def _can_advance_fully_async_version(
        self,
        current_version: int,
        running: deque[_RunningGenable],
    ) -> bool:
        if not running:
            return True
        max_running_staleness = max(current_version - item.launched_version for item in running)
        return max_running_staleness < self.flow_cfg.max_staleness

    @staticmethod
    def _has_future_fully_async_work(
        source: deque[SimulatedGenableItem],
        running: deque[_RunningGenable],
        pre_train: deque[FullyAsyncPromptTrace],
        prompt_per_iter: int,
    ) -> bool:
        return bool(source or running or len(pre_train) >= prompt_per_iter)

    @staticmethod
    def _snapshot_running_state(running: deque[_RunningGenable]) -> tuple[tuple[int, int, int], ...]:
        return tuple((item.item_id, item.infer_cost - item.remaining, item.infer_cost) for item in running)

    @staticmethod
    def _insert_yield_events(
        train_tokens: list[str],
        infer_tokens: list[str],
        yield_events: dict[int, list[str]],
    ) -> tuple[list[str], list[str]]:
        rendered_train: list[str] = []
        rendered_infer: list[str] = []
        FlowControllerSimulator._append_yield_events(rendered_train, rendered_infer, yield_events, at_time=0)
        for current_time, (train_token, infer_token) in enumerate(zip(train_tokens, infer_tokens)):
            rendered_train.append(train_token)
            rendered_infer.append(infer_token)
            FlowControllerSimulator._append_yield_events(
                rendered_train,
                rendered_infer,
                yield_events,
                at_time=current_time + 1,
            )
        return rendered_train, rendered_infer

    def _render_timelines(self, blocks: list[FlowSimulationBlock]) -> tuple[list[str], list[str]]:
        if not blocks:
            return [], []

        final_time = blocks[-1].train_end
        train_by_time = ["N"] * final_time
        infer_by_time = ["N"] * final_time
        yield_events: dict[int, list[str]] = defaultdict(list)

        for block in blocks:
            for offset, current_time in enumerate(range(block.gen_start, block.ready_time)):
                infer_by_time[current_time] = str(block.infer_concurrency[offset])

            for current_time in range(block.train_start, block.train_end):
                train_by_time[current_time] = "T"

            yield_events[block.yield_time].append(str(block.prompt_count))

        train_tokens: list[str] = []
        infer_tokens: list[str] = []

        self._append_yield_events(train_tokens, infer_tokens, yield_events, at_time=0)
        for current_time in range(final_time):
            train_tokens.append(train_by_time[current_time])
            infer_tokens.append(infer_by_time[current_time])
            self._append_yield_events(train_tokens, infer_tokens, yield_events, at_time=current_time + 1)

        return train_tokens, infer_tokens

    @staticmethod
    def _append_yield_events(
        train_tokens: list[str],
        infer_tokens: list[str],
        yield_events: dict[int, list[str]],
        at_time: int,
    ) -> None:
        for prompt_count in yield_events.get(at_time, []):
            train_tokens.append(prompt_count)
            infer_tokens.append("Y")


def simulate_flow_controller(
    flow_cfg: FlowControllerConfig | FullyAsyncFlowControllerConfig,
    infer_costs: Sequence[int] | None = None,
    train_cost: int = 0,
    dataloader: Nextable | None = None,
    max_concurrent: int | None = None,
) -> FlowSimulationResult:
    simulator = FlowControllerSimulator(flow_cfg=flow_cfg)
    return simulator.simulate(
        infer_costs=infer_costs,
        train_cost=train_cost,
        dataloader=dataloader,
        max_concurrent=max_concurrent,
    )


def _normalize_genables(genables: Any) -> list[int]:
    if isinstance(genables, str):
        tokens = [token.strip() for token in genables.replace(",", " ").split()]
        if not tokens:
            raise ValueError("genables string must contain at least one integer")
        return [int(token) for token in tokens]

    if isinstance(genables, Sequence) and not isinstance(genables, (bytes, bytearray)):
        if not genables:
            raise ValueError("genables sequence must be non-empty")
        return [int(token) for token in genables]

    raise TypeError(f"Unsupported genables type: {type(genables).__name__}")


class FlowControllerSimulatorCLI:
    """Fire entrypoint for rendering flow-controller timelines."""

    @staticmethod
    def _build_flow_cfg(
        strategy: str,
        prompt_per_iter: int,
        max_untrained_prompts: int | None,
        max_staleness: int | None,
    ) -> FlowControllerConfig | FullyAsyncFlowControllerConfig:
        if strategy == "fully-async":
            flow_cfg = FullyAsyncFlowControllerConfig()
            flow_cfg.max_untrained_prompts = max_untrained_prompts
            flow_cfg.max_staleness = max_staleness
        else:
            flow_cfg = FlowControllerConfig()
            flow_cfg.async_strategy = strategy
        flow_cfg.prompt_per_iter = prompt_per_iter
        return flow_cfg

    def render(
        self,
        *,
        strategy: str,
        prompt_per_iter: int,
        train_cost: int,
        genables: Any,
        max_concurrent: int | None = None,
        max_untrained_prompts: int | None = None,
        max_staleness: int | None = None,
    ) -> str:
        flow_cfg = self._build_flow_cfg(
            strategy=strategy,
            prompt_per_iter=prompt_per_iter,
            max_untrained_prompts=max_untrained_prompts,
            max_staleness=max_staleness,
        )
        result = simulate_flow_controller(
            flow_cfg,
            infer_costs=_normalize_genables(genables),
            train_cost=train_cost,
            max_concurrent=max_concurrent,
        )
        return result.render()


if __name__ == "__main__":
    fire.Fire(FlowControllerSimulatorCLI)

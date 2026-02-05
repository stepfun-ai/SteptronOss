# Copyright (c) 2026, STEPFUN CORPORATION.  All rights reserved.

"""Megatron timers."""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

import torch
from loguru import logger
from torch.utils._contextlib import _DecoratorContextManager

from steptronoss.exp.base_exp import ProfilerConfig


class TimerBase(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def start(self, barrier=False):
        pass

    @abstractmethod
    def stop(self, barrier=False):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False, sync_device=False):
        pass


class DummyTimer(TimerBase):
    def __init__(self):
        super().__init__("dummy timer")

    def start(self, barrier=False, sync_device=True):
        return

    def stop(self, barrier=False, sync_device=False):
        return

    def summarize_event_time(self):
        return

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False, sync_device=False):
        raise Exception("dummy timer should not be used to calculate elapsed time")


class Timer(TimerBase):
    def __init__(self, name, use_event):
        super().__init__(name)
        self._elapsed = 0.0
        self._started = False
        # Note that None will default to the global process group
        self._barrier_group = None
        self._start_time = time.time()
        self._use_event = use_event
        self._s_events = []
        self._e_events = []

    def update_config(self, use_event: bool):
        self._use_event = use_event

    def set_barrier_group(self, barrier_group):
        self._barrier_group = barrier_group

    def start(self, barrier=False, sync_device=True):
        """Start the timer."""
        assert not self._started, "timer has already been started"
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        if self._use_event:
            curr_event = torch.cuda.Event(enable_timing=True)
            curr_event.record()
            self._s_events.append(curr_event)
        else:
            if sync_device:
                torch.cuda.synchronize()
            self._start_time = time.time()
        self._started = True

    def stop(self, barrier=False, sync_device=False):
        """Stop the timer.

        +-------------+-----------+-------+-------------------+-----------+
        | sync_device | use_event | sync  | summarize_event   | cpu time()|
        +-------------+-----------+-------+-------------------+-----------+
        |    False    |   False   |   x   |         x         |     √     |
        |    False    |   True    |   x   |      x(!!!)       |     x     |
        |    True     |   False   |   √   |         x         |     √     |
        |    True     |   True    |   √   |         √         |     x     |
        +-------------+-----------+-------+-------------------+-----------+
        Legend:
            √ : Yes / Enabled
            x : No / Disabled
            (!!!): Special case, see code for details

        Columns:
            sync_device      - Whether device synchronization is performed
            use_event        - Whether CUDA events are used
            sync             - Actual synchronization performed
            summarize_event  - Whether event summary is performed
            cpu time()       - Whether CPU time is recorded

        """
        assert self._started, "timer is not started"
        if barrier:
            torch.distributed.barrier(group=self._barrier_group)
        if self._use_event:
            curr_event = torch.cuda.Event(enable_timing=True)
            curr_event.record()
            self._e_events.append(curr_event)
        if sync_device:
            torch.cuda.synchronize()
            if self._use_event:
                # Make sure calling torch.cuda.synchronize() before summarize_event_time
                self.summarize_event_time()
        if not self._use_event:
            self._elapsed += time.time() - self._start_time
        self._started = False

    def summarize_event_time(self):
        # should call torch.cuda.synchronize() before using this method
        if not self._use_event:
            return
        n = len(self._s_events)
        assert n == len(self._e_events), len(self._e_events)
        if n == 0:
            return
        # in seconds
        for i in range(n):
            self._elapsed += self._s_events[i].elapsed_time(self._e_events[i]) / 1e3
        self._s_events.clear()
        self._e_events.clear()

    def reset(self):
        """Reset timer."""
        self._elapsed = 0.0
        self._started = False

    def elapsed(self, reset=True, barrier=False, sync_device=False):
        """Calculate the elapsed time."""
        _started = self._started
        # If the timing in progress, end it first.
        if self._started:
            self.stop(barrier=barrier, sync_device=sync_device)
        # If there're pending events, summarize
        self.summarize_event_time()
        # Get the elapsed time.
        _elapsed = self._elapsed
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if _started:
            self.start(barrier=barrier)
        return _elapsed


class DummyTimers:
    @contextmanager
    def record(self, name, log_level=None, barrier=False, sync_device=False, use_event=None):
        yield

    def __call__(self, name, log_level=None, use_event=None):
        return DummyTimer()


class LocalTimers(DummyTimers):
    """Timers Class without any communication."""

    def __init__(self, log_level, use_event=False):
        self._log_level = log_level
        self._timers: dict[str, Timer] = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2
        self._use_event = use_event

    def __call__(self, name, log_level=None, use_event=None):
        # If timer does not exist and no log level is provided,
        # set it to the max log level which is 2.
        if log_level is None:
            log_level = self._log_levels.get(name, self._max_log_level)
        assert log_level <= self._max_log_level, (
            f"log level {log_level} is larger than max supported log level {self._max_log_level}"
        )
        # Now if the input log level is larger than the one set for
        # the timers class, just ignore it and return a dummy timer.
        if log_level > self._log_level:
            return self._dummy_timer

        # If the timer has already been set, then check if the log-level
        # is provided, it matches the one that the timer was created with.
        if name in self._timers:
            if log_level is not None:
                assert log_level == self._log_levels[name], (
                    f"input log level {log_level} does not match already existing "
                    f"log level {self._log_levels[name]} for {name} timer"
                )
            return self._timers[name]

        # Otherwise, initalize the timer and set the level.
        use_event = (
            False if log_level < 0 else (self._use_event if use_event is None else use_event)
        )  # iter-time disable event
        self._timers[name] = Timer(name, use_event)
        self._log_levels[name] = log_level
        return self._timers[name]

    def _get_elapsed_time(self, reset, normalizer):
        """Report only min and max times across all ranks."""

        torch.cuda.synchronize()  # synchronize events
        all_elapsed_times = {k: (v.elapsed(reset=reset) / normalizer) for k, v in self._timers.items()}
        all_elapsed_times = dict(
            filter(
                lambda x: self._log_levels[x[0]] >= 0 and x[1] > 0,
                all_elapsed_times.items(),
            )
        )

        return all_elapsed_times

    @contextmanager
    def record(self, name, log_level=None, barrier=False, sync_device=False, use_event=None):
        timer = self(name, log_level, use_event=use_event)
        timer.start(barrier=barrier, sync_device=sync_device)
        yield
        timer.stop(barrier=barrier, sync_device=sync_device)

    def write(self, writer, iteration, normalizer=1.0, reset=False, barrier=False):
        """Write timers to a tensorboard writer
        Note that we only report maximum time across ranks to tensorboard.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_elapses = self._get_elapsed_time(reset, normalizer)
        if not name_elapses:
            return
        if writer is not None:
            for name in name_elapses:
                writer.add_scalar("Timers/" + name, name_elapses[name], iteration)

    def log(self, iteration: int, normalizer=1.0, reset=True, prefix=""):
        """Log a group of timers."""

        name_elapses = self._get_elapsed_time(reset, normalizer)

        if name_elapses:
            # Print.
            log_str = f"{prefix} | Activated Timers:\n"
            for name, elapse in name_elapses.items():
                log_str += f"{name:}".ljust(48, ".") + f"{elapse:.4e} s\n"
            logger.info(log_str, at=0)

    def reload(self, new_log_level: int, new_use_event: bool):
        """Reload configuration"""

        if new_log_level > self._max_log_level:
            new_log_level = self._max_log_level
        self._log_level = new_log_level
        self._use_event == new_use_event
        # update config of each real timer
        for name, timer in self._timers.items():
            if self._log_levels[name] >= 0:
                timer.update_config(new_use_event)


GLOBAL_TIMERS: LocalTimers | DummyTimers = DummyTimers()


class timeit(_DecoratorContextManager):
    def __init__(
        self,
        name=None,
        log_level=None,
        barrier=False,
        sync_device=False,
        use_event=False,
    ) -> None:
        self.name = name
        self.log_level = log_level
        self.barrier = barrier
        self.sync_device = sync_device
        self.use_event = use_event

    def __call__(self, orig_func):
        if self.name is None:
            self.name = orig_func.__name__
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        GLOBAL_TIMERS(self.name, log_level=self.log_level, use_event=self.use_event).start(
            barrier=self.barrier, sync_device=self.sync_device
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        GLOBAL_TIMERS(self.name, log_level=self.log_level, use_event=self.use_event).stop(
            barrier=self.barrier, sync_device=self.sync_device
        )

    def clone(self) -> "timeit":
        return self.__class__(self.name, self.log_level, self.barrier, self.sync_device, self.use_event)


def get_timers() -> LocalTimers:
    global GLOBAL_TIMERS
    return GLOBAL_TIMERS


def init_timers(config: ProfilerConfig):
    global GLOBAL_TIMERS
    if isinstance(GLOBAL_TIMERS, DummyTimers):
        GLOBAL_TIMERS = LocalTimers(config.timing_log_level, use_event=config.timing_use_event)
    return GLOBAL_TIMERS

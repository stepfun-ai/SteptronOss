"""Async helpers for accelerating SlowFastNextable data sources."""

import multiprocessing as mp
import queue
import threading
import time
from typing import Any

import torch
from loguru import logger
from torch.utils.data._utils.pin_memory import pin_memory

from .nextable import SlowFastNextable

PREFETCH_QUEUE_SIZE: int = 10
DEFAULT_READ_FROM_WORKER_TIMEOUT: int = 10
WARN_INTERVAL_SECONDS: float = 60.0


def soft_kill_process(process: mp.Process, timeout: float = 2) -> None:
    """Terminate a process if possible, otherwise kill it."""
    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()


class PrefetchMuxInputMultiProcessQueue:
    def __init__(
        self,
        mux_size: int = 1,
        queue_size: int = 0,
        pin_memory: bool = True,
        alternative_queue_list: list[Any] | None = None,
    ) -> None:
        """A multi-in, single-out multiprocess queues group with prefetch."""
        if alternative_queue_list is not None:
            self.input_queues: list[Any] = alternative_queue_list
        else:
            single_queue_size = 0 if queue_size == 0 else max(1, queue_size // mux_size)
            self.input_queues = [mp.Queue(single_queue_size) for _ in range(mux_size)]
        self.mux_size = len(self.input_queues)
        self.prefetch_output_queue: queue.Queue = queue.Queue(PREFETCH_QUEUE_SIZE)
        self.pin_emory = pin_memory
        self.idx = -1
        self.thread_stop_signal = False
        self._last_prefetch_empty_warn = 0.0
        self._last_prefetch_duration_warn = 0.0
        self._last_input_empty_warn = [0.0 for _ in range(self.mux_size)]

        self._prefetch_thread: threading.Thread = threading.Thread(
            name="PrefetchLoop",
            target=self._prefetch_loop,
            daemon=True,
        )
        self._prefetch_thread.start()

    def _get_one_data(self, block: bool = True, timeout: float | None = None) -> Any:
        self.idx = (self.idx + 1) % self.mux_size
        if self.input_queues[self.idx].empty():
            now = time.time()
            if now - self._last_input_empty_warn[self.idx] >= WARN_INTERVAL_SECONDS:
                logger.warning(f"InputQueue {self.idx} is empty")
                self._last_input_empty_warn[self.idx] = now
        if timeout is not None:
            return self.input_queues[self.idx].get(block=block, timeout=timeout)

        # Use the default timeout to print warning logs
        timeout = DEFAULT_READ_FROM_WORKER_TIMEOUT
        wanring_log_interval = DEFAULT_READ_FROM_WORKER_TIMEOUT
        start_time = time.time()
        while not self.thread_stop_signal:
            try:
                return self.input_queues[self.idx].get(block=block, timeout=timeout)
            except queue.Empty:
                empty_duration = time.time() - start_time
                if empty_duration > wanring_log_interval:
                    now = time.time()
                    if now - self._last_input_empty_warn[self.idx] >= WARN_INTERVAL_SECONDS:
                        logger.warning(f"InputQueue {self.idx} remains empty after {empty_duration:.2f} seconds.")
                        self._last_input_empty_warn[self.idx] = now
                    wanring_log_interval *= 10

    def _prefetch_loop(self) -> None:
        try:
            while not self.thread_stop_signal:
                data = self._get_one_data()
                if self.pin_emory and not isinstance(data, Exception):
                    # Pin tensors to speed up host->device transfers.
                    data = pin_memory(data)
                while not self.thread_stop_signal:
                    try:
                        self.prefetch_output_queue.put(data, timeout=DEFAULT_READ_FROM_WORKER_TIMEOUT)
                        break
                    except queue.Full:
                        pass
        except Exception as e:
            logger.exception("Error when prefetching data:")
            self.prefetch_output_queue.put(e)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        if not self.prefetch_output_queue.empty():
            return self.prefetch_output_queue.get(block=block, timeout=timeout)
        now = time.time()
        if now - self._last_prefetch_empty_warn >= WARN_INTERVAL_SECONDS:
            logger.warning("Prefetch queue is empty.")
            self._last_prefetch_empty_warn = now
        if timeout is not None:
            return self.prefetch_output_queue.get(block=block, timeout=timeout)

        # Use the default timeout to print warning logs
        timeout = DEFAULT_READ_FROM_WORKER_TIMEOUT
        start_time = time.time()
        while True:
            try:
                data = self.prefetch_output_queue.get(block=block, timeout=timeout)
                now = time.time()
                if now - self._last_prefetch_duration_warn >= WARN_INTERVAL_SECONDS:
                    logger.warning(f"Prefetch queue empty duration {time.time() - start_time:.2f} seconds.")
                    self._last_prefetch_duration_warn = now
                return data
            except queue.Empty:
                now = time.time()
                if now - self._last_prefetch_duration_warn >= WARN_INTERVAL_SECONDS:
                    logger.warning(f"Prefetch queue remains empty after {time.time() - start_time:.2f} seconds.")
                    self._last_prefetch_duration_warn = now
                timeout *= 10

    def __del__(self) -> None:
        self.thread_stop_signal = True


class AsyncAcceleratedSlowFast(SlowFastNextable):
    def __init__(
        self,
        slowfast: SlowFastNextable,
        queue_size: int = 100,
        num_workers: int = 2,
        pin_memory: bool = False,
    ) -> None:
        """```
        Main: ckpt -> fast(0), fast(1), fast(2), fast(3) -> ckpt
                ↓        ↑        ↑        ↑        ↑
            worker0:  slow(0)     ↑        ↑        ↑
            worker1:  slow(1)  ----        ↑        ↑
            worker2:  slow(2)  -------------        ↑
            worker3:  slow(3)  ----------------------

        Args:
            queue_size (int, optional): Size of the MP Queue. Defaults to 100.
            num_workers (int, optional): number of MP workers. Defaults to 2.
        ```"""
        # super().__init__(*args, **kwargs)
        self.nextable: SlowFastNextable = slowfast
        self._shadow_process: mp.Process | None = None
        self._data_process: list[mp.Process] | None = None
        self._data_queue: PrefetchMuxInputMultiProcessQueue | None = None
        self._shadow_worker_queues: tuple[Any, Any] | None = None
        self.queue_size = queue_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _data_worker(self, mux_id: int = 0, mux_size: int = 1) -> None:
        torch.set_num_threads(1)
        qid = mux_id % len(self._data_queue.input_queues)
        i = 0
        while True:
            # process data on my rank only
            if i % mux_size == mux_id:
                data = self.nextable.__next__()
                self._data_queue.input_queues[qid].put(data)
            else:
                self.nextable.fast_step()
            i += 1

    def _shadow_worker(self) -> None:
        # NOTE: in mp context, data index is not shared for data workers
        # we use a shadow worker for index reference.
        input_queue, output_queue = self._shadow_worker_queues
        while True:
            cmd = input_queue.get()
            if cmd == "next":
                self.nextable.fast_step()
            elif cmd == "state":
                output_queue.put(self.nextable.state_dict())
            elif cmd == "repr":
                output_queue.put("Asynced " + self.nextable.__repr__())
            elif cmd == "exit":
                break

    def _reset(self) -> None:
        self._shadow_worker_queues = (
            mp.Queue(self.queue_size),
            mp.Queue(1),
        )
        num_local_workers = self.num_workers

        self._data_queue = PrefetchMuxInputMultiProcessQueue(num_local_workers, self.queue_size, self.pin_memory)

        if self._data_process is not None:
            for p in self._data_process:
                soft_kill_process(p)
            self._data_process = None

        self._data_process = [
            mp.Process(
                target=self._data_worker,
                kwargs=dict(mux_id=i, mux_size=self.num_workers),
            )
            for i in range(self.num_workers)
        ]
        for p in self._data_process:
            p.start()

        if self._shadow_process is not None:
            soft_kill_process(self._shadow_process)
            self._shadow_process = None
        self._shadow_process = mp.Process(target=self._shadow_worker)
        self._shadow_process.start()

    def __next__(self) -> Any:
        if self._data_process is None:
            self._reset()
        data = self._data_queue.get()
        self._shadow_worker_queues[0].put("next")
        return data

    def __len__(self) -> int:
        return len(self.nextable)

    def __del__(self) -> None:
        if self._data_process is not None:
            for p in self._data_process:
                soft_kill_process(p)
        if self._shadow_process is not None:
            soft_kill_process(self._shadow_process)
        if self._data_queue is not None:
            del self._data_queue
        if hasattr(self.nextable, "__del__"):
            self.nextable.__del__()

    def __repr__(self) -> str:
        if self._data_process:
            self._shadow_worker_queues[0].put("repr")
            info = self._shadow_worker_queues[1].get()
            return info
        else:
            return self.nextable.__repr__()

    def state_dict(self) -> dict:
        self._shadow_worker_queues[0].put("state")
        return self._shadow_worker_queues[1].get()

    def load_state_dict(self, state_dict: dict) -> None:
        self.nextable.load_state_dict(state_dict)
        self._reset()


def async_accelearte_slowfast(
    slowfast: SlowFastNextable,
    queue_size: int = 100,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> AsyncAcceleratedSlowFast:
    """Wraps a SlowFastNextable dataloader to async dataloader."""
    assert isinstance(slowfast, SlowFastNextable), "Only SlowFastNextable can be accelerated."

    return AsyncAcceleratedSlowFast(
        slowfast=slowfast,
        queue_size=queue_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

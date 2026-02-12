import asyncio
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Iterable
from queue import Empty, Queue
from typing import Any, NoReturn, Optional
from uuid import uuid4

from loguru import logger
from tqdm import tqdm

from steptronoss.exp.rl import EnvTrajectory
from steptronoss.generation.base_generatable import TrainableItem
from steptronoss.utils import run_async


class SingleGenerationController:
    def __init__(self):
        self.input_queue = Queue()

        def run_gen():
            run_async(self._generate_loop())

        self._alive = True
        self._worker = threading.Thread(target=run_gen, daemon=True)
        self._worker.start()

    @staticmethod
    async def work_on_item(item: TrainableItem, callback, for_train) -> NoReturn:
        try:
            if for_train:
                result = await item.generate_for_train()
            else:
                result = await item.generate()
        except Exception as e:
            import traceback

            logger.error("\n".join(traceback.format_exception(e)))
            result = e

        callback(item, result)

    def generate(
        self, gen_items: list[TrainableItem], for_train: bool = False
    ) -> Iterable[tuple[TrainableItem, list[EnvTrajectory] | Any]]:
        out_queue = Queue()
        for item in gen_items:
            self.input_queue.put((item, lambda a, b: out_queue.put((a, b)), for_train))

        for _item in gen_items:
            output = out_queue.get()
            yield output

    def submit_with_callback(
        self,
        genable: TrainableItem,
        for_train=False,
        callback: Callable[[tuple[TrainableItem, Any]], NoReturn] | None = None,
        task_meta: dict | None = None,
    ) -> NoReturn:
        if callback is None:
            callback = print
        if task_meta:
            genable.meta.update(task_meta)
        # genable.meta["async_gen_callback"] = callback
        self.input_queue.put((genable, callback, for_train))

    async def _generate_loop(self):
        tasks = []
        while self._alive:
            while True:
                try:
                    task = self.input_queue.get_nowait()
                    tasks.append(asyncio.create_task(self.work_on_item(*task)))
                except Empty:
                    break

            if tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
                tasks = list(pending)
            else:
                await asyncio.sleep(1)
        for t in tasks:
            if isinstance(t, asyncio.Task):
                t.cancel()

    def shutdown(self):
        self._alive = False
        self._worker.join()


def _generation_worker_process(input_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Worker process entrypoint; uses only picklable args for spawn."""
    worker = SingleGenerationController()
    while True:
        try:
            task = input_queue.get()
            if task is None:
                break

            item, task_id, for_train = task
            item.meta["_gc_task_id"] = task_id
            worker.submit_with_callback(
                item,
                for_train=for_train,
                callback=lambda x, r, q=result_queue: q.put((x.meta.pop("_gc_task_id"), x, r)),
            )
        except Exception as e:
            logger.error(f"Worker process error: {e}")


class GenerationController:
    def __init__(self, num_workers: int | None = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.input_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.callback_map = {}

        self._alive = True

        # 启动工作进程
        self.worker_processes: list[mp.Process] = []
        for _ in range(self.num_workers):
            p = mp.Process(
                target=_generation_worker_process,
                args=(self.input_queue, self.result_queue),
                daemon=True,
            )
            p.start()
            self.worker_processes.append(p)

        # 启动回调处理线程（在主进程的主线程中运行）
        self.callback_thread = threading.Thread(target=self._callback_loop, daemon=True)
        self.callback_thread.start()

    def _callback_loop(self):
        """在主进程的主线程中运行的回调处理循环"""
        pbar = tqdm(desc="Cumulative Completed")

        while self._alive:
            task_id, item, result = self.result_queue.get()
            assert task_id in self.callback_map
            callback = self.callback_map.pop(task_id)
            callback(item, result)
            pbar.update()

    def generate(
        self, gen_items: list[TrainableItem], for_train: bool = False
    ) -> Iterable[tuple[TrainableItem, list[EnvTrajectory] | Any]]:
        """批量生成方法（保持原有接口）"""
        out_queue = Queue()
        task_ids = list(range(len(gen_items)))

        # 提交所有任务
        for task_id, item in zip(task_ids, gen_items):
            self.submit_with_callback(
                item,
                for_train,
                lambda item, result, q=out_queue: q.put((item, result)),
                task_id=task_id,
            )

        # 按提交顺序收集结果
        for _ in gen_items:
            yield out_queue.get()

    def submit_with_callback(
        self,
        genable: TrainableItem,
        for_train: bool = False,
        callback: Callable[[tuple[TrainableItem, Any]], NoReturn] | None = None,
    ) -> NoReturn:
        """提交单个任务（支持自定义回调）"""
        if callback is None:
            callback = print

        # 生成唯一任务ID
        task_id = str(uuid4())

        # 保存回调函数（将在主线程执行）
        self.callback_map[task_id] = callback

        # 将任务发送给工作进程
        self.input_queue.put((genable, task_id, for_train))

    def shutdown(self):
        """关闭控制器"""
        self._alive = False

        # 发送退出信号给工作进程
        for _ in self.worker_processes:
            self.input_queue.put(None)

        # 等待工作进程结束
        for p in self.worker_processes:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()

        # 等待回调线程结束
        self.callback_thread.join(timeout=0.5)

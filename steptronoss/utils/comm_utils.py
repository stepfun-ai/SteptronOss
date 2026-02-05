import os
import pickle
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable
from functools import cached_property
from queue import Empty

import redis
from loguru import logger
from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError
from redis.retry import Retry
from torch.distributed import barrier, get_rank, get_world_size

from steptronoss.timers import get_timers
from steptronoss.utils import get_exp_id
from steptronoss.utils.dist_utils import broadcast_tensors

REDIS_STARTED = False
THREAD_LOCAL = threading.local()

IS_REDIS_MASTER = False
REDIS_ADDR = None
REDIS_PORT = None


def safely_start_redis(retries=3, wait_per_retry=1):
    """Select a random port to start a redis server.

    Check if the server has start properly, retry when there's a port conflict.
    Return the redis' port.
    """
    KEY = "redis_start_mark"
    VALUE = str(os.getpid())
    for _ in range(retries):
        try:
            # 1. 找一个可用端口
            sock = socket.socket()
            sock.bind(("0.0.0.0", 0))
            port = sock.getsockname()[1]
            sock.close()

            # 2. 日志文件
            log_file = tempfile.NamedTemporaryFile(delete=False)
            log_path = log_file.name
            log_file.close()

            # 3. 启动 Redis
            cmd = [
                "redis-server",
                "--daemonize",
                "yes",
                "--bind",
                "0.0.0.0",
                "--port",
                str(port),
                "--timeout",
                "0",
                "--protected-mode",
                "no",
                "--logfile",
                log_path,
            ]
            cmd_str = " ".join(cmd)
            logger.info(f"Starting redis server with command: {cmd_str}")
            subprocess.run(cmd, check=True)

            # 4. 等待客户端可用并校验 KEY 不存在
            for _ in range(20):  # 最多等待 2 秒
                try:
                    client = Redis(host="127.0.0.1", port=port)
                    if client.exists(KEY):
                        # 已存在，不可靠，关闭重试
                        raise RuntimeError("Redis has stall data, not our redis.")
                    client.set(KEY, VALUE)
                    assert client.get(KEY).decode() == VALUE, f"Redis server launcher pid should be {VALUE}"
                    logger.info("Successuflly started redis server.")
                    return port
                except redis.ConnectionError:
                    pass
                time.sleep(0.1)
        except Exception as e:
            logger.exception(f"Failed to start redis server: {e}")
            time.sleep(wait_per_retry)

    raise RuntimeError(f"Failed to start redis server after {retries} retries")


def block_get_redis(client: Redis, key: str, timeout: int = 1200, interval: int = 1):
    """Block get value from redis."""
    start_time = time.time()
    while True:
        result = client.get(key)
        if result is not None:
            return result
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {key}")
        time.sleep(interval)


def get_exp_redis() -> Redis:
    """Get or initialize the per-experiment Redis client.

    Rendezvous uses the shared filesystem directory from STEPTRON_MEET_DIR
    to elect a master and publish the Redis server port.
    """
    global REDIS_STARTED, IS_REDIS_MASTER, REDIS_ADDR, REDIS_PORT

    def rendezvous(exp_id):
        global REDIS_STARTED, IS_REDIS_MASTER, REDIS_ADDR, REDIS_PORT
        try:
            meet_dir = os.environ["STEPTRON_MEET_DIR"]
        except KeyError as exc:
            raise RuntimeError("STEPTRON_MEET_DIR is not set; cannot use shared filesystem rendezvous.") from exc

        os.makedirs(meet_dir, exist_ok=True)

        host = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        my_process = f"{host}:{os.getpid()}"

        rendezvous_path = os.path.join(meet_dir, f"redis-rendezvous-{exp_id}.pkl")

        def read_state():
            try:
                with open(rendezvous_path, "rb") as handle:
                    return pickle.load(handle)
            except FileNotFoundError:
                return None
            except Exception as exc:
                logger.warning(f"Failed to read rendezvous file {rendezvous_path}: {exc}")
                return None

        def write_state(state: dict):
            tmp_path = f"{rendezvous_path}.{uuid.uuid4().hex}.tmp"
            with open(tmp_path, "wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, rendezvous_path)

        def is_state_stale(state: dict) -> bool:
            created_at = state.get("created_at")
            if created_at is None:
                try:
                    created_at = os.path.getmtime(rendezvous_path)
                except OSError:
                    return True
            return time.time() - created_at > 3600 * 24

        def try_claim_master() -> int | None:
            try:
                fd = os.open(rendezvous_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                return None

            state = {
                "exp_id": exp_id,
                "master": my_process,
                "port": None,
                "created_at": time.time(),
            }
            write_state(state)
            try:
                port = safely_start_redis()
            except Exception:
                try:
                    os.remove(rendezvous_path)
                except FileNotFoundError:
                    pass
                raise

            state["port"] = port
            write_state(state)
            return port

        cannot_be_master = os.environ.get("CANNOT_BE_REDIS_SERVER", None) == "1"
        master_process = None
        port = None
        wait_start = time.monotonic()
        port_wait_start = None

        while True:
            state = read_state()
            if state is not None and is_state_stale(state):
                try:
                    os.remove(rendezvous_path)
                except FileNotFoundError:
                    pass
                state = None

            if state is None:
                if not cannot_be_master:
                    port = try_claim_master()
                    if port is not None:
                        master_process = my_process
                        IS_REDIS_MASTER = True
                        break
            else:
                if master_process is None:
                    master_process = state.get("master")
                port = state.get("port")
                if port is not None:
                    port = int(port)
                    break
                if master_process is not None and port_wait_start is None:
                    port_wait_start = time.monotonic()
                if port_wait_start is not None and time.monotonic() - port_wait_start > 120:
                    raise TimeoutError("Timeout waiting for redis port in rendezvous file")

            if time.monotonic() - wait_start > 3600:
                raise TimeoutError("Timeout waiting for shared filesystem rendezvous")
            time.sleep(1)

        if master_process is None or port is None:
            raise ConnectionError("Cannot establish redis-server for this exp!")

        master_host = master_process.split(":")[0]
        logger.info(f"Connecting to exp redis server at {master_host}:{port}")

        REDIS_ADDR = master_host
        REDIS_PORT = port

        REDIS_STARTED = Redis(
            host=master_host,
            port=port,
            socket_connect_timeout=1200,
            socket_timeout=1200,
            retry_on_timeout=True,
            retry_on_error=[
                BusyLoadingError,
                ConnectionError,
                TimeoutError,
            ],
            retry=Retry(ExponentialBackoff(), 3),  # 重试3次
        )

        _ = REDIS_STARTED.keys()  # test
        # logger.info(f"Initialized Redis for this exp, server: {master_host}")

    if not REDIS_STARTED:
        exp_id = f"{{exp}}redis-master-for-{get_exp_id()}"
        rendezvous(exp_id)
    return REDIS_STARTED


def get_exp_aredis():
    from inspect import get_annotations

    from aioredis import Redis as aRedis

    global THREAD_LOCAL

    if not hasattr(THREAD_LOCAL, "aredis_instance"):
        redis = get_exp_redis()
        aredis_args = get_annotations(aRedis.__init__)
        aredis = aRedis(**{k: v for k, v in redis.get_connection_kwargs().items() if k in aredis_args})
        THREAD_LOCAL.aredis_instance = aredis

    return THREAD_LOCAL.aredis_instance


class RedisQueue:
    """Just act like a Queue, but support sync on multi-nodes.
    NOTE: a RedisQueue is closed only when ALL rank calls close()
    """

    def __init__(self):
        self.redis = get_exp_redis()
        self.qname = str(uuid.uuid4())
        self.qname = broadcast_tensors(self.qname, src_rank=0, group=None)

        self.counter_key = self.qname + "_counter"
        self.closed_key = self.qname + "_closed"
        self.subqueue_list_key = self.qname + "_subqueue_list"

        self._closed = False
        self._local_closed = False

        self.world_size = get_world_size()

    def _build_qname(self, subqueue: str | None = None):
        if subqueue is None:
            return self.qname
        return self.qname + ":" + subqueue

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if get_rank() == 0:
            # TODO: is qname matched?
            keys = self.redis.keys(self.qname + "*")
            for key in keys:
                self.redis.expire(str(key), 3600)

    def register_subqueue(self, subqueue: str):
        self.redis.zadd(self.subqueue_list_key, {subqueue: 0})

    def put(self, data, subqueue: str | None = None, auto_balance: bool = False) -> bool:
        if auto_balance and subqueue is None:
            # NOTE(sxn): Get the subqueue with least push count. Subqueue would be None
            # if there is no subqueue list, leading to default queue.
            #
            # Currently, load balancing only considers push count. In future we can take
            # count of finished requests into account.
            subqueue = self.redis.eval(
                """
                local subqueue_list = redis.call('zrange', KEYS[1], 0, 0)
                if #subqueue_list == 0 then
                    return nil
                end
                -- lua index starts from 1
                local subqueue = subqueue_list[1]
                redis.call('zincrby', KEYS[1], 1, subqueue)
                return subqueue
            """,
                1,
                self.subqueue_list_key,
            )
            if isinstance(subqueue, bytes):
                subqueue = subqueue.decode()
        elif subqueue is not None:
            self.redis.zincrby(self.subqueue_list_key, 1, subqueue)
        pipeline = self.redis.pipeline()
        pipeline.lpush(self._build_qname(subqueue), pickle.dumps(data))
        pipeline.incr(self.counter_key)
        qlen = int(pipeline.execute()[-1])
        return qlen > 0

    def get(self, subqueue_list: list[str] | None = None) -> tuple[int, object]:
        """
        By default, get from the default queue name (qname);
        If subqueue_list is provided, get from the subqueue list in order, return on first found.
        The default queue can be included using `None` in subqueue_list.
        """
        if subqueue_list is None:
            subqueue_list = [None]
        for subqueue in subqueue_list:
            data_bytes = self.redis.rpop(self._build_qname(subqueue))
            if data_bytes is not None:
                self.redis.decr(self.counter_key)
                return pickle.loads(data_bytes)
        raise Empty

    @property
    def closed(self) -> bool:
        if self._closed:
            return True
        self._closed = bool(self.redis.llen(self.closed_key) >= self.world_size)
        return self._closed

    def query_closed(self) -> bool:
        # keep for backward compatibility
        return self.closed

    def qsize(self) -> int:
        return int(self.redis.get(self.counter_key) or 0)

    def close(self):
        assert not self._local_closed, "Cannot close a RedisQueue twice!"
        self.redis.lpush(self.closed_key, get_rank())
        self._local_closed = True


class ReusableDistributor:
    """Redis-backed distributor for multi-node all-to-all streams.

    Extra Support: push_back data to start of queue.

    - a ReusableDistributor is **closed** when all rank called close()
    - Iteration yields data from the shared queue; ordering is not guaranteed.

    Data[a, b, c] @ 0 ->
    Data[d, e, f] @ 1 ->  Distributor()   ←
                            ↓       ↓      ↑
                            0       1      ↑ push_back()
                          a,c,d   b,e,f   →
    """

    def __init__(self, data: list[object] = []):
        self.queue = RedisQueue()
        self.queue.__enter__()

        self._get_counter = 0

        with get_timers().record("put_prompt_to_distributor", log_level=1, use_event=False, sync_device=False):
            for d in data:
                self.push_back(d)
        # NOTE Distributor must be called on all ranks.
        barrier()
        logger.info(f"Distributor Initilized with {self.remains} samples")

        self.prefered_subqueue_list = None

    def __next__(self):
        try:
            data = self.queue.get(self.prefered_subqueue_list)
            self._get_counter += 1
            return data
        except Empty:
            raise StopIteration

    def register_subqueue(self, subqueue: str):
        self.queue.register_subqueue(subqueue)

    def push_back(self, data, subqueue: str | None = None, auto_balance: bool = False):
        return self.queue.put(data, subqueue, auto_balance)

    @property
    def remains(self):
        return self.queue.qsize()

    @property
    def closed(self):
        return self.queue.closed

    def close(self):
        if not self.queue._local_closed:
            return self.queue.close()

    def __del__(self):
        self.queue.__exit__()


class LocalFuture:
    class Pending:
        pass

    def __init__(self):
        self.value = LocalFuture.Pending()

    @property
    def completed(self) -> bool:
        return not isinstance(self.value, LocalFuture.Pending)

    def set_result(self, value: object):
        assert not self.completed, "Cannot set result of a completed Future!"
        self.value = value

    def result(self, timeout: float | None = None):
        return self.get_result(timeout=timeout)

    def get_result(self, timeout: float | None = None):
        start_time = time.monotonic()
        poll_interval = 0.1
        while not self.completed:
            elapsed_time = time.monotonic() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise TimeoutError(f"LocalFuture timed out after {timeout} seconds")

            time.sleep(poll_interval)

        return self.value


class RemoteFuture(LocalFuture):
    """全局可访问的Future对象，仅依赖task_id和Redis。

    任何进程只要知道 task_id 就可以等待并获取结果，不依赖于提交进程的本地状态。
    结果通过 set_result 写入 Redis，get_result 轮询读取并在读取后清理。
    支持通过 __getstate__ / __setstate__ 进行 pickle。
    """

    def __init__(
        self,
        task_id: str | None = None,
    ):
        """初始化RemoteFuture

        Args:
            task_id: 全局唯一的任务ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        self.task_id = task_id
        self.value = LocalFuture.Pending()
        self._set_default()

    @cached_property
    def PENDING_STRING(self):
        return pickle.dumps("[|<PENDING>|]")

    def _set_default(self):
        redis_client = get_exp_redis()
        redis_client.set(self.task_id, self.PENDING_STRING, nx=True)

    def handle_value(self, result_data: bytes):
        if result_data is None:
            raise Exception(f"Future: {self.task_id} waited for more than once!")

        if result_data != self.PENDING_STRING:
            # logger.debug(f"任务 {self.task_id} 结果在 Redis 中找到 (轮询)")
            fetched_value = pickle.loads(result_data)
            get_exp_redis().expire(self.task_id, 0)
            try:
                if isinstance(fetched_value, dict) and fetched_value.get("status") == "error":
                    # Store the actual exception to be raised later
                    error_message = fetched_value.get("error", "未知 Worker 错误")
                    self.value = Exception(f"任务 {self.task_id} 执行失败: {error_message}")
                else:
                    self.value = fetched_value

                if isinstance(self.value, Exception):
                    raise self.value  # Raise the stored exception
                return self.value
            except pickle.UnpicklingError as e:
                logger.error(f"反序列化结果缓存失败 for {self.task_id}: {e}")
                exc = Exception(f"获取任务 {self.task_id} 结果失败: 无法反序列化")
                self.value = exc  # Store exception
                raise exc
            except Exception as e:
                # Catch other potential errors during processing
                logger.error(f"处理任务 {self.task_id} 结果时发生错误: {e}")
                self.value = e  # Store exception
                raise e
        else:
            return self.PENDING_STRING

    def get_result(self, timeout: float | None = None):
        """阻塞等待并获取结果，get_result应在全局最多调用一次。"""
        if self.completed:
            if isinstance(self.value, Exception):  # Re-raise if stored value is exception
                raise self.value
            return self.value

        redis_client = get_exp_redis()
        result_key = self.task_id
        start_time = time.monotonic()
        poll_interval = 0.1

        while True:
            result_data = redis_client.get(result_key)
            result = self.handle_value(result_data)
            if result is not self.PENDING_STRING:
                return result

            # 检查超时
            elapsed_time = time.monotonic() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise TimeoutError(f"等待任务 {self.task_id} 结果超时 (timeout={timeout}s, 轮询)")

            time.sleep(poll_interval)

    async def aget_result(self, timeout: float | None = None):
        import asyncio

        if self.completed:
            if isinstance(self.value, Exception):  # Re-raise if stored value is exception
                raise self.value
            return self.value

        redis_client = get_exp_aredis()
        result_key = self.task_id
        start_time = time.monotonic()
        poll_interval = 0.5

        while True:
            result_data = await redis_client.get(result_key)
            result = self.handle_value(result_data)
            if result is not self.PENDING_STRING:
                return result

            # 检查超时
            elapsed_time = time.monotonic() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise TimeoutError(f"等待任务 {self.task_id} 结果超时 (timeout={timeout}s, 轮询)")

            await asyncio.sleep(poll_interval)

    def set_result(self, value) -> None:
        """设置任务结果，由任务执行者调用 (仅使用 SET)"""

        redis_client = get_exp_redis()

        if isinstance(value, Exception):
            # Include traceback info if helpful?
            value_to_serialize = {"status": "error", "error": str(value)}
        else:
            value_to_serialize = value
        serialized_result = pickle.dumps(value_to_serialize)

        # 设置结果缓存，使用默认 TTL
        success = redis_client.set(self.task_id, serialized_result)
        if not success:
            # This usually indicates a connection issue rather than logical failure in redis-py >= 3
            logger.warning(f"同步设置任务 {self.task_id} 的结果缓存可能失败 (SET returned non-True)")

    def __getstate__(self):
        return (self.task_id, self.value)

    def __setstate__(self, state):
        self.task_id, self.value = state


def get_ready(
    futures: list[LocalFuture | RemoteFuture],
) -> list[LocalFuture | RemoteFuture]:
    """POP futures that are ready from the input list."""
    ready, pending = [], []
    if not futures:
        return ready

    redis_client = get_exp_redis()

    remote_futures = []
    for f in futures:
        if isinstance(f, RemoteFuture):
            remote_futures.append(f)
        else:
            (ready if f.completed else pending).append(f)

    num_futures = len(remote_futures)

    # Use pipeline for efficient batch checking
    pipeline = redis_client.pipeline(transaction=False)
    for future_id in range(num_futures):
        result_key = remote_futures[future_id].task_id
        pipeline.get(result_key)
    exists_results = pipeline.execute()

    for future, result in zip(remote_futures, exists_results):
        if isinstance(result, bytes) and result == future.PENDING_STRING:
            pending.append(future)
        else:
            future.handle_value(result)
            ready.append(future)
    futures.clear()
    futures.extend(pending)

    return ready


def wait_by_finish_order(futures: list[RemoteFuture]) -> Iterable[RemoteFuture]:
    """Iteratively yield future from list according to finish order."""
    while futures:
        ready = get_ready(futures)
        yield from ready
        if not ready:
            time.sleep(0.1)


if __name__ == "__main__":
    # Test future use
    import threading

    futures = [LocalFuture(), RemoteFuture(), LocalFuture(), RemoteFuture()]

    futures_for_worker = [*futures]

    def worker():
        counter = 0
        while True:
            if futures_for_worker:
                futures_for_worker.pop(0).set_result(counter)
                counter += 1
            time.sleep(1)

    threading.Thread(target=worker, daemon=True).start()

    for done_future in wait_by_finish_order(futures):
        print(done_future, done_future.get_result())
        if done_future.get_result() % 2 == 0:
            f = LocalFuture()
            futures.append(f)
            futures_for_worker.append(f)

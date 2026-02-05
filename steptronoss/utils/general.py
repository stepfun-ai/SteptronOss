"""
Standalone utils that has nothing to do with Steptron.
"""

import contextlib
import functools
import os
import subprocess
from typing import TypeVar

import torch
from configurize import DataClass
from loguru import logger

T = TypeVar("Any")


def safediv(n, d):
    q, r = divmod(n, d)
    assert r == 0
    return q


def make_divisible(number: int, divisible_by: int):
    return (number + divisible_by - 1) // divisible_by * divisible_by


def shift_left(tensor: torch.Tensor, pad=0, shift=1) -> torch.Tensor:
    """[1, 2, 3] -> [2, 3, pad]"""
    out = torch.full_like(tensor, pad)
    out[:-shift] = tensor[shift:]
    return out.contiguous()


def shift_right(tensor: torch.Tensor, pad=0, shift=1) -> torch.Tensor:
    """[1, 2, 3] -> [pad, 1, 2]"""
    out = torch.full_like(tensor, pad)
    out[shift:] = tensor[:-shift]
    return out.contiguous()


def get_position_id_from_cu_seqlens(cu_seqlens: torch.IntTensor) -> torch.IntTensor:
    """Generate position ids from cumulative seq lens.
    - cu_seqlens: [N, ] e.g. [0, 3, 5]
    - return: [S, ] e.g. [0, 1, 2, 0, 1]
    """
    position_id = torch.arange(
        cu_seqlens[-1], device=cu_seqlens.device, dtype=cu_seqlens.dtype
    ) - torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff())
    return position_id


def convert_num(num: int | float, nfp=3, G=False) -> str:
    if abs(num) > 10**12:
        return f"{round(num / 10**12, nfp)}T"
    elif abs(num) > 10**8:
        return f"{round(num / 10**9, nfp)}" + ("G" if G else "B")
    elif abs(num) > 10**6:
        return f"{round(num / 10**6, nfp)}M"
    elif abs(num) > 10**3:
        return f"{round(num / 10**3, nfp)}K"
    else:
        return f"{round(num, nfp)}"


def pad_tensor(tensor: torch.Tensor, dim: int, to: int, value=0):
    """pad_tensor([1, 2, 3], dim=0, to=5, value=-1) -> [1, 2, 3, -1, -1]

    Pad a Tensor on **dim** to **to** long, use **value** for padding.
    If the tensor is longer than **to**, do nothing.
    """
    shape = list(tensor.shape)
    dst_size = to - shape[dim]
    if dst_size <= 0:
        return tensor
    shape[dim] = dst_size
    padder = tensor.new_full(shape, fill_value=value)
    return torch.cat([tensor, padder], dim=dim)


def lens_to_cum_len(size_list: list[int] | torch.IntTensor, dtype=None, device=None) -> torch.IntTensor:
    """[3, 5] -> [0, 3, 8]

    inverse func of torch.diff() with support of list input
    """
    if isinstance(size_list, list):
        size_list = torch.tensor([0] + size_list, dtype=dtype, device=device)
    else:
        size_list = torch.cat([size_list.new_zeros(1, dtype=dtype, device=device), size_list], 0)
    cum_sizes = torch.cumsum(size_list, dim=0, dtype=dtype)
    return cum_sizes


def list_split(data: list[T], split: int) -> list[list[T]]:
    """
    split `data` into `split` sub lists in row-major order
    E.g. list_split(torch.arange(13).tolist(), 3) ->
    [
      [ 0, 1, 2, 3],
      [ 4, 5, 6, 7],
      [ 8, 9,10,11,12],
    ]
    """
    chunk_size = len(data) // split
    extra_data = len(data) % split
    output = [data[i * chunk_size : i * chunk_size + chunk_size] for i in range(split - extra_data)]
    data = data[(split - extra_data) * chunk_size :]
    chunk_size += 1
    output.extend([data[i * chunk_size : i * chunk_size + chunk_size] for i in range(extra_data)])
    return output


def list_split_T(data: list[T], split: int) -> list[list[T]]:
    """
    Column-major version of list_split.
    E.g. list_split_T(torch.arange(13).tolist(), 3) ->
    [
      [0,3,6,9,12],
      [1,4,7,10],
      [2,5,8,11],
    ]
    """
    return [data[i::split] for i in range(split)]


def balanced_list_split(data: list[T], sizes: list[int], split: int, number_balance=False) -> list[list[T]]:
    """Greedy, order-agnostic split of `data` into `split` buckets.

    Items are processed from largest to smallest `size` and placed in the bucket
    whose score is minimal at that step.

    Args:
        data: Items to distribute. Must have the same length as ``sizes``.
        sizes: Per-item weights used for balancing.
        split: Number of buckets to create.
        number_balance: When ``True``, also penalizes buckets with more items to
            keep counts roughly even; when ``False`` balances only total weight.

    Returns:
        A list of ``split`` buckets containing the selected items; input order is
        not preserved.
    """
    import numpy as np

    new_data: list[list[T]] = [[] for i in range(split)]
    new_size = np.zeros(split, dtype=int)
    new_cout = np.zeros(split, dtype=int)
    total_sizes = sum(sizes) if number_balance else 0
    for i in np.argsort(sizes)[::-1]:
        dst = (total_sizes * new_cout + new_size).argmin()
        new_data[dst].append(data[i])
        new_size[dst] += sizes[i]
        new_cout[dst] += 1
    return new_data


def balanced_list_split_keep_order(data: list[T], sizes: list[int], k: int) -> list[list[T]]:
    """Split data: list[Any] with size: list[int], minimize max(chunked_size)."""
    assert len(data) == len(sizes)
    n = len(sizes)
    assert k and n, "Both sizes and k cannot be empty!"
    if k >= n:  # more buckets than items → some empties
        return [[x] for x in sizes] + [[] for _ in range(k - n)]

    low, high = max(sizes), sum(sizes)  # 1 elms at least, all elms at most

    def need_buckets(cumsum_limit):
        buckets, cur = 1, 0
        for x in sizes:
            if cur + x > cumsum_limit:
                buckets += 1
                cur = 0
            cur += x
        return buckets

    while low < high:
        mid = (low + high) // 2
        if need_buckets(mid) <= k:
            high = mid
        else:
            low = mid + 1
    max_allowed = low

    # Reconstruct chunks: keep as many items as possible in each bucket
    res, cur, cur_size, remaining = [], [], [], k - 1
    for i, x in enumerate(sizes):
        items_left = n - i
        if cur and (sum(cur_size) + x > max_allowed or items_left == remaining):
            res.append(cur)
            cur, cur_size = [], []
            remaining -= 1
        cur.append(data[i])
        cur_size.append(x)
    res.append(cur)
    return res


def list_chunk(data: list[T], chunk_size: int) -> list[list[T]]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def sync_point(string: str):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    logger.info(f"Sync point [ {string} ]")


@contextlib.contextmanager
def fork_rng_state(seed=2718):
    """Fork the cuda rng state, perform operations, and exit with
    the original state.
        seed = 2718 is just for fun and any POSITIVE value will work.
    """
    # Store current rng state.
    orig_cuda_rng_state = torch.cuda.get_rng_state()
    # Set rng state to the desired one
    torch.cuda.manual_seed(seed)
    _ = torch.cuda.get_rng_state()  # is necessary?
    # Do the stuff we wanted to do.
    try:
        yield
    finally:
        # backload rng state for later use.
        torch.cuda.set_rng_state(orig_cuda_rng_state)


def recur_to(sample, to: str | torch.device = "cpu"):
    """Recursively move all tensors in the python object
    from host/device to device/host
    """
    if isinstance(sample, dict):
        return {k: recur_to(v, to) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [recur_to(v, to) for v in sample]
    elif isinstance(sample, tuple):
        return tuple(recur_to(v, to) for v in sample)
    elif isinstance(sample, torch.Tensor):
        return sample.to(to)
    elif isinstance(sample, DataClass):
        sample.__dict__ = recur_to(sample.__dict__, to)
        return sample
    else:
        return sample


def check_and_link(src: str, dst: str):
    """create a symbol link from SRC to DST

    Args:
        src (str): SrcPath
        dst (str): DstPath

    Raises:
        e: raise Exception if link fail.
    """
    try:
        os.symlink(src.rstrip("/"), dst.rstrip("/"))
    except Exception as e:
        if os.readlink(dst) != src:
            raise e


def try_save(*args, **kwargs):
    try:
        return torch.save(*args, **kwargs)
    except Exception as e:
        logger.error(e)


def get_object_from_file(file: str, name: str = "Exp") -> object:
    """
    get object by file.
    Args:
        file (str): file path.
        name (str): object name.
    """
    import importlib

    module_name_without_ext = os.path.splitext(os.path.basename(file))[0]
    directory_path = os.path.dirname(file)
    module_path_parts = directory_path.replace(os.sep, ".")
    if module_path_parts:
        module_name = f"{module_path_parts}.{module_name_without_ext}"
    else:
        module_name = module_name_without_ext
    current_exp = importlib.import_module(module_name)
    obj = getattr(current_exp, name)
    return obj


def hack_pickle_load():
    import pickle

    pickle.Unpickler = pickle._Unpickler
    raw_func = pickle.Unpickler.find_class

    class LoadFailure:
        def __init__(self, *args, **kwargs):
            pass

    def find_class(self, module, name):
        try:
            return raw_func(self, module, name)
        except:
            print(f"Cannot Load {module}.{name}, but we hacked to pass.")
            return LoadFailure

    pickle._Unpickler.find_class = find_class


def deprecated(message: str = "This function is deprecated."):
    """
    Decorator to mark a function or method as deprecated.

    Args:
        message: The deprecation message to display when the function is called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if getattr(func, "_deprecated_warned", True):
                logger.warning(message)
                func._deprecated_warned = True
            return func(*args, **kwargs)

        return wrapper

    return decorator


def run_async(coro):
    """Just like asyncio.run, get rid of 'Event Loop Closed'"""
    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return asyncio.get_event_loop().run_until_complete(coro)


def retry_on(
    exceptions: tuple[type[BaseException], ...] | type[BaseException],
    for_times: int = 3,
    delay: float = 0.1,
    max_delay: float | None = None,
    backoff: float = 1.0,
    jitter: float = 0.0,
):
    """Retry a function (sync or async) with exponential backoff and optional jitter.

    Args:
        exceptions: Exception type(s) that trigger a retry.
        for_times: Total number of attempts. Must be >= 1.
        delay: Initial delay (seconds) before the first retry.
        max_delay:Upper bound for any individual sleep interval.
        backoff: Multiplier applied to the delay after each failed attempt (classic exponential backoff).
        jitter: Random jitter factor. The actual sleep becomes ``delay * backoff**n + U(0, delay * backoff**n * jitter)``.

    Usage:
    ```
    @retry_on(Exception, for_times=3)
    def my_func(data):
        raise Exception('HAHA')
    ```

    ```
    @retry_on((TimeoutError,), for_times=2, delay=1.0, max_delay=30.0, backoff=2.0, jitter=0.1)
    async def fetch():
        ...
    ```
    """
    assert for_times > 0, f"Expected for_times > 0, got {for_times}"
    if max_delay is None:
        max_delay = 1e9  # sufficient large to keep backward compatibility

    import asyncio
    import random
    import time
    from functools import wraps

    def _compute_sleep(attempt_index: int) -> tuple[float, float, float]:
        base_delay = min(max_delay, delay * (backoff**attempt_index))
        jitter_delay = random.uniform(0, base_delay * jitter) if jitter > 0 else 0.0
        sleep_for = base_delay + jitter_delay
        return sleep_for, base_delay, jitter_delay

    def wrapper(func):
        func_name = getattr(func, "__qualname__", func.__name__)

        @wraps(func)
        def wrapped(*args, **kwargs):
            last_err = None
            for try_time in range(1, for_times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    if try_time >= for_times:
                        logger.error(f"{func_name} failed after {for_times} attempts: {e}")
                        break
                    sleep_for, base_delay, jitter_delay = _compute_sleep(try_time - 1)
                    logger.warning(
                        f"{func_name} attempt {try_time}/{for_times} raised {type(e).__name__}: {e}. "
                        f"Retrying in {sleep_for:.2f}s (base={base_delay:.2f}s, jitter={jitter_delay:.2f}s)."
                    )
                    time.sleep(sleep_for)
            raise last_err

        @wraps(func)
        async def awrapped(*args, **kwargs):
            last_err = None
            for try_time in range(1, for_times + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    if try_time >= for_times:
                        logger.error(f"{func_name} failed after {for_times} attempts: {e}")
                        break
                    sleep_for, base_delay, jitter_delay = _compute_sleep(try_time - 1)
                    logger.warning(
                        f"{func_name} attempt {try_time}/{for_times} raised {type(e).__name__}: {e}. "
                        f"Retrying in {sleep_for:.2f}s (base={base_delay:.2f}s, jitter={jitter_delay:.2f}s)."
                    )
                    await asyncio.sleep(sleep_for)
            raise last_err

        if asyncio.iscoroutinefunction(func):
            return awrapped
        else:
            return wrapped

    return wrapper


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@functools.cache
def require_package(package: str):
    import pkg_resources
    from packaging import version

    if ">=" in package:
        package, required_version = package.split(">=")
        package, required_version = package.strip(), required_version.strip()
    elif ">" in package:
        raise ValueError("Use '>=' instead of '>' in requirement!")
    else:
        required_version = "0"

    try:
        current_version = pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        current_version = None

    if current_version is None:  # not exists
        raise RuntimeError(f"{package} required but not installed!")
    # PEP have (1.2.3.dev2 >= 1.2.3) is False, but we didnt follow

    current_version = current_version.replace("dev", "")
    required_version = required_version.replace("dev", "")

    if version.parse(current_version) < version.parse(required_version):
        raise RuntimeError(f"{package} >= {required_version} required but got {current_version}!")
    return True

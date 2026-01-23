from typing import Callable, Optional, TypedDict

from loguru import logger


class OptimizeMeta(TypedDict):
    alternatives: dict[str, Callable]
    use_optimize: Optional[str]


OPTIMIZABLE_REGISTER: dict[str, OptimizeMeta] = {}


def optimizable(alternatives: dict[str, Callable] = {}):
    """Mark a callable as 'optimizable', optimized alternatives can be set.

    Usage:
    ```
    @optimizable([optimized_my_func])
    def my_func(data):
        return sum(data)
    ```
    optimized_my_func & my_func should take exactly same arguments and give allclose output.
    """

    from functools import wraps

    def wrapper(func: Callable):
        global OPTIMIZABLE_REGISTER

        module_name = getattr(func, "__module__", "UnknownModule")
        func_name = getattr(func, "__qualname__", func.__name__)
        reg_name = f"{module_name}.{func_name}"
        OPTIMIZABLE_REGISTER[reg_name] = OptimizeMeta(
            alternatives=alternatives,
            use_optimize=None,
        )

        @wraps(func)
        def wrapped(*args, **kwargs):
            global OPTIMIZABLE_REGISTER
            if use_optimize := OPTIMIZABLE_REGISTER[reg_name]["use_optimize"]:
                return OPTIMIZABLE_REGISTER[reg_name]["alternatives"][use_optimize](*args, **kwargs)
            return func(*args, **kwargs)  # not optimize

        return wrapped

    return wrapper

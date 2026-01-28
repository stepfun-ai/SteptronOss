from typing import Callable, Optional, TypedDict

import torch
from loguru import logger


class OptimizeMeta(TypedDict):
    alternatives: dict[str, Callable]
    use_optimize: Optional[str]


OPTIMIZABLE_REGISTER: dict[str, OptimizeMeta] = {}


def optimizable(alternatives: dict[str, Callable] = None):
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
        nonlocal alternatives

        module_name = getattr(func, "__module__", "UnknownModule")
        func_name = getattr(func, "__qualname__", func.__name__)
        reg_name = f"{module_name}.{func_name}"

        func_alters = alternatives or {}
        func_alters["torch_compile"] = torch.compile(func)
        OPTIMIZABLE_REGISTER[reg_name] = OptimizeMeta(
            alternatives=func_alters,
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


def set_optimization(default=None, **kwargs):
    from pprint import pformat

    global OPTIMIZABLE_REGISTER
    op_abbr_map = {}
    for k in OPTIMIZABLE_REGISTER:
        func_name = k.split(".")[-1]
        if func_name in op_abbr_map:  # already exists, disable abbr
            op_abbr_map[func_name] = None
        else:
            op_abbr_map[func_name] = k
        op_abbr_map[k] = k
    op_abbr_map = {k: v for k, v in op_abbr_map.items() if v is not None}

    for k, v in OPTIMIZABLE_REGISTER.items():
        if default in v["alternatives"]:
            v["use_optimize"] = default

    for k, v in kwargs.items():
        if k in op_abbr_map:
            register = OPTIMIZABLE_REGISTER[op_abbr_map[k]]
            if v in register["alternatives"]:
                register["use_optimize"] = v
            else:
                logger.error(f"Func {k} has no alter named {v}! Availables: \n{pformat(register['alternatives'])}")
                raise KeyError()
        else:
            logger.error(f"Cannot find Func {k} in optimizable register! Availables: \n{pformat(op_abbr_map)}")
            raise KeyError()
    optim_map = "\n".join(f"{k} -> {v['use_optimize']}" for k, v in OPTIMIZABLE_REGISTER.items())
    logger.info(f"Optimization set: \n{optim_map}", at=0)

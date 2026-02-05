from collections.abc import Callable

from loguru import logger
from torch import Tensor
from torch.nn import Module

from steptronoss.utils import convert_num


def default_wd_cond(name: str, param: Tensor) -> float:
    if name.endswith(".bias") or len(param.shape) == 1:
        return 0.0
    return 1.0


def default_lr_cond(name: str, param: Tensor) -> float:
    return 1.0


def advanced_get_param_groups(
    module: Module,
    scale_wd_cond: Callable[[str, Tensor], float] = default_wd_cond,
    scale_lr_cond: Callable[[str, Tensor], float] = default_lr_cond,
    **special_tags,
) -> list[dict]:
    """creates param groups based on weight decay condition (regularized vs non regularized)
    and learning rate scale condition (args.lr vs lr_mult * args.lr)
    scale_lr_cond is used during finetuning where head of the network requires a scaled
    version of the base learning rate.
    """
    groups: dict[tuple, list] = {}
    special_tag_list = list(special_tags.keys())
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        wd_scale = scale_wd_cond(name, param)

        lr_scale = scale_lr_cond(name, param)

        group_key = ["default", wd_scale, lr_scale]
        extra_tags = [None for _ in range(len(special_tags))]

        if (prefix := getattr(param, "manual_grad_bucket_prefix", None)) is not None:
            group_key[0] = prefix

        for i, tag_name in enumerate(special_tag_list):
            extra_tags[i] = special_tags[tag_name](name, param)

        group_key = tuple(group_key + [tuple(extra_tags)])

        if group_key not in groups:
            groups[group_key] = [param]
        else:
            groups[group_key].append(param)

    param_groups = []
    for group_key, params in groups.items():
        param_groups.append(
            {"params": params, "wd_mult": group_key[1], "lr_mult": group_key[2]}
            | dict(zip(special_tag_list, group_key[3]))
        )

    del groups
    for idx, group in enumerate(param_groups):
        extra = "; ".join([f"{k}: {v}" for k, v in group.items() if k != "params"])
        logger.info(
            f"Optim group {idx} -> # params: {convert_num(sum([p.nelement() for p in group['params']]))}; {extra}"
        )
    return param_groups

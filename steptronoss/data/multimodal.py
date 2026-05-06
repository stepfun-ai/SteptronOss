from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch

from steptronoss.model.common.parallel_embedding import ImageForInsert

IMAGE_ITEM_TYPE = 0
"""Image item type used by multimodal data transforms."""

PATCH_ITEM_TYPE = 1
"""Patch item type used by multimodal data transforms."""

RopeArgsFn = Callable[[Sequence[torch.Tensor]], tuple[torch.Tensor, int]]


def _stack_images(
    images: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype | None,
    to_cuda: bool,
) -> torch.Tensor:
    stacked = torch.stack(list(images), dim=0)
    if dtype is not None:
        stacked = stacked.to(dtype=dtype)
    if to_cuda:
        stacked = stacked.to("cuda", non_blocking=True)
    return stacked


def build_image_for_insert(
    images_and_types: Iterable[tuple[torch.Tensor, int]],
    *,
    patch_start_id: int,
    image_start_id: int,
    limit_images: int | None = None,
    limit_patches: int | None = None,
    rope_args_fn: RopeArgsFn | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    to_cuda: bool = True,
) -> list[ImageForInsert]:
    """Pack multimodal data-transform output into language-model insert payloads."""

    images: list[torch.Tensor] = []
    patches: list[torch.Tensor] = []
    for image_tensor, image_type in images_and_types:
        if image_type == IMAGE_ITEM_TYPE:
            images.append(image_tensor)
        elif image_type == PATCH_ITEM_TYPE:
            patches.append(image_tensor)
        else:
            raise ValueError(f"Unsupported multimodal image type: {image_type}")

    if limit_images is not None:
        images = images[: int(limit_images)]
    if limit_patches is not None:
        patches = patches[: int(limit_patches)]

    result: list[ImageForInsert] = []
    for start_token, tensors in ((patch_start_id, patches), (image_start_id, images)):
        if not tensors:
            continue
        extra = {}
        if rope_args_fn is not None:
            rope_cu_seqlens, rope_max_seq_len = rope_args_fn(tensors)
            extra = {
                "rope_cu_seqlens": rope_cu_seqlens,
                "rope_max_seq_len": int(rope_max_seq_len),
            }
        result.append(
            ImageForInsert(
                insert_start_token=start_token,
                images=_stack_images(tensors, dtype=dtype, to_cuda=to_cuda),
                **extra,
            )
        )
    return result


def compute_rope_args(
    images: Sequence[torch.Tensor],
    patch_size: int,
    *,
    to_cuda: bool = True,
) -> tuple[torch.Tensor, int]:
    """Compute per-image patch cu_seqlens for multimodal RoPE users."""

    if not images:
        raise ValueError("images must be non-empty when computing RoPE args")

    patch_counts = []
    for image in images:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        patch_counts.append((height // patch_size) * (width // patch_size))

    cu_seqlens = torch.zeros(len(patch_counts) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(patch_counts, dtype=torch.int32), dim=0)
    if to_cuda:
        cu_seqlens = cu_seqlens.to("cuda")
    return cu_seqlens, max(patch_counts)

from __future__ import annotations

import torch

from steptronoss.data.multimodal import (
    IMAGE_ITEM_TYPE,
    PATCH_ITEM_TYPE,
    build_image_for_insert,
    compute_rope_args,
)


def test_build_image_for_insert_groups_and_limits_cpu():
    image_a = torch.zeros(3, 28, 28)
    image_b = torch.ones(3, 28, 28)
    patch = torch.full((3, 14, 14), 2.0)

    packed = build_image_for_insert(
        [
            (image_a, IMAGE_ITEM_TYPE),
            (patch, PATCH_ITEM_TYPE),
            (image_b, IMAGE_ITEM_TYPE),
        ],
        patch_start_id=10,
        image_start_id=7,
        limit_images=1,
        limit_patches=1,
        dtype=torch.float32,
        to_cuda=False,
    )

    assert [item.insert_start_token for item in packed] == [10, 7]
    assert packed[0].images is not None
    assert packed[1].images is not None
    assert packed[0].images.shape == (1, 3, 14, 14)
    assert packed[1].images.shape == (1, 3, 28, 28)
    torch.testing.assert_close(packed[1].images[0], image_a)


def test_compute_rope_args_handles_variable_image_sizes_cpu():
    cu_seqlens, max_seq_len = compute_rope_args(
        [
            torch.zeros(3, 28, 28),
            torch.zeros(3, 14, 28),
        ],
        patch_size=14,
        to_cuda=False,
    )

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 6]
    assert max_seq_len == 4

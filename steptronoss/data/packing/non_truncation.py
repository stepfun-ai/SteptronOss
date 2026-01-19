from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass
class PackingResult:
    num_packed_samples: int
    num_droped: int
    packed_sample_ranges: list[tuple[int, int]]
    """Offset & Num of samples packed for each packed-sample, e.g. [(0, 2), (3, 2), ...]"""


def pack(sizes: list[int], max_len: int, oversize_policy: Literal["drop", "extend"]) -> PackingResult:
    total = len(sizes)
    packed_sample_ranges: list[tuple[int, int]] = []
    packed_ids, packed_size, consumed, droped = [], 0, 0, 0

    def flush() -> None:
        nonlocal packed_ids, packed_size, consumed
        if packed_ids:
            packed_sample_ranges.append((packed_ids[0], len(packed_ids)))
            packed_ids = []
            packed_size = 0

    while consumed < total:
        sample_len = sizes[consumed]

        if packed_size + sample_len <= max_len:
            packed_ids.append(consumed)
            packed_size += sample_len
            if packed_size == max_len:
                flush()
        else:  # too long with new sample
            flush()
            if sample_len <= max_len:
                consumed -= 1  # putback
            else:  # oversize
                if oversize_policy == "extend":
                    packed_ids.append(consumed)
                    packed_size = sample_len
                    flush()
                else:
                    droped += 1  # drop

        consumed += 1

    flush()
    return PackingResult(
        num_packed_samples=len(packed_sample_ranges),
        num_droped=droped,
        packed_sample_ranges=packed_sample_ranges,
    )

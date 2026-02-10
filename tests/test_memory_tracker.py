import time

import pytest
import torch

from steptronoss.utils.memory_tracker import CudaMemoryTracker

pytestmark = pytest.mark.cpu


def _setup_fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 128 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 256 * 1024**2)


def test_cmt_report_clears_tracks(monkeypatch):
    _setup_fake_cuda(monkeypatch)
    tracker = CudaMemoryTracker()

    tracker.mark("first")
    tracker.mark("second")

    assert len(tracker.tracks) == 2

    tracker.report(topk=2)

    assert tracker.tracks == []


def test_cmt_report_over_world(monkeypatch):
    _setup_fake_cuda(monkeypatch)

    tracker = CudaMemoryTracker()
    tracker.mark("peak")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def _fake_all_gather_object(obj, group=None):
        other = {
            "rank": 1,
            "mark": "other",
            "time": time.time(),
            "allocated": float(64 * 1024**2),
            "reserved": float(128 * 1024**2),
        }
        return [obj, other]

    from steptronoss.utils import dist_utils

    monkeypatch.setattr(dist_utils, "all_gather_object", _fake_all_gather_object)

    tracker.report_over_world(topk=1, topk_ranks=1)

    assert tracker.tracks == []

import pytest

from steptronoss.utils import memory_tracker
from steptronoss.utils.memory_tracker import CudaMemoryTracker

pytestmark = pytest.mark.cpu


def _setup_fake_cuda(monkeypatch):
    monkeypatch.setattr(memory_tracker.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(memory_tracker.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(memory_tracker.torch.cuda, "memory_reserved", lambda: 0)


def test_cmt_report_clears_tracks(monkeypatch):
    _setup_fake_cuda(monkeypatch)
    monkeypatch.setenv("MEM_DIAGNOSE", "1")
    monkeypatch.setattr(memory_tracker, "_get_cpu_rss_bytes", lambda: 1234.0)
    tracker = CudaMemoryTracker()

    tracker.mark("first")
    tracker.mark("second")

    assert len(tracker.tracks) == 2

    tracker.report(topk=2)

    assert tracker.tracks == []


def test_cmt_report_over_world(monkeypatch):
    _setup_fake_cuda(monkeypatch)
    monkeypatch.setenv("MEM_DIAGNOSE", "1")
    monkeypatch.setattr(memory_tracker, "_get_cpu_rss_bytes", lambda: 4321.0)

    tracker = CudaMemoryTracker()
    tracker.mark("peak")

    monkeypatch.setattr(memory_tracker.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(memory_tracker.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(memory_tracker.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(memory_tracker.torch.distributed, "get_world_size", lambda: 2)

    def _fake_all_gather_object(obj, group=None):
        other = {
            "rank": 1,
            "mark": "other",
            "time": 0.0,
            "allocated": float(64 * 1024**2),
            "reserved": float(128 * 1024**2),
            "cpu_rss": 1111.0,
        }
        return [obj, other]

    from steptronoss.utils import dist_utils

    monkeypatch.setattr(dist_utils, "all_gather_object", _fake_all_gather_object)

    tracker.report_over_world(topk=1, topk_ranks=1)

    assert tracker.tracks == []


def test_cmt_records_cpu_rss(monkeypatch):
    _setup_fake_cuda(monkeypatch)
    monkeypatch.setenv("MEM_DIAGNOSE", "1")
    monkeypatch.setattr(memory_tracker, "_get_cpu_rss_bytes", lambda: 2048.0)

    tracker = CudaMemoryTracker()
    tracker.mark("cpu")

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0]["cpu_rss"] == 2048.0

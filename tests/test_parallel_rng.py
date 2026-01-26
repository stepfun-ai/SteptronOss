import random

import numpy as np
import pytest
import torch

from steptronoss.core.parallel_state import PM

pytestmark = pytest.mark.cpu


def _assert_np_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_parallel_manager_rng_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(PM, "parallels", {"TP": object(), "PP": object()})
    monkeypatch.setattr(PM, "rank_in", lambda name: {"TP": 1, "PP": 0}[name])
    monkeypatch.setattr(PM, "size_of", lambda name: {"TP": 4, "PP": 2}[name])
    monkeypatch.setattr(PM, "_rng_seeds", {})
    monkeypatch.setattr(PM, "rng_states", {})

    name = "TEST_PM_RNG"
    seed = 1234
    PM.register_rng(name, seed, diff_across=["TP", "PP"])
    assert PM._rng_seeds[name] == seed + 1

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    with PM.use_rng(name):
        seq1 = (
            random.random(),
            float(np.random.rand()),
            torch.rand(1).item(),
        )

    state_after_first = PM.rng_states[name]

    with PM.use_rng(name):
        seq2 = (
            random.random(),
            float(np.random.rand()),
            torch.rand(1).item(),
        )

    state_after_second = PM.rng_states[name]

    assert seq1[0] != seq2[0]
    assert seq1[1] != seq2[1]
    assert seq1[2] != seq2[2]
    assert random.getstate() == py_state
    _assert_np_state_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert not torch.equal(state_after_first[0], state_after_second[0])

    PM._rng_seeds.pop(name, None)


def _sample_for_ranks(monkeypatch, tp_rank: int, pp_rank: int, dp_rank: int):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(PM, "parallels", {"TP": object(), "PP": object(), "DP": object()})
    monkeypatch.setattr(PM, "rank_in", lambda name: {"TP": tp_rank, "PP": pp_rank, "DP": dp_rank}[name])
    monkeypatch.setattr(PM, "size_of", lambda name: {"TP": 4, "PP": 2, "DP": 2}[name])
    monkeypatch.setattr(PM, "_rng_seeds", {})
    monkeypatch.setattr(PM, "rng_states", {})

    name = "TEST_PM_RNG"
    PM.register_rng(name, 777, diff_across=["TP", "PP"])

    with PM.use_rng(name):
        return (
            random.random(),
            float(np.random.rand()),
            torch.rand(1).item(),
        )


def test_parallel_manager_rng_diff_across(monkeypatch):
    base = _sample_for_ranks(monkeypatch, tp_rank=0, pp_rank=0, dp_rank=0)
    same_dp = _sample_for_ranks(monkeypatch, tp_rank=0, pp_rank=0, dp_rank=1)
    diff_tp = _sample_for_ranks(monkeypatch, tp_rank=1, pp_rank=0, dp_rank=0)
    diff_pp = _sample_for_ranks(monkeypatch, tp_rank=0, pp_rank=1, dp_rank=0)

    assert base[0] == same_dp[0]
    assert base[1] == same_dp[1]
    assert base[2] == same_dp[2]
    assert base[0] != diff_tp[0]
    assert base[1] != diff_tp[1]
    assert base[2] != diff_tp[2]
    assert base[0] != diff_pp[0]
    assert base[1] != diff_pp[1]
    assert base[2] != diff_pp[2]

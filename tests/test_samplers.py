import pytest

from steptronoss.data.samplers.base_sampler import (
    LoopedSequentialSampler,
    LoopedShuffleSampler,
    WeightedRandomSampler,
)

pytestmark = pytest.mark.cpu


def test_looped_sequential_sampler_cycles_and_state():
    sampler = LoopedSequentialSampler(size=3)
    seq = [next(sampler) for _ in range(7)]
    assert seq == [0, 1, 2, 0, 1, 2, 0]

    state = sampler.state_dict()
    sampler2 = LoopedSequentialSampler(size=3)
    sampler2.load_state_dict(state)
    assert next(sampler2) == state["data_idx"] % 3


def test_looped_shuffle_sampler_same_order_each_epoch():
    sampler = LoopedShuffleSampler(size=5, base_seed=123, same_order_for_each_epoch=True)
    seq = [next(sampler) for _ in range(10)]
    first_epoch = seq[:5]
    second_epoch = seq[5:]
    assert sorted(first_epoch) == list(range(5))
    assert first_epoch == second_epoch


def test_weighted_random_sampler_balances_and_respects_history():
    sampler = WeightedRandomSampler(size=3, base_seed=11, weights=[1, 1, 1])
    sampler.load_state_dict({"size": 3, "data_idx": 0, "counts": [5, 2, 3]})
    assert sampler.get() == 1
    sampler.update()
    state = sampler.state_dict()
    assert state["counts"][1] == 3
    assert state["data_idx"] == 1

    sampler = WeightedRandomSampler(size=3, base_seed=11, weights=[1, 1, 1])
    for _ in range(60):
        next(sampler)
    counts = sampler.state_dict()["counts"]
    assert max(counts) - min(counts) <= 1

    sampler_a = WeightedRandomSampler(size=3, base_seed=11, weights=[1, 1, 1])
    sampler_b = WeightedRandomSampler(size=3, base_seed=11, weights=[1, 1, 1])
    history = {"size": 3, "data_idx": 0, "counts": [0, 0, 1]}
    sampler_a.load_state_dict(history)
    sampler_b.load_state_dict(history)
    assert sampler_a.get() == sampler_b.get()

import importlib.util
import os

import pytest

# nextable.py itself has no torch dependency, but the package __init__.py
# imports async_accelerate which requires torch.  Load the module directly
# so these pure-Python tests can run without torch installed.
_spec = importlib.util.spec_from_file_location(
    "nextable",
    os.path.join(os.path.dirname(__file__), os.pardir, "steptronoss", "data", "nextable", "nextable.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DPMux = _mod.DPMux
LazyUpdateNextable = _mod.LazyUpdateNextable
Nextable = _mod.Nextable

pytestmark = pytest.mark.cpu


class SimpleNextable(LazyUpdateNextable):
    """A minimal LazyUpdateNextable backed by a fixed list that loops."""

    def __init__(self, items: list):
        self._items = items
        self._idx = 0

    def get(self):
        return self._items[self._idx % len(self._items)]

    def update(self):
        self._idx += 1

    def state_dict(self):
        return {"idx": self._idx}

    def load_state_dict(self, state):
        self._idx = state["idx"]

    def __len__(self):
        return len(self._items)


class TestDPMuxBasic:
    def test_each_rank_gets_correct_item(self):
        items = [10, 20, 30, 40]
        dp_size = 4
        for rank in range(dp_size):
            source = SimpleNextable(items)
            mux = DPMux(source, dp_size=dp_size, dp_rank=rank)
            assert next(mux) == items[rank]

    def test_sequential_batches_across_steps(self):
        items = list(range(8))
        dp_size = 2
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=dp_size, dp_rank=0)

        # rank 0 should get items 0, 2, 4, 6
        results = [next(mux) for _ in range(4)]
        assert results == [0, 2, 4, 6]

    def test_rank1_gets_odd_items(self):
        items = list(range(8))
        dp_size = 2
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=dp_size, dp_rank=1)

        results = [next(mux) for _ in range(4)]
        assert results == [1, 3, 5, 7]

    def test_dp_size_1_is_passthrough(self):
        items = [10, 20, 30]
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=1, dp_rank=0)

        results = [next(mux) for _ in range(3)]
        assert results == items


class TestDPMuxStateDict:
    def test_state_roundtrip(self):
        items = list(range(12))
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=3, dp_rank=1)

        # Advance a few steps
        for _ in range(2):
            next(mux)

        state = mux.state_dict()

        # Create a fresh mux and restore
        source2 = SimpleNextable(items)
        mux2 = DPMux(source2, dp_size=3, dp_rank=1)
        mux2.load_state_dict(state)

        # Both should produce the same items going forward
        seq1 = [next(mux) for _ in range(3)]
        seq2 = [next(mux2) for _ in range(3)]
        assert seq1 == seq2

    def test_state_dict_reflects_underlying_source(self):
        items = list(range(6))
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=2, dp_rank=0)

        next(mux)  # consumes 2 items from source (dp_size=2)
        state = mux.state_dict()
        assert state["idx"] == 2


class TestDPMuxFastStep:
    def test_fast_step_advances_same_as_next(self):
        items = list(range(12))

        source_slow = SimpleNextable(items)
        mux_slow = DPMux(source_slow, dp_size=3, dp_rank=0)

        source_fast = SimpleNextable(items)
        mux_fast = DPMux(source_fast, dp_size=3, dp_rank=0)

        # Advance mux_slow with __next__, mux_fast with fast_step
        next(mux_slow)
        mux_fast.fast_step()

        # Internal state should match
        assert mux_slow.state_dict() == mux_fast.state_dict()

        # Both should produce the same next item
        assert next(mux_slow) == next(mux_fast)

    def test_fast_step_multiple_times(self):
        items = list(range(20))
        dp_size = 4

        source_slow = SimpleNextable(items)
        mux_slow = DPMux(source_slow, dp_size=dp_size, dp_rank=2)

        source_fast = SimpleNextable(items)
        mux_fast = DPMux(source_fast, dp_size=dp_size, dp_rank=2)

        for _ in range(3):
            next(mux_slow)
            mux_fast.fast_step()

        assert mux_slow.state_dict() == mux_fast.state_dict()
        assert next(mux_slow) == next(mux_fast)


class TestDPMuxLen:
    def test_len_delegates_to_source(self):
        items = list(range(10))
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=4, dp_rank=0)
        assert len(mux) == len(source)


class TestDPMuxRequiresSlowFast:
    def test_rejects_non_slowfast(self):
        # An object without fast_step should fail the isinstance check
        class NotSlowFast:
            pass

        with pytest.raises(AssertionError, match="SlowFastNextable Required"):
            DPMux(NotSlowFast())


class TestDPMuxIterProtocol:
    def test_works_with_for_loop(self):
        items = list(range(6))
        source = SimpleNextable(items)
        mux = DPMux(source, dp_size=2, dp_rank=0)

        results = []
        for item in mux:
            results.append(item)
            if len(results) == 3:
                break
        assert results == [0, 2, 4]

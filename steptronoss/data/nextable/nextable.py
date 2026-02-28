"""Iterator interfaces that support state save/restore and fast/slow steps."""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

# Alias for items yielded by Nextable implementations.
DataItem = object


class Nextable(metaclass=ABCMeta):
    """Abstract iterator that can save and restore its progress."""

    def __iter__(self) -> Iterator[DataItem]:
        return self

    @abstractmethod
    def __next__(self) -> DataItem:
        """Yield the next item in the sequence."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the iterator state."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state previously produced by ``state_dict``."""
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        def has_callable(obj: Any, name: str) -> bool:
            return hasattr(obj, name) and isinstance(getattr(obj, name), Callable)

        return (
            has_callable(__subclass, "__next__")
            and has_callable(__subclass, "state_dict")
            and has_callable(__subclass, "load_state_dict")
        )


class SlowFastNextable(Nextable):
    """Iterator that separates fast state updates from slow work."""

    def fast_step(self) -> None:
        # Fast path: only mutate state, no blocking work.
        raise NotImplementedError

    def __next__(self) -> DataItem:
        # Slow path: perform work, then return the current index.
        self.fast_step()
        raise NotImplementedError


class LazyUpdateNextable(SlowFastNextable):
    def get(self) -> DataItem:
        # Return the data item without mutating internal state.
        raise NotImplementedError

    def update(self) -> None:
        # Advance internal state without doing expensive work.
        raise NotImplementedError

    def fast_step(self) -> None:
        self.update()

    def __next__(self) -> DataItem:
        data = self.get()
        self.update()
        return data


class DPMux(SlowFastNextable):
    def __init__(self, slowfast: SlowFastNextable, dp_size=1, dp_rank=0):
        assert isinstance(slowfast, SlowFastNextable), "SlowFastNextable Required!"
        self.nextable = slowfast

        self.dp_size = dp_size
        self.dp_rank = dp_rank

    def fast_step(self):
        for _i in range(self.dp_size):
            self.nextable.fast_step()

    def __next__(self):
        for i in range(self.dp_size):
            example = next(self.nextable)
            if i == self.dp_rank:
                my_batch = example
        return my_batch

    def __len__(self):
        return len(self.nextable)

    def state_dict(self):
        return self.nextable.state_dict()

    def load_state_dict(self, state):
        self.nextable.load_state_dict(state)

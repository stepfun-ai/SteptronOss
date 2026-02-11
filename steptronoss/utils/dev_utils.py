"""Utils for development, auto-analyze, regression tests."""


class TBRecord(dict):
    def __init__(self, path=""):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        self.cache = {}
        self.ea = EventAccumulator(path)
        self.refresh()

    def refresh(self):
        self.ea.Reload()
        self._keys = self.ea.Tags()["scalars"]

    def keys(self):
        return self._keys

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __getitem__(self, key):
        self.refresh()
        return self.ea.Scalars(key)

    def asciichart(self, scalar_key: str, height: int = 18) -> str:
        import asciichartpy as ac

        events = self[scalar_key]
        values = [evt.value for evt in events]

        class Repr(str):
            def __repr__(self):
                return self

        return Repr(ac.plot(values, {"height": height, "format": "{:.3e}"}))

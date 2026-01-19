from torch.utils.data import Dataset

from steptronoss.data.dataloader import MixedDataloader


class DummyDataset(Dataset):
    def __init__(self, domain_id: int, size: int):
        self.domain_id = domain_id
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return (self.domain_id, idx)


def test_mixed_dataloader_balances_by_weight():
    datasets = [DummyDataset(0, 4), DummyDataset(1, 2)]
    epochs = [1.0, 1.0]
    loader = MixedDataloader(datasets=datasets, epochs=epochs)

    counts = {0: 0, 1: 0}
    for _ in range(60):
        domain_id, _ = next(loader)
        counts[domain_id] += 1

    weights = [4.0, 2.0]
    yields = [counts[0] / weights[0], counts[1] / weights[1]]
    diff = abs(yields[0] - yields[1])
    assert diff <= 1.0 / min(weights) + 1e-6
    assert len(loader) == 6


def test_mixed_dataloader_state_roundtrip():
    datasets = [DummyDataset(0, 3), DummyDataset(1, 2)]
    epochs = [1.0, 1.0]
    loader = MixedDataloader(datasets=datasets, epochs=epochs)

    for _ in range(7):
        next(loader)

    state = loader.state_dict()
    loader2 = MixedDataloader(datasets=datasets, epochs=epochs)
    loader2.load_state_dict(state)

    seq1 = [next(loader) for _ in range(10)]
    seq2 = [next(loader2) for _ in range(10)]
    assert seq1 == seq2

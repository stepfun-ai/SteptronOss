from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _mock_distributed(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, group=None: None)
    mock_pm = MagicMock()
    mock_pm.group_of.return_value = None
    import steptronoss.utils.utils as utils_mod

    monkeypatch.setattr(utils_mod, "PM", mock_pm)


def test_normal_input():
    from steptronoss.utils.utils import get_normalizer

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std = get_normalizer(x)
    assert torch.isfinite(mean)
    assert torch.isfinite(std)
    assert torch.isclose(mean, torch.tensor(3.0))


def test_nan_input_raises():
    from steptronoss.utils.utils import get_normalizer

    x = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        get_normalizer(x)


def test_inf_input_raises():
    from steptronoss.utils.utils import get_normalizer

    x = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        get_normalizer(x)


def test_constant_input():
    from steptronoss.utils.utils import get_normalizer

    x = torch.tensor([5.0, 5.0, 5.0])
    mean, std = get_normalizer(x)
    assert torch.isclose(mean, torch.tensor(5.0))
    assert torch.isclose(std, torch.tensor(0.0))

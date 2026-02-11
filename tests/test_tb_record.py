import time
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from steptronoss.utils.dev_utils import TBRecord

pytestmark = pytest.mark.cpu


def test_tb_record_reads_scalars(tmp_path: Path):
    log_dir = tmp_path / "runs"
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("train/loss", 1.0, 0)
    writer.add_scalar("train/loss", 0.5, 1)
    writer.flush()
    writer.close()

    time.sleep(0.1)

    record = TBRecord(path=str(log_dir))
    assert "train/loss" in record

    scalars = record["train/loss"]
    assert len(scalars) >= 2
    assert scalars[0].value == pytest.approx(1.0)
    assert scalars[1].value == pytest.approx(0.5)

import os

import pytest
import torch.distributed as dist


@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
def test_torchrun_launcher_sets_world_size():
    if os.environ.get("STEPTRON_TORCHRUN") != "1":
        pytest.skip("torchrun launcher not active in parent pytest process")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size >= 2
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    assert dist.get_world_size() == world_size
    dist.barrier()
    dist.destroy_process_group()

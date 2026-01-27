import pickle

import pytest

from steptronoss.utils import comm_utils

pytestmark = pytest.mark.cpu


class _DummyRedis:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def keys(self):
        return []


def _reset_comm_utils():
    comm_utils.REDIS_STARTED = False
    comm_utils.IS_REDIS_MASTER = False
    comm_utils.REDIS_ADDR = None
    comm_utils.REDIS_PORT = None


def test_rendezvous_requires_meet_dir(monkeypatch):
    _reset_comm_utils()
    monkeypatch.delenv("STEPTRON_MEET_DIR", raising=False)
    monkeypatch.setattr(comm_utils, "get_exp_id", lambda *args, **kwargs: "test-exp")
    monkeypatch.setattr(comm_utils, "Redis", _DummyRedis)

    with pytest.raises(RuntimeError, match="STEPTRON_MEET_DIR is not set"):
        comm_utils.get_exp_redis()


def test_rendezvous_shared_fs_master_and_follower(tmp_path, monkeypatch):
    _reset_comm_utils()
    monkeypatch.setenv("STEPTRON_MEET_DIR", str(tmp_path))
    monkeypatch.delenv("CANNOT_BE_REDIS_SERVER", raising=False)
    monkeypatch.setattr(comm_utils, "safely_start_redis", lambda: 12345)
    monkeypatch.setattr(comm_utils, "Redis", _DummyRedis)
    monkeypatch.setattr(comm_utils, "get_exp_id", lambda *args, **kwargs: "test-exp")

    redis_client = comm_utils.get_exp_redis()
    assert isinstance(redis_client, _DummyRedis)
    assert comm_utils.IS_REDIS_MASTER is True
    assert comm_utils.REDIS_PORT == 12345

    exp_id = "{exp}redis-master-for-test-exp"
    rendezvous_path = tmp_path / f"redis-rendezvous-{exp_id}.pkl"
    assert rendezvous_path.exists()
    state = pickle.loads(rendezvous_path.read_bytes())
    assert state["port"] == 12345

    _reset_comm_utils()
    monkeypatch.setenv("CANNOT_BE_REDIS_SERVER", "1")

    redis_client = comm_utils.get_exp_redis()
    assert isinstance(redis_client, _DummyRedis)
    assert comm_utils.IS_REDIS_MASTER is False
    assert comm_utils.REDIS_PORT == 12345

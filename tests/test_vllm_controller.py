import io

import pytest

from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.generation.vllm.vllm_controller import VLLMController

pytestmark = pytest.mark.cpu


class _FakeProcess:
    def __init__(self, stdout=None, stderr=None):
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 12345
        self._returncode = None

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        return 0


class _FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _make_cfg():
    cfg = VLLMDeployConfig()
    cfg.model_config_path = "dummy"
    cfg.model_name_template = "dummy"
    cfg.vllm_tp = 1
    cfg.vllm_dp = 1
    return cfg


def test_run_vllm_spawns_process(monkeypatch):
    cfg = _make_cfg()
    monkeypatch.setenv("NNODES", "1")
    controller = VLLMController(cfg)

    fake_proc = _FakeProcess(stdout=io.StringIO(""), stderr=io.StringIO(""))
    captured = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0]
        captured["env"] = kwargs.get("env")
        return fake_proc

    monkeypatch.setattr("steptronoss.generation.vllm.vllm_controller.get_free_port", lambda: 50001)
    monkeypatch.setattr("steptronoss.generation.vllm.vllm_controller.subprocess.Popen", _fake_popen)

    controller.run_vllm()

    assert controller.vllm_port == 50001
    assert "--port $PORT_SERVING" in captured["cmd"]
    assert captured["env"]["PORT_SERVING"] == "50001"


def test_register_and_deregister(monkeypatch):
    cfg = _make_cfg()
    monkeypatch.setenv("NNODES", "1")
    controller = VLLMController(cfg)
    controller.vllm_port = 8001

    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse(status_code=200, text="ok")

    monkeypatch.setattr("requests.get", _fake_get)

    class _FakeRedis:
        def set(self, *_args, **_kwargs):
            return True

        def rpush(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(
        "steptronoss.generation.vllm.vllm_controller.get_exp_redis",
        lambda: _FakeRedis(),
    )
    monkeypatch.setattr(
        "steptronoss.generation.vllm.vllm_controller.block_get_redis",
        lambda _client, _key: b"127.0.0.1:9000",
    )
    monkeypatch.setattr(
        "steptronoss.generation.vllm.vllm_controller.socket.gethostbyname",
        lambda _name: "10.0.0.1",
    )

    controller.register()
    controller.deregister()

    register_calls = [c for c in calls if c[0].endswith("/register")]
    unregister_calls = [c for c in calls if c[0].endswith("/unregister")]

    assert register_calls, "missing register call"
    assert unregister_calls, "missing unregister call"
    assert register_calls[0][0] == "http://127.0.0.1:9000/register"
    assert unregister_calls[0][0] == "http://127.0.0.1:9000/unregister"
    assert register_calls[0][1]["endpoint"] == "10.0.0.1:8001"
    assert unregister_calls[0][1]["endpoint"] == "10.0.0.1:8001"

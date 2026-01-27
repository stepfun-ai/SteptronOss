import pytest
from fastapi.testclient import TestClient

from steptronoss.generation.vllm_utils import VLLMRouter, VLLMRouterConfig

pytestmark = pytest.mark.cpu


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeUpstream:
    def __init__(self, status=200, headers=None, body=b"", chunks=None):
        self.status = status
        self.headers = headers or {}
        self._body = body
        self.content = _FakeContent(chunks or [])
        self.released = False

    async def read(self):
        return self._body

    def release(self):
        self.released = True


class _FakeSession:
    def __init__(self, upstream):
        self._upstream = upstream
        self.closed = False

    async def request(self, **kwargs):
        return self._upstream


def _build_router(monkeypatch, upstream):
    cfg = VLLMRouterConfig()
    cfg.endpoints = ["127.0.0.1:8000"]
    router = VLLMRouter(cfg)

    async def _get_session():
        return _FakeSession(upstream)

    monkeypatch.setattr(router, "_get_session", _get_session)
    return router


def test_vllm_router_streaming(monkeypatch):
    upstream = _FakeUpstream(
        status=200,
        headers={"content-type": "text/event-stream"},
        chunks=[b"data: 1\n\n", b"data: 2\n\n"],
    )
    router = _build_router(monkeypatch, upstream)
    client = TestClient(router.app)

    response = client.post("/v1/completions", params={"stream": "true"}, json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.content == b"data: 1\n\ndata: 2\n\n"
    assert upstream.released is True


def test_vllm_router_non_stream(monkeypatch):
    body = b'{"ok": true}'
    upstream = _FakeUpstream(
        status=200,
        headers={"content-type": "application/json"},
        body=body,
    )
    router = _build_router(monkeypatch, upstream)
    client = TestClient(router.app)

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.content == body
    assert upstream.released is True

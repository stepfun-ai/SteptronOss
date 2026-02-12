from __future__ import annotations

import json
import threading
from collections.abc import Iterable

import aiohttp
import uvicorn
from configurize import Config
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from steptronoss.utils.comm_utils import get_exp_redis
from steptronoss.utils.general import get_free_port, get_my_ip


class VLLMRouterConfig(Config):
    router_addr_key: str = "default"

    routed_methods: dict[str, list[str]] = {"completions": ["POST"]}

    def run(self):
        router = VLLMRouter(cfg=self)
        router.serve()


class VLLMRouter:
    """vLLM request router.

    APIs:
    - GET /register?endpoint=host:port
    - GET /unregister?endpoint=host:port
    - GET /get_info  -> [{"endpoint": "host:port"}, ...]
    - /v1/{route} proxies requests to registered vLLM servers (default: /v1/completions).

    Discovery:
    On startup, the router picks an open port, then publishes its address to the
    experiment Redis via get_exp_redis() under key f"VLLM_ROUTER_ADDR_PORT_{cfg.router_addr_key}"
    (format: "{ip}:{port}"). Clients should use block_get_redis(exp_redis,
    f"VLLM_ROUTER_ADDR_PORT_{cfg.router_addr_key}") to fetch the router address.
    """

    def __init__(self, cfg: VLLMRouterConfig):
        self.cfg = cfg
        self._endpoints: list[str] = list()
        self._lock = threading.Lock()
        self._rr_index = 0
        self._session: aiohttp.ClientSession | None = None

        app = FastAPI()
        app.add_api_route("/register", self._register_api, methods=["GET"])
        app.add_api_route("/unregister", self._unregister_api, methods=["GET"])
        app.add_api_route("/get_info", self._get_info_api, methods=["GET"])

        for route_name, methods in self.cfg.routed_methods.items():
            app.add_api_route(
                f"/v1/{route_name}",
                self._make_proxy(route_name),
                methods=methods,
            )

        self.app = app

    def register(self, endpoint: str) -> bool:
        with self._lock:
            if endpoint in self._endpoints:
                return False
            self._endpoints.append(endpoint)
            return True

    def unregister(self, endpoint: str) -> bool:
        with self._lock:
            if endpoint not in self._endpoints:
                return False
            self._endpoints.remove(endpoint)
            return True

    def get_info(self) -> list[dict]:
        with self._lock:
            return [{"endpoint": endpoint} for endpoint in self._endpoints]

    def serve(self):
        my_ip = get_my_ip()
        my_port = get_free_port()

        exp_redis = get_exp_redis()
        # if exist_info is not None and exist_info.decode() != my_info:
        #     raise RuntimeError("It seems an VLLM router has already running!")
        exp_redis.set(f"VLLM_ROUTER_ADDR_PORT_{self.cfg.router_addr_key}", f"{my_ip}:{my_port}")

        uvicorn.run(self.app, host="0.0.0.0", port=my_port, access_log=False)

    def _register_api(self, endpoint: str) -> bool:
        return self.register(endpoint)

    def _unregister_api(self, endpoint: str) -> bool:
        return self.unregister(endpoint)

    def _get_info_api(self) -> list[dict]:
        return self.get_info()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=None)
        return self._session

    def _make_proxy(self, route_name: str):
        async def _proxy(request: Request):
            endpoint = self._select_endpoint()
            url = f"{self._normalize_endpoint(endpoint)}/v1/{route_name}"
            params = dict(request.query_params)
            headers = self._filter_request_headers(request.headers)
            body = await request.body()

            try:
                session = await self._get_session()
                upstream = await session.request(
                    method=request.method,
                    url=url,
                    params=params,
                    data=body,
                    headers=headers,
                )
            except aiohttp.ClientError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            response_headers = self._filter_response_headers(upstream.headers)

            is_stream = False
            if params.get("stream") in {"1", "true", "True"}:
                is_stream = True
            else:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type.lower() and body:
                    try:
                        payload = json.loads(body)
                        if payload.get("stream") is True:
                            is_stream = True
                    except json.JSONDecodeError:
                        pass

            if "text/event-stream" in upstream.headers.get("content-type", "").lower():
                is_stream = True

            if is_stream:

                async def streamer():
                    try:
                        async for chunk in upstream.content.iter_any():
                            if chunk:
                                yield chunk
                    finally:
                        upstream.release()

                return StreamingResponse(
                    streamer(),
                    status_code=upstream.status,
                    headers=response_headers,
                    media_type=upstream.headers.get("content-type"),
                )

            try:
                content = await upstream.read()
            finally:
                upstream.release()

            return Response(
                content=content,
                status_code=upstream.status,
                headers=response_headers,
            )

        return _proxy

    def _select_endpoint(self) -> str:
        with self._lock:
            if not self._endpoints:
                raise HTTPException(status_code=503, detail="No endpoints registered")
            endpoint = self._endpoints[self._rr_index % len(self._endpoints)]
            self._rr_index = (self._rr_index + 1) % len(self._endpoints)
            return endpoint

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"http://{endpoint}"

    @staticmethod
    def _filter_request_headers(
        headers: Iterable[tuple[str, str]] | dict,
    ) -> dict[str, str]:
        if hasattr(headers, "items"):
            items = headers.items()
        else:
            items = headers
        filtered = {}
        for key, value in items:
            lower_key = key.lower()
            if lower_key in {"host", "content-length"}:
                continue
            filtered[key] = value
        return filtered

    @staticmethod
    def _filter_response_headers(
        headers: Iterable[tuple[str, str]] | dict,
    ) -> dict[str, str]:
        if hasattr(headers, "items"):
            items = headers.items()
        else:
            items = headers
        filtered = {}
        for key, value in items:
            lower_key = key.lower()
            if lower_key in {"content-length", "transfer-encoding", "connection"}:
                continue
            filtered[key] = value
        return filtered

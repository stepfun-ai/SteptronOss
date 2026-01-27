from __future__ import annotations

import json
import threading
from typing import Iterable

import aiohttp
from configurize import Config
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse


class VLLMRouterConfig(Config):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float | None = 120.0
    endpoints: list[str] = []


class VLLMRouter:
    ROUTED_METHODS = {"completions": ["POST"]}

    def __init__(self, cfg: VLLMRouterConfig):
        self.cfg = cfg
        self._endpoints: list[str] = list(cfg.endpoints)
        self._lock = threading.Lock()
        self._rr_index = 0
        self._session: aiohttp.ClientSession | None = None

        app = FastAPI()
        app.add_api_route("/register", self._register_api, methods=["GET"])
        app.add_api_route("/unregister", self._unregister_api, methods=["GET"])
        app.add_api_route("/get_info", self._get_info_api, methods=["GET"])

        for route_name, methods in self.ROUTED_METHODS.items():
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
        import uvicorn

        uvicorn.run(self.app, host=self.cfg.host, port=self.cfg.port)

    def _register_api(self, endpoint: str) -> bool:
        return self.register(endpoint)

    def _unregister_api(self, endpoint: str) -> bool:
        return self.unregister(endpoint)

    def _get_info_api(self) -> list[dict]:
        return self.get_info()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = None
            if self.cfg.request_timeout is not None:
                timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
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
    def _filter_request_headers(headers: Iterable[tuple[str, str]] | dict) -> dict[str, str]:
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
    def _filter_response_headers(headers: Iterable[tuple[str, str]] | dict) -> dict[str, str]:
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

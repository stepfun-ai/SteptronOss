import asyncio
import time
from functools import cached_property
from typing import Any

import aiohttp
import requests
from aiohttp.client_exceptions import ClientConnectionError
from loguru import logger

from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.utils.comm_utils import block_get_redis, get_exp_redis
from steptronoss.utils.general import retry_on, run_async

ROUTER_ADDR_KEY = "VLLM_ROUTER_ADDR_PORT"


class VLLMClient:
    def __init__(self, cfg: VLLMDeployConfig):
        self.cfg = cfg

        self.session = requests.Session()
        self.session.proxies = {"http": ""}

        self.asession = None

    async def aclose(self) -> None:
        """Close underlying HTTP sessions to avoid aiohttp resource warnings."""
        if self.asession is not None:
            try:
                await self.asession.close()
            finally:
                self.asession = None
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    def close(self) -> None:
        """Sync close wrapper for callers in non-async contexts."""
        if self.asession is not None:
            try:
                run_async(self.asession.close())
            finally:
                self.asession = None
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    @retry_on((ConnectionError, IOError), for_times=3)
    def post(self, url: str, **kwargs):
        response = self.session.post(url, **kwargs)
        return response.json()

    @retry_on((ConnectionError, IOError), for_times=3)
    def get(self, url: str, params: dict | None = None):
        response = self.session.get(url, params=params or {})
        return response.json()

    @retry_on((ConnectionError, IOError, ClientConnectionError), for_times=3)
    async def apost(self, url: str, decode_json=True, **kwargs):
        if self.asession is None:
            self.asession = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600, connect=60, sock_read=600))
        async with self.asession.post(url, **kwargs) as response:
            if decode_json:
                return await response.json()
            else:
                return await response.read()

    @retry_on((ConnectionError, IOError, ClientConnectionError), for_times=3)
    async def aget(self, url: str, params: dict | None = None):
        if self.asession is None:
            self.asession = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600, connect=60, sock_read=600))
        async with self.asession.get(url, params=params or {}) as response:
            return await response.json()

    @cached_property
    def _router_addr(self) -> str:
        exp_redis = get_exp_redis()
        router_addr_key = f"{ROUTER_ADDR_KEY}_{self.cfg.router_addr_key}"

        router_addr = block_get_redis(exp_redis, router_addr_key).decode()
        return f"http://{router_addr}"

    def check_server_alive(self) -> bool:
        try:
            completion = self.post(
                f"{self._router_addr}/v1/completions",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.cfg.model_name,
                    "prompt": "你还活着吗？",
                    "max_tokens": 3,
                },
            )
            if completion["choices"][0]:
                return True
        except Exception:
            pass
        return False

    def wait_for_server(self, timeout: int = 3600) -> list[str]:
        """Wait till all vLLM servers ready or timeout; return endpoints list."""
        exp_redis = get_exp_redis()
        expected_replica = block_get_redis(
            exp_redis,
            f"VLLM_NUM_WORKERS_OF_{self.cfg.model_name}",
            timeout=timeout,
        )
        expected_replica = int(expected_replica)

        start_time = time.time()
        while time.time() - start_time < timeout:
            endpoints = self.get(f"{self._router_addr}/get_info")
            endpoint_addrs = [ep["endpoint"] for ep in endpoints]
            logger.info(f"[wait_for_server] {len(endpoint_addrs)}/{expected_replica} endpoints ready: {endpoint_addrs}")
            if self.check_server_alive() and len(endpoints) >= expected_replica:
                assert len(endpoint_addrs) == len(set(endpoint_addrs)), "Duplicate endpoints found in endpoint_addrs"
                return endpoint_addrs
            time.sleep(max(2, timeout / 1200))
        raise TimeoutError("VLLM wait server timeout!")

    def completion(self, prompt_or_request: str | list[int] | dict[str, Any], max_tokens=None) -> dict[str, Any]:
        """发送聊天完成请求进行测试"""
        if hasattr(prompt_or_request, "model_dump"):
            payload = prompt_or_request.model_dump()
        elif isinstance(prompt_or_request, dict):
            payload = prompt_or_request.copy()
        else:
            payload = {"prompt": prompt_or_request, "return_token_ids": True}
        payload["model"] = self.cfg.model_name
        if max_tokens:
            payload["max_tokens"] = max_tokens
        result = self.post(
            url=f"{self._router_addr}/v1/completions",
            json=payload,
        )
        return result

    async def _get_model_endpoints(self) -> list[str]:
        """获取所有模型端点的IP和端口"""
        result = await self.aget(f"{self._router_addr}/get_info")
        return [ep["endpoint"] for ep in result]

    async def _update_and_reload_for_endpoint(self, endpoint: str, new_path: str | None = None) -> dict[str, Any]:
        """为特定端点执行更新配置和重新加载权重的任务组合"""
        overrides = {
            "load_config": {"load_format": "auto"},
        }
        if new_path:
            overrides.update({"model_config": {"model": new_path}})

        await self.apost(
            f"http://{endpoint}/collective_rpc",
            json={
                # "model": self.cfg.model_name,
                "method": "update_config",
                "kwargs": {"overrides": overrides},
            },
        )

        await self.apost(
            f"http://{endpoint}/collective_rpc",
            json={
                # "model": self.cfg.model_name,
                "method": "reload_weights",
            },
        )
        await self.apost(f"http://{endpoint}/reset_prefix_cache", decode_json=False)
        return True

    async def _update_all_workers(self, endpoints: list[str], new_path: str | None = None):
        tasks = [
            self._update_and_reload_for_endpoint(
                endpoint,
                new_path=new_path,
            )
            for endpoint in endpoints
        ]
        results = await asyncio.gather(*tasks)
        return all(results)

    def reload_weights(self, new_path: str | None = None):
        endpoints = self.wait_for_server()

        try:
            return run_async(self._update_all_workers(endpoints, new_path))

        finally:
            # Deleting without close() will leak connectors and trigger
            # "Unclosed client session/connector" warnings at shutdown.
            if self.asession is not None:
                try:
                    run_async(self.asession.close())
                finally:
                    self.asession = None

import atexit
import os
import signal
import socket
import subprocess
import sys
import time

import requests
from loguru import logger

from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.utils.comm_utils import block_get_redis, get_exp_redis
from steptronoss.utils.general import get_free_port

HEALTHY_CHECK_INTERVAL = 10


class VLLMController:
    def __init__(self, cfg: VLLMDeployConfig):
        self.cfg = cfg
        self.process: subprocess.Popen | None = None
        self.vllm_port: int | None = None
        self.endpoint: str | None = None
        self.router_url: str = None
        self._shutdown_called = False

        # get total replica, so we know how many replica we are expecting
        self.nnodes = int(os.environ["NNODES"])

    def start(self):
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        atexit.register(self.shutdown)
        self.run_vllm()
        self.wait_for_health()
        self.register()
        self.process.wait()

    def run_vllm(self):
        """run vllm in subprocess, just like run 'vllm serve xxx'"""
        cmd, env = self.cfg.get_entrypoint_command_and_envs()

        self.vllm_port = get_free_port()
        env["PORT_SERVING"] = str(self.vllm_port)

        logger.info(f"Launching vLLM with command: {cmd}")
        self.process = subprocess.Popen(  # noqa: S602
            cmd,
            shell=True,
            env=os.environ | env | {"VLLM_SERVER_DEV_MODE": "1"},  # must in DEV MODE
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin,
            preexec_fn=os.setsid,
        )

    def wait_for_health(self):
        """Wait for health check to pass, then register service, and finally periodically execute hc_pass"""
        health_url = f"http://localhost:{self.vllm_port}/health"

        # First phase: wait for health check to pass
        logger.info("Waiting for health check to pass...")
        while True:
            time.sleep(HEALTHY_CHECK_INTERVAL)
            if self.process.poll() is not None:
                raise RuntimeError("VLLM process dead.")
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Health check passed, status code: 200")
                    break
                    # logger.info(f"Health check failed, status code: {response.status_code}, continue waiting...")
            except requests.exceptions.RequestException as e:
                logger.info(f"Health check request failed: {e}, continue waiting...")

    def register(self):
        """Register service to exp-specific router. Wait for router ready here."""
        exp_redis = get_exp_redis()
        logger.info("Register to router: waiting for router ready...")
        vllm_router_endpoint = block_get_redis(exp_redis, f"VLLM_ROUTER_ADDR_PORT_{self.cfg.router_addr_key}").decode()
        # should be like "127.127.127.127:4321"

        if self.vllm_port is None:
            raise RuntimeError("vLLM port is not set; cannot register endpoint.")

        self.router_url = f"http://{vllm_router_endpoint}"

        my_ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        self.endpoint = f"{my_ip}:{self.vllm_port}"
        model_name = self.cfg.model_name
        exp_redis.set(f"VLLM_NUM_WORKERS_OF_{model_name}", self.nnodes)
        exp_redis.rpush(f"VLLM_WORKERS_OF_{model_name}", self.endpoint)
        try:
            response = requests.get(
                f"{self.router_url}/register",
                params={"endpoint": self.endpoint},
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(f"Registered endpoint {self.endpoint} to router {self.router_url}.")
            else:
                logger.warning(f"Router register failed: {response.status_code} {response.text}")
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Router register request failed: {exc}")

    def deregister(self):
        # self._wait_for_active_requests()

        try:
            response = requests.get(
                f"{self.router_url}/unregister",
                params={"endpoint": self.endpoint},
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(f"Unregistered endpoint {self.endpoint} from router.")
            else:
                logger.warning(f"Router unregister failed: {response.status_code} {response.text}")
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Router unregister request failed: {exc}")

    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.deregister()
        self._terminate_vllm()

    def _terminate_vllm(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not exit in time; force killing.")
            os.killpg(self.process.pid, signal.SIGKILL)

    def _handle_sigterm(self, signum, frame):
        logger.info("Received SIGTERM; shutting down vLLM controller.")
        self.shutdown()
        raise SystemExit(0)

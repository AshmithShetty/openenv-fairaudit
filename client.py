import os
import json
import asyncio
import subprocess
import time
import urllib.request
from typing import Optional, Tuple

import websockets

from server.models import BiasAuditObservation, BiasAuditAction


class StepResult:
    def __init__(self, observation: BiasAuditObservation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class ResetResult:
    def __init__(self, observation: BiasAuditObservation):
        self.observation = observation


class FairAuditEnv:
    def __init__(self, base_url: str = None):
        if base_url:
            ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
            self.uri = ws_url.rstrip("/") + "/ws/evaluator"
        else:
            host = os.environ.get("OPENENV_HOST", "localhost")
            port = os.environ.get("OPENENV_PORT", "7860")
            self.uri = f"ws://{host}:{port}/ws/evaluator"

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._container_name: Optional[str] = None

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "FairAuditEnv":
        container_name = f"fairaudit-{int(time.time())}"

        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "-d",
                "--name", container_name,
                "-p", f"{port}:{port}",
                image_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()

        health_url = f"http://localhost:{port}/health"
        for _ in range(30):
            try:
                urllib.request.urlopen(health_url, timeout=2)
                break
            except Exception:
                await asyncio.sleep(1)

        instance = cls(base_url=f"http://localhost:{port}")
        instance._container_name = container_name
        return instance

    async def __aenter__(self) -> "FairAuditEnv":
        for attempt in range(15):
            try:
                self.ws = await websockets.connect(self.uri)
                return self
            except Exception:
                await asyncio.sleep(1)
        raise ConnectionError(f"Failed to connect to environment at {self.uri}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def reset(self, task_name: str = "dataset-scan") -> ResetResult:
        if not self.ws:
            await self.__aenter__()

        payload = {"command": "reset", "task_name": task_name}
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())

        obs_data = response.get("data", response)
        obs = BiasAuditObservation(**obs_data)
        return ResetResult(observation=obs)

    async def step(self, action: BiasAuditAction) -> StepResult:
        if not self.ws:
            raise RuntimeError("Environment not connected. Call reset() first.")

        payload = {"command": "step", "action": action.model_dump()}
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())

        obs_data = response.get("observation", {})
        obs = BiasAuditObservation(**obs_data)
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", True))
        info = response.get("info", {})

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    async def state(self) -> dict:
        if not self.ws:
            return {"status": "uninitialized"}

        payload = {"command": "state"}
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())
        return response.get("data", {})

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        if self._container_name:
            subprocess.run(
                ["docker", "stop", self._container_name],
                capture_output=True,
                timeout=15,
            )
            self._container_name = None
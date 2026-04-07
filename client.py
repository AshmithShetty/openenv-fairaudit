import os
import json
import asyncio
import websockets
from typing import Optional, Dict, Any

from openenv.client import BaseEnvironment
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

class FairAuditEnv(BaseEnvironment):
    def __init__(self, base_url: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if base_url:
            # Replaces http:// with ws:// for websocket connections
            self.uri = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/evaluator"
        else:
            self.port = os.environ.get("OPENENV_PORT", 7860)
            self.host = os.environ.get("OPENENV_HOST", "localhost")
            self.uri = f"ws://{self.host}:{self.port}/ws/evaluator"
            
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def __aenter__(self):
        retries = 15
        for _ in range(retries):
            try:
                self.ws = await websockets.connect(self.uri)
                return self
            except Exception:
                await asyncio.sleep(1)
        raise ConnectionError(f"Failed to connect to environment server at {self.uri}")

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
            raise RuntimeError("Environment connection not established. Call reset() first.")
            
        payload = {"command": "step", "action": action.model_dump()}
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())
        
        obs = BiasAuditObservation(**response.get("observation", {}))
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
            await self.ws.close()
            self.ws = None
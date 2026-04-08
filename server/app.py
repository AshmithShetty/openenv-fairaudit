import json
from typing import Dict, Tuple

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from server.environment import FairAuditEnvironment
from server.models import BiasAuditAction, BiasAuditObservation

app = FastAPI(title="FairAudit OpenEnv", version="1.0.0")

sessions: Dict[str, FairAuditEnvironment] = {}


def _get_or_create(session_id: str) -> FairAuditEnvironment:
    if session_id not in sessions:
        sessions[session_id] = FairAuditEnvironment()
    return sessions[session_id]


@app.get("/")
def root():
    return {"message": "FairAudit environment is online."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def http_reset(body: dict = None):
    body = body or {}
    task_name = body.get("task_name", "dataset-scan")
    env = FairAuditEnvironment()
    sessions["http"] = env
    obs = env.reset(task_name)
    return {"type": "observation", "data": obs.model_dump()}


@app.post("/step")
def http_step(body: dict):
    env = sessions.get("http")
    if env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first."})
    action = BiasAuditAction(**body.get("action", {}))
    obs, reward, done, info = env.step(action)
    return {
        "type": "step_result",
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def http_state():
    env = sessions.get("http")
    if env is None:
        return {"status": "uninitialized"}
    return {"type": "state", "data": env.state_dump()}


class _ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)


manager = _ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    env = _get_or_create(client_id)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            command = payload.get("command")

            if command == "reset":
                task_name = payload.get("task_name", "dataset-scan")
                sessions[client_id] = FairAuditEnvironment()
                env = sessions[client_id]
                obs = env.reset(task_name)
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})

            elif command == "step":
                action_data = payload.get("action", {})
                action = BiasAuditAction(**action_data)
                obs, reward, done, info = env.step(action)
                await websocket.send_json({
                    "type": "step_result",
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                })

            elif command == "state":
                await websocket.send_json({"type": "state", "data": env.state_dump()})

            else:
                await websocket.send_json({"error": f"Unknown command: {command}"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        sessions.pop(client_id, None)
    except Exception as exc:
        manager.disconnect(client_id)
        sessions.pop(client_id, None)
        print(f"[DEBUG] WebSocket error for {client_id}: {exc}", flush=True)


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
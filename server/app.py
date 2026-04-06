from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import json
from client import FairAuditEnv
from server.models import BiasAuditAction

app = FastAPI()
env_manager = FairAuditEnv()

@app.get("/")
def read_root():
    return {"message": "FairAudit Environment is online. API is ready."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            command = payload.get("command")
            
            if command == "reset":
                task_name = payload.get("task_name", "dataset-scan")
                obs = await env_manager.reset(task_name=task_name, session_id=client_id)
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})
                
            elif command == "step":
                action_data = payload.get("action")
                action = BiasAuditAction(**action_data)
                obs, reward, done, info = await env_manager.step(action, session_id=client_id)
                await websocket.send_json({
                    "type": "step_result",
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info
                })
                
            elif command == "state":
                state_data = await env_manager.state(session_id=client_id)
                await websocket.send_json({"type": "state", "data": state_data})
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if client_id in env_manager.active_sessions:
            del env_manager.active_sessions[client_id]
        print(f"Client #{client_id} disconnected.")

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
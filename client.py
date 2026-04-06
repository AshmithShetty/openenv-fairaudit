from typing import Any, Tuple, Dict
from server.models import BiasAuditObservation, BiasAuditAction
from server.environment import FairAuditEnvironment

class FairAuditEnv:
    def __init__(self):
        # Track independent environments by WebSocket connection ID
        self.active_sessions: Dict[str, FairAuditEnvironment] = {}

    async def reset(self, task_name: str = "dataset-scan", session_id: str = "default") -> BiasAuditObservation:
        self.active_sessions[session_id] = FairAuditEnvironment()
        return self.active_sessions[session_id].reset(task_name)

    async def step(self, action: BiasAuditAction, session_id: str = "default") -> Tuple[BiasAuditObservation, float, bool, dict]:
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not initialized. Call reset first.")
        return self.active_sessions[session_id].step(action)

    async def state(self, session_id: str = "default") -> dict:
        if session_id not in self.active_sessions:
            return {"status": "uninitialized"}
        return self.active_sessions[session_id].state_dump()
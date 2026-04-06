from pydantic import BaseModel
from typing import Any, Dict

class BiasAuditObservation(BaseModel):
    task_name: str
    step_number: int
    progress_hint: str

class BiasAuditAction(BaseModel):
    action_type: str
    parameters: Dict[str, Any]
    reasoning: str

class FairAuditEnv:
    def __init__(self):
        pass

    async def reset(self, task_name: str = "dataset-scan") -> Any:
        return BiasAuditObservation(
            task_name=task_name,
            step_number=0,
            progress_hint="Environment initialized."
        )

    async def step(self, action: BiasAuditAction) -> Any:
        pass

    async def state(self) -> dict:
        return {"status": "initialized"}
from typing import Any, Tuple
from server.models import BiasAuditObservation, BiasAuditAction, BiasAuditReward
from server.environment import FairAuditEnvironment

class FairAuditEnv:
    def __init__(self):
        self.env = FairAuditEnvironment()

    async def reset(self, task_name: str = "dataset-scan") -> BiasAuditObservation:
        return self.env.reset(task_name)

    async def step(self, action: BiasAuditAction) -> Tuple[BiasAuditObservation, float, bool, dict]:
        return self.env.step(action)

    async def state(self) -> dict:
        return self.env.state_dump()
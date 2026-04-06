import pytest
from server.environment import FairAuditEnvironment
from server.models import BiasAuditAction

def test_defensive_wrapper_handles_hallucinations():
    env = FairAuditEnvironment()
    env.reset(task_name="dataset-scan")
    
    action = BiasAuditAction(
        action_type="inspect_column",
        parameters={"column": "made_up_hallucinated_column_name"},
        reasoning="I am hallucinating a column to crash the container."
    )
    
    obs, reward, done, info = env.step(action)
    
    assert done is False
    assert reward == -0.05
    assert "error" in obs.last_action_result
    assert "made_up_hallucinated_column_name" in obs.last_action_result["error"]

def test_submission_triggers_grader():
    env = FairAuditEnvironment()
    env.reset(task_name="dataset-scan")
    
    action = BiasAuditAction(
        action_type="submit",
        parameters={},
        reasoning="I am submitting without doing anything."
    )
    
    obs, reward, done, info = env.step(action)
    
    assert done is True
    assert reward == 0.0 
    assert info["reward_details"]["accumulated_total"] == 0.0
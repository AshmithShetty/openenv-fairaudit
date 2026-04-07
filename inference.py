import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from client import FairAuditEnv
from server.models import BiasAuditAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "dummy_key"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "fairaudit:latest")
BENCHMARK = "openenv-fairaudit"

# Task-specific step limits based on difficulty
TASK_STEPS = {
    "dataset-scan": 10,
    "model-audit": 15,
    "bias-mitigation": 25
}

# Derived from reward.py normalization standard (1.5 max points)
MAX_TOTAL_REWARD = 1.5
SUCCESS_SCORE_THRESHOLD = 0.5

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

async def get_model_action(client: AsyncOpenAI, obs: Dict[str, Any], history: List[str]) -> str:
    system_prompt = """You are an AI fairness auditing agent.
Your goal is to complete the Bias Audit task by taking logical steps.
Always respond with valid JSON matching this schema:
{
  "action_type": "string",
  "parameters": {},
  "reasoning": "string"
}
Available actions are listed in the observation. To finish, use action_type: 'submit'."""

    prompt = f"Observation:\n{json.dumps(obs)}\n\nAction History:\n{json.dumps(history)}\n\nProvide the next action as JSON:"
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"action_type": "submit", "parameters": {}, "reasoning": f"Error: {str(e)}"})

async def run_task(client: AsyncOpenAI, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = TASK_STEPS.get(task_name, 15)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Initialize isolated OpenEnv container
        env = await FairAuditEnv.from_docker_image(IMAGE_NAME)
        
        result = await env.reset(task_name=task_name)
        obs_data = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation
        
        for step in range(1, max_steps + 1):

            

            action_json = await get_model_action(client, obs_data, history)
            
            try:
                action_dict = json.loads(action_json)
            except json.JSONDecodeError:
                action_dict = {"action_type": "submit", "parameters": {}, "reasoning": "JSON Parse Error"}
                action_json = json.dumps(action_dict)

            # Map the parsed JSON back to the OpenEnv Action Model
            action_obj = BiasAuditAction(**action_dict)
            step_result = await env.step(action_obj)
            
            obs_data = step_result.observation.model_dump() if hasattr(step_result.observation, "model_dump") else step_result.observation
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)
            error = step_result.info.get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_json, reward=reward, done=done, error=str(error) if error else None)
            history.append(f"Step {step}: {action_json} -> Reward {reward}")

            if done:
                break

        # Calculate final score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error running task {task_name}: {e}", flush=True)
        
    finally:
        try:
            if "env" in locals():
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task in ["dataset-scan", "model-audit", "bias-mitigation"]:
        await run_task(client, task)

if __name__ == "__main__":
    asyncio.run(main())
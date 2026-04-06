import os
import json
import asyncio
import websockets
from typing import List, Dict, Any
from openai import AsyncOpenAI


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "dummy_key"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "openenv-fairaudit"

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

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

async def run_task(ws: websockets.WebSocketClientProtocol, client: AsyncOpenAI, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        await ws.send(json.dumps({"command": "reset", "task_name": task_name}))
        reset_res = json.loads(await ws.recv())
        obs = reset_res.get("data", reset_res)
        
        for step in range(1, MAX_STEPS + 1):
            action_json = await get_model_action(client, obs, history)
            
            try:
                action_dict = json.loads(action_json)
            except json.JSONDecodeError:
                action_dict = {"action_type": "submit", "parameters": {}, "reasoning": "JSON Parse Error"}
                action_json = json.dumps(action_dict)

            await ws.send(json.dumps({"command": "step", "action": action_dict}))
            
            step_res = json.loads(await ws.recv())
            obs = step_res.get("observation", {})
            reward = float(step_res.get("reward", 0.0))
            done = step_res.get("done", True)
            error = step_res.get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_json, reward=reward, done=done, error=str(error) if error else None)
            history.append(f"Step {step}: {action_json} -> Reward {reward}")

            if done:
                score = reward
                break

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error running task {task_name}: {e}", flush=True)
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    uri = "ws://localhost:7860/ws/inference_agent"
    
    try:
        async with websockets.connect(uri) as ws:
            for task in ["dataset-scan", "model-audit", "bias-mitigation"]:
                await run_task(ws, client, task)
    except Exception as e:
        print(f"Failed to connect to environment: {e}")

if __name__ == "__main__":
    asyncio.run(main())
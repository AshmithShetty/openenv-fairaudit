import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI

from client import FairAuditEnv
from server.models import BiasAuditAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "dummy_key"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "fairaudit:latest")
BENCHMARK = "openenv-fairaudit"

TASK_STEPS = {
    "dataset-scan": 10,
    "model-audit": 15,
    "bias-mitigation": 25,
}

MAX_TOTAL_REWARD = 1.5
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def get_model_action(client: OpenAI, obs: Dict[str, Any], history: List[str]) -> str:
    system_prompt = (
        "You are an AI fairness auditing agent.\n"
        "Your goal is to complete the Bias Audit task by taking logical steps.\n"
        "Always respond with valid JSON matching this schema:\n"
        '{"action_type": "string", "parameters": {}, "reasoning": "string"}\n'
        "Available actions are listed in the observation. To finish, use action_type: 'submit'."
    )
    prompt = (
        f"Observation:\n{json.dumps(obs)}\n\n"
        f"Action History:\n{json.dumps(history)}\n\n"
        "Provide the next action as JSON:"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return json.dumps({"action_type": "submit", "parameters": {}, "reasoning": f"Error: {str(exc)}"})


async def run_task(client: OpenAI, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    max_steps = TASK_STEPS.get(task_name, 15)
    env = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await FairAuditEnv.from_docker_image(IMAGE_NAME)

        result = await env.reset(task_name=task_name)
        obs_data = (
            result.observation.model_dump()
            if hasattr(result.observation, "model_dump")
            else result.observation
        )

        for step in range(1, max_steps + 1):
            action_json = get_model_action(client, obs_data, history)

            try:
                action_dict = json.loads(action_json)
            except json.JSONDecodeError:
                action_dict = {
                    "action_type": "submit",
                    "parameters": {},
                    "reasoning": "JSON Parse Error",
                }
                action_json = json.dumps(action_dict)

            action_obj = BiasAuditAction(**action_dict)
            step_result = await env.step(action_obj)

            obs_data = (
                step_result.observation.model_dump()
                if hasattr(step_result.observation, "model_dump")
                else step_result.observation
            )
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)
            error = step_result.info.get("error", None) if step_result.info else None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_json,
                reward=reward,
                done=done,
                error=str(error) if error else None,
            )
            history.append(f"Step {step}: {action_json} -> Reward {reward}")

            if done:
                break

        raw_score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(raw_score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Error running task {task_name}: {exc}", flush=True)

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ["dataset-scan", "model-audit", "bias-mitigation"]:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
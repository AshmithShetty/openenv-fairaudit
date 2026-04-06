---
title: FairAudit
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# OpenEnv: FairAudit

## Environment Description & Motivation
FairAudit is a comprehensive Machine Learning Operations (MLOps) environment designed to evaluate and train AI agents in identifying, measuring, and mitigating algorithmic bias. While the core framework, action spaces, and fairness metrics (e.g., Disparate Impact, Demographic Parity) are inherently domain-agnostic and designed to scale across industries such as healthcare, hiring, and criminal justice, this specific benchmark environment simulates a **Financial Loan Approval Pipeline**.

AI agents are tasked with navigating a complete, end-to-end fairness audit on a highly realistic loan dataset. While the tasks represent a continuous chronological workflow (Data Scan -> Model Audit -> Bias Mitigation), the environment utilizes a **State Checkpointing Architecture**. When an agent resets to a specific task, the environment automatically loads a perfectly cleaned dataset or pre-trained model required for that stage. This guarantees that automated LLM evaluators do not trigger cascading failures across the pipeline, ensuring robust and reproducible scoring.

### The Pareto Trap
In the final task, the environment introduces the "Pareto Trap." Agents must mitigate algorithmic bias without destroying the model's predictive accuracy. If an agent mindlessly drops highly predictive financial indicators (like Debt-to-Income ratio) to brute-force a perfect fairness score, the model's accuracy crashes, simulating massive financial losses for the bank, and the agent receives a severe reward penalty.

## Tasks & Difficulty
1. **Dataset Scan (Easy):** Inspect raw data and identify hidden proxy variables (e.g., zip codes masking redlining).
2. **Model Audit (Medium):** Compute the Disparate Impact of a pre-trained baseline model and flag severity.
3. **Bias Mitigation (Hard):** Apply mitigation strategies to the biased model, verify fairness metrics, and navigate the Pareto Trap to maintain utility > 70%.

## Action Space
The agent submits JSON payloads to the `step()` function. Key actions include:
* `inspect_column`: Returns dtype, null counts, and statistical distributions.
* `check_correlation`: Computes Pearson's r or Cramer's V between features.
* `compute_metric`: Calculates Disparate Impact or Demographic Parity.
* `flag_bias` / `flag_violation`: Submits findings to the internal ledger.
* `evaluate_model`: Trains a Logistic Regression model and returns accuracy.
* `apply_mitigation`: Drops features or applies reweighing strategies.
* `verify_fairness`: Re-calculates metrics post-mitigation.
* `submit`: Terminates the episode and triggers the automated grader.

## Observation Space
The environment returns a strict Pydantic model (`BiasAuditObservation`) containing:
* `task_name` and `step_number`
* `dataset_info`: Current active columns and row counts.
* `findings`: The agent's submitted flags and violation reports.
* `last_action_result`: The direct JSON response from the previous action.
* `progress_hint`: System feedback for partial progress.

## Setup and Usage

### Containerized Execution
Build and run the environment locally using Docker:
```bash
docker build -t fairaudit .
docker run -d --name fairaudit-container -p 7860:7860 fairaudit
```

### Baseline Inference
The repository includes an official `inference.py` script that uses the standard `AsyncOpenAI` client to run an LLM against the 3 tasks.
```bash
export API_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)"
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
uv run python inference.py
```
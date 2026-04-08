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

## Environment Description and Motivation

FairAudit is a real-world OpenEnv environment that trains and evaluates AI agents to audit algorithmic bias in machine learning systems. It simulates a complete end-to-end fairness audit on a synthetic financial loan approval dataset, covering the full compliance workflow used by data scientists and regulators: data scanning, model auditing, and bias mitigation.

The environment is grounded in documented real-world cases. The CFPB issued a consent order against Upstart in 2023 for exactly the type of zip-code-to-race proxy discrimination this environment simulates. Agents trained here learn the same forensic skills compliance teams apply under the Equal Credit Opportunity Act and the EU AI Act.

## Action Space

The agent submits a JSON payload with three fields to every `step()` call:

- `action_type`: one of the 13 valid action types listed below
- `parameters`: a dict of arguments specific to the action
- `reasoning`: a string explaining the agent's reasoning (used in reward shaping)

Valid action types and their parameters:

| Action | Parameters | Description |
|---|---|---|
| `inspect_column` | `col: str` | Returns dtype, null counts, distribution stats |
| `check_correlation` | `col1: str, col2: str` | Computes Pearson r or Cramer's V |
| `sample_rows` | `filter: str` | Returns up to 20 rows matching a filter |
| `flag_bias` | `col: str, attr: str, reason: str` | Marks a column as a biased proxy |
| `compute_metric` | `metric: str, group_col: str` | Computes DI, DP, EO, or EqOdds |
| `compare_groups` | `col: str, val1: str, val2: str` | Compares outcome rates between groups |
| `get_confusion_matrix` | `group: str` | Returns TPR/FPR breakdown for a group |
| `flag_violation` | `metric: str, val: float, severity: str` | Logs a compliance violation |
| `apply_mitigation` | `technique: str, params: dict` | Applies reweighting or threshold adjustment |
| `evaluate_model` | `{}` | Returns accuracy and all 4 fairness metrics post-mitigation |
| `verify_fairness` | `metric: str` | Checks if a metric now passes its legal threshold |
| `write_report` | `section: str, content: str` | Writes one of 4 audit report sections |
| `submit` | `{}` | Ends the episode and triggers the grader |

## Observation Space

Every `step()` and `reset()` returns a `BiasAuditObservation` Pydantic model:

| Field | Type | Description |
|---|---|---|
| `task_name` | str | Current task identifier |
| `step_number` | int | Current step count |
| `max_steps` | int | Maximum steps for this task |
| `dataset_info` | dict | Column names and row count |
| `model_info` | ModelInfo or null | Model type, features, accuracy (Tasks 2 and 3 only) |
| `findings` | List[Finding] | All bias flags and violations submitted so far |
| `fairness_metrics` | FairnessMetrics | Computed DI, DP, EO, EqOdds values |
| `available_actions` | List[str] | Valid action types for this step |
| `last_action_result` | dict | Direct result from the previous action |
| `progress_hint` | str | Plain-English feedback on current progress |

## Tasks and Difficulty

**Task 1: Dataset Scan (Easy) — max 10 steps**
Agent receives a raw loan dataset and must identify which columns are proxies for the protected attribute (race via zip code). Graded on precision and recall of flagged columns against pre-computed ground truth. Score range: 0.0–1.0.

**Task 2: Model Audit (Medium) — max 15 steps**
Agent receives a pre-trained biased LogisticRegression model and must compute all four fairness metrics (Disparate Impact, Demographic Parity, Equal Opportunity, Equalized Odds) and classify their severity against legal thresholds. Score range: 0.0–1.0.

**Task 3: Bias Mitigation (Hard) — max 25 steps**
Agent must apply mitigation techniques to reduce bias while maintaining model accuracy above 70%. Wrong mitigation choices that destroy accuracy trigger a penalty. Agent must also write a 4-section audit report. This task directly tests reasoning about the Pareto tradeoff between fairness and accuracy. Score range: 0.0–1.0.

## Setup and Usage

### Local Docker execution

```bash
docker build -t fairaudit:latest .
docker run -d --name fairaudit-container -p 7860:7860 fairaudit:latest
```

### Running inference

Set the required environment variables then run the inference script:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export HF_TOKEN="your-key-here"
export MODEL_NAME="gpt-4o-mini"
export IMAGE_NAME="fairaudit:latest"
python inference.py
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `HF_TOKEN` | Yes | API key |
| `MODEL_NAME` | Yes | Model identifier |
| `IMAGE_NAME` | Yes | Docker image name for the environment |

## Baseline Scores

Scores produced by running `inference.py` with `gpt-4o-mini` against the local Docker container.

| Task | Difficulty | Score (0.0–1.0) | Success |
|---|---|---|---|
| `dataset-scan` | Easy | 0.62 | true |
| `model-audit` | Medium | 0.41 | false |
| `bias-mitigation` | Hard | 0.28 | false |
import json
from typing import Tuple, Dict, Any
from server.state import EpisodeState

class RewardEngine:
    def calculate_step_reward(self, action_type: str, params: Dict[str, Any], result: Dict[str, Any], state: EpisodeState) -> Tuple[float, str]:
        #Negative Signal: Invalid action or hallucinated param
        if "error" in result:
            return -0.05, f"Invalid action or hallucinated parameters: {result['error']}"

        #Negative Signal: Duplicate action
        if not hasattr(state, "unique_actions"):
            state.unique_actions = set()
            
        action_sig = f"{action_type}_{json.dumps(params, sort_keys=True)}"
        if action_sig in state.unique_actions:
            return -0.10, "Penalty: Duplicate action performed."
            
        state.unique_actions.add(action_sig)

        reward = 0.0
        feedback = "Action executed successfully."

        #Positive Signals
        if action_type == "inspect_column":
            reward = 0.02
            feedback = "Good job. Inspected column structure."

        elif action_type == "check_correlation":
            val = abs(result.get("value", 0.0))
            if val >= 0.3:
                reward = 0.05
                feedback = f"Significant correlation found ({val:.2f})."
            else:
                reward = 0.01
                feedback = "Correlation checked, but not highly significant."

        elif action_type == "flag_bias":
            col = params.get("column")
            if col in state.ground_truth.get("biased_columns", []):
                reward = 0.15
                feedback = f"Correct! {col} is a valid biased proxy variable."
            else:
                reward = -0.05
                feedback = f"Incorrect. {col} is not a primary biased proxy."

        elif action_type == "compute_metric":
            reward = 0.08
            feedback = "Valid fairness metric computed."

        elif action_type == "flag_violation":
            attr = params.get("protected_attribute", "").split("_")[-1].lower()
            expected_sev = None
            for k, v in state.ground_truth.get("violation_severities", {}).items():
                if attr in k.lower():
                    expected_sev = v
                    break

            if expected_sev and params.get("severity") == expected_sev:
                reward = 0.12
                feedback = "Correct severity flag applied based on metrics."
            else:
                reward = -0.05
                feedback = "Incorrect severity flag applied."

        elif action_type == "evaluate_model":
            acc = state.model.test_accuracy if state.model else 0.0
            if acc < 0.70:
                if not getattr(state, "accuracy_penalty_applied", False):
                    reward = -0.15
                    feedback = "Warning: Model accuracy dropped below 70%. Utility severely degraded."
                    state.accuracy_penalty_applied = True
                else:
                    reward = 0.0
                    feedback = "Model evaluated. Accuracy remains poor."
            else:
                reward = 0.05
                feedback = "Model evaluated with acceptable accuracy."

        elif action_type == "verify_fairness":
            gt_metrics = state.ground_truth.get("true_metric_values", {})
            orig_di = 1.0
            if "race_metrics" in gt_metrics:
                orig_di = gt_metrics["race_metrics"].get("disparate_impact", 1.0)
                
            new_di = result.get("post_mitigation_metrics", {}).get("disparate_impact", orig_di)

            if new_di > orig_di and new_di <= 1.0:
                delta = new_di - orig_di
                reward = 0.30 * delta
                feedback = f"Mitigation successfully improved Disparate Impact by {delta:.2f}."
            else:
                reward = 0.0
                feedback = "Fairness verified. No significant improvement in Disparate Impact."

        #Clamp & Output
        reward = max(-0.5, min(1.0, reward))
        return reward, feedback

    def calculate_final_score(self, accumulated: float, grader_score: float) -> float:
        # Normalize accumulated score to [0, 1] assuming approx - 1.5 points is an excellent run
        normalized_accumulated = max(0.0, min(1.0, accumulated / 1.5))
        return (0.30 * normalized_accumulated) + (0.70 * grader_score)
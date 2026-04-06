from typing import Tuple, Dict, Any
from server.models import BiasAuditObservation, BiasAuditAction, BiasAuditReward, FairnessMetrics
from server.state import EpisodeState
from server.data.generator import DataGenerator
import server.actions.handlers as handlers
import server.graders.grader_t1 as grader_t1
import server.graders.grader_t2 as grader_t2
import server.graders.grader_t3 as grader_t3

class FairAuditEnvironment:
    def __init__(self):
        self.generator = DataGenerator()
        self.state = None

    def reset(self, task_name: str = "dataset-scan") -> BiasAuditObservation:
        scenario_name = "hiring"
        if task_name == "model-audit":
            scenario_name = "loan"
        elif task_name == "bias-mitigation":
            scenario_name = "medical"
            
        df, ground_truth = self.generator.generate(scenario_name, seed=42)
        
        self.state = EpisodeState(
            task_name=task_name,
            scenario_name=scenario_name,
            df=df,
            ground_truth=ground_truth,
            current_step=0,
            max_steps=15
        )
        
        return BiasAuditObservation(
            task_name=self.state.task_name,
            step_number=self.state.current_step,
            max_steps=self.state.max_steps,
            dataset_info={"columns": list(df.columns), "rows": len(df)},
            findings=[],
            fairness_metrics=FairnessMetrics(),
            available_actions=["inspect_column", "check_correlation", "sample_rows", "flag_bias", "submit"],
            last_action_result={"status": "initialized"},
            progress_hint="Environment initialized."
        )

    def step(self, action: BiasAuditAction) -> Tuple[BiasAuditObservation, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step.")
            
        self.state.current_step += 1
        
        action_map = {
            "inspect_column": handlers.handle_inspect_column,
            "check_correlation": handlers.handle_check_correlation,
            "sample_rows": handlers.handle_sample_rows,
            "flag_bias": handlers.handle_flag_bias,
            "submit": handlers.handle_submit,
            "compute_metric": handlers.handle_compute_metric,
            "compare_groups": handlers.handle_compare_groups,
            "get_confusion_matrix": handlers.handle_get_confusion_matrix,
            "flag_violation": handlers.handle_flag_violation,
            "apply_mitigation": handlers.handle_apply_mitigation,
            "evaluate_model": handlers.handle_evaluate_model,
            "verify_fairness": handlers.handle_verify_fairness,
            "write_report": handlers.handle_write_report
        }
        
        handler = action_map.get(action.action_type)
        if handler:
            result, reward_val = handler(self.state, action.parameters)
        else:
            result, reward_val = {"error": f"Unknown action: {action.action_type}"}, -0.1
            
        self.state.action_history.add(f"{action.action_type}_{self.state.current_step}")
        
        done = self.state.current_step >= self.state.max_steps or action.action_type == "submit"
        self.state.is_done = done
        
        # Grading execution upon submission
        if action.action_type == "submit":
            if self.state.task_name == "dataset-scan":
                reward_val = grader_t1.grade(self.state)
            elif self.state.task_name == "model-audit":
                reward_val = grader_t2.grade(self.state)
            elif self.state.task_name == "bias-mitigation":
                reward_val = grader_t3.grade(self.state)
            else:
                reward_val = 0.0
                
            self.state.accumulated_reward = reward_val
        else:
            self.state.accumulated_reward += reward_val
        
        obs = BiasAuditObservation(
            task_name=self.state.task_name,
            step_number=self.state.current_step,
            max_steps=self.state.max_steps,
            dataset_info={"columns": list(self.state.df.columns), "rows": len(self.state.df)},
            findings=self.state.findings,
            fairness_metrics=FairnessMetrics(),
            available_actions=list(action_map.keys()),
            last_action_result=result,
            progress_hint="Action executed." if not done else f"Episode complete. Final Score: {self.state.accumulated_reward}"
        )
        
        reward_obj = BiasAuditReward(
            value=reward_val,
            breakdown={action.action_type: reward_val},
            accumulated_total=self.state.accumulated_reward,
            feedback="Execution complete."
        )
        
        return obs, reward_obj.value, done, {"reward_details": reward_obj.model_dump()}

    def state_dump(self) -> Dict[str, Any]:
        if not self.state:
            return {"status": "uninitialized"}
        return {
            "current_step": self.state.current_step,
            "is_done": self.state.is_done,
            "accumulated_reward": self.state.accumulated_reward
        }
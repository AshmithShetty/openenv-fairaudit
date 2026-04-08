from typing import Tuple, Dict, Any
import pandas as pd
from server.models import BiasAuditObservation, BiasAuditAction, BiasAuditReward, FairnessMetrics, ModelInfo
from server.state import EpisodeState
from server.data.generator import DataGenerator
from server.reward import RewardEngine
import server.actions.handlers as handlers
import server.graders.grader_t1 as grader_t1
import server.graders.grader_t2 as grader_t2
import server.graders.grader_t3 as grader_t3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TASK_MAX_STEPS = {
    "dataset-scan": 10,
    "model-audit": 15,
    "bias-mitigation": 25,
}


class FairAuditEnvironment:
    def __init__(self):
        self.generator = DataGenerator()
        self.reward_engine = RewardEngine()
        self.state = None

    def _train_checkpoint_model(self, df: pd.DataFrame) -> Tuple[ModelInfo, pd.Series]:
        if "zip_code" in df.columns:
            df = df.drop(columns=["zip_code"])

        target = "loan_approved"
        features = [c for c in df.columns if c != target]

        X = df[features]
        X = pd.get_dummies(X, drop_first=True)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        model_info = ModelInfo(
            model_type="LogisticRegression",
            feature_columns=features,
            target_column=target,
            train_accuracy=float(train_acc),
            test_accuracy=float(test_acc),
        )

        predictions = pd.Series(clf.predict(X), index=X.index)
        return model_info, predictions

    def reset(self, task_name: str = "dataset-scan") -> BiasAuditObservation:
        df, ground_truth = self.generator.generate("loan", seed=42)

        max_steps = TASK_MAX_STEPS.get(task_name, 15)

        self.state = EpisodeState(
            task_name=task_name,
            scenario_name="loan",
            df=df,
            ground_truth=ground_truth,
            current_step=0,
            max_steps=max_steps,
        )

        progress_hint = "Environment initialized. Raw dataset loaded."

        if task_name == "model-audit":
            self.state.df = self.state.df.drop(columns=["zip_code"], errors="ignore")
            model_info, preds = self._train_checkpoint_model(self.state.df)
            self.state.model = model_info
            self.state.latest_predictions = preds
            progress_hint = "Checkpoint Loaded: Dataset cleaned. Baseline model trained and ready for audit."

        elif task_name == "bias-mitigation":
            self.state.df = self.state.df.drop(columns=["zip_code"], errors="ignore")
            model_info, preds = self._train_checkpoint_model(self.state.df)
            self.state.model = model_info
            self.state.latest_predictions = preds

            protected_col = ground_truth.get("protected_attribute", "race")
            res = ground_truth.get("true_metric_values", {}).get(f"{protected_col}_metrics", {})

            if not hasattr(self.state, "computed_metrics"):
                self.state.computed_metrics = {}
            self.state.computed_metrics[protected_col] = res

            progress_hint = "Checkpoint Loaded: Model audited. Bias identified. Ready for mitigation."

        return BiasAuditObservation(
            task_name=self.state.task_name,
            step_number=self.state.current_step,
            max_steps=self.state.max_steps,
            dataset_info={"columns": list(self.state.df.columns), "rows": len(self.state.df)},
            findings=[],
            fairness_metrics=FairnessMetrics(),
            available_actions=["inspect_column", "check_correlation", "sample_rows", "flag_bias", "submit"],
            last_action_result={"status": "initialized"},
            progress_hint=progress_hint,
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
            "write_report": handlers.handle_write_report,
        }

        handler = action_map.get(action.action_type)
        if handler:
            result, _ = handler(self.state, action.parameters)
        else:
            result = {"error": f"Unknown action: {action.action_type}"}

        if not hasattr(self.state, "action_history"):
            self.state.action_history = set()

        self.state.action_history.add(f"{action.action_type}_{self.state.current_step}")
        done = self.state.current_step >= self.state.max_steps or action.action_type == "submit"
        self.state.is_done = done

        if action.action_type == "submit":
            if self.state.task_name == "dataset-scan":
                grader_score = grader_t1.grade(self.state)
            elif self.state.task_name == "model-audit":
                grader_score = grader_t2.grade(self.state)
            elif self.state.task_name == "bias-mitigation":
                grader_score = grader_t3.grade(self.state)
            else:
                grader_score = 0.0

            final_reward = self.reward_engine.calculate_final_score(self.state.accumulated_reward, grader_score)
            step_reward = final_reward
            feedback = f"Episode complete. Grader Score: {grader_score:.2f}. Final Weighted Score: {final_reward:.2f}"
            self.state.accumulated_reward = final_reward
        else:
            step_reward, feedback = self.reward_engine.calculate_step_reward(
                action.action_type, action.parameters, result, self.state
            )
            self.state.accumulated_reward += step_reward

        obs = BiasAuditObservation(
            task_name=self.state.task_name,
            step_number=self.state.current_step,
            max_steps=self.state.max_steps,
            dataset_info={"columns": list(self.state.df.columns), "rows": len(self.state.df)},
            findings=self.state.findings,
            fairness_metrics=FairnessMetrics(),
            available_actions=list(action_map.keys()),
            last_action_result=result,
            progress_hint=feedback,
        )

        reward_obj = BiasAuditReward(
            value=float(step_reward),
            breakdown={action.action_type: float(step_reward)},
            accumulated_total=float(self.state.accumulated_reward),
            feedback=feedback,
        )

        return obs, reward_obj.value, done, {"reward_details": reward_obj.model_dump()}

    def state_dump(self) -> Dict[str, Any]:
        if not self.state:
            return {"status": "uninitialized"}
        return {
            "current_step": self.state.current_step,
            "is_done": self.state.is_done,
            "accumulated_reward": self.state.accumulated_reward,
        }
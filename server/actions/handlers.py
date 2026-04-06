import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from functools import wraps
from server.state import EpisodeState
from server.models import Finding, ModelInfo
from server.fairness.metrics import calculate_cramers_v, calculate_historical_fairness

def defensive_handler(func):
    @wraps(func)
    def wrapper(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        try:
            return func(state, params)
        except Exception as e:
            return {"error": f"Action failed: {str(e)}"}, -0.05
    return wrapper

@defensive_handler
def handle_inspect_column(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    col = params["column"]
    if col not in state.df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
        
    series = state.df[col]
    is_num = pd.api.types.is_numeric_dtype(series)
    
    info = {
        "name": col,
        "dtype": str(series.dtype),
        "unique_count": int(series.nunique()),
        "null_count": int(series.isnull().sum()),
        "sample_values": series.dropna().sample(min(5, len(series))).tolist()
    }
    
    if is_num:
        info["mean"] = float(series.mean())
        info["std"] = float(series.std())
        info["min"] = float(series.min())
        info["max"] = float(series.max())
        
    return info, 0.0

@defensive_handler
def handle_check_correlation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    col1, col2 = params["column1"], params["column2"]
    if col1 not in state.df.columns or col2 not in state.df.columns:
        raise ValueError("One or both columns not found in dataset.")
        
    is_num1 = pd.api.types.is_numeric_dtype(state.df[col1])
    is_num2 = pd.api.types.is_numeric_dtype(state.df[col2])
    
    if is_num1 and is_num2:
        val = float(state.df[col1].corr(state.df[col2]))
        metric_type = "pearson_r"
    else:
        val = calculate_cramers_v(state.df[col1], state.df[col2])
        metric_type = "cramers_v"
        
    return {"metric": metric_type, "value": val, "columns": [col1, col2]}, 0.0

@defensive_handler
def handle_sample_rows(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    col = params.get("column")
    op = params.get("operator")
    val = params.get("value")
    
    df = state.df
    if col and op and val is not None:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if op == "==":
            df = df[df[col] == val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == "!=":
            df = df[df[col] != val]
            
    sample = df.sample(min(10, len(df))).to_dict(orient="records")
    return {"rows_returned": len(sample), "data": sample}, 0.0

@defensive_handler
def handle_flag_bias(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    col = params["column"]
    reason = params.get("reason", "No reason provided")
    
    if col not in state.df.columns:
        raise ValueError(f"Column '{col}' not found.")
        
    finding = Finding(
        column=col,
        attribute="N/A",
        finding_type="bias_proxy",
        reason=reason
    )
    state.findings.append(finding)
    return {"status": "success", "flagged_column": col}, 0.0

@defensive_handler
def handle_submit(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "submitted", "message": "Task complete, triggering grader."}, 0.0

@defensive_handler
def handle_compute_metric(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    protected_col = params.get("protected_attribute")
    target_col = params.get("target_column")
    priv_val = params.get("privileged_value")
    pos_val = params.get("positive_outcome")

    if not all([protected_col, target_col, priv_val is not None, pos_val is not None]):
        raise ValueError("Missing required parameters for metric computation.")

    if protected_col not in state.df.columns:
        if protected_col == state.ground_truth.get("protected_attribute"):
            res = state.ground_truth.get("true_metric_values", {}).get(f"{protected_col}_metrics", {})
            if not hasattr(state, "computed_metrics"):
                state.computed_metrics = {}
            if protected_col not in state.computed_metrics:
                state.computed_metrics[protected_col] = {}
            state.computed_metrics[protected_col].update(res)
            return {"computed_metrics": res}, 0.0
        else:
            raise ValueError(f"Column '{protected_col}' not found in dataset.")

    if target_col not in state.df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    res = calculate_historical_fairness(state.df, protected_col, target_col, priv_val, pos_val)

    if not hasattr(state, "computed_metrics"):
        state.computed_metrics = {}
    if protected_col not in state.computed_metrics:
        state.computed_metrics[protected_col] = {}
    state.computed_metrics[protected_col].update(res)

    return {"computed_metrics": res}, 0.0

@defensive_handler
def handle_compare_groups(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    group_col = params["group_column"]
    target_col = params["target_column"]

    if group_col not in state.df.columns or target_col not in state.df.columns:
        raise ValueError("Columns not found.")

    grouped = state.df.groupby(group_col)[target_col].value_counts(normalize=True).unstack().fillna(0)
    return {"distributions": grouped.to_dict()}, 0.0

@defensive_handler
def handle_get_confusion_matrix(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    col1 = params.get("actual_column")
    col2 = params.get("predicted_column") or params.get("group_column")

    if not col1 or not col2 or col1 not in state.df.columns or col2 not in state.df.columns:
        raise ValueError("Valid columns required for matrix.")

    cm = pd.crosstab(state.df[col1], state.df[col2])
    return {"matrix": cm.to_dict()}, 0.0

@defensive_handler
def handle_flag_violation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    attr = params["protected_attribute"]
    severity = params.get("severity", "MEDIUM")
    reason = params.get("reason", "No reason provided")

    finding = Finding(
        column="N/A",
        attribute=attr,
        finding_type="violation",
        severity=severity,
        reason=reason
    )
    state.findings.append(finding)
    return {"status": "violation_flagged", "attribute": attr, "severity": severity}, 0.0

@defensive_handler
def handle_apply_mitigation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    strategy = params.get("strategy")
    
    if not hasattr(state, "mitigations"):
        state.mitigations = []
        
    if strategy == "drop_feature":
        col = params.get("feature")
        if col in state.df.columns:
            state.df = state.df.drop(columns=[col])
            state.mitigations.append(f"dropped_{col}")
            return {"status": "success", "mitigation": "feature_dropped", "feature": col}, 0.0
        return {"error": f"Feature '{col}' not found"}, -0.05
        
    elif strategy in ["reweighing", "threshold_shift"]:
        state.mitigations.append(strategy)
        return {"status": "success", "mitigation": strategy}, 0.0
        
    return {"error": "Unknown mitigation strategy"}, -0.05

@defensive_handler
def handle_evaluate_model(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    features = params.get("feature_columns", [])
    target = params.get("target_column")
    
    if not features or target not in state.df.columns:
        raise ValueError("Valid features and target column required.")
        
    missing = [f for f in features if f not in state.df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")
        
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    X = state.df[features]
    X = pd.get_dummies(X, drop_first=True)
    y = state.df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    if state.scenario_name == "loan" and hasattr(state, "mitigations"):
        if "dropped_debt_to_income_ratio" in state.mitigations:
            test_acc = min(test_acc, 0.65)
            
    state.model = ModelInfo(
        model_type="LogisticRegression",
        feature_columns=features,
        target_column=target,
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc)
    )
    
    state.latest_predictions = pd.Series(clf.predict(X), index=X.index)
    return state.model.model_dump(), 0.0

@defensive_handler
def handle_verify_fairness(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    prot_attr = params.get("protected_attribute")
    priv_val = params.get("privileged_value")
    pos_val = params.get("positive_outcome")
    
    if not hasattr(state, "latest_predictions"):
        raise ValueError("Must evaluate model before verifying fairness metrics.")
        
    if prot_attr not in state.df.columns:
        if prot_attr == state.ground_truth.get("protected_attribute"):
            orig_di = state.ground_truth.get("true_metric_values", {}).get(f"{prot_attr}_metrics", {}).get("disparate_impact", 1.0)
            improved_di = min(1.0, orig_di + 0.20) if hasattr(state, "mitigations") and state.mitigations else orig_di
            res = {"disparate_impact": improved_di, "demographic_parity": 0.05}
            state.mitigated_metrics = res
            return {"post_mitigation_metrics": res}, 0.0
        else:
            raise ValueError(f"Column '{prot_attr}' not found in dataset.")
            
    temp_df = state.df.copy()
    temp_df['model_predictions'] = state.latest_predictions
    
    res = calculate_historical_fairness(temp_df, prot_attr, 'model_predictions', priv_val, pos_val)
    
    state.mitigated_metrics = res
    return {"post_mitigation_metrics": res}, 0.0

@defensive_handler
def handle_write_report(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    section = params.get("section")
    content = params.get("content")
    
    if not section or not content:
        raise ValueError("Both section title and content are required.")
        
    state.report_sections[section] = content
    return {"status": "saved", "section": section, "length": len(content)}, 0.0
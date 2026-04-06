from typing import Tuple, Dict, Any
from server.state import EpisodeState

def handle_inspect_column(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "inspect_column"}, 0.0

def handle_check_correlation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "check_correlation"}, 0.0

def handle_sample_rows(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "sample_rows"}, 0.0

def handle_flag_bias(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "flag_bias"}, 0.0

def handle_submit(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "submit"}, 0.0

def handle_compute_metric(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "compute_metric"}, 0.0

def handle_compare_groups(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "compare_groups"}, 0.0

def handle_get_confusion_matrix(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "get_confusion_matrix"}, 0.0

def handle_flag_violation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "flag_violation"}, 0.0

def handle_apply_mitigation(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "apply_mitigation"}, 0.0

def handle_evaluate_model(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "evaluate_model"}, 0.0

def handle_verify_fairness(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "verify_fairness"}, 0.0

def handle_write_report(state: EpisodeState, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return {"status": "stub", "action": "write_report"}, 0.0
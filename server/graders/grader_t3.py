from server.state import EpisodeState

def grade(state: EpisodeState) -> float:
    score = 0.0
    
    if len(state.report_sections) >= 3:
        score += 0.2
    elif len(state.report_sections) > 0:
        score += 0.1
        
    if state.model:
        acc = state.model.test_accuracy
        if state.scenario_name == "loan" and acc < 0.70:
            score -= 0.15 
        elif acc >= 0.75:
            score += 0.4
        elif acc >= 0.70:
            score += 0.2
            
    gt_metrics = state.ground_truth.get("true_metric_values", {})
    orig_di = 1.0
    
    if "race_metrics" in gt_metrics:
        orig_di = gt_metrics["race_metrics"].get("disparate_impact", 1.0)
        
    new_di = getattr(state, "mitigated_metrics", {}).get("disparate_impact", orig_di)
    
    if new_di > orig_di and new_di <= 1.0:
        score += 0.4
    elif new_di > orig_di:
        score += 0.2
        
    return max(0.01, min(0.99, score))
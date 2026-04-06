from server.state import EpisodeState

def grade(state: EpisodeState) -> float:
    true_biased_columns = set(state.ground_truth.get("biased_columns", []))
    
    agent_flagged_columns = set()
    for finding in state.findings:
        if finding.finding_type == "bias_proxy":
            agent_flagged_columns.add(finding.column)
            
    if not true_biased_columns and not agent_flagged_columns:
        return 1.0
        
    if not agent_flagged_columns and true_biased_columns:
        return 0.0

    true_positives = len(agent_flagged_columns.intersection(true_biased_columns))
    false_positives = len(agent_flagged_columns - true_biased_columns)
    false_negatives = len(true_biased_columns - agent_flagged_columns)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    score = (0.5 * precision) + (0.5 * recall)
    
    if precision == 1.0 and recall == 1.0:
        score += 0.1
        
    return max(0.0, min(1.0, score))
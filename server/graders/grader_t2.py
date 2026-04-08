from server.state import EpisodeState

def grade(state: EpisodeState) -> float:
    gt_metrics = state.ground_truth.get("true_metric_values", {})
    true_severities = state.ground_truth.get("violation_severities", {})

    if not gt_metrics or not true_severities:
        return 0.01

    score = 0.0
    max_score = 0.0

    computed_di_values = []
    if hasattr(state, "computed_metrics"):
        for k, v in state.computed_metrics.items():
            if "disparate_impact" in v:
                computed_di_values.append(v["disparate_impact"])

    for gt_group, gt_vals in gt_metrics.items():
        max_score += 0.5 
        gt_di = gt_vals.get("disparate_impact", 1.0)

        best_diff = 1.0
        for c_di in computed_di_values:
            diff = abs(c_di - gt_di)
            if diff < best_diff:
                best_diff = diff

        if best_diff <= 0.02:
            score += 0.5
        elif best_diff <= 0.05:
            score += 0.25

    agent_severities = {}
    for f in state.findings:
        if f.finding_type == "violation":
            agent_severities[f.attribute.lower()] = f.severity

    for gt_key, gt_sev in true_severities.items():
        max_score += 0.5
        attr = gt_key.split("_")[-1].lower()

        if attr in agent_severities:
            if agent_severities[attr] == gt_sev:
                score += 0.5
            elif gt_sev != "LOW" and agent_severities[attr] != "LOW":
                score += 0.25

    return max(0.01, min(0.99, score / max_score)) if max_score > 0 else 0.01
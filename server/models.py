from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unique_count: int
    null_count: int
    sample_values: List[Any]

class FairnessMetrics(BaseModel):
    disparate_impact: Optional[float] = None
    demographic_parity: Optional[float] = None
    equal_opportunity: Optional[float] = None
    equalized_odds: Optional[float] = None
    accuracy: Optional[float] = None

class Finding(BaseModel):
    column: str
    attribute: str
    finding_type: str = Field(pattern="^(bias_proxy|violation)$")
    severity: Optional[str] = Field(None, pattern="^(HIGH|MEDIUM|LOW)$")
    reason: str
    is_correct: Optional[bool] = None

class ModelInfo(BaseModel):
    model_type: str
    feature_columns: List[str]
    target_column: str
    train_accuracy: float
    test_accuracy: float

class BiasAuditObservation(BaseModel):
    task_name: str
    step_number: int
    max_steps: int
    dataset_info: Dict[str, Any]
    model_info: Optional[ModelInfo] = None
    findings: List[Finding]
    fairness_metrics: FairnessMetrics
    available_actions: List[str]
    last_action_result: Dict[str, Any]
    progress_hint: str

class BiasAuditAction(BaseModel):
    action_type: str = Field(
        pattern="^(inspect_column|check_correlation|sample_rows|flag_bias|submit|compute_metric|compare_groups|get_confusion_matrix|flag_violation|apply_mitigation|evaluate_model|verify_fairness|write_report)$"
    )
    parameters: Dict[str, Any]
    reasoning: str

class BiasAuditReward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    accumulated_total: float
    feedback: str
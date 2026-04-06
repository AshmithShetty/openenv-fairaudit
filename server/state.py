from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional
import pandas as pd
from server.models import Finding

@dataclass
class EpisodeState:
    task_name: str
    scenario_name: str
    df: pd.DataFrame
    ground_truth: Dict[str, Any]
    current_step: int
    max_steps: int
    findings: List[Finding] = field(default_factory=list)
    computed_metrics: Dict[str, Any] = field(default_factory=dict)
    report_sections: Dict[str, str] = field(default_factory=dict)
    model: Optional[Any] = None
    mitigated_model: Optional[Any] = None
    is_done: bool = False
    action_history: Set[str] = field(default_factory=set)
    accumulated_reward: float = 0.0
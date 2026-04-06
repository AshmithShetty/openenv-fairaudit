from abc import ABC, abstractmethod
from typing import Tuple, Dict
import pandas as pd

class BaseScenario(ABC):
    MAX_ROWS = 10000

    @abstractmethod
    def generate(self, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        pass

    def _seal_ground_truth(self, 
                           df: pd.DataFrame, 
                           protected_attribute: str, 
                           biased_columns: list, 
                           metrics: dict, 
                           severities: dict) -> Dict:
        return {
            "protected_attribute": protected_attribute,
            "biased_columns": biased_columns,
            "true_metric_values": metrics,
            "violation_severities": severities
        }
from typing import Tuple, Dict
import pandas as pd
from server.data.hiring import HiringScenario
from server.data.loan import LoanScenario
from server.data.medical import MedicalScenario

class DataGenerator:
    def __init__(self):
        self.scenarios = {
            "hiring": HiringScenario(),
            "loan": LoanScenario(),
            "medical": MedicalScenario()
        }

    def generate(self, scenario_name: str, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario requested: {scenario_name}")
            
        return self.scenarios[scenario_name].generate(seed=seed, bias_magnitude=bias_magnitude)
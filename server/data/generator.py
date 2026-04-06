from typing import Tuple, Dict
import pandas as pd
from server.data.loan import LoanScenario

class DataGenerator:
    def __init__(self):
        self.scenario = LoanScenario()

    def generate(self, scenario_name: str, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        return self.scenario.generate(seed=seed, bias_magnitude=bias_magnitude)
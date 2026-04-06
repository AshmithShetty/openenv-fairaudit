import pandas as pd
import numpy as np
from typing import Tuple, Dict
from server.data.base import BaseScenario
from server.fairness.metrics import calculate_historical_fairness

class LoanScenario(BaseScenario):
    def generate(self, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        rng = np.random.RandomState(seed)
        n_samples = self.MAX_ROWS
        
        race_latent = rng.choice([0, 1], size=n_samples, p=[0.65, 0.35])
        
        zip_maj = ['75001', '75002', '75003']
        zip_min = ['75004', '75005', '75006']
        zip_code = np.where(
            race_latent == 0,
            rng.choice(zip_maj + zip_min, size=n_samples, p=[0.3, 0.3, 0.3, 0.033, 0.033, 0.034]),
            rng.choice(zip_maj + zip_min, size=n_samples, p=[0.05, 0.05, 0.05, 0.283, 0.283, 0.284])
        )
        
        base_income = rng.lognormal(mean=11.0, sigma=0.5, size=n_samples)
        income_penalty = race_latent * (15000 * bias_magnitude)
        annual_income = np.maximum(20000, base_income - income_penalty).round(-3)
        
        base_fico = 600 + (annual_income / 1000) * 1.5 + rng.normal(0, 30, size=n_samples)
        fico_score = np.clip(base_fico, 300, 850).astype(int)
        
        debt_to_income_ratio = rng.uniform(0.1, 0.6, size=n_samples)
        employment_length = rng.randint(0, 15, size=n_samples)
        home_ownership = rng.choice(['MORTGAGE', 'RENT', 'OWN'], size=n_samples, p=[0.4, 0.5, 0.1])
        education_level = rng.choice(['Bachelors', 'High School', 'Masters', 'Associate'], size=n_samples)
        
        approval_score = (fico_score * 0.5) - (debt_to_income_ratio * 500) + (employment_length * 5)
        
        threshold = np.percentile(approval_score, 60)
        loan_approved = (approval_score >= threshold).astype(int)
        
        df = pd.DataFrame({
            'annual_income': annual_income,
            'debt_to_income_ratio': debt_to_income_ratio.round(2),
            'fico_score': fico_score,
            'employment_length': employment_length,
            'home_ownership': home_ownership,
            'zip_code': zip_code,
            'education_level': education_level,
            'loan_approved': loan_approved,
            '_latent_race': race_latent
        })
        
        race_metrics = calculate_historical_fairness(df, '_latent_race', 'loan_approved', priv_value=0, pos_outcome=1)
        
        severities = {
            "disparate_impact_race": "HIGH" if race_metrics["disparate_impact"] < 0.8 else "LOW"
        }
        
        df = df.drop(columns=['_latent_race'])
        
        ground_truth = self._seal_ground_truth(
            df=df,
            protected_attribute="race",
            biased_columns=["zip_code", "annual_income", "fico_score"],
            metrics={"race_metrics": race_metrics},
            severities=severities
        )
        
        return df, ground_truth
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from server.data.base import BaseScenario
from server.fairness.metrics import calculate_historical_fairness

class MedicalScenario(BaseScenario):
    def generate(self, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        rng = np.random.RandomState(seed)
        n_samples = self.MAX_ROWS
        
        # 1. Latent Protected Attribute (Race: 0=Majority, 1=Minority)
        race_latent = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # 2. True Health Need (Minority actually has slightly higher true need in this scenario)
        comorbidity_count = rng.poisson(lam=1.5 + (race_latent * 0.5 * bias_magnitude), size=n_samples)
        vitals_score = rng.normal(loc=60 - (comorbidity_count * 2), scale=10, size=n_samples)
        vitals_score = np.clip(vitals_score, 0, 100).astype(int)
        
        # 3. Biased Proxy Variable (Cost of prior care)
        # Despite higher actual need, minority group has LOWER historical cost
        base_cost = 500 + (comorbidity_count * 1500)
        cost_penalty = race_latent * (1200 * bias_magnitude)
        cost_of_care_prior_year = np.maximum(0, base_cost - cost_penalty + rng.normal(0, 500, size=n_samples))
        
        # 4. Standard Features
        age = rng.normal(loc=65, scale=12, size=n_samples).astype(int)
        age = np.clip(age, 18, 95)
        insurance_type = rng.choice(['Medicare', 'Private', 'Medicaid', 'Uninsured'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
        primary_language = np.where(
            race_latent == 0,
            rng.choice(['English', 'Spanish', 'Other'], size=n_samples, p=[0.9, 0.05, 0.05]),
            rng.choice(['English', 'Spanish', 'Other'], size=n_samples, p=[0.6, 0.3, 0.1])
        )
        zip_code = rng.randint(10000, 99999, size=n_samples).astype(str)
        
        # 5. Target Variable (Escalated to ICU / High Risk Care)
        # The algorithm makes decisions heavily weighted by historical cost
        approval_score = (cost_of_care_prior_year * 0.6) + (vitals_score * 0.4)
        
        # Top 20% are escalated
        threshold = np.percentile(approval_score, 80)
        escalated_to_icu = (approval_score >= threshold).astype(int)
        
        df = pd.DataFrame({
            'age': age,
            'vitals_score': vitals_score,
            'insurance_type': insurance_type,
            'primary_language': primary_language,
            'zip_code': zip_code,
            'comorbidity_count': comorbidity_count,
            'cost_of_care_prior_year': cost_of_care_prior_year.round(2),
            'escalated_to_icu': escalated_to_icu,
            '_latent_race': race_latent
        })
        
        # 6. Compute Ground Truth Metrics
        race_metrics = calculate_historical_fairness(df, '_latent_race', 'escalated_to_icu', priv_value=0, pos_outcome=1)
        
        severities = {
            "disparate_impact_race": "HIGH" if race_metrics["disparate_impact"] < 0.8 else "LOW"
        }
        
        df = df.drop(columns=['_latent_race'])
        
        ground_truth = self._seal_ground_truth(
            df=df,
            protected_attribute="race",
            biased_columns=["cost_of_care_prior_year"],
            metrics={"race_metrics": race_metrics},
            severities=severities
        )
        
        return df, ground_truth
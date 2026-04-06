import pandas as pd
import numpy as np
from typing import Tuple, Dict
from server.data.base import BaseScenario
from server.fairness.metrics import calculate_cramers_v, calculate_historical_fairness

class HiringScenario(BaseScenario):
    def generate(self, seed: int, bias_magnitude: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        rng = np.random.RandomState(seed)
        n_samples = self.MAX_ROWS
        
        # 1. Latent Protected Attributes (Not exposed to the agent)
        # Race: 0 (Majority/Privileged), 1 (Minority/Unprivileged)
        race_latent = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        # Gender: 0 (Male/Privileged), 1 (Female/Unprivileged)
        gender_latent = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        
        # 2. Proxy Variables Generation
        zip_codes_maj = ['90210', '10012', '60614']
        zip_codes_min = ['90059', '10453', '60621']
        
        zip_code = np.where(
            race_latent == 0,
            rng.choice(zip_codes_maj + zip_codes_min, size=n_samples, p=[0.28, 0.28, 0.29, 0.05, 0.05, 0.05]),
            rng.choice(zip_codes_maj + zip_codes_min, size=n_samples, p=[0.05, 0.05, 0.05, 0.28, 0.28, 0.29])
        )
        
        first_names_m = ['James', 'John', 'Robert', 'Michael', 'William']
        first_names_f = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth']
        unisex_names = ['Taylor', 'Jordan', 'Casey', 'Riley', 'Peyton']
        
        name_gender_proxy = np.where(
            gender_latent == 0,
            rng.choice(first_names_m + unisex_names, size=n_samples, p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04]),
            rng.choice(first_names_f + unisex_names, size=n_samples, p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04])
        )
        
        # 3. Legitimate Features
        age = rng.normal(loc=35, scale=10, size=n_samples).astype(int)
        age = np.clip(age, 22, 65)
        education = rng.choice(['Bachelors', 'Masters', 'PhD', 'High School'], size=n_samples, p=[0.5, 0.3, 0.05, 0.15])
        job_level = rng.choice(['Entry', 'Mid', 'Senior'], size=n_samples, p=[0.4, 0.4, 0.2])
        years_experience = np.maximum(0, age - 22 + rng.normal(0, 2, size=n_samples)).astype(int)
        
        # 4. Target Variable Generation (with injected bias)
        base_score = 50 + (years_experience * 2) + rng.normal(0, 10, size=n_samples)
        
        # Inject bias magnitude penalty against unprivileged groups
        race_penalty = race_latent * (15 * bias_magnitude)
        gender_penalty = gender_latent * (10 * bias_magnitude)
        
        interview_score = base_score - race_penalty - gender_penalty
        interview_score = np.clip(interview_score, 0, 100)
        
        # Top 30% are hired
        threshold = np.percentile(interview_score, 70)
        hired = (interview_score >= threshold).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'zip_code': zip_code,
            'first_name': name_gender_proxy,
            'education': education,
            'job_level': job_level,
            'years_experience': years_experience,
            'interview_score': interview_score.round(1),
            'hired': hired,
            '_latent_race': race_latent,
            '_latent_gender': gender_latent
        })
        
        # 5. Compute Ground Truth Metrics BEFORE dropping latent columns
        race_metrics = calculate_historical_fairness(df, '_latent_race', 'hired', priv_value=0, pos_outcome=1)
        gender_metrics = calculate_historical_fairness(df, '_latent_gender', 'hired', priv_value=0, pos_outcome=1)
        
        # Determine severities based on standard legal thresholds (DI < 0.8 is a violation)
        severities = {
            "disparate_impact_race": "HIGH" if race_metrics["disparate_impact"] < 0.6 else ("MEDIUM" if race_metrics["disparate_impact"] < 0.8 else "LOW"),
            "disparate_impact_gender": "HIGH" if gender_metrics["disparate_impact"] < 0.6 else ("MEDIUM" if gender_metrics["disparate_impact"] < 0.8 else "LOW")
        }
        
        # 6. Drop latent columns to seal the dataset
        df = df.drop(columns=['_latent_race', '_latent_gender'])
        
        ground_truth = self._seal_ground_truth(
            df=df,
            protected_attribute="race", 
            biased_columns=["zip_code", "first_name"],
            metrics={"race_metrics": race_metrics, "gender_metrics": gender_metrics},
            severities=severities
        )
        
        return df, ground_truth
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0:
        return 0.0
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    if n <= 1 or min(k-1, r-1) == 0:
        return 0.0
        
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    denominator = min((kcorr-1), (rcorr-1))
    if denominator <= 0:
        return 0.0
        
    return float(np.sqrt(phi2corr / denominator))

def calculate_historical_fairness(df: pd.DataFrame, protected_col: str, target_col: str, priv_value: int, pos_outcome: int) -> dict:
    priv_mask = (df[protected_col] == priv_value)
    unpriv_mask = (df[protected_col] != priv_value)
    
    priv_total = priv_mask.sum()
    unpriv_total = unpriv_mask.sum()
    
    if priv_total == 0 or unpriv_total == 0:
        return {"disparate_impact": 1.0, "demographic_parity": 0.0}
        
    priv_pos = (df[priv_mask][target_col] == pos_outcome).sum()
    unpriv_pos = (df[unpriv_mask][target_col] == pos_outcome).sum()
    
    priv_rate = priv_pos / priv_total
    unpriv_rate = unpriv_pos / unpriv_total
    
    di = unpriv_rate / priv_rate if priv_rate > 0 else 1.0
    dp = unpriv_rate - priv_rate
    
    return {
        "disparate_impact": float(di),
        "demographic_parity": float(dp)
    }
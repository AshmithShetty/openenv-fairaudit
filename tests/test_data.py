import pytest
import pandas as pd
from server.data.generator import DataGenerator

@pytest.fixture
def generator():
    return DataGenerator()

@pytest.mark.parametrize("scenario", ["hiring", "loan", "medical"])
def test_scenario_generation_and_keys(generator, scenario):
    df, ground_truth = generator.generate(scenario, seed=42)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    
    assert "protected_attribute" in ground_truth
    assert "biased_columns" in ground_truth
    assert "true_metric_values" in ground_truth
    assert "violation_severities" in ground_truth

@pytest.mark.parametrize("scenario", ["hiring", "loan", "medical"])
def test_strict_determinism(generator, scenario):
    df1, gt1 = generator.generate(scenario, seed=100)
    df2, gt2 = generator.generate(scenario, seed=100)
    df3, gt3 = generator.generate(scenario, seed=101)
    
    pd.testing.assert_frame_equal(df1, df2)
    assert gt1 == gt2
    
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)

def test_hiring_metrics_validity(generator):
    _, ground_truth = generator.generate("hiring", seed=42)
    
    race_di = ground_truth["true_metric_values"]["race_metrics"]["disparate_impact"]
    gender_di = ground_truth["true_metric_values"]["gender_metrics"]["disparate_impact"]
    
    assert 0.0 <= race_di <= 1.0
    assert 0.0 <= gender_di <= 1.0
    assert "zip_code" in ground_truth["biased_columns"]
    assert "first_name" in ground_truth["biased_columns"]
"""
Unit tests for the feature engineering step in model2_loan_default_prediction.py.

Covers the two engineered features:
    Income_per_Emp_Year    = person_income / (person_emp_exp + 1)
    Credit_History_per_Age = cb_person_cred_hist_length / (person_age + 1)

These formulas are re-implemented here (rather than importing the main
script) because model2_loan_default_prediction.py is a top-to-bottom
script that loads loan_data.csv and trains models on import — not
suitable for isolated, fast unit tests. The formulas below mirror
lines 87 and 90 of model2_loan_default_prediction.py exactly.
"""

import numpy as np
import pandas as pd
import pytest


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the feature engineering step (Section 3) of model2_loan_default_prediction.py."""
    df = df.copy()
    df['Income_per_Emp_Year'] = df['person_income'] / (df['person_emp_exp'] + 1)
    df['Credit_History_per_Age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
    return df


@pytest.fixture
def sample_df():
    """3-row fake dataset with known expected results, including edge cases
    person_emp_exp = 0 (row 2) and person_age = 0 (row 2)."""
    return pd.DataFrame({
        'person_income': [50000, 60000, 90000],
        'person_emp_exp': [4, 0, 9],
        'cb_person_cred_hist_length': [5, 10, 2],
        'person_age': [4, 0, 19],
    })


def test_income_per_emp_year_correct_values(sample_df):
    result = add_engineered_features(sample_df)
    expected = [50000 / 5, 60000 / 1, 90000 / 10]  # [10000.0, 60000.0, 9000.0]
    assert result['Income_per_Emp_Year'].tolist() == pytest.approx(expected)


def test_credit_history_per_age_correct_values(sample_df):
    result = add_engineered_features(sample_df)
    expected = [5 / 5, 10 / 1, 2 / 20]  # [1.0, 10.0, 0.1]
    assert result['Credit_History_per_Age'].tolist() == pytest.approx(expected)


def test_income_per_emp_year_zero_emp_exp_no_zero_division(sample_df):
    """Row 2 has person_emp_exp = 0. The +1 in the denominator must prevent
    a division-by-zero error and produce a finite, correct value."""
    result = add_engineered_features(sample_df)
    value = result.loc[1, 'Income_per_Emp_Year']
    assert np.isfinite(value)
    assert value == pytest.approx(60000 / 1)


def test_credit_history_per_age_zero_age_no_zero_division(sample_df):
    """Row 2 has person_age = 0. The +1 in the denominator must prevent
    a division-by-zero error and produce a finite, correct value."""
    result = add_engineered_features(sample_df)
    value = result.loc[1, 'Credit_History_per_Age']
    assert np.isfinite(value)
    assert value == pytest.approx(10 / 1)


def test_no_inf_or_nan_in_engineered_columns(sample_df):
    """Across the whole fake dataset, neither engineered column should ever
    contain inf or NaN, confirming the +1 offset is safe in general, not
    just for the two hand-picked edge-case rows."""
    result = add_engineered_features(sample_df)
    engineered = result[['Income_per_Emp_Year', 'Credit_History_per_Age']]
    assert not np.isinf(engineered.to_numpy()).any()
    assert not engineered.isna().any().any()

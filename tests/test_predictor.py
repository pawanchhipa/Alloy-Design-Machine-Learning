import pytest
import numpy as np
import pandas as pd
from alloy_ml_prediction import AlloyPredictor

@pytest.fixture
def predictor():
    return AlloyPredictor()

@pytest.fixture
def sample_data():
    # Create a small sample dataset for testing
    X = pd.DataFrame({
        'C1': [0.1, 0.2, 0.3, 0.4],
        'C2': [0.2, 0.3, 0.4, 0.5],
        'C3': [0.7, 0.5, 0.3, 0.1]
    })
    y = pd.Series([100, 150, 200, 250])
    return X, y

def test_predictor_initialization(predictor):
    assert predictor is not None
    assert predictor.model is None

def test_analytical_solution(predictor, sample_data):
    X, y = sample_data
    coefficients = predictor.analytical_solution(X, y)
    assert isinstance(coefficients, np.ndarray)
    assert len(coefficients) == X.shape[1] + 1  # +1 for bias term

def test_gradient_descent(predictor, sample_data):
    X, y = sample_data
    coefficients = predictor.gradient_descent(X, y)
    assert isinstance(coefficients, np.ndarray)
    assert len(coefficients) == X.shape[1] + 1  # +1 for bias term

def test_kfold_analysis(predictor, sample_data):
    X, y = sample_data
    results = predictor.perform_kfold_analysis(X, y, n_splits=2)
    
    assert 'rmse_train' in results
    assert 'rmse_test' in results
    assert 'r2_train' in results
    assert 'r2_test' in results
    assert 'fold_results' in results
    
    assert len(results['fold_results']) == 2  # As we specified n_splits=2

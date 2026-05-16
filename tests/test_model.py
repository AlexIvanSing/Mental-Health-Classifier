import pytest
import tempfile
import os
import yaml
from xgboost import XGBClassifier
from src.model import build_model

# --- FIXTURES ---

@pytest.fixture
def temp_config():
    config = {
        "model": {
            "n_estimators": 10,  # pequeño para que el test corra rapido
            "max_depth": 3,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 42,
            "scale_pos_weight": 0.93
        }
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        yield path
    finally:
        os.remove(path)

@pytest.fixture
def sample_config(temp_config):
    import yaml
    with open(temp_config, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def synthetic_dataset():
    """Dataset sintetico pequeño para pruebas de comportamiento."""
    import numpy as np
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)
    return X, y

# --- PRUEBAS build_model ---

def test_build_model_returns_xgb_instance(sample_config):
    model = build_model(sample_config)
    assert isinstance(model, XGBClassifier)

def test_build_model_n_estimators(sample_config):
    model = build_model(sample_config)
    assert model.n_estimators == 10

def test_build_model_max_depth(sample_config):
    model = build_model(sample_config)
    assert model.max_depth == 3

def test_build_model_random_state(sample_config):
    model = build_model(sample_config)
    assert model.random_state == 42

def test_build_model_eval_metric(sample_config):
    model = build_model(sample_config)
    assert model.eval_metric == "auc"

# --- PRUEBAS comportamiento fit + predict_proba ---

def test_model_fit_does_not_crash(sample_config, synthetic_dataset):
    X, y = synthetic_dataset
    model = build_model(sample_config)
    model.fit(X, y)

def test_predict_proba_returns_two_columns(sample_config, synthetic_dataset):
    X, y = synthetic_dataset
    model = build_model(sample_config)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape[1] == 2

def test_predict_proba_values_between_0_and_1(sample_config, synthetic_dataset):
    X, y = synthetic_dataset
    model = build_model(sample_config)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0

def test_predict_proba_rows_sum_to_1(sample_config, synthetic_dataset):
    import numpy as np
    X, y = synthetic_dataset
    model = build_model(sample_config)
    model.fit(X, y)
    proba = model.predict_proba(X)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0)

def test_predict_returns_binary_labels(sample_config, synthetic_dataset):
    X, y = synthetic_dataset
    model = build_model(sample_config)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

def test_same_random_state_same_predictions(sample_config, synthetic_dataset):
    import numpy as np
    X, y = synthetic_dataset
    model1 = build_model(sample_config)
    model2 = build_model(sample_config)
    model1.fit(X, y)
    model2.fit(X, y)
    assert np.array_equal(model1.predict(X), model2.predict(X))

# --- PRUEBAS importlib + backward compat ---

def test_build_model_without_class_key():
    config = {
        "model": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 0,
            "scale_pos_weight": 1.0,
        }
    }
    model = build_model(config)
    assert isinstance(model, XGBClassifier)

def test_build_model_with_explicit_class_key():
    config = {
        "model": {
            "class": "xgboost.XGBClassifier",
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 0,
            "scale_pos_weight": 1.0,
        }
    }
    model = build_model(config)
    assert isinstance(model, XGBClassifier)

def test_early_stopping_rounds_not_passed_to_constructor():
    config = {
        "model": {
            "class": "xgboost.XGBClassifier",
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 0,
            "scale_pos_weight": 1.0,
            "early_stopping_rounds": 30,
        }
    }
    model = build_model(config)
    assert model.early_stopping_rounds is None
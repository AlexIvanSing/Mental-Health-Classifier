"""
Shared pytest fixtures across the test suite.

Anything fixture-shaped that more than one test module needs (toy pipeline
config, a trained-and-serialized pipeline, a synthetic input CSV, a full
config dict with all `paths` keys pointed at tmp_path) lives here so the
individual test files stay focused.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipeline import build_pipeline


@pytest.fixture
def pipeline_config():
    """Minimal config dict for building a fast toy pipeline."""
    return {
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "max_features": 50,
        },
        "model": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 42,
            "scale_pos_weight": 1.0,
        },
    }


@pytest.fixture
def trained_pipeline(pipeline_config):
    """Fit a tiny pipeline on a toy corpus so it's ready for inference."""
    X = [
        "i want to end my life",
        "i cannot take this anymore",
        "today is a beautiful day",
        "what a wonderful weekend",
        "everything hurts and feels empty",
        "i love spending time with friends",
    ]
    y = np.array([1, 1, 0, 0, 1, 0])
    pipe = build_pipeline(pipeline_config)
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def model_on_disk(trained_pipeline, tmp_path):
    """Serialize the trained pipeline to a temp .joblib path."""
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    return str(model_path)


@pytest.fixture
def input_csv(tmp_path):
    """Synthetic labeled CSV matching the project's expected schema."""
    df = pd.DataFrame({
        "user_id":    ["u1", "u2", "u3"],
        "text_id":    ["t1", "t2", "t3"],
        "title":      ["help me", "good morning", "the void"],
        "text":       ["I cannot go on", "I love today", "darkness inside"],
        "is_suicide": ["yes", "no", "yes"],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def unlabeled_input_csv(tmp_path):
    """CSV missing the target column — used to test evaluate's error path."""
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "text_id": ["t1", "t2"],
        "title":   ["a", "b"],
        "text":    ["foo", "bar"],
    })
    path = tmp_path / "unlabeled.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def full_config(model_on_disk, tmp_path):
    """
    Full config dict with every path under `paths` rooted at tmp_path
    so tests stay hermetic and never touch the real reports/ folder.
    """
    reports_dir = tmp_path / "reports"
    return {
        "data": {
            "expected_columns": ["user_id", "text_id", "title", "text", "is_suicide"],
            "text_columns":     ["title", "text"],
            "target_column":    "is_suicide",
            "value_true":       "yes",
            "value_false":      "no",
        },
        "paths": {
            "model_output":        model_on_disk,
            "reports_dir":         str(reports_dir),
            "roc_val_output":      str(reports_dir / "roc_val.png"),
            "cm_val_output":       str(reports_dir / "cm_val.png"),
            "metrics_val_output":  str(reports_dir / "metrics_val.json"),
            "report_val_output":   str(reports_dir / "training_report.md"),
            "roc_test_output":     str(reports_dir / "roc_test.png"),
            "cm_test_output":      str(reports_dir / "cm_test.png"),
            "metrics_test_output": str(reports_dir / "metrics_test.json"),
            "report_test_output":  str(reports_dir / "final_evaluation.md"),
        },
    }


@pytest.fixture
def config_file(full_config, tmp_path):
    """Serialize full_config to disk as YAML so CLI wrappers can load it."""
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(full_config, f)
    return str(path)

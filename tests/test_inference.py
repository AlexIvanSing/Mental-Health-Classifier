import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import yaml
import pytest

from src.inference import load_pipeline, predict, run_inference, run_inference_cli
from src.pipeline import build_pipeline


# --- shared fixtures ---

@pytest.fixture
def pipeline_config():
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
    """Synthetic CSV matching the project's expected schema."""
    df = pd.DataFrame({
        "user_id":   ["u1", "u2", "u3"],
        "text_id":   ["t1", "t2", "t3"],
        "title":     ["help me", "good morning", "the void"],
        "text":      ["I cannot go on", "I love today", "darkness inside"],
        "is_suicide": ["yes", "no", "yes"],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def full_config(model_on_disk):
    return {
        "data": {
            "expected_columns": ["user_id", "text_id", "title", "text", "is_suicide"],
            "text_columns":     ["title", "text"],
        },
        "paths": {
            "model_output": model_on_disk,
        },
    }


@pytest.fixture
def config_file(full_config, tmp_path):
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(full_config, f)
    return str(path)


# --- load_pipeline ---

def test_load_pipeline_returns_fitted_object(model_on_disk):
    pipe = load_pipeline(model_on_disk)
    # smoke-check it can predict (i.e. is actually fitted)
    preds = pipe.predict(["hello world"])
    assert len(preds) == 1


def test_load_pipeline_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pipeline("nonexistent_model.joblib")


# --- predict ---

def test_predict_returns_two_lists(trained_pipeline):
    preds, probas = predict(trained_pipeline, ["sample text"])
    assert isinstance(preds, list)
    assert isinstance(probas, list)


def test_predict_lengths_match_input(trained_pipeline):
    texts = ["a", "b", "c", "d"]
    preds, probas = predict(trained_pipeline, texts)
    assert len(preds) == len(texts)
    assert len(probas) == len(texts)


def test_predict_returns_binary_labels(trained_pipeline):
    preds, _ = predict(trained_pipeline, ["hello", "world"])
    assert set(preds).issubset({0, 1})


def test_predict_probabilities_in_range(trained_pipeline):
    _, probas = predict(trained_pipeline, ["foo", "bar", "baz"])
    assert all(0.0 <= p <= 1.0 for p in probas)


# --- run_inference (full flow with dict config) ---

def test_run_inference_writes_output_csv(full_config, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    run_inference(input_csv, str(out), full_config)
    assert out.exists()


def test_run_inference_output_has_expected_columns(full_config, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    run_inference(input_csv, str(out), full_config)
    df = pd.read_csv(out)
    assert list(df.columns) == ["text_id", "prediction", "probability"]


def test_run_inference_output_row_count_matches_input(full_config, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    run_inference(input_csv, str(out), full_config)
    df = pd.read_csv(out)
    assert len(df) == 3


def test_run_inference_predictions_are_binary(full_config, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    run_inference(input_csv, str(out), full_config)
    df = pd.read_csv(out)
    assert set(df["prediction"].tolist()).issubset({0, 1})


def test_run_inference_probabilities_in_range(full_config, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    run_inference(input_csv, str(out), full_config)
    df = pd.read_csv(out)
    assert df["probability"].between(0.0, 1.0).all()


# --- run_inference_cli (wrapper that loads YAML path) ---

def test_run_inference_cli_end_to_end(config_file, input_csv, tmp_path):
    out = tmp_path / "preds.csv"
    result = run_inference_cli(input_csv, str(out), config_file)
    assert out.exists()
    assert len(result) == 3
    assert list(result.columns) == ["text_id", "prediction", "probability"]

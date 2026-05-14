import pandas as pd
import pytest

from src.inference import load_pipeline, predict, run_inference, run_inference_cli

# Fixtures (pipeline_config, trained_pipeline, model_on_disk, input_csv,
# full_config, config_file) come from tests/conftest.py.


# --- load_pipeline ---

def test_load_pipeline_returns_fitted_object(model_on_disk):
    pipe = load_pipeline(model_on_disk)
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

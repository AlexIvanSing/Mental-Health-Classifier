# ============================================================================
# Tests — CLI de evaluación
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import json
import os

import pytest

from src.evaluate_cli import run_evaluation, run_evaluation_cli


EXPECTED_METRIC_KEYS = {
    "TP", "TN", "FP", "FN", "TPR", "FPR",
    "AUC", "precision", "recall", "F1",
}


# --- run_evaluation (dict config) ---

def test_run_evaluation_returns_metrics_dict_with_expected_keys(full_config, input_csv):
    metrics = run_evaluation(input_csv, full_config)
    assert isinstance(metrics, dict)
    assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())


def test_run_evaluation_writes_roc_png(full_config, input_csv):
    run_evaluation(input_csv, full_config)
    roc_path = full_config["paths"]["roc_test_output"]
    assert os.path.exists(roc_path)
    assert os.path.getsize(roc_path) > 0


def test_run_evaluation_writes_cm_png(full_config, input_csv):
    run_evaluation(input_csv, full_config)
    cm_path = full_config["paths"]["cm_test_output"]
    assert os.path.exists(cm_path)
    assert os.path.getsize(cm_path) > 0


def test_run_evaluation_writes_metrics_json(full_config, input_csv):
    run_evaluation(input_csv, full_config)
    metrics_path = full_config["paths"]["metrics_test_output"]
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        data = json.load(f)
    assert EXPECTED_METRIC_KEYS.issubset(data.keys())


def test_run_evaluation_writes_markdown_report(full_config, input_csv):
    run_evaluation(input_csv, full_config)
    report_path = full_config["paths"]["report_test_output"]
    assert os.path.exists(report_path)
    content = open(report_path, encoding="utf-8").read()
    # Sanity: report mentions both metrics and embedded plots.
    assert "AUC" in content
    assert "Matriz de confusión" in content
    assert "Curva ROC" in content
    assert "![ROC Curve]" in content
    assert "![Confusion Matrix]" in content


def test_run_evaluation_metric_values_in_valid_range(full_config, input_csv):
    metrics = run_evaluation(input_csv, full_config)
    assert 0.0 <= metrics["AUC"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["F1"] <= 1.0


def test_run_evaluation_raises_if_target_column_missing(full_config, unlabeled_input_csv):
    # ingestion() returns None when the schema doesn't match; run_evaluation
    # must surface that as a ValueError instead of crashing on a None deref.
    with pytest.raises(ValueError, match="target column"):
        run_evaluation(unlabeled_input_csv, full_config)


# --- run_evaluation_cli (YAML path wrapper) ---

def test_run_evaluation_cli_end_to_end(config_file, input_csv, full_config):
    metrics = run_evaluation_cli(input_csv, config_file)
    assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())
    assert os.path.exists(full_config["paths"]["roc_test_output"])
    assert os.path.exists(full_config["paths"]["cm_test_output"])
    assert os.path.exists(full_config["paths"]["metrics_test_output"])

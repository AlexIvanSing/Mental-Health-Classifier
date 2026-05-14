import os
import tempfile
import numpy as np
import pytest
from src.evaluation import (
    compute_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    evaluate,
)


# --- fixtures: hand-computed ground truth ---
#
# 10 samples — labels and predictions chosen so we can compute the
# confusion matrix by hand:
#
#   y_true: 1 1 1 1 1 0 0 0 0 0
#   y_pred: 1 1 1 0 0 0 0 0 1 1
#
#   TP = 3 (idx 0,1,2)
#   FN = 2 (idx 3,4)
#   TN = 3 (idx 5,6,7)
#   FP = 2 (idx 8,9)

@pytest.fixture
def y_true():
    return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


@pytest.fixture
def y_pred():
    return np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])


@pytest.fixture
def y_proba():
    return np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.1, 0.2, 0.15, 0.6, 0.55])


# --- compute_metrics ---

def test_compute_metrics_returns_dict(y_true, y_pred, y_proba):
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert isinstance(metrics, dict)


def test_compute_metrics_has_expected_keys(y_true, y_pred, y_proba):
    metrics = compute_metrics(y_true, y_pred, y_proba)
    expected = {"TP", "TN", "FP", "FN", "TPR", "FPR", "AUC", "precision", "recall", "F1"}
    assert expected.issubset(metrics.keys())


def test_confusion_counts_match_manual_calculation(y_true, y_pred, y_proba):
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["TP"] == 3
    assert metrics["FN"] == 2
    assert metrics["TN"] == 3
    assert metrics["FP"] == 2


def test_tpr_matches_manual_calculation(y_true, y_pred, y_proba):
    # TPR = TP / (TP + FN) = 3 / 5 = 0.6
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["TPR"] == pytest.approx(0.6, abs=1e-4)


def test_fpr_matches_manual_calculation(y_true, y_pred, y_proba):
    # FPR = FP / (FP + TN) = 2 / 5 = 0.4
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["FPR"] == pytest.approx(0.4, abs=1e-4)


def test_precision_matches_manual_calculation(y_true, y_pred, y_proba):
    # precision = TP / (TP + FP) = 3 / 5 = 0.6
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == pytest.approx(0.6, abs=1e-4)


def test_recall_equals_tpr(y_true, y_pred, y_proba):
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["recall"] == pytest.approx(metrics["TPR"], abs=1e-4)


def test_f1_matches_manual_calculation(y_true, y_pred, y_proba):
    # F1 = 2 * P * R / (P + R) = 2 * 0.6 * 0.6 / 1.2 = 0.6
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["F1"] == pytest.approx(0.6, abs=1e-4)


def test_auc_in_valid_range(y_true, y_pred, y_proba):
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["AUC"] <= 1.0


def test_perfect_predictions_give_auc_one():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["AUC"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["F1"] == pytest.approx(1.0, abs=1e-4)


# --- plotting smoke tests ---

def test_plot_roc_curve_creates_file(y_true, y_proba):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "roc.png")
        plot_roc_curve(y_true, y_proba, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_plot_confusion_matrix_creates_file(y_true, y_pred):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cm.png")
        plot_confusion_matrix(y_true, y_pred, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_evaluate_returns_metrics_and_writes_plots(y_true, y_pred, y_proba):
    with tempfile.TemporaryDirectory() as tmp:
        roc = os.path.join(tmp, "roc.png")
        cm = os.path.join(tmp, "cm.png")
        metrics = evaluate(y_true, y_pred, y_proba, roc_path=roc, cm_path=cm)
        assert isinstance(metrics, dict)
        assert metrics["TP"] == 3
        assert os.path.exists(roc)
        assert os.path.exists(cm)

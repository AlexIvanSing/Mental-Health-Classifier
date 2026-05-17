# ============================================================================
# Tests — Evaluación del modelo
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import os
import tempfile
import numpy as np
import pytest
from src.evaluation import (
    compute_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    evaluate,
    generate_report,
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


# --- generate_report ---

def _full_metrics_fixture():
    return {
        "TP": 87, "TN": 86, "FP": 36, "FN": 43,
        "TPR": 0.6692, "FPR": 0.2951,
        "AUC": 0.739, "precision": 0.7073, "recall": 0.6692, "F1": 0.6877,
        "cv_auc_mean": 0.7596, "cv_auc_std": 0.0275,
        "cv_fold_scores": [0.81, 0.75, 0.75, 0.75, 0.73],
    }


def test_generate_report_creates_markdown_file(tmp_path):
    out = tmp_path / "report.md"
    generate_report(
        _full_metrics_fixture(),
        roc_path=str(tmp_path / "roc.png"),
        cm_path=str(tmp_path / "cm.png"),
        output_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_report_contains_metric_values(tmp_path):
    out = tmp_path / "report.md"
    generate_report(
        _full_metrics_fixture(),
        roc_path=str(tmp_path / "roc.png"),
        cm_path=str(tmp_path / "cm.png"),
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    for key in ("0.739", "0.6877", "0.7073", "0.6692", "0.2951"):
        assert key in content
    for cm_value in ("87", "86", "36", "43"):
        assert cm_value in content


def test_generate_report_embeds_plot_paths_as_relative(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    out = reports_dir / "report.md"
    roc = reports_dir / "roc.png"
    cm  = reports_dir / "cm.png"
    generate_report(
        _full_metrics_fixture(),
        roc_path=str(roc),
        cm_path=str(cm),
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    # PNGs sit next to the report -> markdown should use plain filenames, not absolute paths.
    assert "![ROC Curve](roc.png)" in content
    assert "![Confusion Matrix](cm.png)" in content


def test_generate_report_includes_cv_section_when_folds_present(tmp_path):
    out = tmp_path / "report.md"
    generate_report(
        _full_metrics_fixture(),
        roc_path=str(tmp_path / "roc.png"),
        cm_path=str(tmp_path / "cm.png"),
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    assert "Validación cruzada" in content
    assert "Fold 1" in content
    assert "Fold 5" in content


def test_generate_report_omits_cv_section_when_no_folds(tmp_path):
    metrics = _full_metrics_fixture()
    metrics.pop("cv_fold_scores")
    metrics.pop("cv_auc_mean")
    metrics.pop("cv_auc_std")
    out = tmp_path / "report.md"
    generate_report(
        metrics,
        roc_path=str(tmp_path / "roc.png"),
        cm_path=str(tmp_path / "cm.png"),
        output_path=str(out),
    )
    content = out.read_text(encoding="utf-8")
    assert "Validación cruzada" not in content
    # The other sections still render.
    assert "Matriz de confusión" in content
    assert "Curva ROC" in content


def test_generate_report_creates_output_directory_if_missing(tmp_path):
    out = tmp_path / "nested" / "sub" / "report.md"
    generate_report(
        _full_metrics_fixture(),
        roc_path=str(tmp_path / "roc.png"),
        cm_path=str(tmp_path / "cm.png"),
        output_path=str(out),
    )
    assert out.exists()

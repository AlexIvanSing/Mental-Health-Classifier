import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -> no Tk/Qt dependency
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Compute classification metrics from predictions.

    Parameters
    ----------
    y_true  : array-like — ground truth binary labels (0/1)
    y_pred  : array-like — binary predictions at threshold 0.5
    y_proba : array-like — predicted probabilities for the positive class

    Returns
    -------
    dict with keys: TP, TN, FP, FN, TPR, FPR, AUC, precision, recall, F1
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # fall-out

    return {
        "TP":        int(tp),
        "TN":        int(tn),
        "FP":        int(fp),
        "FN":        int(fn),
        "TPR":       round(tpr, 4),
        "FPR":       round(fpr, 4),
        "AUC":       round(roc_auc_score(y_true, y_proba), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def plot_roc_curve(y_true, y_proba, output_path: str = "reports/roc_curve.png") -> None:
    """
    Generate and save a ROC curve plot.

    Parameters
    ----------
    y_true      : array-like — ground truth binary labels
    y_proba     : array-like — predicted probabilities for the positive class
    output_path : str — path to save the PNG
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="#01696f", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#bab9b4", lw=1, linestyle="--", label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved -> {output_path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list = None,
    output_path: str = "reports/confusion_matrix.png",
) -> None:
    """
    Generate and save a labeled confusion matrix plot.

    Parameters
    ----------
    y_true      : array-like — ground truth binary labels
    y_pred      : array-like — binary predictions
    labels      : list of str — display names for classes (default: ["Non-suicidal", "Suicidal"])
    output_path : str — path to save the PNG
    """
    if labels is None:
        labels = ["Non-suicidal", "Suicidal"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {output_path}")


def evaluate(
    y_true,
    y_pred,
    y_proba,
    roc_path: str = "reports/roc_curve.png",
    cm_path: str = "reports/confusion_matrix.png",
) -> dict:
    """
    Full evaluation: metrics dict + ROC PNG + confusion matrix PNG.

    Parameters
    ----------
    y_true   : array-like — ground truth binary labels
    y_pred   : array-like — binary predictions at threshold 0.5
    y_proba  : array-like — predicted probabilities for the positive class
    roc_path : str — output path for ROC curve PNG
    cm_path  : str — output path for confusion matrix PNG

    Returns
    -------
    dict — same as compute_metrics()
    """
    metrics = compute_metrics(y_true, y_pred, y_proba)
    plot_roc_curve(y_true, y_proba, roc_path)
    plot_confusion_matrix(y_true, y_pred, output_path=cm_path)
    return metrics


def generate_report(
    metrics: dict,
    roc_path: str = "reports/roc_curve.png",
    cm_path: str = "reports/confusion_matrix.png",
    output_path: str = "reports/final_evaluation.md",
    title: str = "Reporte de Evaluación — Detección de Ideación Suicida",
) -> None:
    """
    Render a markdown evaluation report from a metrics dict and existing plot PNGs.

    The report includes a metrics table, the confusion-matrix counts in tabular
    form, optional K-fold CV scores (if `cv_fold_scores` is present in metrics),
    and embedded references to the ROC and confusion-matrix PNGs (paths are
    rewritten relative to the report's directory so the markdown renders
    correctly when opened from anywhere).

    Parameters
    ----------
    metrics : dict
        Output of `compute_metrics` / `evaluate`. Expected keys:
        AUC, F1, precision, recall, FPR, TP, TN, FP, FN. Optional:
        cv_fold_scores, cv_auc_mean, cv_auc_std.
    roc_path : str
        Path to the ROC curve PNG (must already exist).
    cm_path : str
        Path to the confusion-matrix PNG (must already exist).
    output_path : str
        Where to write the markdown report.
    title : str
        Top-level title rendered at the top of the report.
    """
    fold_scores = metrics.get("cv_fold_scores", [])
    fold_rows = "\n".join(
        f"| Fold {i+1} | {score:.4f} |"
        for i, score in enumerate(fold_scores)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    roc_rel = os.path.relpath(roc_path, os.path.dirname(output_path)).replace("\\", "/")
    cm_rel  = os.path.relpath(cm_path,  os.path.dirname(output_path)).replace("\\", "/")

    cv_section = ""
    if fold_scores:
        cv_section = f"""## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
{fold_rows}
| **Promedio** | **{metrics.get('cv_auc_mean', 'N/A')}** |
| **Std** | {metrics.get('cv_auc_std', 'N/A')} |

"""

    content = f"""# {title}

_Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **{metrics['AUC']}** |
| F1 | {metrics['F1']} |
| Precision | {metrics['precision']} |
| Recall (TPR) | {metrics['recall']} |
| FPR | {metrics['FPR']} |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = {metrics['TN']} | FP = {metrics['FP']} |
| **Real Positivo** | FN = {metrics['FN']} | TP = {metrics['TP']} |

{cv_section}## Curva ROC

![ROC Curve]({roc_rel})

## Matriz de confusión (visualización)

![Confusion Matrix]({cm_rel})
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Reporte guardado -> {output_path}")
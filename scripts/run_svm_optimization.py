# ============================================================================
# Optimización de hiperparámetros — SVM (Stage 2-B)
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
"""
Stage 2-B: tune SVC hyperparameters on the winning variant from Stage 1.

Steps:
  1. Detect the winning variant (highest Test AUC in
     ``reports/optimization_results.json``) — unless ``--variant`` is given.
  2. Run ``optimize_model`` with _SVM_SPACE and checkpointing to
     ``reports/optuna_svm_<winner>.journal``.
  3. Write best params + fixed keys to ``configs/optimized/svm_<winner>.yaml``.
  4. Train end-to-end and evaluate on the held-out test fold.
  5. Append results to the consolidated JSON under "svm_optimization".

Run as:
  python scripts/run_svm_optimization.py
  python scripts/run_svm_optimization.py --variant base
  python scripts/run_svm_optimization.py --n-trials 30
  python scripts/run_svm_optimization.py --no-resume
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_cli import run_evaluation
from src.optimization import optimize_model, _SVM_SPACE, pick_winner_variant
from src.training import train_pipeline
from src.utils import load_config


VARIANT_CONFIGS = {
    "base":             "configs/default.yaml",
    "stopwords_nltk":   "configs/variant_stopwords_nltk.yaml",
    "stopwords_domain": "configs/variant_stopwords_domain.yaml",
    "stemming":         "configs/variant_stemming.yaml",
}

OPTIMIZED_CONFIGS = {
    "base":             "configs/optimized/base.yaml",
    "stopwords_nltk":   "configs/optimized/stopwords_nltk.yaml",
    "stopwords_domain": "configs/optimized/stopwords_domain.yaml",
    "stemming":         "configs/optimized/stemming.yaml",
}

CONSOLIDATED = "reports/optimization_results.json"
MODEL_KEY = "svm"
JSON_KEY = "svm_optimization"

FIXED_KEYS = {
    "class":       "sklearn.svm.SVC",
    "probability": True,
    "kernel":      "rbf",
    "random_state": 42,
}


def _write_svm_yaml(base_yaml_path: str, out_yaml_path: str, best_params: dict, variant: str) -> None:
    cfg = load_config(base_yaml_path)
    cfg["model"] = {**FIXED_KEYS, **best_params}
    cfg["paths"]["model_output"]        = f"models/svm_{variant}.joblib"
    cfg["paths"]["roc_val_output"]      = f"reports/svm_{variant}_roc_val.png"
    cfg["paths"]["cm_val_output"]       = f"reports/svm_{variant}_cm_val.png"
    cfg["paths"]["metrics_val_output"]  = f"reports/svm_{variant}_training_metrics.json"
    cfg["paths"]["report_val_output"]   = f"reports/svm_{variant}_training_report.md"
    cfg["paths"]["roc_test_output"]     = f"reports/svm_{variant}_roc_test.png"
    cfg["paths"]["cm_test_output"]      = f"reports/svm_{variant}_cm_test.png"
    cfg["paths"]["metrics_test_output"] = f"reports/svm_{variant}_test_metrics.json"
    cfg["paths"]["report_test_output"]  = f"reports/svm_{variant}_test_report.md"
    Path(out_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_py(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def main(variant: str | None, n_trials: int, timeout: int | None, resume: bool) -> None:
    # --- Step 1: pick winner ---
    if variant is None:
        winner = pick_winner_variant(CONSOLIDATED)
        print(f"Auto-detected winner from {CONSOLIDATED}: {winner!r}")
    else:
        if variant not in VARIANT_CONFIGS:
            raise SystemExit(
                f"Unknown variant {variant!r}. Expected one of: {sorted(VARIANT_CONFIGS)}"
            )
        winner = variant
        print(f"Variant override: {winner!r}")

    opt_path = OPTIMIZED_CONFIGS[winner]
    out_path = f"configs/optimized/svm_{winner}.yaml"
    print(f"Tuning SVM for variant {winner!r} (base config: {opt_path})\n")

    # --- Step 2: Optuna ---
    storage_path = f"reports/optuna_{MODEL_KEY}_{winner}.journal"
    optuna_result = optimize_model(
        winner,
        opt_path,
        space=_SVM_SPACE,
        model_key=MODEL_KEY,
        model_overrides=FIXED_KEYS,
        n_trials=n_trials,
        timeout=timeout,
        storage_path=storage_path,
        resume=resume,
    )

    # --- Step 3: write SVM yaml ---
    print(f"\nWriting best SVM hyperparameters to {out_path}")
    _write_svm_yaml(opt_path, out_path, optuna_result["best_params"], winner)

    # --- Step 4 & 5: train + evaluate ---
    print(f"\nTraining and evaluating SVM pipeline ({winner})...")
    train_metrics = train_pipeline(out_path)
    cfg_after = load_config(out_path)
    test_metrics = run_evaluation(cfg_after["data"]["test_path"], cfg_after)

    # --- Step 6: append to consolidated JSON ---
    final = {
        "variant":      winner,
        "yaml_path":    out_path,
        "n_trials":     optuna_result["n_trials"],
        "best_cv_auc":  optuna_result["best_cv_auc"],
        "best_params":  optuna_result["best_params"],
        "fixed_keys":   FIXED_KEYS,
        "storage_path": storage_path,
        "train": {
            "cv_auc_mean":   train_metrics.get("cv_auc_mean"),
            "cv_auc_std":    train_metrics.get("cv_auc_std"),
            "val_auc":       train_metrics.get("val_auc"),
            "val_AUC":       train_metrics.get("AUC"),
            "val_F1":        train_metrics.get("F1"),
            "val_precision": train_metrics.get("precision"),
            "val_recall":    train_metrics.get("recall"),
            "val_TPR":       train_metrics.get("TPR"),
            "val_FPR":       train_metrics.get("FPR"),
        },
        "test": {
            "AUC":       test_metrics["AUC"],
            "F1":        test_metrics["F1"],
            "precision": test_metrics["precision"],
            "recall":    test_metrics["recall"],
            "TPR":       test_metrics["TPR"],
            "FPR":       test_metrics["FPR"],
            "TP":        test_metrics["TP"],
            "TN":        test_metrics["TN"],
            "FP":        test_metrics["FP"],
            "FN":        test_metrics["FN"],
        },
        "all_trial_scores": optuna_result["all_trial_scores"],
    }

    consolidated_path = Path(CONSOLIDATED)
    consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    if consolidated_path.exists():
        with open(consolidated_path) as f:
            data = json.load(f)
    else:
        data = {}
    data[JSON_KEY] = _to_py(final)
    with open(consolidated_path, "w") as f:
        json.dump(data, f, indent=2)

    # --- Summary ---
    print()
    print("=" * 72)
    print("SVM tuning summary")
    print("=" * 72)
    print(f"Variant:               {winner}")
    print(f"Best CV AUC (Optuna):  {optuna_result['best_cv_auc']:.4f}")
    print(f"Val AUC (holdout 20%): {train_metrics.get('val_auc'):.4f}")
    print(f"Test AUC:              {test_metrics['AUC']:.4f}")
    print(f"Recall (test):         {test_metrics['recall']:.4f}")
    print(f"FPR (test):            {test_metrics['FPR']:.4f}")
    print(f"F1 (test):             {test_metrics['F1']:.4f}")
    print()
    print(f"Best hyperparameters written to {out_path}")
    print(f"Consolidated results updated:   {CONSOLIDATED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None, help="Override winner detection")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds per study")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Ignore existing journal and start a fresh study",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    main(variant=args.variant, n_trials=args.n_trials, timeout=args.timeout, resume=args.resume)

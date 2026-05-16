"""
Stage 2 orchestrator: tune the XGBClassifier hyperparameters on the
winning variant from stage 1 (TF-IDF tuning).

Steps:
  1. Detect the winning variant (highest Test AUC in
     ``reports/optimization_results.json``) — unless ``--variant`` is given.
  2. Run ``optimize_xgb`` with checkpointing to
     ``reports/optuna_xgb_<winner>.journal``.
  3. Write the best hyperparameters + ``early_stopping_rounds=30`` to
     the optimized YAML in ``configs/optimized/`` (originals are never
     modified).
  4. Train end-to-end on the winner's variant (early stopping kicks in
     thanks to the new YAML field).
  5. Evaluate on the held-out test fold.
  6. Append the run to the consolidated JSON under "xgb_optimization".

Run as:
  python scripts/run_xgb_optimization.py                 # auto-detect winner
  python scripts/run_xgb_optimization.py --variant base  # override
  python scripts/run_xgb_optimization.py --n-trials 50
  python scripts/run_xgb_optimization.py --no-resume     # ignore checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_cli import run_evaluation
from src.optimization import optimize_xgb, pick_winner_variant
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
DEFAULT_EARLY_STOPPING_ROUNDS = 30


def write_optimized_model_block(yaml_path: str, best_params: dict, fixed_keys: dict) -> None:
    """
    Read the (already-optimized) YAML at *yaml_path*, replace its ``model``
    block with the merged Optuna best params + fixed knobs, and write it
    back. The original configs in ``configs/`` are never touched — this
    function only modifies files inside ``configs/optimized/``.
    """
    cfg = load_config(yaml_path)
    cfg["model"] = {**fixed_keys, **best_params}
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def _to_py(o):
    """Recursively convert numpy scalars to Python primitives."""
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_py(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def main(variant: str | None, n_trials: int, resume: bool) -> None:
    # --- Step 1: pick winner ---
    if variant is None:
        winner = pick_winner_variant(CONSOLIDATED)
        print(f"Auto-detected winner from {CONSOLIDATED}: {winner!r}")
    else:
        if variant not in VARIANT_CONFIGS:
            raise SystemExit(
                f"Unknown variant {variant!r}. Expected one of: "
                f"{sorted(VARIANT_CONFIGS)}"
            )
        winner = variant
        print(f"Variant override: {winner!r}")

    opt_path = OPTIMIZED_CONFIGS[winner]
    print(f"Tuning XGB for variant {winner!r} (config: {opt_path})\n")

    # --- Step 2: Optuna ---
    storage_path = f"reports/optuna_xgb_{winner}.journal"
    optuna_result = optimize_xgb(
        winner,
        opt_path,
        n_trials=n_trials,
        storage_path=storage_path,
        resume=resume,
    )

    # --- Step 3: write best XGB params to the optimized YAML ---
    print(f"\nWriting best XGB hyperparameters to {opt_path}")
    base_cfg = load_config(opt_path)
    fixed_keys = {
        "class":                 base_cfg["model"].get("class", "xgboost.XGBClassifier"),
        "scale_pos_weight":      base_cfg["model"]["scale_pos_weight"],
        "eval_metric":           base_cfg["model"]["eval_metric"],
        "random_state":          base_cfg["model"]["random_state"],
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
    }
    write_optimized_model_block(opt_path, optuna_result["best_params"], fixed_keys)

    # --- Step 4 & 5: train + evaluate the tuned pipeline ---
    print(f"\nTraining and evaluating tuned pipeline ({winner})...")
    train_metrics = train_pipeline(opt_path)
    cfg_after = load_config(opt_path)
    test_metrics = run_evaluation(cfg_after["data"]["test_path"], cfg_after)

    # --- Step 6: append to consolidated JSON ---
    final = {
        "variant":      winner,
        "yaml_path":    opt_path,
        "n_trials":     optuna_result["n_trials"],
        "best_cv_auc":  optuna_result["best_cv_auc"],
        "best_params":  optuna_result["best_params"],
        "fixed_keys":   fixed_keys,
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
    data["xgb_optimization"] = _to_py(final)
    with open(consolidated_path, "w") as f:
        json.dump(data, f, indent=2)

    # --- Summary ---
    print()
    print("=" * 72)
    print("XGB tuning summary")
    print("=" * 72)
    print(f"Variant:               {winner}")
    print(f"Best CV AUC (Optuna):  {optuna_result['best_cv_auc']:.4f}")
    print(f"Val AUC (holdout 20%): {train_metrics.get('val_auc'):.4f}")
    print(f"Test AUC:              {test_metrics['AUC']:.4f}")
    print(f"Recall (test):         {test_metrics['recall']:.4f}")
    print(f"FPR (test):            {test_metrics['FPR']:.4f}")
    print(f"F1 (test):             {test_metrics['F1']:.4f}")
    print()
    print(f"Best hyperparameters written to {opt_path}")
    print(f"Consolidated results updated:   {CONSOLIDATED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None, help="Override winner detection")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Ignore existing journal and start a fresh study",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    main(variant=args.variant, n_trials=args.n_trials, resume=args.resume)

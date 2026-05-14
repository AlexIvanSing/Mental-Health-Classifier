"""
Full optimization pipeline orchestrator.

For each preprocessing variant (base, stopwords_nltk, stopwords_domain,
stemming):
  1. Run Optuna over the TF-IDF hyperparameter space (n_trials=30, 5-fold CV).
  2. Overwrite the variant's YAML vectorizer block with the best params.
  3. Train end-to-end on `data/data_train.csv` using `train_pipeline`.
  4. Evaluate the trained model on `data/data_test_fold1.csv` via
     `run_evaluation`.
  5. Collect all metrics into a single consolidated JSON for the report.

Run as:  python scripts/run_optimization_pipeline.py
"""

import json
import sys
from pathlib import Path

import yaml

# Allow running as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_cli import run_evaluation
from src.optimization import run_all_optimizations
from src.training import train_pipeline
from src.utils import load_config


VARIANT_CONFIGS = {
    "base":             "configs/default.yaml",
    "stopwords_nltk":   "configs/variant_stopwords_nltk.yaml",
    "stopwords_domain": "configs/variant_stopwords_domain.yaml",
    "stemming":         "configs/variant_stemming.yaml",
}

CONSOLIDATED_OUTPUT = "reports/optimization_results.json"


def overwrite_vectorizer_block(yaml_path: str, best_params: dict) -> None:
    """
    Replace the vectorizer block of a YAML config with Optuna's best params,
    preserving every other section verbatim.
    """
    cfg = load_config(yaml_path)
    cfg["vectorizer"] = {
        "ngram_range":  best_params["ngram_range"],
        "min_df":       best_params["min_df"],
        "max_df":       best_params["max_df"],
        "sublinear_tf": best_params["sublinear_tf"],
        "max_features": best_params["max_features"],
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main(n_trials: int = 30) -> None:
    print("=" * 72)
    print(f"Step 1/4: Optuna search over {len(VARIANT_CONFIGS)} variants  "
          f"(n_trials={n_trials} each)")
    print("=" * 72)

    # All variants share the same data + XGB hyperparameters; we use the
    # default config as the source of truth for those during the search.
    optuna_results = run_all_optimizations(
        config_path="configs/default.yaml",
        n_trials=n_trials,
    )

    print("=" * 72)
    print("Step 2/4: Writing best hyperparameters back to each YAML")
    print("=" * 72)
    for variant, yaml_path in VARIANT_CONFIGS.items():
        overwrite_vectorizer_block(yaml_path, optuna_results[variant]["best_params"])
        print(f"  [{variant}] vectorizer block updated in {yaml_path}")

    print()
    print("=" * 72)
    print("Step 3/4: Training and evaluating each variant end-to-end")
    print("=" * 72)
    final_results = {}
    for variant, yaml_path in VARIANT_CONFIGS.items():
        print(f"\n--- {variant} ---")
        train_metrics = train_pipeline(yaml_path)

        cfg = load_config(yaml_path)
        test_metrics = run_evaluation(cfg["data"]["test_path"], cfg)

        final_results[variant] = {
            "yaml_path":      yaml_path,
            "best_params":    optuna_results[variant]["best_params"],
            "optuna": {
                "best_cv_auc":      optuna_results[variant]["best_cv_auc"],
                "n_trials":         optuna_results[variant]["n_trials"],
                "all_trial_scores": optuna_results[variant]["all_trial_scores"],
            },
            "train": {
                # train_pipeline returns the merged metrics dict
                "cv_auc_mean": train_metrics.get("cv_auc_mean"),
                "cv_auc_std":  train_metrics.get("cv_auc_std"),
                "val_auc":     train_metrics.get("val_auc"),
                # full classification metrics on the holdout val set:
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
        }

    print()
    print("=" * 72)
    print(f"Step 4/4: Writing consolidated results to {CONSOLIDATED_OUTPUT}")
    print("=" * 72)

    # Convert numpy types to Python primitives so json.dump is happy.
    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_py(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    Path(CONSOLIDATED_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATED_OUTPUT, "w") as f:
        json.dump(_to_py(final_results), f, indent=2)

    print(f"\nDone. Summary:\n")
    print(f"{'variant':<20s} {'CV AUC':>8s} {'Val AUC':>8s} {'Test AUC':>9s} "
          f"{'Recall':>8s} {'FPR':>8s}")
    for variant, r in final_results.items():
        print(f"{variant:<20s} "
              f"{r['optuna']['best_cv_auc']:>8.4f} "
              f"{r['train']['val_auc']:>8.4f} "
              f"{r['test']['AUC']:>9.4f} "
              f"{r['test']['recall']:>8.4f} "
              f"{r['test']['FPR']:>8.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    main(n_trials=args.n_trials)

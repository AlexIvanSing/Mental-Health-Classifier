import argparse
import json
import os

import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline
from src.data_ingestion import ingestion
from src.preprocessing import clean_text
from src.evaluation import evaluate, generate_report
from src.utils import load_config


def _fit_with_early_stopping(
    pipeline: Pipeline,
    X,
    y,
    early_stopping_rounds: int,
    random_state: int,
) -> Pipeline:
    """
    Fit `pipeline` with an inner 90/10 split used as eval_set for XGBoost
    early stopping. The cleaner+TF-IDF steps are fitted on the 90% inner
    train; the XGB step is fitted with the 10% inner eval as `eval_set`
    so trees stop growing once validation AUC plateaus.

    This preserves the train/serve consistency property of the Pipeline:
    the returned object is a fully-fitted sklearn Pipeline that accepts
    raw strings end-to-end.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted pipeline produced by `build_pipeline`. Must have steps
        named "cleaner", "tfidf", "clf".
    X, y : array-like
        Outer training partition (the 80% from the train/val split).
    early_stopping_rounds : int
        Passed verbatim to `XGBClassifier.fit`.
    random_state : int
        Seed for the inner stratified split.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The same pipeline, fully fitted.
    """
    X_in, X_es, y_in, y_es = train_test_split(
        X, y, test_size=0.10, random_state=random_state, stratify=y,
    ) #Solo existe para el mecanismo de early stopping de XGBoost., no crea un set de testing.

    cleaner = pipeline.named_steps["cleaner"]
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # FunctionTransformer is stateless; "fit" is a no-op. We still call it
    # so any future stateful cleaner picks up the train distribution.
    X_in_clean = cleaner.fit_transform(X_in)
    X_es_clean = cleaner.transform(X_es)

    X_in_vec = tfidf.fit_transform(X_in_clean)
    X_es_vec = tfidf.transform(X_es_clean)

    # XGBoost >= 2.0 reads `early_stopping_rounds` from the constructor;
    # set_params() before fit() is the supported way to inject it.
    clf.set_params(early_stopping_rounds=early_stopping_rounds)
    clf.fit(
        X_in_vec, y_in,
        eval_set=[(X_es_vec, y_es)],
        verbose=False,
    )

    best_iter = getattr(clf, "best_iteration", None)
    n_grown = (best_iter + 1) if best_iter is not None else clf.n_estimators
    print(
        f"Early stopping: trees grown = {n_grown} "
        f"(cap was {clf.n_estimators}, patience = {early_stopping_rounds})"
    )

    return pipeline


def train(X: list, y, config: dict) -> tuple[object, dict, dict]:
    """
    Train the pipeline with train/val split and StratifiedKFold CV.

    Parameters
    ----------
    X : list
        Raw text samples.
    y : array-like
        Binary labels (0/1).
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    tuple
        (fitted_pipeline, metrics_dict, eval_artifacts_dict)
        - metrics_dict: cv_auc_mean, cv_auc_std, val_auc, cv_fold_scores.
        - eval_artifacts_dict: y_val, y_pred_val, y_proba_val — kept so the
          orchestrator (`train_pipeline`) can hand them to `evaluation.evaluate`
          without re-running prediction. Disk I/O lives in the orchestrator,
          not here.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    pipeline = build_pipeline(config)

    cv = StratifiedKFold(
        n_splits=config["training"]["n_splits"],
        shuffle=True,
        random_state=config["training"]["random_state"]
    )

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        return_train_score=True,
        n_jobs=-1
    )

    auc_mean = cv_results["test_score"].mean()
    auc_std  = cv_results["test_score"].std()
    print(f"CV AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    early_stopping = config["model"].get("early_stopping_rounds")
    if early_stopping:
        pipeline = _fit_with_early_stopping(
            pipeline, X_train, y_train,
            early_stopping_rounds=int(early_stopping),
            random_state=config["training"]["random_state"],
        )
    else:
        pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred = pipeline.predict(X_val)
    val_auc = roc_auc_score(y_val, y_proba)
    print(f"Val AUC: {val_auc:.4f}")

    metrics = {
        "cv_auc_mean":    round(auc_mean, 4),
        "cv_auc_std":     round(auc_std, 4),
        "val_auc":        round(val_auc, 4),
        "cv_fold_scores": cv_results["test_score"].tolist()
    }

    eval_artifacts = {
        "y_val":        list(y_val),
        "y_pred_val":   y_pred.tolist(),
        "y_proba_val":  y_proba.tolist(),
    }

    joblib.dump(pipeline, config["paths"]["model_output"])
    print(f"Modelo guardado en {config['paths']['model_output']}")

    return pipeline, metrics, eval_artifacts


def train_pipeline(config_path: str) -> dict:
    """
    Full training entrypoint: load config → ingest → preprocess → train → evaluate → persist.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Merged metrics: CV/val AUC from `train()` plus the full evaluation
        report (TP/TN/FP/FN/TPR/FPR/AUC/precision/recall/F1) from
        `evaluation.evaluate()` on the holdout split. Also writes ROC PNG,
        confusion-matrix PNG, and a JSON metrics dump to `paths.reports_dir`.
    """
    config = load_config(config_path)

    df = ingestion(
        data_url=config["data"]["train_path"],
        expected_columns=config["data"]["expected_columns"],
        text_columns=config["data"]["text_columns"],
        target_column=config["data"]["target_column"],
        value_true=config["data"]["value_true"],
        value_false=config["data"]["value_false"],
    )

    text_col = "_".join(config["data"]["text_columns"])

    X = df[text_col].tolist()
    y = df[config["data"]["target_column"]].tolist()

    _, metrics, artifacts = train(X, y, config)

    os.makedirs(config["paths"]["reports_dir"], exist_ok=True)
    eval_metrics = evaluate(
        artifacts["y_val"],
        artifacts["y_pred_val"],
        artifacts["y_proba_val"],
        roc_path=config["paths"]["roc_val_output"],
        cm_path=config["paths"]["cm_val_output"],
    )
    metrics.update(eval_metrics)

    with open(config["paths"]["metrics_val_output"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metricas guardadas en {config['paths']['metrics_val_output']}")

    generate_report(
        metrics,
        roc_path=config["paths"]["roc_val_output"],
        cm_path=config["paths"]["cm_val_output"],
        output_path=config["paths"]["report_val_output"],
        title="Reporte de Entrenamiento — Detección de Ideación Suicida",
    )

    print(metrics)
    return metrics


def main():
    # para llamar directo con   python src/training.py --config mi.yaml
    parser = argparse.ArgumentParser(description="Train the suicide detection pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)


if __name__ == "__main__":
    main()
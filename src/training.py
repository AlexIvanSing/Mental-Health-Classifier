import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score

from src.pipeline import build_pipeline
from src.data_ingestion import ingestion
from src.preprocessing import clean_text
from src.utils import load_config


def train(X: list, y, config: dict) -> tuple[object, dict]:
    """
    Train the pipeline with train/val split and StratifiedKFold CV.

    Parameters
    ----------
    X : list
        Raw text samples (preprocesados con clean_text).
    y : array-like
        Binary labels (0/1).
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    tuple
        (fitted_pipeline, metrics_dict)
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

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_proba)
    print(f"Val AUC: {val_auc:.4f}")

    metrics = {
        "cv_auc_mean":    round(auc_mean, 4),
        "cv_auc_std":     round(auc_std, 4),
        "val_auc":        round(val_auc, 4),
        "cv_fold_scores": cv_results["test_score"].tolist()
    }

    joblib.dump(pipeline, config["paths"]["model_output"])
    print(f"Modelo guardado en {config['paths']['model_output']}")

    return pipeline, metrics


def train_pipeline(config_path: str) -> dict:
    """
    Full training entrypoint: load config → ingest → preprocess → train → persist.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Metrics produced by `train()`.
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
    df[text_col] = df[text_col].apply(clean_text)

    X = df[text_col].tolist()
    y = df[config["data"]["target_column"]].tolist()

    _, metrics = train(X, y, config)
    print(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the suicide detection pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)


if __name__ == "__main__":
    main()
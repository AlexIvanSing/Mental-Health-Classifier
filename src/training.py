import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score

from src.pipeline import build_pipeline
from src.utils import load_config


def train(X: list, y, config: dict) -> tuple[object, dict]:
    """
    Train the pipeline with train/val split and StratifiedKFold CV.

    Parameters
    ----------
    X : list
        Raw text samples (ya preprocesados con clean_text).
    y : array-like
        Binary labels (0/1).
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    tuple
        (fitted_pipeline, metrics_dict)
    """
    # --- Split estratificado 80/20 --x-
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    pipeline = build_pipeline(config)

    # --- K-Fold para validación robusta ---
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

    # --- Fit final sobre todo el train ---
    pipeline.fit(X_train, y_train)

    # --- Validación sobre el val set ---
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_proba)
    print(f"Val AUC: {val_auc:.4f}")

    metrics = {
        "cv_auc_mean": round(auc_mean, 4),
        "cv_auc_std":  round(auc_std, 4),
        "val_auc":     round(val_auc, 4),
        "cv_fold_scores": cv_results["test_score"].tolist()
    }

    # --- Persistencia ---
    joblib.dump(pipeline, config["paths"]["model_output"])
    print(f"Modelo guardado en {config['paths']['model_output']}")

    return pipeline, metrics
from src.data_ingestion import ingestion


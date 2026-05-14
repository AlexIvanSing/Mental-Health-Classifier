from xgboost import XGBClassifier


def build_model(config: dict) -> XGBClassifier:
    """
    Build an XGBoost classifier from a configuration dictionary.

    Reads the `model` section of the config and returns an unfitted
    `XGBClassifier`. The first six keys are required; the other six
    (introduced for Optuna-driven XGB tuning) fall back to sensible
    defaults when absent, so older configs keep working unchanged.

    The `scale_pos_weight` parameter compensates for class imbalance in
    the training set (`n_neg / n_pos` ≈ 0.93 in this corpus). The
    `early_stopping_rounds` field is intentionally **not** consumed here
    — it's handled by `training.train()`, which builds an inner eval
    split and passes the rounds to `.fit()` directly.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML. Must contain a `model`
        section with at least: `n_estimators`, `max_depth`, `learning_rate`,
        `eval_metric`, `scale_pos_weight`, `random_state`. May also contain
        the tunable knobs: `subsample`, `colsample_bytree`, `gamma`,
        `min_child_weight`, `reg_alpha`, `reg_lambda`.

    Returns
    -------
    xgboost.XGBClassifier
        Unfitted classifier ready to be plugged into a Pipeline.
    """
    m = config["model"]
    return XGBClassifier(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        learning_rate=m["learning_rate"],
        eval_metric=m["eval_metric"],
        scale_pos_weight=m["scale_pos_weight"],
        random_state=m["random_state"],
        subsample=m.get("subsample", 1.0),
        colsample_bytree=m.get("colsample_bytree", 1.0),
        gamma=m.get("gamma", 0.0),
        min_child_weight=m.get("min_child_weight", 1),
        reg_alpha=m.get("reg_alpha", 0.0),
        reg_lambda=m.get("reg_lambda", 1.0),
    )

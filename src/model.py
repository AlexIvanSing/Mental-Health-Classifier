from xgboost import XGBClassifier


def build_model(config: dict) -> XGBClassifier:
    """
    Build an XGBoost classifier from a configuration dictionary.

    Reads the `model` section of the config and returns an unfitted
    `XGBClassifier`. The `scale_pos_weight` parameter compensates for
    class imbalance in the training set.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML. Must contain a
        `model` key with: `n_estimators`, `max_depth`, `learning_rate`,
        `eval_metric`, `scale_pos_weight`, `random_state`.

    Returns
    -------
    xgboost.XGBClassifier
        Unfitted classifier ready to be plugged into a Pipeline.
    """
    return XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        eval_metric=config["model"]["eval_metric"],
        scale_pos_weight=config["model"]["scale_pos_weight"],
        random_state=config["model"]["random_state"],
    )
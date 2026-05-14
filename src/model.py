from xgboost import XGBClassifier
    
def build_model(config: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        #early_stopping_rounds=config["model"]["early_stopping_rounds"] va en  training
        eval_metric=config["model"]["eval_metric"],
        scale_pos_weight=config["model"]["scale_pos_weight"],
        random_state=config["model"]["random_state"]
    )
# ============================================================================
# Constructor del modelo
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import importlib

_EXCLUDED_KEYS = frozenset({"class", "early_stopping_rounds"})


def build_model(config: dict):
    """
    Build a classifier from a configuration dictionary using dynamic import.

    Reads ``config["model"]["class"]`` (dotted path, e.g.
    ``"xgboost.XGBClassifier"``) and instantiates it with every other key
    in the block as a constructor argument — except ``early_stopping_rounds``,
    which is handled by ``training.train()`` via ``set_params()``.

    Falls back to ``xgboost.XGBClassifier`` when ``class`` is absent, so
    older configs keep working unchanged.
    """
    m = config["model"]
    class_path = m.get("class", "xgboost.XGBClassifier")
    module_path, class_name = class_path.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)
    params = {k: v for k, v in m.items() if k not in _EXCLUDED_KEYS}
    return ModelClass(**params)

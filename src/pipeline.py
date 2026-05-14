import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.vectorizer import build_vectorizer
from src.model import build_model
from src.preprocessing import clean_text


def _clean_series(X) -> list:
    """
    Apply `clean_text` element-wise to an iterable of raw strings.

    Defined at module level (not as a lambda) so the enclosing
    `FunctionTransformer` is picklable by `joblib.dump`.
    """
    return pd.Series(X).apply(clean_text).tolist()


def build_pipeline(config: dict) -> Pipeline:
    """
    Build the end-to-end sklearn Pipeline: text cleaner → TF-IDF → XGBoost.

    The cleaner stage wraps `clean_text` in a `FunctionTransformer` so the
    pipeline accepts raw strings directly — preprocessing is applied
    consistently at train time and inference time, removing the risk of
    train/serve skew.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML. Must contain
        `vectorizer` and `model` sections.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for training.
    """
    cleaner = FunctionTransformer(_clean_series)
    vectorizer = build_vectorizer(config)
    model = build_model(config)

    return Pipeline([
        ("cleaner", cleaner),
        ("tfidf", vectorizer),
        ("clf", model),
    ])
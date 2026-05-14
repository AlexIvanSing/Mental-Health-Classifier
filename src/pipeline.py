from sklearn.pipeline import Pipeline
from src.vectorizer import build_vectorizer
from src.model import build_model


def build_pipeline(config: dict) -> Pipeline:
    """
    Build the full sklearn Pipeline: TF-IDF vectorizer + XGBoost classifier.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for training.
    """
    vectorizer = build_vectorizer(config)
    model = build_model(config)

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", model)
    ])
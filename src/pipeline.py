import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.vectorizer import build_vectorizer
from src.model import build_model
from src.preprocessing import (
    clean_text,
    clean_text_with_stopwords_nltk,
    clean_text_with_stopwords_domain,
    clean_text_with_stemming,
)


# Registry of preprocessing variants. The active one is selected via
# config["preprocessing"]["variant"] (defaulting to "base"). Keeping the
# registry at module level means the choice is data-only — no `if`s in the
# Pipeline construction — and adding a new variant is a one-line change.
CLEANERS = {
    "base":             clean_text,
    "stopwords_nltk":   clean_text_with_stopwords_nltk,
    "stopwords_domain": clean_text_with_stopwords_domain,
    "stemming":         clean_text_with_stemming,
}


# Module-level wrappers (not lambdas / closures) so the FunctionTransformer
# remains picklable by joblib.dump. One per variant.

def _clean_series_base(X) -> list:
    """Apply baseline `clean_text` element-wise. Used by the default variant."""
    return pd.Series(X).apply(clean_text).tolist()


def _clean_series_stopwords_nltk(X) -> list:
    """Apply `clean_text_with_stopwords_nltk` element-wise."""
    return pd.Series(X).apply(clean_text_with_stopwords_nltk).tolist()


def _clean_series_stopwords_domain(X) -> list:
    """Apply `clean_text_with_stopwords_domain` element-wise."""
    return pd.Series(X).apply(clean_text_with_stopwords_domain).tolist()


def _clean_series_stemming(X) -> list:
    """Apply `clean_text_with_stemming` element-wise."""
    return pd.Series(X).apply(clean_text_with_stemming).tolist()


_SERIES_CLEANERS = {
    "base":             _clean_series_base,
    "stopwords_nltk":   _clean_series_stopwords_nltk,
    "stopwords_domain": _clean_series_stopwords_domain,
    "stemming":         _clean_series_stemming,
}


# Back-compat alias: existing tests / code may still call `_clean_series`.
_clean_series = _clean_series_base


def build_pipeline(config: dict) -> Pipeline:
    """
    Build the end-to-end sklearn Pipeline: text cleaner → TF-IDF → XGBoost.

    The cleaner stage is selected by `config["preprocessing"]["variant"]`
    (default: `"base"`). Whatever variant is chosen, the cleaning runs
    *inside* the Pipeline as a FunctionTransformer — guaranteeing that the
    same preprocessing applied at training time is applied at inference
    time, eliminating train/serve skew regardless of variant.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML. Must contain
        `vectorizer` and `model` sections. May contain a `preprocessing`
        section with `variant ∈ {base, stopwords_nltk, stopwords_domain,
        stemming}`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for training.
    """
    variant = config.get("preprocessing", {}).get("variant", "base")
    if variant not in _SERIES_CLEANERS:
        raise ValueError(
            f"Unknown preprocessing variant {variant!r}. "
            f"Expected one of: {sorted(_SERIES_CLEANERS)}"
        )

    cleaner = FunctionTransformer(_SERIES_CLEANERS[variant])
    vectorizer = build_vectorizer(config)
    model = build_model(config)

    return Pipeline([
        ("cleaner", cleaner),
        ("tfidf", vectorizer),
        ("clf", model),
    ])

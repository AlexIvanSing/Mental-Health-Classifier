import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.pipeline import build_pipeline


@pytest.fixture
def sample_config():
    return {
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "max_features": 50,
        },
        "model": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "eval_metric": "auc",
            "random_state": 42,
            "scale_pos_weight": 1.0,
        },
    }


@pytest.fixture
def toy_corpus():
    X = [
        "I want to end my life today",
        "I cannot take this anymore please help",
        "the sun is shining and i feel great",
        "today was such a beautiful day at the park",
        "everything hurts and nothing makes sense anymore",
        "happy birthday to my best friend",
        "i feel hopeless and empty inside",
        "what a wonderful weekend with family",
    ]
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    return X, y


# --- structure ---

def test_build_pipeline_returns_sklearn_pipeline(sample_config):
    pipe = build_pipeline(sample_config)
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_expected_steps(sample_config):
    pipe = build_pipeline(sample_config)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["cleaner", "tfidf", "clf"]


# --- fit / predict end-to-end with raw strings ---

def test_pipeline_fit_does_not_crash(sample_config, toy_corpus):
    X, y = toy_corpus
    pipe = build_pipeline(sample_config)
    pipe.fit(X, y)


def test_pipeline_predict_returns_binary(sample_config, toy_corpus):
    X, y = toy_corpus
    pipe = build_pipeline(sample_config)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds.tolist()).issubset({0, 1})


def test_pipeline_predict_proba_in_range(sample_config, toy_corpus):
    X, y = toy_corpus
    pipe = build_pipeline(sample_config)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_pipeline_predict_proba_rows_sum_to_one(sample_config, toy_corpus):
    X, y = toy_corpus
    pipe = build_pipeline(sample_config)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_accepts_raw_text_directly(sample_config, toy_corpus):
    """The cleaner step runs inside the pipeline — no manual preprocessing needed."""
    X, y = toy_corpus
    dirty = [t.upper() + "  https://example.com  @user" for t in X]
    pipe = build_pipeline(sample_config)
    pipe.fit(dirty, y)
    preds = pipe.predict(dirty)
    assert len(preds) == len(dirty)


def test_pipeline_is_deterministic(sample_config, toy_corpus):
    X, y = toy_corpus
    pipe1 = build_pipeline(sample_config)
    pipe2 = build_pipeline(sample_config)
    pipe1.fit(X, y)
    pipe2.fit(X, y)
    assert np.array_equal(pipe1.predict(X), pipe2.predict(X))

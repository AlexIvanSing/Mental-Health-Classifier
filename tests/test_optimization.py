"""
Tests for src/optimization.py.

We don't run a full 30-trial Optuna search here — that would be both slow and
non-deterministic. Instead we run `n_trials=1` against a tiny synthetic
labeled CSV so the search loop executes end-to-end, and we assert on the
*shape* and *contracts* of the returned dict.

This catches the most common regressions: missing keys, wrong dtypes,
crashes inside the objective function, breaking the variant registry, etc.
"""

import pandas as pd
import pytest
import yaml

from src.optimization import (
    VARIANTS,
    optimize_vectorizer,
    run_all_optimizations,
)


# --- Fixtures: a self-contained tiny project rooted in tmp_path -----------

@pytest.fixture
def tiny_train_csv(tmp_path):
    """A 12-row CSV with the project schema. Balanced classes, varied text."""
    rows = [
        ("u1",  "t1",  "help",          "I want to end my life tonight",         "yes"),
        ("u2",  "t2",  "alone",         "Nothing matters anymore I am done",     "yes"),
        ("u3",  "t3",  "tired",         "I cannot keep going like this",         "yes"),
        ("u4",  "t4",  "darkness",      "Everything hurts please make it stop",  "yes"),
        ("u5",  "t5",  "empty",         "I dont want to be here anymore",        "yes"),
        ("u6",  "t6",  "hopeless",      "There is no point in continuing",       "yes"),
        ("u7",  "t7",  "happy day",     "The sun is shining and I feel great",   "no"),
        ("u8",  "t8",  "wonderful",     "Today was such a beautiful day",        "no"),
        ("u9",  "t9",  "good news",     "I got the promotion at work",           "no"),
        ("u10", "t10", "friends",       "Spent the weekend with my best friend", "no"),
        ("u11", "t11", "vacation",      "Heading to the beach next month",       "no"),
        ("u12", "t12", "family",        "My nephew turned five today",           "no"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "text_id", "title", "text", "is_suicide"])
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tiny_config_file(tmp_path, tiny_train_csv):
    """A YAML config that points to the tiny CSV. Vectorizer values are placeholders."""
    cfg = {
        "data": {
            "train_path":       tiny_train_csv,
            "test_path":        tiny_train_csv,
            "expected_columns": ["user_id", "text_id", "title", "text", "is_suicide"],
            "text_columns":     ["title", "text"],
            "target_column":    "is_suicide",
            "value_true":       "yes",
            "value_false":      "no",
        },
        "preprocessing": {"variant": "base"},
        "paths": {
            "model_output": str(tmp_path / "model.joblib"),
        },
        "vectorizer": {
            "ngram_range":  [1, 2],
            "min_df":       1,
            "max_df":       0.95,
            "sublinear_tf": True,
            "max_features": 50,
        },
        "model": {
            "n_estimators":     10,
            "max_depth":        3,
            "learning_rate":    0.1,
            "eval_metric":      "auc",
            "random_state":     42,
            "scale_pos_weight": 1.0,
        },
        "training": {
            "test_size":    0.2,
            "random_state": 42,
            "n_splits":     5,
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


# --- optimize_vectorizer --------------------------------------------------

EXPECTED_RESULT_KEYS = {"variant", "best_cv_auc", "best_params", "n_trials", "all_trial_scores"}
EXPECTED_PARAM_KEYS  = {"ngram_range", "min_df", "max_df", "sublinear_tf", "max_features"}


def test_optimize_vectorizer_returns_expected_top_level_keys(tiny_config_file):
    result = optimize_vectorizer("base", tiny_config_file, n_trials=1)
    assert isinstance(result, dict)
    assert EXPECTED_RESULT_KEYS.issubset(result.keys())


def test_optimize_vectorizer_records_variant(tiny_config_file):
    for v in VARIANTS:
        result = optimize_vectorizer(v, tiny_config_file, n_trials=1)
        assert result["variant"] == v


def test_optimize_vectorizer_best_params_have_correct_shape(tiny_config_file):
    result = optimize_vectorizer("base", tiny_config_file, n_trials=2)
    params = result["best_params"]
    assert EXPECTED_PARAM_KEYS == set(params.keys())

    # types ready for YAML round-trip
    assert isinstance(params["ngram_range"], list)
    assert len(params["ngram_range"]) == 2
    assert isinstance(params["min_df"], int)
    assert isinstance(params["max_df"], float)
    assert isinstance(params["sublinear_tf"], bool)
    assert isinstance(params["max_features"], int)


def test_optimize_vectorizer_best_params_within_search_space(tiny_config_file):
    result = optimize_vectorizer("base", tiny_config_file, n_trials=3)
    p = result["best_params"]
    assert p["ngram_range"] in ([1, 1], [1, 2], [1, 3])
    assert 1 <= p["min_df"] <= 5
    assert 0.70 <= p["max_df"] <= 0.99
    assert p["max_features"] in (5000, 10000, 20000, 50000)


def test_optimize_vectorizer_cv_auc_is_in_unit_interval(tiny_config_file):
    result = optimize_vectorizer("base", tiny_config_file, n_trials=2)
    assert 0.0 <= result["best_cv_auc"] <= 1.0


def test_optimize_vectorizer_records_all_trial_scores(tiny_config_file):
    result = optimize_vectorizer("base", tiny_config_file, n_trials=3)
    assert isinstance(result["all_trial_scores"], list)
    assert len(result["all_trial_scores"]) == 3
    assert all(0.0 <= s <= 1.0 for s in result["all_trial_scores"])


def test_optimize_vectorizer_rejects_unknown_variant(tiny_config_file):
    with pytest.raises(ValueError, match="Unknown variant"):
        optimize_vectorizer("nonexistent_variant", tiny_config_file, n_trials=1)


# --- run_all_optimizations ------------------------------------------------

def test_run_all_optimizations_returns_one_entry_per_variant(tiny_config_file):
    results = run_all_optimizations(
        config_path=tiny_config_file,
        n_trials=1,
        variants=("base", "stopwords_nltk"),  # subset to keep this test fast
    )
    assert set(results.keys()) == {"base", "stopwords_nltk"}
    for v, res in results.items():
        assert res["variant"] == v
        assert EXPECTED_RESULT_KEYS.issubset(res.keys())


def test_run_all_optimizations_default_covers_all_four_variants(tiny_config_file):
    """We don't actually run all four (slow). Just confirm VARIANTS is the default."""
    from src.optimization import VARIANTS
    assert VARIANTS == ("base", "stopwords_nltk", "stopwords_domain", "stemming")

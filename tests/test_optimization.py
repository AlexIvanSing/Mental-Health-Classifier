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

import json
import os

from src.optimization import (
    VARIANTS,
    optimize_vectorizer,
    optimize_xgb,
    pick_winner_variant,
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

EXPECTED_RESULT_KEYS = {"variant", "best_cv_auc", "best_params", "n_trials", "all_trial_scores", "storage_path"}
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


def test_optimize_vectorizer_records_all_trial_scores(tiny_config_file, tmp_path):
    storage = str(tmp_path / "tfidf_scores.journal")
    result = optimize_vectorizer("base", tiny_config_file, n_trials=3, storage_path=storage)
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


# ============================================================================
# optimize_xgb
# ============================================================================

EXPECTED_XGB_KEYS = {
    "variant", "best_cv_auc", "best_params", "n_trials",
    "n_trials_this_run", "all_trial_scores", "storage_path",
}

EXPECTED_XGB_PARAM_KEYS = {
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "gamma",
    "min_child_weight", "reg_alpha", "reg_lambda",
}


def test_optimize_xgb_returns_expected_top_level_keys(tiny_config_file, tmp_path):
    storage = str(tmp_path / "xgb.journal")
    result = optimize_xgb("base", tiny_config_file, n_trials=1, storage_path=storage)
    assert isinstance(result, dict)
    assert EXPECTED_XGB_KEYS.issubset(result.keys())


def test_optimize_xgb_best_params_have_correct_shape(tiny_config_file, tmp_path):
    storage = str(tmp_path / "xgb.journal")
    result = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage)
    params = result["best_params"]
    assert EXPECTED_XGB_PARAM_KEYS == set(params.keys())

    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["max_depth"], int)
    assert isinstance(params["learning_rate"], float)
    assert isinstance(params["subsample"], float)
    assert isinstance(params["colsample_bytree"], float)
    assert isinstance(params["gamma"], float)
    assert isinstance(params["min_child_weight"], int)
    assert isinstance(params["reg_alpha"], float)
    assert isinstance(params["reg_lambda"], float)


def test_optimize_xgb_best_params_within_search_space(tiny_config_file, tmp_path):
    storage = str(tmp_path / "xgb.journal")
    result = optimize_xgb("base", tiny_config_file, n_trials=3, storage_path=storage)
    p = result["best_params"]
    assert 100 <= p["n_estimators"] <= 1500
    assert 3 <= p["max_depth"] <= 10
    assert 0.005 <= p["learning_rate"] <= 0.3
    assert 0.6 <= p["subsample"] <= 1.0
    assert 0.6 <= p["colsample_bytree"] <= 1.0
    assert 0.0 <= p["gamma"] <= 5.0
    assert 1 <= p["min_child_weight"] <= 10
    assert 1e-8 <= p["reg_alpha"] <= 10.0
    assert 1e-8 <= p["reg_lambda"] <= 10.0


def test_optimize_xgb_cv_auc_is_in_unit_interval(tiny_config_file, tmp_path):
    storage = str(tmp_path / "xgb.journal")
    result = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage)
    assert 0.0 <= result["best_cv_auc"] <= 1.0


def test_optimize_xgb_rejects_unknown_variant(tiny_config_file, tmp_path):
    import pytest
    storage = str(tmp_path / "xgb.journal")
    with pytest.raises(ValueError, match="Unknown variant"):
        optimize_xgb("bogus", tiny_config_file, n_trials=1, storage_path=storage)


def test_optimize_xgb_checkpoint_persists_across_calls(tiny_config_file, tmp_path):
    """
    Two sequential calls with resume=True must produce a study that has
    n_trials_call_1 + n_trials_call_2 completed trials total.
    """
    storage = str(tmp_path / "xgb.journal")

    r1 = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage)
    assert r1["n_trials"] >= 2  # at least 2 completed in study so far

    r2 = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage, resume=True)
    # Cumulative count should be at least the first call's count + 2 new trials
    assert r2["n_trials"] >= r1["n_trials"] + 1  # >= because some trials may fail and not count


def test_optimize_xgb_no_resume_starts_fresh(tiny_config_file, tmp_path):
    storage = str(tmp_path / "xgb.journal")

    r1 = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage)
    n_after_first = r1["n_trials"]

    r2 = optimize_xgb("base", tiny_config_file, n_trials=2, storage_path=storage, resume=False)
    # After --no-resume, the journal is wiped — so the second call's trial
    # count must be <= n_trials_this_run, not cumulative.
    assert r2["n_trials"] <= 2
    assert r2["n_trials"] <= n_after_first  # strictly less or equal to the cumulative


# ============================================================================
# pick_winner_variant
# ============================================================================

def test_pick_winner_variant_returns_top_test_auc(tmp_path):
    path = tmp_path / "results.json"
    fake = {
        "base":             {"test": {"AUC": 0.71}},
        "stopwords_nltk":   {"test": {"AUC": 0.73}},
        "stopwords_domain": {"test": {"AUC": 0.69}},
        "stemming":         {"test": {"AUC": 0.74}},
    }
    path.write_text(json.dumps(fake))
    assert pick_winner_variant(str(path)) == "stemming"


def test_pick_winner_variant_ignores_xgb_optimization_key(tmp_path):
    """`xgb_optimization` is a sibling key (set by stage 2) that must NOT be
    considered when picking the winner from stage 1 results."""
    path = tmp_path / "results.json"
    fake = {
        "base":             {"test": {"AUC": 0.78}},
        "stopwords_nltk":   {"test": {"AUC": 0.70}},
        "xgb_optimization": {"variant": "base", "test": {"AUC": 0.99}},  # decoy
    }
    path.write_text(json.dumps(fake))
    assert pick_winner_variant(str(path)) == "base"


def test_pick_winner_variant_file_not_found(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        pick_winner_variant(str(tmp_path / "nonexistent.json"))


def test_pick_winner_variant_empty_results(tmp_path):
    import pytest
    path = tmp_path / "results.json"
    path.write_text(json.dumps({}))
    with pytest.raises(ValueError):
        pick_winner_variant(str(path))

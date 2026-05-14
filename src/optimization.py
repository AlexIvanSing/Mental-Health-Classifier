"""
Optuna-driven hyperparameter search, in two stages.

Stage 1 — TF-IDF (`optimize_vectorizer`)
    For each variant in {base, stopwords_nltk, stopwords_domain, stemming}
    this module searches independently over the vectorizer space:

        ngram_range  ∈ [(1,1), (1,2), (1,3)]
        min_df       ∈ [1, 5]                  (int)
        max_df       ∈ [0.70, 0.99]            (float)
        sublinear_tf ∈ [True, False]
        max_features ∈ [5000, 10000, 20000, 50000]

    The XGBClassifier hyperparameters are NOT tuned here — they're taken
    verbatim from the YAML config.

Stage 2 — XGBoost (`optimize_xgb`)
    Tunes the 9-dimensional XGB space on the winner of stage 1 (or any
    variant passed explicitly), holding the TF-IDF hyperparameters fixed
    at the values already written to the YAML by stage 1. Uses
    `optuna.storages.JournalStorage` so a killed run resumes from disk.

Objective for both stages: mean 5-fold Stratified CV ROC-AUC on the
training corpus. The test fold is never touched.

Authors (equipo del proyecto TC3002B):
    Aislinn Ruiz Sandoval, Iván Alexander Ramos Ramírez,
    Miguel Ángel Galicia Sánchez, Víctor Alejandro Morales García.
"""

import json
import os
from pathlib import Path

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_ingestion import ingestion
from src.model import build_model
from src.pipeline import CLEANERS
from src.utils import load_config
from src.vectorizer import build_vectorizer

# Mute the noisy per-trial logging — we print our own summary at the end.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# The four variants are the keys of CLEANERS, but we materialize the order
# so the public API surface is explicit and testable.
VARIANTS = ("base", "stopwords_nltk", "stopwords_domain", "stemming")


def _load_corpus(config: dict) -> tuple[list, list]:
    """
    Load and ingest the training CSV, returning the concatenated-text column
    plus the binary target. Does NOT apply any cleaning — that's the variant's
    job.
    """
    df = ingestion(
        data_url=config["data"]["train_path"],
        expected_columns=config["data"]["expected_columns"],
        text_columns=config["data"]["text_columns"],
        target_column=config["data"]["target_column"],
        value_true=config["data"]["value_true"],
        value_false=config["data"]["value_false"],
    )
    text_col = "_".join(config["data"]["text_columns"])
    X = df[text_col].astype(str).tolist()
    y = df[config["data"]["target_column"]].tolist()
    return X, y


def _build_trial_config(base_config: dict, trial: optuna.Trial) -> dict:
    """
    Construct a config dict for one trial by overlaying Optuna-suggested
    vectorizer hyperparameters on top of the base config.
    """
    ngram_choices = [(1, 1), (1, 2), (1, 3)]
    ngram_idx = trial.suggest_categorical("ngram_idx", [0, 1, 2])
    ngram = list(ngram_choices[ngram_idx])

    cfg = {
        "vectorizer": {
            "ngram_range":   ngram,
            "min_df":        trial.suggest_int("min_df", 1, 5),
            "max_df":        trial.suggest_float("max_df", 0.70, 0.99),
            "sublinear_tf":  trial.suggest_categorical("sublinear_tf", [True, False]),
            "max_features":  trial.suggest_categorical("max_features", [5000, 10000, 20000, 50000]),
        },
        "model": base_config["model"],
    }
    return cfg


def optimize_vectorizer(
    variant: str,
    config_path: str,
    n_trials: int = 30,
    timeout: int | None = None
) -> dict:
    """
    Run an Optuna study over TF-IDF hyperparameters for one preprocessing variant.

    The chosen cleaner is applied **once** to the entire training corpus before
    the cross-validation loop. This is a deliberate optimization: the cleaning
    is a deterministic, leakage-free function of each row in isolation, so
    running it outside the CV folds is mathematically equivalent to running it
    inside the Pipeline (as `build_pipeline` does at deployment time) but ~5x
    faster during the search.

    Parameters
    ----------
    variant : str
        One of the keys of `src.pipeline.CLEANERS`.
    config_path : str
        Path to the YAML config (used for data paths and XGB hyperparameters).
    n_trials : int, default 30
        Number of Optuna trials.

    Returns
    -------
    dict
        {
            "variant":           <str>,
            "best_cv_auc":       <float>,
            "best_params":       <dict>,   # vectorizer hyperparams as YAML-friendly types
            "n_trials":          <int>,
            "all_trial_scores":  <list[float]>,
        }
    """
    if variant not in CLEANERS:
        raise ValueError(
            f"Unknown variant {variant!r}. Expected one of: {sorted(CLEANERS)}"
        )

    base_config = load_config(config_path)
    X_raw, y = _load_corpus(base_config)

    cleaner = CLEANERS[variant]
    X_clean = [cleaner(x) for x in X_raw]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=base_config["training"]["random_state"],
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = _build_trial_config(base_config, trial)
        vectorizer = build_vectorizer(trial_cfg)
        model = build_model(trial_cfg)

        from sklearn.pipeline import Pipeline as SkPipeline
        pipe = SkPipeline([("tfidf", vectorizer), ("clf", model)])

        try:
            scores = cross_val_score(
                pipe, X_clean, y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                error_score="raise",
            )
        except ValueError:
            # Pathological params (e.g. min_df too high for the corpus) prune
            # the vocabulary to zero. Treat as worst-case AUC so Optuna avoids
            # this region of the search space rather than crashing the study.
            return 0.0

        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=base_config["training"]["random_state"]
        ),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # Resolve the ngram_idx back to a real tuple for human consumption /
    # YAML overwrite. Everything else is already a primitive.
    ngram_choices = [[1, 1], [1, 2], [1, 3]]
    best_raw = study.best_params
    best_params = {
        "ngram_range":  ngram_choices[best_raw["ngram_idx"]],
        "min_df":       int(best_raw["min_df"]),
        "max_df":       float(round(best_raw["max_df"], 4)),
        "sublinear_tf": bool(best_raw["sublinear_tf"]),
        "max_features": int(best_raw["max_features"]),
    }

    all_scores = [t.value for t in study.trials if t.value is not None]

    print(
        f"[{variant}] best CV AUC = {study.best_value:.4f}  "
        f"({n_trials} trials, params: {best_params})"
    )

    return {
        "variant":          variant,
        "best_cv_auc":      float(round(study.best_value, 4)),
        "best_params":      best_params,
        "n_trials":         n_trials,
        "all_trial_scores": [float(round(s, 4)) for s in all_scores],
    }


def run_all_optimizations(
    config_path: str = "configs/default.yaml",
    n_trials: int = 30,
    timeout: int | None = None,
    variants: tuple[str, ...] = VARIANTS,
) -> dict:
    """
    Run `optimize_vectorizer` for every variant in sequence and aggregate
    the results.

    Parameters
    ----------
    config_path : str
        Path to the base YAML config (data paths and XGB hyperparameters
        come from here; vectorizer params will be overwritten by Optuna).
    n_trials : int
        Number of trials per variant.
    variants : tuple of str
        Variants to optimize. Defaults to all four.

    Returns
    -------
    dict
        Keyed by variant name; each value is the dict returned by
        `optimize_vectorizer`.
    """
    print(f"Running Optuna optimization for variants: {variants}")
    print(f"n_trials per variant: {n_trials}  |  timeout per variant: {timeout}s")
    print(f"Base config: {config_path}\n")

    results = {}
    for v in variants:
        print(f"--- Optimizing variant: {v} ---")
        results[v] = optimize_vectorizer(v, config_path, n_trials=n_trials, timeout=timeout)
        print()

    print("All variants done.")
    return results


# ============================================================================
# Stage 2 — XGBoost tuning on the winning variant
# ============================================================================

# Search space for the 9 tunable XGBClassifier hyperparameters.
# `scale_pos_weight`, `eval_metric` and `random_state` are NOT tuned —
# they're domain decisions (n_neg/n_pos, "auc", 42).
_XGB_SPACE = {
    "n_estimators":     ("int",   100,  1500),
    "max_depth":        ("int",   3,    10),
    "learning_rate":    ("logf",  0.005, 0.3),
    "subsample":        ("float", 0.6,  1.0),
    "colsample_bytree": ("float", 0.6,  1.0),
    "gamma":            ("float", 0.0,  5.0),
    "min_child_weight": ("int",   1,    10),
    "reg_alpha":        ("logf",  1e-8, 10.0),
    "reg_lambda":       ("logf",  1e-8, 10.0),
}


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Map the search-space spec to Optuna `suggest_*` calls."""
    params = {}
    for name, spec in _XGB_SPACE.items():
        kind, lo, hi = spec
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)
        elif kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "logf":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            raise ValueError(f"Unknown space kind: {kind}")
    return params


def optimize_xgb(
    variant: str,
    config_path: str,
    n_trials: int = 30,
    storage_path: str | None = None,
    resume: bool = True,
) -> dict:
    """
    Run an Optuna study over XGBClassifier hyperparameters for one variant.

    The TF-IDF vectorizer is held fixed at the values already in the YAML
    (typically those written by `optimize_vectorizer`). The corpus is
    cleaned once with the variant's cleaner and then TF-IDF-transformed
    once into a sparse matrix — that matrix is reused across all trials
    and all CV folds, so each trial only pays for XGBoost training.

    The study uses **JournalStorage** so that:
      - Killed runs resume from the last completed trial on the next call.
      - The full trial history is auditable on disk.

    A progress callback prints `[xgb-<variant>] Trial X/N  AUC=...  best=...`
    after every completed trial.

    Parameters
    ----------
    variant : str
        One of the keys of `src.pipeline.CLEANERS`.
    config_path : str
        Path to the YAML config. Used for data paths, the (fixed) TF-IDF
        hyperparameters, and the fixed XGB knobs (`scale_pos_weight`,
        `eval_metric`, `random_state`).
    n_trials : int, default 30
        Number of Optuna trials to run **in this invocation**. If the
        journal already contains completed trials and `resume=True`,
        the next `n_trials` are appended to the study.
    storage_path : str, optional
        Path to the journal file. Defaults to
        `reports/optuna_xgb_<variant>.journal`.
    resume : bool, default True
        If False, deletes any existing journal before starting.

    Returns
    -------
    dict
        {
            "variant":           <str>,
            "best_cv_auc":       <float>,
            "best_params":       <dict>,   # XGB hyperparams as YAML-friendly types
            "n_trials":          <int>,   # total trials in the study (cumulative)
            "n_trials_this_run": <int>,
            "all_trial_scores":  <list[float]>,
            "storage_path":      <str>,
        }
    """
    if variant not in CLEANERS:
        raise ValueError(
            f"Unknown variant {variant!r}. Expected one of: {sorted(CLEANERS)}"
        )

    base_config = load_config(config_path)
    X_raw, y = _load_corpus(base_config)

    cleaner = CLEANERS[variant]
    X_clean = [cleaner(x) for x in X_raw]

    # Vectorize ONCE upfront so every trial reuses the same sparse matrix.
    # This is mathematically equivalent to vectorizing inside each fold
    # only because the TF-IDF hyperparams are fixed across trials — if
    # they were being tuned, we'd be leaking the val fold into the
    # vocabulary. Here they're frozen, so this is sound.
    vectorizer = build_vectorizer(base_config)
    X_mat = vectorizer.fit_transform(X_clean)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=base_config["training"]["random_state"],
    )

    # --- Journal storage (checkpointing) ---
    if storage_path is None:
        storage_path = f"reports/optuna_xgb_{variant}.journal"
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    if not resume and os.path.exists(storage_path):
        os.remove(storage_path)
        print(f"  [{variant}] --no-resume: deleted existing journal at {storage_path}")

    storage = JournalStorage(JournalFileBackend(storage_path))
    study_name = f"xgb_{variant}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=base_config["training"]["random_state"]
        ),
    )

    n_already = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_already > 0 and resume:
        print(f"  [{variant}] resuming study with {n_already} completed trials in journal")

    def objective(trial: optuna.Trial) -> float:
        xgb_params = _suggest_xgb_params(trial)

        trial_cfg = {
            "model": {
                # required keys for build_model
                **base_config["model"],
                # overrides from Optuna
                **xgb_params,
            },
        }

        model = build_model(trial_cfg)

        try:
            scores = cross_val_score(
                model, X_mat, y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                error_score="raise",
            )
        except (ValueError, Exception):
            return 0.0

        return float(scores.mean())

    # --- Progress callback ---
    progress_state = {"trial_idx": n_already}

    def _progress_cb(_study, trial):
        progress_state["trial_idx"] += 1
        idx = progress_state["trial_idx"]
        total = n_already + n_trials
        best = _study.best_value if len(_study.trials) > 0 else float("nan")
        val = trial.value if trial.value is not None else float("nan")
        print(
            f"  [xgb-{variant}] Trial {idx}/{total}  AUC={val:.4f}  best={best:.4f}",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_progress_cb],
    )

    # Snapshot best params as YAML-friendly Python primitives.
    best_raw = study.best_params
    best_params = {
        "n_estimators":     int(best_raw["n_estimators"]),
        "max_depth":        int(best_raw["max_depth"]),
        "learning_rate":    float(round(best_raw["learning_rate"], 6)),
        "subsample":        float(round(best_raw["subsample"], 4)),
        "colsample_bytree": float(round(best_raw["colsample_bytree"], 4)),
        "gamma":            float(round(best_raw["gamma"], 4)),
        "min_child_weight": int(best_raw["min_child_weight"]),
        "reg_alpha":        float(round(best_raw["reg_alpha"], 8)),
        "reg_lambda":       float(round(best_raw["reg_lambda"], 8)),
    }

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    all_scores = [t.value for t in completed if t.value is not None]

    print(
        f"[xgb-{variant}] best CV AUC = {study.best_value:.4f}  "
        f"({len(completed)} completed trials in study; params: {best_params})"
    )

    return {
        "variant":            variant,
        "best_cv_auc":        float(round(study.best_value, 4)),
        "best_params":        best_params,
        "n_trials":           len(completed),
        "n_trials_this_run":  n_trials,
        "all_trial_scores":   [float(round(s, 4)) for s in all_scores],
        "storage_path":       storage_path,
    }


def pick_winner_variant(
    consolidated_path: str = "reports/optimization_results.json",
) -> str:
    """
    Inspect the consolidated stage-1 results and return the variant with
    the highest Test AUC.

    Parameters
    ----------
    consolidated_path : str
        Path to the JSON produced by `scripts/run_optimization_pipeline.py`.

    Returns
    -------
    str
        The winning variant name.

    Raises
    ------
    FileNotFoundError
        If the consolidated results file doesn't exist yet.
    ValueError
        If the file exists but contains no usable results.
    """
    path = Path(consolidated_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated results not found at {consolidated_path}. "
            f"Run `python scripts/run_optimization_pipeline.py` first."
        )

    with open(path) as f:
        data = json.load(f)

    # The top-level keys are variant names; each value has a "test" block
    # with an "AUC" key. (The optional "xgb_optimization" key, if present,
    # is intentionally ignored — we always pick from the TF-IDF stage.)
    candidates = {
        k: v.get("test", {}).get("AUC")
        for k, v in data.items()
        if k != "xgb_optimization" and isinstance(v, dict) and "test" in v
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}

    if not candidates:
        raise ValueError(
            f"No usable variant results found in {consolidated_path}"
        )

    winner = max(candidates, key=candidates.get)
    return winner

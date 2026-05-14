"""
Optuna-driven TF-IDF hyperparameter search per preprocessing variant.

For each variant in {base, stopwords_nltk, stopwords_domain, stemming} this
module searches independently over the vectorizer space:

    ngram_range  ∈ [(1,1), (1,2), (1,3)]
    min_df       ∈ [1, 5]                  (int)
    max_df       ∈ [0.70, 0.99]            (float)
    sublinear_tf ∈ [True, False]
    max_features ∈ [5000, 10000, 20000, 50000]

The XGBClassifier hyperparameters are NOT tuned here — they are taken
verbatim from the YAML config. The optimization metric is 5-fold Stratified
CV ROC-AUC on the training corpus.

Authors (equipo del proyecto TC3002B):
    Aislinn Ruiz Sandoval, Iván Alexander Ramos Ramírez,
    Miguel Ángel Galicia Sánchez, Víctor Alejandro Morales García.
"""

import optuna
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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
    print(f"n_trials per variant: {n_trials}")
    print(f"Base config: {config_path}\n")

    results = {}
    for v in variants:
        print(f"--- Optimizing variant: {v} ---")
        results[v] = optimize_vectorizer(v, config_path, n_trials=n_trials)
        print()

    print("All variants done.")
    return results

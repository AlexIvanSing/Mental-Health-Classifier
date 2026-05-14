# Mental Health Classifier — Suicide Ideation Detection

> Binary text classifier that flags suicide-related ideation in Reddit-style posts (title + body), built on a reproducible scikit-learn pipeline with XGBoost. End-to-end CLI for training, evaluation against a labeled test fold, and blind inference — plus two-stage Optuna optimization (TF-IDF + XGBoost), checkpointable studies, and auto-generated markdown reports.

---

## Overview

This repository implements an end-to-end machine learning pipeline for the binary classification task **`is_suicide ∈ {0, 1}`** on user-generated text. The system is designed to be:

- **Reproducible** — every hyperparameter, path and split lives in YAML configs under `configs/`. No magic numbers in code.
- **Modular** — ingestion, preprocessing, vectorization, model, evaluation, reporting and optimization are independent modules, each unit-tested in isolation.
- **Configurable preprocessing** — four interchangeable text-cleaning variants (`base`, `stopwords_nltk`, `stopwords_domain`, `stemming`) selected via the YAML config. Each can be A/B-tested against the others.
- **Two-stage tuning with Optuna** — first the TF-IDF hyperparameters are tuned independently per variant; then the XGBoost hyperparameters are tuned on the winning variant. Studies persist to disk so a killed process resumes where it left off.
- **CLI-driven** — three subcommands cover the day-to-day lifecycle (`train`, `evaluate`, `predict`); two orchestration scripts cover the optimization lifecycle.
- **Train/serve consistent** — text cleaning lives *inside* the sklearn `Pipeline` as a picklable `FunctionTransformer`, so the exact same preprocessing runs at training time and at inference time. No skew, regardless of variant.
- **Self-documenting runs** — every training and evaluation invocation drops JSON metrics + ROC PNG + confusion-matrix PNG + a `*.md` report into a variant-specific folder under `reports/`. Optimization runs also produce a comparison report.

The repository is academic in scope (TC3002B, Tecnológico de Monterrey) but follows the structural conventions of a production ML codebase.

**Team:** Aislinn Ruiz Sandoval · Iván Alexander Ramos Ramírez · Miguel Ángel Galicia Sánchez · Víctor Alejandro Morales García.

---

## How it works

```
            ┌─────────────────────────────────────────────────────────┐
            │            configs/<variant>.yaml + Optuna              │
            └─────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────┬───────────┴───────────┬──────────────────┐
        ▼                  ▼                       ▼                  ▼
   TRAIN FLOW         EVALUATE FLOW          PREDICT FLOW       OPTIMIZATION FLOW
   (labeled)          (labeled test)         (unlabeled csv)    (Optuna)
        │                  │                       │                  │
        ▼                  ▼                       ▼                  ▼
   ingestion()        ingestion()             ingestion()        Stage 1: TF-IDF
        │                  │                       │              ──────────────
        ▼                  ▼                       ▼              For each of 4
   build_pipeline       load_pipeline()         load_pipeline()   variants:
   ┌──────────────┐         │                       │             optimize_vectorizer
   │ Cleaner      │         │                       │              → 30 trials
   │  (variant-   │         ▼                       ▼              → 5-fold CV AUC
   │   specific)  │   predict + predict_proba  predict + proba    → writes best
   │      ↓       │         │                       │                hyperparams to
   │ TfidfVect    │         ▼                       ▼                its YAML
   │      ↓       │   evaluation.evaluate     write CSV            
   │ XGBClassifier│   + generate_report        (text_id,           Stage 2: XGB
   └──────────────┘         │                  prediction,         ──────────────
        │                   │                  probability)        pick_winner_variant()
        ▼                   ▼                                      → load winner YAML
   StratifiedKFold     reports/<variant>/                          optimize_xgb
   CV + holdout fit    ├─ roc_curve_test.png                        → 30 trials
        │              ├─ confusion_matrix_test.png                 → JournalStorage
        ▼              ├─ test_metrics.json                           checkpoint
   joblib.dump         └─ final_evaluation.md                       → progress per
        │                                                             trial printed
        ▼                                                           → writes best XGB
   evaluation.evaluate                                                hyperparams to
   + generate_report                                                  winner's YAML
        │                                                              + early stopping
        ▼                                                            
   reports/<variant>/                                              Final train+evaluate
   ├─ roc_curve_val.png                                            on tuned pipeline,
   ├─ confusion_matrix_val.png                                     reports/optimization.md
   ├─ training_metrics.json                                        renders the comparison.
   └─ training_report.md
```

---

## Project structure

```
Suicide_classifier/
├── configs/
│   ├── default.yaml                       # Baseline variant (preprocessing.variant: "base")
│   ├── variant_stopwords_nltk.yaml        # NLTK stopwords (negations preserved)
│   ├── variant_stopwords_domain.yaml      # Curated grammar-only stopword list
│   └── variant_stemming.yaml              # Porter stemmer
├── data/
│   ├── data_train.csv                     # Training split (1,516 rows)
│   └── data_test_fold1.csv                # Held-out test fold (252 rows)
├── models/                                # Trained .joblib artifacts (one per variant)
├── notebooks/
│   ├── EDA.ipynb                          # Exploratory data analysis
│   └── pipeline_test.ipynb                # Pipeline smoke tests
├── reports/                               # Auto-populated by train/evaluate/optimization
│   ├── <variant>/                         # One subfolder per variant after training
│   │   ├─ roc_curve_{val,test}.png
│   │   ├─ confusion_matrix_{val,test}.png
│   │   ├─ {training,test}_metrics.json
│   │   ├─ training_report.md
│   │   └─ final_evaluation.md
│   ├── optimization_results.json          # Consolidated Optuna results (both stages)
│   ├── optimization.md                    # Auto-rendered comparison report
│   └── optuna_xgb_<winner>.journal        # XGB study checkpoint
├── scripts/
│   ├── run_optimization_pipeline.py       # Stage 1: tune TF-IDF on all 4 variants
│   ├── run_xgb_optimization.py            # Stage 2: tune XGB on the winning variant
│   └── generate_optimization_report.py    # Render reports/optimization.md
├── src/
│   ├── __main__.py                        # CLI: python -m src {train|evaluate|predict}
│   ├── data_ingestion.py                  # Schema validation, loading, NA handling, label mapping
│   ├── preprocessing.py                   # clean_text + 3 variants (stopwords_nltk, _domain, stemming)
│   ├── vectorizer.py                      # TfidfVectorizer factory
│   ├── model.py                           # XGBClassifier factory (9-param config)
│   ├── pipeline.py                        # sklearn Pipeline assembly + CLEANERS registry
│   ├── training.py                        # train() + train_pipeline() orchestrator
│   ├── inference.py                       # run_inference() + CLI wrapper (blind prediction)
│   ├── evaluate_cli.py                    # run_evaluation() + CLI wrapper (labeled scoring)
│   ├── evaluation.py                      # compute_metrics, plots, evaluate, generate_report
│   ├── optimization.py                    # optimize_vectorizer + optimize_xgb + pick_winner_variant
│   └── utils.py                           # YAML loader
├── tests/                                 # pytest suite (one file per src module)
└── pyproject.toml
```

---

## Module reference

| Module | Public API | Purpose |
|---|---|---|
| `data_ingestion` | `ingestion`, `split_dataset`, `schema_validation`, `data_loader`, `data_mapping`, `handle_missing_data`, `concatenate_df` | Load CSV, validate schema, resolve nulls, concatenate text columns, map labels, stratified train/test split. |
| `preprocessing` | `clean_text`, `clean_text_with_stopwords_nltk`, `clean_text_with_stopwords_domain`, `clean_text_with_stemming`, `tokenize_text`, `preprocessing` | Baseline cleaner + three variants. Negations always preserved across all variants. |
| `vectorizer` | `build_vectorizer` | Config-driven `TfidfVectorizer`. |
| `model` | `build_model` | Config-driven `XGBClassifier`. Supports the full 9-hyperparameter space (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`) plus `scale_pos_weight` and `eval_metric`. |
| `pipeline` | `build_pipeline`, `CLEANERS` | Assemble `cleaner → tfidf → xgboost` Pipeline. Cleaner picked from `CLEANERS` registry via `config["preprocessing"]["variant"]`. |
| `training` | `train`, `train_pipeline`, `main` | Stratified 80/20 split + 5-fold Stratified CV (AUC) + final fit. **Early stopping** activated when `config["model"]["early_stopping_rounds"]` is present: refits the XGB step with an inner 90/10 eval_set so trees stop growing once validation AUC plateaus. |
| `inference` | `run_inference`, `run_inference_cli`, `load_pipeline`, `predict` | **Blind** prediction (no ground truth required). |
| `evaluate_cli` | `run_evaluation`, `run_evaluation_cli` | **Scored** prediction: requires ground-truth labels, computes full metrics, writes plots + JSON + markdown. |
| `evaluation` | `compute_metrics`, `plot_roc_curve`, `plot_confusion_matrix`, `evaluate`, `generate_report` | TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1; ROC and confusion-matrix PNGs; self-contained markdown report. |
| `optimization` | `optimize_vectorizer`, `optimize_xgb`, `pick_winner_variant`, `run_all_optimizations` | Two-stage Optuna search. Stage 1 tunes the 5-param TF-IDF space per variant. Stage 2 tunes the 9-param XGB space on the winning variant with `JournalStorage` checkpointing. |
| `utils` | `load_config` | Parse a YAML config file into a dict. |

> **Two distinct "pipelines"** — `pipeline.build_pipeline()` returns the **sklearn Pipeline** (the serializable model object). `training.train_pipeline()` is the **workflow orchestrator**. The first lives inside the second.
>
> **`predict` vs `evaluate`** — `predict` is for *unlabeled* data: emits `text_id, prediction, probability`. `evaluate` is for *labeled* data: computes the full metrics dict against ground truth and dumps a markdown report.

---

## Preprocessing variants

The four cleaners are selected via `config["preprocessing"]["variant"]`:

| Variant | What it does | Rationale |
|---|---|---|
| `base` | Encoding repair + URL/mention/hashtag/emoji/special-char removal + lowercase + whitespace normalization. | Minimal-loss baseline. Relies on `max_df=0.95` to filter corpus-wide stopwords implicitly. |
| `stopwords_nltk` | `base` + remove NLTK English stopwords **except** negations (`no`, `not`, `never`, `nor`, `neither`, `without`, `nothing`, `nobody`, `nowhere`, `none`). | Standard stopword filtering with a critical exception: negations flip polarity in suicide-ideation text and must be preserved. |
| `stopwords_domain` | `base` + remove a hand-curated list of **only** purely grammatical tokens (articles, prepositions, conjunctions, non-negated auxiliaries). Emotional content explicitly kept. | More conservative than NLTK: never drops tokens like "alone", "die", "hopeless" that generic lists sometimes include. |
| `stemming` | `base` + Porter stemming token-by-token. | Collapses morphological variants (`die`/`died`/`dying` → `die`) to reduce vocabulary in a small-corpus setting. |

All variants are exposed in `src.pipeline.CLEANERS`; adding a new one is a one-line registry entry.

---

## Configuration

Every runtime parameter lives in YAML. Each variant has its own config; they share schema but differ in `preprocessing.variant`, `paths.*` and the post-Optuna `vectorizer`/`model` blocks.

```yaml
data:
  train_path: "data/data_train.csv"
  test_path:  "data/data_test_fold1.csv"
  expected_columns: [user_id, text_id, title, text, is_suicide]
  text_columns:     [title, text]
  target_column:    "is_suicide"
  value_true:       "yes"
  value_false:      "no"

preprocessing:
  variant: "base"                         # base | stopwords_nltk | stopwords_domain | stemming

paths:
  model_output:        "models/model.joblib"
  reports_dir:         "reports"
  roc_val_output:      "reports/roc_curve_val.png"
  cm_val_output:       "reports/confusion_matrix_val.png"
  metrics_val_output:  "reports/training_metrics.json"
  report_val_output:   "reports/training_report.md"
  roc_test_output:     "reports/roc_curve_test.png"
  cm_test_output:      "reports/confusion_matrix_test.png"
  metrics_test_output: "reports/test_metrics.json"
  report_test_output:  "reports/final_evaluation.md"

# Filled by Optuna stage 1 (per-variant).
vectorizer:
  ngram_range:   [1, 2]
  min_df:        2
  max_df:        0.95
  sublinear_tf:  true
  max_features:  10000

# Filled by Optuna stage 2 (only the winning variant).
# Fields beyond the first six are optional; build_model() uses sensible defaults.
model:
  n_estimators:      500
  max_depth:         6
  learning_rate:     0.05
  eval_metric:       "auc"
  scale_pos_weight:  0.93
  random_state:      42
  subsample:         1.0       # optional; default 1.0
  colsample_bytree:  1.0       # optional; default 1.0
  gamma:             0.0       # optional; default 0.0
  min_child_weight:  1         # optional; default 1
  reg_alpha:         0.0       # optional; default 0.0 (L1)
  reg_lambda:        1.0       # optional; default 1.0 (L2)
  early_stopping_rounds: 30    # optional; if present, training uses inner 90/10 eval_set

training:
  test_size:    0.2
  random_state: 42
  n_splits:     5
```

---

## Quick start

### 1. Install

```powershell
venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.11+. Runtime deps: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `nltk`, `joblib`, `pyyaml`, `matplotlib`, `optuna`, `ftfy`.

### 2. Day-to-day commands

```powershell
# Train on the active variant (reads configs/default.yaml by default)
python -m src train

# Evaluate the saved model against the labeled test fold
python -m src evaluate --input data/data_test_fold1.csv

# Predict on unlabeled data
python -m src predict --input <csv> --output predictions.csv
```

Each subcommand accepts `--config configs/<variant>.yaml` to switch variants.

### 3. Two-stage Optuna optimization

```powershell
# Stage 1: tune TF-IDF for ALL 4 variants (30 trials each), then train + evaluate each
python scripts/run_optimization_pipeline.py --n-trials 30
# → writes best vectorizer hyperparams back to each variant's YAML
# → produces reports/<variant>/* and reports/optimization_results.json

# Stage 2: tune XGB on the winning variant (auto-detected from stage-1 results)
# it has to be run as an administrator 
python scripts/run_xgb_optimization.py --n-trials 30
# → writes best model hyperparams (incl. early_stopping_rounds) to the winner's YAML
# → checkpoints to reports/optuna_xgb_<winner>.journal
# → appends results to reports/optimization_results.json

# Render the comparison markdown
python scripts/generate_optimization_report.py
# → reports/optimization.md
```

Override the auto-detection of the winner:
```powershell
python scripts/run_xgb_optimization.py --variant base
```

### 4. Resuming an interrupted optimization

The XGB study uses **`optuna.storages.JournalStorage`**, which persists every completed trial to a journal file. If the process is killed mid-run, simply re-running the same command resumes from the last completed trial — no flags needed.

```powershell
python scripts/run_xgb_optimization.py --n-trials 30
# Ctrl+C after trial 14...
python scripts/run_xgb_optimization.py --n-trials 30
# Resumes from trial 15. Total trials will be 30 across both runs.
```

To start over from scratch (ignoring the checkpoint):
```powershell
python scripts/run_xgb_optimization.py --no-resume
```

Stage 1 (TF-IDF) is in-memory and not checkpointed — its trials are fast (<10 s each) so resuming a partial run by re-doing the missing trials is cheap.

---

## Optuna search spaces

**Stage 1 — TF-IDF (5 dimensions, per variant):**

| Param | Space |
|---|---|
| `ngram_range` | `{(1,1), (1,2), (1,3)}` |
| `min_df` | int in `[1, 5]` |
| `max_df` | float in `[0.70, 0.99]` |
| `sublinear_tf` | `{True, False}` |
| `max_features` | `{5000, 10000, 20000, 50000}` |

**Stage 2 — XGBoost (9 dimensions, winning variant only):**

| Param | Space |
|---|---|
| `n_estimators` | int in `[100, 1500]` |
| `max_depth` | int in `[3, 10]` |
| `learning_rate` | log-uniform in `[0.005, 0.3]` |
| `subsample` | float in `[0.6, 1.0]` |
| `colsample_bytree` | float in `[0.6, 1.0]` |
| `gamma` | float in `[0.0, 5.0]` |
| `min_child_weight` | int in `[1, 10]` |
| `reg_alpha` | log-uniform in `[1e-8, 10.0]` |
| `reg_lambda` | log-uniform in `[1e-8, 10.0]` |

`scale_pos_weight`, `eval_metric` and `random_state` are not tuned — they're domain decisions (`n_neg/n_pos`, `auc`, `42`).

Both studies use TPE sampler with `random_state=42`. Objective: mean **5-fold StratifiedKFold ROC-AUC** on `data/data_train.csv`. The test fold (`data_test_fold1.csv`) is never touched during search — only at the very end for the reported Test AUC.

**Early stopping** is **not** applied during CV (the `n_estimators` proposed by Optuna is the exact tree count used in each fold, keeping fold scores apples-to-apples). It **is** applied in the final training run via an inner 90/10 split of the 80% train partition, when `early_stopping_rounds` is present in the YAML.

---

## Latest results

> Refer to `reports/optimization.md` for the auto-generated comparison after running stage 1 + stage 2. The baseline pre-Optuna numbers below are from the initial commit (no tuning).

| | Val AUC | Test AUC | Recall | FPR |
|---|---|---|---|---|
| Baseline (defaults, no tuning) | 0.7264 | 0.739 | 0.669 | 0.295 |

---

## Testing

```powershell
pytest
```

Coverage spans every module: ingestion, preprocessing (all 4 variants), vectorizer, model, pipeline, evaluation (incl. `generate_report`), inference, evaluate_cli, and optimization (incl. `optimize_xgb` with `n_trials=1` smoke tests + a checkpoint-resume test that verifies the journal persists trials across calls).

Shared fixtures live in `tests/conftest.py` (toy pipeline config, trained-and-serialized pipeline on `tmp_path`, synthetic labeled CSV, full config dict with every path rooted in `tmp_path` so tests stay hermetic).

---

## Design decisions

- **TF-IDF + XGBoost over deep learning.** For ~1,500 training rows of English-language Reddit-style posts, a sparse linear feature space with a gradient-boosted tree classifier reaches competitive AUC at a fraction of the training cost — and is much easier to debug, version, and deploy.
- **`scale_pos_weight = 0.93`.** Empirical class ratio `n_neg / n_pos` in the training set; passed to XGBoost rather than oversampling because it preserves the original distribution at evaluation time.
- **Cleaning inside the `Pipeline`.** Wrapping the chosen `clean_text*` in a `FunctionTransformer` guarantees inference applies the *same* preprocessing as training — no separate "serve" path can drift, regardless of variant.
- **Two-stage tuning instead of joint.** TF-IDF (5 dims) + XGBoost (9 dims) = 14-dim joint space, intractable in 30 trials. Tuning the vectorizer first and freezing it before tuning the model decomposes the problem and converges on a much smaller budget. Trade-off: ignores interactions between the two — but those interactions are empirically minor at this scale.
- **Early stopping only at final fit.** Inside CV, every fold trains exactly the `n_estimators` Optuna proposes, so fold scores are directly comparable. After Optuna picks the winner, the final training run uses an inner eval split + `early_stopping_rounds=30` so the deployed model stops growing trees once validation AUC plateaus, avoiding overfitting at deployment without distorting the CV signal.
- **`predict` and `evaluate` as separate subcommands.** `predict` is for production-style blind inference (no labels). `evaluate` is for offline scoring against a labeled fold. Splitting them keeps `run_inference` ignorant of ground truth and makes `evaluate` the obvious entry point for the final test report.
- **JournalStorage for XGB checkpoints.** XGB studies are 5–10× slower than TF-IDF studies (deeper trees, more rounds). Persisting the study to a journal file means a Ctrl+C or laptop sleep doesn't waste the trials already done.

---

## Roadmap

`[x]` is shipped, `[ ]` is pending.

### Shipped

- [x] Project scaffolding (`pyproject.toml`, `src/`, `tests/`, `configs/`, `notebooks/`, `models/`, `data/`, `reports/`, `scripts/`).
- [x] `data_ingestion`: schema validation, CSV loader, NaN handling, `title + text` concatenation, `yes/no → 1/0` mapping, stratified `split_dataset`, early `None` return on schema mismatch.
- [x] `preprocessing`: baseline `clean_text` + 3 variants (`stopwords_nltk`, `stopwords_domain`, `stemming`). Negation whitelist enforced everywhere.
- [x] `vectorizer`: YAML-driven `TfidfVectorizer`.
- [x] `model`: YAML-driven `XGBClassifier` with 9 tunable hyperparameters + back-compat defaults.
- [x] `pipeline`: end-to-end `cleaner → tfidf → xgboost` with a `CLEANERS` registry. Variant chosen via config.
- [x] `training`: stratified 80/20 + 5-fold Stratified CV (AUC); optional `early_stopping_rounds` activated by YAML.
- [x] `evaluation`: TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1; ROC and confusion-matrix PNGs; `generate_report` markdown renderer.
- [x] `inference`: blind CLI (`python -m src predict`).
- [x] `evaluate_cli`: labeled-scoring CLI (`python -m src evaluate`).
- [x] `optimization` stage 1: `optimize_vectorizer` (5-dim TF-IDF space, 30 trials, 5-fold CV).
- [x] `optimization` stage 2: `optimize_xgb` (9-dim XGB space, 30 trials, 5-fold CV) with `JournalStorage` checkpointing and per-trial progress logging.
- [x] `pick_winner_variant` to wire stage 1 → stage 2 automatically.
- [x] Orchestrator scripts under `scripts/`: `run_optimization_pipeline.py`, `run_xgb_optimization.py`, `generate_optimization_report.py`.
- [x] EDA notebook with honest outlier analysis and decision rationale.
- [x] Full pytest coverage of every module (~215 tests passing).

### Next steps (short term)

- [ ] **`reports/algorithm_justification.md`** — hand-written: why TF-IDF + XGBoost over deep learning, with literature references.
- [ ] **Threshold tuning**. Default is 0.5; in this domain a false negative is costlier than a false positive. After inspecting the precision-recall curve, evaluate thresholds in `{0.35, 0.40, 0.45}` and pick by recall floor.
- [ ] **Error analysis**. Surface the top-K false negatives and false positives, look for systematic patterns (length, vocabulary, sarcasm), feed findings back into preprocessing.

### Next steps (mid term)

- [ ] **User-disjoint splits**: currently ~14% of test rows come from users also present in train. Stratify by `user_id` instead of by row for a stricter methodological story.
- [ ] **Coverage report**: `pytest --cov=src --cov-report=term-missing`, target ≥ 80%.
- [ ] **Document the train/test split provenance**: the script that produced `data_train.csv` and `data_test_fold1.csv` isn't checked in. Wrap `split_dataset` so the split is reproducible from a single raw source.

### Next steps (stretch)

- [ ] **`TruncatedSVD` post-TF-IDF** — only worth it if the sparse matrix becomes a memory/speed bottleneck.
- [ ] **Strategy pattern for preprocessing**: define a `Preprocessor` interface so cleaning variants can be swapped from config without touching `pipeline.py`.
- [ ] **Combined variants**: e.g., `stemming + stopwords_domain`. Worth trying if individual variants show real lift over baseline.
- [ ] **Pruning during XGB CV**: report intermediate fold scores so Optuna can prune unpromising trials early. Requires reporting per-fold AUCs in the objective.

---

## Disclaimer

This classifier is an academic exercise. It is **not** a clinical screening tool and must not be used to make decisions about anyone's mental health or safety. If you or someone you know is in crisis, please contact a local emergency service or suicide-prevention helpline.

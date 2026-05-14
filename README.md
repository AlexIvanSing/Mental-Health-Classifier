# Mental Health Classifier — Suicide Ideation Detection

> Binary text classifier that flags suicide-related ideation in social media posts (Reddit-style title + body), built on a reproducible scikit-learn pipeline with XGBoost. End-to-end CLI for training, evaluation against a labeled test fold, and blind inference — every run drops metrics, plots and a self-contained markdown report into `reports/`.

---

## Overview

This repository implements an end-to-end machine learning pipeline for the binary classification task **`is_suicide ∈ {0, 1}`** on user-generated text. The system is designed to be:

- **Reproducible** — every hyperparameter, path and split lives in `configs/default.yaml`. No magic numbers in code.
- **Modular** — ingestion, preprocessing, vectorization, model, evaluation and reporting are independent modules, each unit-tested in isolation.
- **CLI-driven** — three subcommands cover the lifecycle: `train`, `evaluate`, `predict`.
- **Train/serve consistent** — text cleaning lives *inside* the sklearn `Pipeline`, so the exact same preprocessing runs at training time and at inference time. No skew.
- **Self-documenting runs** — every `train` and `evaluate` invocation writes JSON metrics + ROC PNG + confusion-matrix PNG + a `*.md` report into `reports/`. The reports embed the plots and the metric table, ready to hand in.

The repository is academic in scope (TC3002B, Tecnológico de Monterrey) but follows the structural conventions of a production ML codebase.

---

## How it works

```
            ┌─────────────────────────────────────────────────────────┐
            │                  configs/default.yaml                    │
            └─────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   TRAIN FLOW                    EVALUATE FLOW                  PREDICT FLOW
   (labeled train.csv)           (labeled test.csv)             (unlabeled csv)
        │                              │                              │
        ▼                              ▼                              ▼
   ingestion() ─► clean_text       ingestion() ─► clean_text       ingestion()
        │                              │                              │
        ▼                              ▼                              ▼
   build_pipeline(config)          load_pipeline()                load_pipeline()
   ┌────────────────────┐               │                              │
   │ FunctionTransformer│               │                              │
   │   _clean_series    │               ▼                              ▼
   │      ↓             │          predict + predict_proba        predict + predict_proba
   │ TfidfVectorizer    │               │                              │
   │      ↓             │               ▼                              ▼
   │ XGBClassifier      │          evaluation.evaluate           write CSV
   └────────────────────┘          + generate_report             (text_id,
        │                              │                          prediction,
        ▼                              ▼                          probability)
   StratifiedKFold CV +            reports/
   train_test_split fit            ├─ roc_curve_test.png
        │                          ├─ confusion_matrix_test.png
        ▼                          ├─ test_metrics.json
   joblib.dump → models/           └─ final_evaluation.md
        │
        ▼
   evaluation.evaluate
   + generate_report
        │
        ▼
   reports/
   ├─ roc_curve_val.png
   ├─ confusion_matrix_val.png
   ├─ training_metrics.json
   └─ training_report.md
```

---

## Project structure

```
Suicide_classifier/
├── configs/
│   └── default.yaml             # Single source of truth for hyperparams & paths
├── data/
│   ├── data_train.csv           # Training split  (1,516 rows)
│   └── data_test_fold1.csv      # Held-out test   (252 rows)
├── models/                      # Trained .joblib artifacts (populated by train)
├── notebooks/
│   ├── EDA.ipynb                # Exploratory data analysis
│   └── pipeline_test.ipynb      # Pipeline smoke tests
├── reports/                     # Auto-populated by train & evaluate
│   ├── roc_curve_val.png
│   ├── confusion_matrix_val.png
│   ├── training_metrics.json
│   ├── training_report.md
│   ├── roc_curve_test.png
│   ├── confusion_matrix_test.png
│   ├── test_metrics.json
│   └── final_evaluation.md
├── src/
│   ├── __main__.py              # CLI dispatcher: python -m src {train|evaluate|predict}
│   ├── data_ingestion.py        # Schema validation, loading, NA handling, label mapping, split
│   ├── preprocessing.py         # Text cleaning (encoding, URLs, mentions, hashtags, emojis, …)
│   ├── vectorizer.py            # TfidfVectorizer factory
│   ├── model.py                 # XGBClassifier factory
│   ├── pipeline.py              # sklearn Pipeline assembly (cleaner → tfidf → clf)
│   ├── training.py              # train() + train_pipeline() orchestrator
│   ├── inference.py             # run_inference() + CLI wrapper (blind prediction)
│   ├── evaluate_cli.py          # run_evaluation() + CLI wrapper (labeled scoring)
│   ├── evaluation.py            # compute_metrics, plots, evaluate, generate_report
│   └── utils.py                 # YAML loader
├── tests/
│   ├── conftest.py              # Shared fixtures (pipeline, model on disk, configs)
│   ├── test_data_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_vectorizer.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_evaluate_cli.py
└── pyproject.toml               # Package metadata, dependencies, console scripts
```

---

## Module reference

| Module | Public API | Purpose |
|---|---|---|
| `data_ingestion` | `ingestion`, `split_dataset`, `schema_validation`, `data_loader`, `data_mapping`, `handle_missing_data`, `concatenate_df` | Load CSV, validate schema, resolve nulls, concatenate text columns, map labels, stratified train/test split. |
| `preprocessing` | `clean_text`, `tokenize_text`, `preprocessing` | Repair encoding, strip URLs / mentions / hashtags / emojis / special chars, lowercase, normalize whitespace. |
| `vectorizer` | `build_vectorizer` | Build a config-driven `TfidfVectorizer` (sublinear TF, n-grams 1–2, `min_df`/`max_df` caps). |
| `model` | `build_model` | Build a config-driven `XGBClassifier` with `scale_pos_weight` for class imbalance. |
| `pipeline` | `build_pipeline` | Assemble `cleaner → tfidf → xgboost` into a single sklearn `Pipeline`. |
| `training` | `train`, `train_pipeline`, `main` | Stratified 80/20 split, 5-fold Stratified CV (AUC), final fit, holdout evaluation, persist to `models/` + `reports/`. |
| `inference` | `run_inference`, `run_inference_cli`, `load_pipeline`, `predict` | **Blind** prediction: load model, preprocess input CSV, write predictions CSV. No labels required. |
| `evaluate_cli` | `run_evaluation`, `run_evaluation_cli` | **Scored** prediction: same as inference but requires ground-truth labels, computes full metrics, writes plots + JSON + markdown report. |
| `evaluation` | `compute_metrics`, `plot_roc_curve`, `plot_confusion_matrix`, `evaluate`, `generate_report` | TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1; ROC and confusion-matrix PNGs; markdown report with embedded plots. |
| `utils` | `load_config` | Parse a YAML config file into a dict. |

> **Two distinct "pipelines"** — `pipeline.build_pipeline()` returns the **sklearn Pipeline** (the model object that gets serialized). `training.train_pipeline()` is the **workflow orchestrator** (config → data → train → evaluate → save). The first lives inside the second.
>
> **`predict` vs `evaluate`** — `predict` is for *unlabeled* data: it only emits `text_id, prediction, probability`. `evaluate` is for *labeled* data (e.g., the held-out test fold): it computes the full metrics dict against ground truth and dumps a markdown report.

---

## Configuration

All runtime parameters live in `configs/default.yaml`:

```yaml
data:
  train_path: "data/data_train.csv"          # Training CSV
  test_path:  "data/data_test_fold1.csv"     # Held-out test CSV
  expected_columns: [user_id, text_id, title, text, is_suicide]
  text_columns:     [title, text]            # Concatenated into "title_text"
  target_column:    "is_suicide"
  value_true:       "yes"                    # Label mapped → 1
  value_false:      "no"                     # Label mapped → 0

paths:
  model_output:        "models/model.joblib"          # Where the trained pipeline is dumped
  reports_dir:         "reports"                      # Auto-created by train/evaluate
  # Train-time artifacts (val split inside data_train.csv)
  roc_val_output:      "reports/roc_curve_val.png"
  cm_val_output:       "reports/confusion_matrix_val.png"
  metrics_val_output:  "reports/training_metrics.json"
  report_val_output:   "reports/training_report.md"
  # Test-time artifacts (held-out test fold)
  roc_test_output:     "reports/roc_curve_test.png"
  cm_test_output:      "reports/confusion_matrix_test.png"
  metrics_test_output: "reports/test_metrics.json"
  report_test_output:  "reports/final_evaluation.md"

vectorizer:
  ngram_range:   [1, 2]                      # Unigrams + bigrams
  min_df:        2                           # Term must appear in ≥ 2 docs
  max_df:        0.95                        # Drop terms appearing in > 95% of docs (de facto stopwords)
  sublinear_tf:  true                        # 1 + log(tf) instead of raw tf
  max_features:  10000                       # Cap vocabulary size

model:
  n_estimators:      500
  max_depth:         6
  learning_rate:     0.05
  eval_metric:       "auc"
  scale_pos_weight:  0.93                    # Compensates class imbalance (n_neg/n_pos)
  random_state:      42

training:
  test_size:    0.2                          # Holdout fraction (inside the train split)
  random_state: 42
  n_splits:     5                            # StratifiedKFold splits for CV
```

---

## Quick start

### 1. Install

```powershell
# Activate venv
venv\Scripts\activate

# Install package + runtime deps
pip install -e .

# (Optional) dev deps for tests
pip install -e ".[dev]"
```

Python 3.11+ is required. Runtime dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `nltk`, `joblib`, `pyyaml`, `matplotlib`, `optuna`, `ftfy`.

### 2. Train

```powershell
python -m src train --config configs/default.yaml
```

Loads the YAML config; ingests `data.train_path`; cleans text; runs 5-fold Stratified CV + 80/20 holdout; fits final model on the 80% split; evaluates on the 20% val split; serializes the pipeline to `models/model.joblib`; and dumps:

- `reports/roc_curve_val.png`
- `reports/confusion_matrix_val.png`
- `reports/training_metrics.json`
- `reports/training_report.md`

### 3. Evaluate against the held-out test fold

```powershell
python -m src evaluate --input data/data_test_fold1.csv --config configs/default.yaml
```

Loads the saved model; predicts on the test CSV (which must contain ground-truth `is_suicide`); computes the full metric dict; and dumps:

- `reports/roc_curve_test.png`
- `reports/confusion_matrix_test.png`
- `reports/test_metrics.json`
- `reports/final_evaluation.md`

### 4. Predict on unlabeled data

```powershell
python -m src predict --input data/data_test_fold1.csv --output predictions.csv --config configs/default.yaml
```

Produces a CSV with columns: `text_id`, `prediction` (0/1), `probability` (P(class=1)). No metrics, no plots — just predictions.

---

## Latest results

Last run on `data/data_test_fold1.csv` (252 rows, held out from training):

| Metric | Value |
|---|---|
| **AUC** | **0.739** |
| F1 | 0.688 |
| Precision | 0.707 |
| Recall (TPR) | 0.669 |
| FPR | 0.295 |
| TP / TN / FP / FN | 87 / 86 / 36 / 43 |

5-fold CV on the train split: **AUC = 0.7596 ± 0.0275**.

> See `reports/final_evaluation.md` for the auto-generated report.

---

## Testing

```powershell
pytest
```

**148 tests passing**, covering every module: ingestion, preprocessing, vectorizer, model, pipeline, evaluation (including `generate_report`), inference, and evaluate_cli.

Shared fixtures live in `tests/conftest.py` (toy pipeline config, trained-and-serialized pipeline on tmp_path, synthetic labeled CSV, full config dict with every path rooted in tmp_path so tests stay hermetic).

---

## Design decisions

- **TF-IDF + XGBoost over deep learning.** For the data volume available (~1,500 train rows of English-language Reddit-style posts), a sparse linear feature space with a gradient-boosted tree classifier reaches competitive AUC at a fraction of the training cost — and is much easier to debug, version, and deploy.
- **`scale_pos_weight = 0.93`.** Empirical class ratio `n_neg / n_pos` in the training set; passed to XGBoost rather than oversampling because it preserves the original distribution at evaluation time.
- **No explicit stopword list.** The combination of `max_df=0.95` + `sublinear_tf=True` + `min_df=2` already suppresses corpus-wide function words. Adding a hand-curated list would risk dropping high-signal tokens specific to the domain (e.g., "die", "alone").
- **Cleaning inside the `Pipeline`.** Wrapping `clean_text` in a `FunctionTransformer` guarantees inference applies the *same* preprocessing as training — no separate "serve" path can drift.
- **`predict` and `evaluate` as separate subcommands.** `predict` is for production-style blind inference (no labels). `evaluate` is for offline scoring against a labeled fold. Splitting them keeps `run_inference` ignorant of ground truth and makes `evaluate` the obvious entry point for the final test report.
- **Markdown reports per run.** `generate_report` consumes the metrics dict + plot PNGs and emits a self-contained `*.md` whose image paths are rewritten relative to the report's directory — so the file renders correctly when opened from anywhere, including GitHub.

---

## Roadmap

Derived from the project plan in `Roadmap Enfoque Clásico TF-IDF + XGBoost.md`. `[x]` is shipped, `[ ]` is pending.

### Shipped

- [x] Repo scaffolding (`pyproject.toml`, `src/`, `tests/`, `configs/`, `notebooks/`, `models/`, `data/`, `reports/`).
- [x] `data_ingestion`: schema validation, CSV loader, NaN handling, `title + text` concatenation, `yes/no → 1/0` mapping, stratified `split_dataset`, early `None` return on schema mismatch.
- [x] `preprocessing`: idempotent `clean_text` (encoding repair, lowercase, URLs, mentions, hashtags, emojis, special chars, whitespace).
- [x] `vectorizer`: YAML-driven `TfidfVectorizer` (`(1,2)` n-grams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=10000`).
- [x] `model`: YAML-driven `XGBClassifier` with `scale_pos_weight=0.93` for class imbalance.
- [x] `pipeline`: end-to-end `cleaner → tfidf → xgboost` Pipeline — TF-IDF only sees train data, no leakage.
- [x] `training`: stratified 80/20 split + 5-fold Stratified CV with AUC, holdout AUC, full evaluation, JSON dump, markdown report, `joblib.dump` persistence.
- [x] `evaluation`: TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1; ROC and confusion-matrix PNGs; `generate_report` markdown renderer with embedded plots.
- [x] `inference`: blind CLI (`python -m src predict`) that loads the pipeline and writes `text_id, prediction, probability`.
- [x] `evaluate_cli`: labeled-scoring CLI (`python -m src evaluate`) that produces full metrics + report against a labeled CSV.
- [x] **First baseline run on the real dataset** — Val AUC 0.7264 · Test AUC 0.739 · 5-fold CV 0.7596 ± 0.0275. Artifacts in `reports/`.
- [x] Unit tests for every module under `tests/` (148 passing).
- [x] EDA notebook + pipeline smoke-test notebook.

### Next steps (short term)

- [ ] **Early stopping** in XGBoost: enable `early_stopping_rounds=30` with an `eval_set` (requires plumbing `eval_set` through `pipeline.fit`).
- [ ] **`reports/optimization.md`** — hand-written: techniques applied, before/after table on AUC + training time, decisions on hyperparameters.
- [ ] **`reports/algorithm_justification.md`** — hand-written: why TF-IDF + XGBoost over deep learning, with literature references.

### Next steps (mid term)

- [ ] **Hyperparameter tuning with Optuna** (already a declared dependency). Search space candidates: `max_depth ∈ [3,8]`, `learning_rate ∈ [0.01, 0.2]`, `n_estimators ∈ [200, 1000]`, `min_df ∈ [1,5]`, `max_features ∈ [5k, 20k]`. Optimize 5-fold CV AUC.
- [ ] **Threshold tuning**. Default is 0.5; in this domain a false negative (undetected crisis) is more costly than a false positive. After inspecting the precision-recall curve, evaluate thresholds in `{0.35, 0.40, 0.45}` and pick by recall floor.
- [ ] **Error analysis**. Surface the top-K false negatives and false positives, look for systematic patterns (length, vocabulary, sarcasm), feed findings back into preprocessing or thresholding.
- [ ] **Coverage report**: `pytest --cov=src --cov-report=term-missing`, target ≥ 80%.
- [ ] **Document the train/test split provenance**: the script that produced `data_train.csv` and `data_test_fold1.csv` isn't checked in. Either commit it or wrap `split_dataset` so the split is reproducible from a single raw source.

### Next steps (stretch)

- [ ] **`TruncatedSVD` post-TF-IDF** as a dimensionality-reduction experiment — only worth it if the sparse matrix becomes a memory/speed bottleneck.
- [ ] **NLTK stopword ablation** to confirm the `max_df=0.95` cap is doing the work an explicit stopword list would do.
- [ ] **Strategy pattern for preprocessing**: define a `Preprocessor` interface so cleaning variants (with/without stemming, lemmatization, stopwords) can be swapped from config without touching `pipeline.py`.
- [ ] **User-disjoint splits**: currently ~14% of test rows come from users also present in train. For a stricter methodological story, stratify by `user_id` instead of by row.

---

## Disclaimer

This classifier is an academic exercise. It is **not** a clinical screening tool and must not be used to make decisions about anyone's mental health or safety. If you or someone you know is in crisis, please contact a local emergency service or suicide-prevention helpline.

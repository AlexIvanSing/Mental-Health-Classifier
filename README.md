# Mental Health Classifier — Suicide Ideation Detection

> Binary text classifier that flags suicide-related ideation in social media posts (Reddit-style title + body), built on a reproducible scikit-learn pipeline with XGBoost as the underlying model.

---

## Overview

This repository implements an end-to-end machine learning pipeline for the binary classification task **`is_suicide ∈ {0, 1}`** on user-generated text. The system is designed to be:

- **Reproducible** — every hyperparameter, path and split lives in `configs/default.yaml`. No magic numbers in code.
- **Modular** — ingestion, preprocessing, vectorization, model and evaluation are independent modules, each unit-tested in isolation.
- **CLI-driven** — training and inference are exposed as subcommands of the `src` package (`python -m src train|predict`).
- **Train/serve consistent** — text cleaning lives *inside* the sklearn `Pipeline`, so the exact same preprocessing runs at training time and at inference time. No skew.

The repository is academic in scope (school project) but follows the structural conventions of a production ML codebase.

---

## How it works

```
            ┌─────────────────────────────────────────────────────────┐
            │                  configs/default.yaml                    │
            └─────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
        TRAINING FLOW           INFERENCE FLOW            EVALUATION
              │                        │                        │
   data/train.csv                 input CSV                   y_true,
        │                              │                      y_pred,
        ▼                              ▼                      y_proba
   ingestion()                   ingestion()                    │
   - schema validation           - schema validation            ▼
   - load                        - load                  compute_metrics()
   - fillna                      - fillna                plot_roc_curve()
   - concat title+text           - concat title+text     plot_confusion_matrix()
   - map yes/no → 1/0                  │                       │
        │                              │                       ▼
        ▼                              ▼                  metrics dict
   build_pipeline(config)        load_pipeline()           + PNG reports
   ┌────────────────────┐               │
   │ FunctionTransformer│               │
   │   clean_text       │               │
   │      ↓             │               │
   │ TfidfVectorizer    │               │
   │      ↓             │               │
   │ XGBClassifier      │               │
   └────────────────────┘               │
        │                               │
        ▼                               ▼
   train() — StratifiedKFold      predict() → CSV
   CV + holdout val + fit              with text_id,
        │                              prediction,
        ▼                              probability
   joblib.dump → models/model.joblib
```

---

## Project structure

```
Suicide_classifier/
├── configs/
│   └── default.yaml            # Single source of truth for hyperparams and paths
├── data/
│   └── data_test_fold1.csv     # Sample dataset (user_id, text_id, title, text, is_suicide)
├── models/                     # Trained artifacts (.joblib), populated by training
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   └── pipeline_test.ipynb     # Pipeline smoke tests
├── src/
│   ├── __main__.py             # CLI entrypoint: python -m src {train|predict}
│   ├── data_ingestion.py       # Schema validation, loading, NA handling, label mapping, split
│   ├── preprocessing.py        # Text cleaning (encoding, URLs, mentions, hashtags, emojis, …)
│   ├── vectorizer.py           # TfidfVectorizer factory
│   ├── model.py                # XGBClassifier factory
│   ├── pipeline.py             # sklearn Pipeline assembly (cleaner → tfidf → clf)
│   ├── training.py             # train() + train_pipeline() orchestrator
│   ├── inference.py            # run_inference() + CLI wrapper
│   ├── evaluation.py           # Metrics + ROC + confusion matrix plots
│   └── utils.py                # YAML loader
├── tests/                      # pytest suite (one file per module)
└── pyproject.toml              # Package metadata, dependencies, console scripts
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
| `training` | `train`, `train_pipeline`, `main` | Stratified train/val split, 5-fold CV (AUC), final fit, holdout AUC, persist to `models/`. |
| `inference` | `run_inference`, `run_inference_cli`, `load_pipeline`, `predict` | Load model, preprocess input CSV, write predictions CSV (`text_id`, `prediction`, `probability`). |
| `evaluation` | `compute_metrics`, `plot_roc_curve`, `plot_confusion_matrix`, `evaluate` | Classification metrics (TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1) and PNG reports. |
| `utils` | `load_config` | Parse a YAML config file into a dict. |

> **Two distinct "pipelines"** — `pipeline.build_pipeline()` returns the **sklearn Pipeline** (the model object). `training.train_pipeline()` is the **workflow orchestrator** (config → data → train → save). The first lives inside the second.

---

## Configuration

All runtime parameters live in `configs/default.yaml`:

```yaml
data:
  train_path: "data/data_train.csv"          # Path to training CSV
  test_path:  "data/data_test_fold1.csv"     # Path to test CSV (held out)
  expected_columns: [user_id, text_id, title, text, is_suicide]
  text_columns:     [title, text]            # Concatenated into "title_text"
  target_column:    "is_suicide"
  value_true:       "yes"                    # Label mapped → 1
  value_false:      "no"                     # Label mapped → 0

paths:
  model_output: "models/model.joblib"        # Where the trained pipeline is dumped

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
  scale_pos_weight:  0.93                    # Compensates class imbalance
  random_state:      42

training:
  test_size:    0.2                          # Holdout fraction
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

This will:
1. Load the YAML config.
2. Ingest the CSV at `data.train_path`, validate schema, fill nulls, concatenate `title + text`, map `yes/no → 1/0`.
3. Build the sklearn Pipeline (cleaner → TF-IDF → XGBoost).
4. Run 5-fold Stratified CV on the train split and report mean ± std AUC.
5. Fit on the full training split and evaluate on the 20% holdout.
6. Serialize the fitted pipeline to `models/model.joblib`.

Expected output:

```
Data Loaded, schema valid
Data ingestion completed.
CV AUC: 0.XXXX ± 0.XXXX
Val AUC: 0.XXXX
Modelo guardado en models/model.joblib
{'cv_auc_mean': ..., 'cv_auc_std': ..., 'val_auc': ..., 'cv_fold_scores': [...]}
```

### 3. Predict

```powershell
python -m src predict --input data/data_test_fold1.csv --output predictions.csv
```

Produces a CSV with columns: `text_id`, `prediction` (0/1), `probability` (P(class=1)).

---

## Testing

```powershell
pytest
```

Runs the full suite under `tests/`. Coverage spans `data_ingestion`, `preprocessing`, `vectorizer` and `model`. Tests for `pipeline`, `evaluation` and `inference` are scaffolded but pending.

---

## Design decisions

- **TF-IDF + XGBoost over deep learning.** For the data volume available (school-scale dataset, English-language Reddit-style posts), a sparse linear feature space with a gradient-boosted tree classifier reaches competitive AUC at a fraction of the training cost — and is much easier to debug, version, and deploy.
- **`scale_pos_weight = 0.93`.** Empirical class ratio in the training set; passed to XGBoost rather than oversampling because it preserves the original distribution at evaluation time.
- **No explicit stopword list.** The combination of `max_df=0.95` + `sublinear_tf=True` + `min_df=2` already suppresses corpus-wide function words. Adding a hand-curated list would risk dropping high-signal tokens specific to the domain (e.g., "die", "alone").
- **Cleaning inside the `Pipeline`.** Wrapping `clean_text` in a `FunctionTransformer` guarantees inference applies the *same* preprocessing as training — no separate "serve" path can drift.

---

## Roadmap

Derived from the project plan in `Roadmap Enfoque Clásico TF-IDF + XGBoost.md`. Items below are grouped by phase; `[x]` is shipped, `[ ]` is pending.

### Shipped

- [x] Repo scaffolding (`pyproject.toml`, `src/`, `tests/`, `configs/`, `notebooks/`, `models/`, `data/`).
- [x] `data_ingestion`: schema validation, CSV loader, NaN handling, `title + text` concatenation, `yes/no → 1/0` mapping, stratified `split_dataset`.
- [x] `preprocessing`: idempotent `clean_text` (encoding repair, lowercase, URLs, mentions, hashtags, emojis, special chars, whitespace).
- [x] `vectorizer`: YAML-driven `TfidfVectorizer` (`(1,2)` n-grams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=10000`).
- [x] `model`: YAML-driven `XGBClassifier` with `scale_pos_weight=0.93` for class imbalance.
- [x] `pipeline`: end-to-end `cleaner → tfidf → xgboost` Pipeline — TF-IDF only sees train data, no leakage.
- [x] `training`: stratified 80/20 split + 5-fold Stratified CV with AUC, holdout AUC, `joblib.dump` persistence.
- [x] `evaluation`: TP/TN/FP/FN, TPR, FPR, AUC, precision, recall, F1; ROC and confusion-matrix PNGs.
- [x] `inference`: CLI script (`python -m src predict`) that loads the pipeline, predicts, and writes `text_id, prediction, probability`.
- [x] Unit tests for `data_ingestion`, `preprocessing`, `vectorizer`, `model`, `pipeline`, `evaluation`, `inference` (134 tests passing).
- [x] EDA notebook + pipeline smoke-test notebook.

### Next steps (short term)

- [ ] **First baseline run** on the real dataset — record AUC, F1, recall and commit the resulting metrics + ROC/CM PNGs under `reports/baseline.md`.
- [ ] **Early stopping** in XGBoost: enable `early_stopping_rounds=30` with an `eval_set` (currently dropped because it needs `eval_set` plumbed through `pipeline.fit`). Will require either splitting the fit into two stages or using `XGBClassifier`'s callbacks.
- [ ] **Document the train/test split provenance**: `data/data_train.csv` (7,002 rows) and `data/data_test_fold1.csv` (1,192 rows) are already disjoint, but the script that produced them isn't checked in. Either commit it or wrap `split_dataset` so the split is reproducible from a single raw source.
- [ ] **Plot generation inside `train_pipeline`**: call `evaluate()` on the holdout right after `train()` so every training run drops fresh `reports/roc_curve.png` and `reports/confusion_matrix.png`.
- [ ] **`reports/` directory** with three markdowns required by the assignment:
  - `optimization.md` — techniques applied, before/after table on AUC + training time.
  - `algorithm_justification.md` — why TF-IDF + XGBoost over deep learning, with literature references.
  - `final_evaluation.md` — TP/TN/FP/FN, TPR, FPR, AUC on the held-out test set + ROC PNG.

### Next steps (mid term)

- [ ] **Hyperparameter tuning with Optuna** (already a declared dependency). Search space candidates: `max_depth ∈ [3,8]`, `learning_rate ∈ [0.01, 0.2]`, `n_estimators ∈ [200, 1000]`, `min_df ∈ [1,5]`, `max_features ∈ [5k, 20k]`. Optimize 5-fold CV AUC.
- [ ] **Threshold tuning**. Default is 0.5; in this domain a false negative (undetected crisis) is more costly than a false positive. After inspecting the precision-recall curve, evaluate thresholds in `{0.35, 0.40, 0.45}` and pick by recall floor.
- [ ] **Error analysis**. Surface the top-K false negatives and false positives, look for systematic patterns (length, vocabulary, sarcasm), and feed the findings back into preprocessing or thresholding.
- [ ] **Coverage report**: `pytest --cov=src --cov-report=term-missing`, target ≥ 80%.

### Next steps (stretch)

- [ ] **`TruncatedSVD` post-TF-IDF** as a dimensionality-reduction experiment — only worth it if the sparse matrix becomes a memory/speed bottleneck.
- [ ] **NLTK stopword ablation** to confirm the `max_df=0.95` cap is actually doing the work an explicit stopword list would do (current hypothesis: yes, it is).
- [ ] **Strategy pattern for preprocessing**: define a `Preprocessor` interface so cleaning variants (with/without stemming, lemmatization, stopwords) can be swapped from config without touching `pipeline.py`.
- [ ] **`.gitignore` for `data/` and `models/`** once the frozen splits are committed once and the artifacts are reproducible from `train`.

---

## Disclaimer

This classifier is an academic exercise. It is **not** a clinical screening tool and must not be used to make decisions about anyone's mental health or safety. If you or someone you know is in crisis, please contact a local emergency service or suicide-prevention helpline.

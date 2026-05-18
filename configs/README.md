# configs/ — Configuración del pipeline

Cada variante de preprocesamiento tiene su propio archivo YAML. Los YAMLs en `configs/optimized/` son generados automáticamente por los scripts de Optuna — **nunca editarlos a mano**.

---

## Archivos disponibles

| Archivo | Variante | Generado por |
|---|---|---|
| `default.yaml` | `base` — limpieza mínima | Manual (baseline) |
| `variant_stopwords_nltk.yaml` | `stopwords_nltk` — stopwords NLTK sin negaciones | Manual |
| `variant_stopwords_domain.yaml` | `stopwords_domain` — lista curada gramatical | Manual |
| `variant_stemming.yaml` | `stemming` — Porter stemmer | Manual |
| `optimized/base.yaml` | `base` con TF-IDF tuneado (Etapa 1) | `run_optimization_pipeline.py` |
| `optimized/stopwords_nltk.yaml` | ídem para variante NLTK | `run_optimization_pipeline.py` |
| `optimized/stopwords_domain.yaml` | ídem para variante domain | `run_optimization_pipeline.py` |
| `optimized/stemming.yaml` | ídem para variante stemming | `run_optimization_pipeline.py` |
| `optimized/lr_base.yaml` | `base` con LR tuneada (Etapa 2-C) | `run_lr_optimization.py` |
| `optimized/svm_base.yaml` | `base` con SVM tuneada (Etapa 2-B) | `run_svm_optimization.py` |

> Los archivos en `configs/` son la **fuente de verdad**. Los archivos en `configs/optimized/` son copias con hiperparámetros tuneados sobreescritos — si se borra un optimizado, se puede regenerar corriendo el script correspondiente.

---

## Estructura completa del YAML

Cada YAML comparte este schema con cinco bloques:

```yaml
data:
  train_path: "data/data_train.csv"          # CSV de entrenamiento
  test_path:  "data/data_test_fold1.csv"     # CSV de test (held-out)
  expected_columns: [user_id, text_id, title, text, is_suicide]
  text_columns:     [title, text]            # Concatenadas en "title_text"
  target_column:    "is_suicide"
  value_true:       "yes"                    # Label que se mapea a → 1
  value_false:      "no"                     # Label que se mapea a → 0

preprocessing:
  variant: "base"                            # base | stopwords_nltk | stopwords_domain | stemming

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

# Llenado por Optuna etapa 1 (por variante).
vectorizer:
  ngram_range:   [1, 2]                      # Unigramas + bigramas
  min_df:        2                           # Término debe aparecer en ≥ 2 documentos
  max_df:        0.95                        # Drop terms que aparecen en > 95% de documentos
  sublinear_tf:  true                        # 1 + log(tf) en lugar de tf crudo
  max_features:  10000                       # Límite del vocabulario

# Llenado por Optuna etapa 2 (solo la variante ganadora).
# Los campos extra son opcionales; build_model() usa defaults sensatos.
model:
  class:             "xgboost.XGBClassifier" # Cualquier clase sklearn-compatible
  n_estimators:      500
  max_depth:         6
  learning_rate:     0.05
  eval_metric:       "auc"
  scale_pos_weight:  0.93                    # Compensa el desbalance (n_neg/n_pos)
  random_state:      42
  subsample:         1.0                     # opcional; default 1.0
  colsample_bytree:  1.0                     # opcional; default 1.0
  gamma:             0.0                     # opcional; default 0.0
  min_child_weight:  1                       # opcional; default 1
  reg_alpha:         0.0                     # opcional; default 0.0 (L1)
  reg_lambda:        1.0                     # opcional; default 1.0 (L2)
  early_stopping_rounds: 30                  # opcional; si está, training usa eval_set 90/10 interno

training:
  test_size:    0.2                          # Fracción del holdout (interno al train)
  random_state: 42
  n_splits:     5                            # Splits de StratifiedKFold para CV
```

---

## Campos obligatorios vs opcionales

**Bloque `model`:** Los seis primeros campos (`class`, `n_estimators`, `max_depth`, `learning_rate`, `eval_metric`, `scale_pos_weight`, `random_state`) son obligatorios para XGBoost. Para SVM y LR usar `class: "sklearn.svm.SVC"` o `class: "sklearn.linear_model.LogisticRegression"` con los parámetros correspondientes.

Los seis campos siguientes (`subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`) y `early_stopping_rounds` son **opcionales**: `build_model()` y `train()` los manejan con defaults razonables si no están presentes. Esto preserva compatibilidad con configs anteriores.

**Bloque `paths`:** Todos los campos son obligatorios. Las rutas de test (`roc_test_output`, etc.) solo se usan en el flujo `evaluate`; las de val se usan en `train`.

> **¿Por qué hay tantos campos en `paths`?** Para que cada variante escriba a su propia subcarpeta (`reports/<variant>/`), evitando que se sobreescriban entre sí. El baseline (`default.yaml`) escribe directo a `reports/`.

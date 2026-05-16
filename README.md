# Mental Health Classifier — Detección de Ideación Suicida

> Clasificador binario de texto que identifica ideación suicida en publicaciones tipo Reddit (título + cuerpo). Construido sobre un pipeline reproducible de scikit-learn con soporte multi-modelo (XGBoost, SVM, Logistic Regression), e incluye CLI end-to-end para entrenamiento, evaluación contra un fold de prueba etiquetado, e inferencia ciega — más optimización en dos etapas con Optuna (TF-IDF + modelo), estudios checkpoint-ables con journals resumibles, y reportes markdown auto-generados con tabla comparativa de modelos.

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Cómo funciona el pipeline](#2-cómo-funciona-el-pipeline)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Referencia de módulos](#4-referencia-de-módulos)
5. [Variantes de preprocesamiento](#5-variantes-de-preprocesamiento)
6. [Configuración (YAML)](#6-configuración-yaml)
7. [Inicio rápido](#7-inicio-rápido)
8. [Espacios de búsqueda de Optuna](#8-espacios-de-búsqueda-de-optuna)
9. [Resultados actuales](#9-resultados-actuales)
10. [Pruebas (testing)](#10-pruebas-testing)
11. [Decisiones de diseño](#11-decisiones-de-diseño)
12. [Roadmap](#12-roadmap)
13. [Aviso ético](#13-aviso-ético)

---

## 1. Visión general

Este repositorio implementa un pipeline de machine learning end-to-end para la tarea de clasificación binaria **`is_suicide ∈ {0, 1}`** sobre texto generado por usuarios. El sistema está diseñado para ser:

- **Reproducible** — todo hiperparámetro, ruta y split vive en archivos YAML bajo `configs/`. No hay números mágicos en el código.
- **Modular** — ingesta, preprocesamiento, vectorización, modelo, evaluación, reporte y optimización son módulos independientes, cada uno con tests unitarios.
- **Preprocesamiento configurable** — cuatro variantes de limpieza de texto intercambiables (`base`, `stopwords_nltk`, `stopwords_domain`, `stemming`) seleccionadas vía la config YAML. Cada una puede compararse A/B contra las demás.
- **Tuning en dos etapas con Optuna** — primero los hiperparámetros del TF-IDF se tunean independientemente por variante; luego los hiperparámetros de cada modelo (XGBoost, SVM, LR) se tunean sobre la variante ganadora. Todos los estudios persisten a disco con `JournalStorage` para que un proceso interrumpido se reanude desde donde se quedó.
- **CLI-driven** — tres subcomandos cubren el ciclo del día a día (`train`, `evaluate`, `predict`); dos scripts orquestan el ciclo de optimización.
- **Train/serve consistente** — la limpieza del texto vive **dentro** del `Pipeline` de sklearn como un `FunctionTransformer` serializable, así que la misma transformación corre en entrenamiento e inferencia. Sin desviaciones (skew), independientemente de la variante.
- **Runs auto-documentados** — cada invocación de entrenamiento y evaluación deja en disco JSON con métricas + PNG con curva ROC + PNG con matriz de confusión + un reporte `*.md` dentro de una carpeta específica de la variante bajo `reports/`. Las corridas de optimización también producen un reporte comparativo unificado.

El alcance es académico (TC3002B, Tecnológico de Monterrey) pero la estructura sigue las convenciones de un proyecto de ML productivo.

**Equipo:** Aislinn Ruiz Sandoval · Iván Alexander Ramos Ramírez · Miguel Ángel Galicia Sánchez · Víctor Alejandro Morales García.

---

## 2. Cómo funciona el pipeline

```
            ┌─────────────────────────────────────────────────────────┐
            │            configs/<variant>.yaml + Optuna              │
            └─────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────┬───────────┴───────────┬──────────────────┐
        ▼                  ▼                       ▼                  ▼
  FLUJO TRAIN          FLUJO EVALUATE          FLUJO PREDICT      FLUJO OPTIMIZACIÓN
  (etiquetado)         (test etiquetado)       (csv sin labels)   (Optuna 2 etapas)
        │                  │                       │                  │
        ▼                  ▼                       ▼                  ▼
   ingestion()        ingestion()             ingestion()        Etapa 1: TF-IDF
        │                  │                       │              ──────────────
        ▼                  ▼                       ▼              Para cada una de
   build_pipeline       load_pipeline()         load_pipeline()   las 4 variantes:
   ┌──────────────┐         │                       │             optimize_vectorizer
   │ Cleaner      │         │                       │              → 30 trials
   │  (variante-  │         │                       │              → 5-fold CV AUC
   │   específico)│         ▼                       ▼              → escribe los
   │      ↓       │   predict + predict_proba  predict + proba       mejores hiper-
   │ TfidfVect    │         │                       │                params en su YAML
   │      ↓       │         ▼                       ▼              
   │ Classifier   │   evaluation.evaluate     write CSV            Etapa 2: Modelos
   └──────────────┘   + generate_report        (text_id,           ──────────────
        │                   │                  prediction,         pick_winner_variant()
        ▼                   ▼                  probability)        → carga YAML ganador
   StratifiedKFold     reports/<variant>/                          Para cada modelo:
   CV + holdout fit    ├─ roc_curve_test.png                        optimize_xgb (9 dims)
        │              ├─ confusion_matrix_test.png                  optimize_model(SVM)
        ▼              ├─ test_metrics.json                          optimize_model(LR)
   joblib.dump         └─ final_evaluation.md                       → JournalStorage
        │                                                           → escribe hiperparams
        ▼                                                              en YAML propio
   evaluation.evaluate                                              
   + generate_report                                               Train+evaluate final
        │                                                           por modelo. Reporte
        ▼                                                           compara los 3.
   reports/<variant>/
   ├─ roc_curve_val.png
   ├─ confusion_matrix_val.png
   ├─ training_metrics.json
   └─ training_report.md
```

### 2.1 División de datos

El entrenamiento usa tres splits con propósitos distintos:

```
data_train.csv (100 %)
│
├── 20 % → Holdout final — apartado desde el inicio.
│           Nunca entra en entrenamiento ni en CV.
│           Se usa al terminar el fit para reportar el Val AUC.
│
└── 80 % → Conjunto de entrenamiento
        │
        ├── StratifiedKFold × 5 → Estima estabilidad del modelo.
        │   Produce cv_auc_mean ± cv_auc_std.
        │   El modelo resultante se descarta; es solo diagnóstico.
        │
        └── Fit final (sobre el 80 % completo)
              │
              └── Si early_stopping_rounds está en el YAML:
                    ├── 90 % del 80 % (≈ 72 % del total) → entrena cleaner + tfidf + XGBoost
                    └── 10 % del 80 % (≈  8 % del total) → eval_set: XGBoost lo observa
                                                            para frenar árboles cuando
                                                            el AUC deja de mejorar.
                                                            No afecta pesos ni parámetros.

data_test_fold1.csv → Test externo. Nunca se toca durante entrenamiento
                      ni durante la búsqueda de Optuna. Solo se usa al
                      final, vía `python -m src evaluate`, para reportar
                      el Test AUC definitivo.
```

| Split | % del total | Propósito |
|---|---|---|
| KFold CV (×5 sobre el 80 %) | 80 % rotando | ¿Es el modelo estable? → `cv_auc_mean ± std` |
| Holdout interno (20 %) | 20 % fijo | ¿Qué tan bueno quedó? → `val_auc` |
| eval_set early stopping (10 % del 80 %) | ≈ 8 % | ¿Cuándo frenar árboles? → no reportado |
| `data_test_fold1.csv` | dataset separado | Examen final sin contaminar |

> **KFold y el holdout responden preguntas distintas.** KFold dice si el AUC es confiable o varía mucho según qué datos ve el modelo. El 20 % holdout dice cuál es ese AUC sobre datos completamente nuevos. El 8 % de early stopping no es evaluación — solo es el freno de XGBoost.

---

Los **tres subcomandos del CLI** (`train`, `evaluate`, `predict`) y los **dos scripts de orquestación** (`run_optimization_pipeline.py`, `run_xgb_optimization.py`) son las superficies que usarás día a día. Todo lo demás es interno.

---

## 3. Estructura del repositorio

```
Suicide_classifier/
├── configs/
│   ├── default.yaml                       # Variante baseline (preprocessing.variant: "base")
│   ├── variant_stopwords_nltk.yaml        # Stopwords NLTK (negaciones preservadas)
│   ├── variant_stopwords_domain.yaml      # Lista curada solo de palabras gramaticales
│   ├── variant_stemming.yaml              # Porter stemmer
│   └── optimized/                         # Generado por los scripts de optimización
│       ├── base.yaml                      #   (nunca editar manualmente — los originales
│       ├── stopwords_nltk.yaml            #    en configs/ son la fuente de verdad;
│       ├── stopwords_domain.yaml          #    estos son copias con hiperparámetros
│       └── stemming.yaml                  #    tuneados por Optuna)
├── data/
│   ├── data_train.csv                     # Split de entrenamiento (1,516 filas)
│   └── data_test_fold1.csv                # Fold de prueba (252 filas)
├── models/                                # Artefactos .joblib entrenados (uno por variante)
├── notebooks/
│   ├── EDA.ipynb                          # Análisis exploratorio
│   └── pipeline_test.ipynb                # Smoke tests del pipeline
├── reports/                               # Auto-poblado por train/evaluate/optimization
│   ├── <variant>/                         # Una subcarpeta por variante después de entrenar
│   │   ├─ roc_curve_{val,test}.png
│   │   ├─ confusion_matrix_{val,test}.png
│   │   └─ {training,test}_metrics.json
│   ├── optimization_results.json          # Resultados consolidados de Optuna (todas las etapas)
│   ├── final_report.md                    # Reporte único consolidado (generate_final_report.py)
│   ├── next_steps.md                      # Consideraciones pendientes
│   ├── optuna_tfidf_<variant>.journal     # Checkpoint estudios TF-IDF (Etapa 1)
│   ├── optuna_xgb_<winner>.journal        # Checkpoint estudio XGBoost (Etapa 2-A)
│   ├── optuna_svm_<winner>.journal        # Checkpoint estudio SVM (Etapa 2-B)
│   └── optuna_lr_<winner>.journal         # Checkpoint estudio LR (Etapa 2-C)
├── scripts/
│   ├── run_optimization_pipeline.py       # Etapa 1: tuning de TF-IDF en las 4 variantes
│   ├── run_xgb_optimization.py            # Etapa 2-A: tuning de XGB en la variante ganadora
│   ├── run_svm_optimization.py            # Etapa 2-B: tuning de SVM en la variante ganadora
│   ├── run_lr_optimization.py             # Etapa 2-C: tuning de LR en la variante ganadora
│   └── generate_final_report.py           # Renderiza reports/final_report.md
├── src/
│   ├── __main__.py                        # CLI: python -m src {train|evaluate|predict}
│   ├── data_ingestion.py                  # Validación de schema, carga, manejo NA, mapeo de labels
│   ├── preprocessing.py                   # clean_text + 3 variantes (stopwords_nltk, _domain, stemming)
│   ├── vectorizer.py                      # Factory de TfidfVectorizer
│   ├── model.py                           # Factory de XGBClassifier (config de 9 parámetros)
│   ├── pipeline.py                        # Ensamble del Pipeline sklearn + registro CLEANERS
│   ├── training.py                        # train() + train_pipeline() orquestador
│   ├── inference.py                       # run_inference() + wrapper CLI (predicción ciega)
│   ├── evaluate_cli.py                    # run_evaluation() + wrapper CLI (evaluación con labels)
│   ├── evaluation.py                      # compute_metrics, plots, evaluate, generate_report
│   ├── optimization.py                    # optimize_vectorizer + optimize_xgb + pick_winner_variant
│   └── utils.py                           # Loader de YAML
├── tests/                                 # Suite pytest (un archivo por módulo de src)
└── pyproject.toml
```

---

## 4. Referencia de módulos

| Módulo | API pública | Propósito |
|---|---|---|
| `data_ingestion` | `ingestion`, `split_dataset`, `schema_validation`, `data_loader`, `data_mapping`, `handle_missing_data`, `concatenate_df` | Cargar CSV, validar schema, resolver nulos, concatenar columnas de texto, mapear labels, split estratificado train/test. |
| `preprocessing` | `clean_text`, `clean_text_with_stopwords_nltk`, `clean_text_with_stopwords_domain`, `clean_text_with_stemming`, `tokenize_text`, `preprocessing` | Cleaner baseline + tres variantes. La whitelist de negaciones se respeta en todas las variantes. |
| `vectorizer` | `build_vectorizer` | `TfidfVectorizer` configurado por YAML. |
| `model` | `build_model` | Clasificador configurado por YAML. Usa `importlib` para instanciar cualquier clase sklearn-compatible desde `config["model"]["class"]` (default: `xgboost.XGBClassifier`). Soporta XGBoost, SVM (`sklearn.svm.SVC`), LR (`sklearn.linear_model.LogisticRegression`) o cualquier estimador con `fit`/`predict_proba`. |
| `pipeline` | `build_pipeline`, `CLEANERS` | Ensambla el Pipeline `cleaner → tfidf → xgboost`. El cleaner se elige del registro `CLEANERS` vía `config["preprocessing"]["variant"]`. |
| `training` | `train`, `train_pipeline`, `main` | Split estratificado 80/20 + 5-fold Stratified CV (AUC) + fit final. **Early stopping** se activa cuando `config["model"]["early_stopping_rounds"]` está presente: re-fittea el step XGB con un eval_set interno 90/10 para que los árboles dejen de crecer cuando el AUC de validación se estanca. |
| `inference` | `run_inference`, `run_inference_cli`, `load_pipeline`, `predict` | Predicción **ciega** (no requiere ground truth). |
| `evaluate_cli` | `run_evaluation`, `run_evaluation_cli` | Predicción **con score**: requiere labels reales, calcula métricas completas, escribe plots + JSON + markdown. |
| `evaluation` | `compute_metrics`, `plot_roc_curve`, `plot_confusion_matrix`, `evaluate`, `generate_report` | TP/TN/FP/FN, TPR, FPR, AUC, precisión, recall, F1; PNGs de curva ROC y matriz de confusión; reporte markdown auto-contenido. |
| `optimization` | `optimize_vectorizer`, `optimize_xgb`, `optimize_model`, `pick_winner_variant`, `run_all_optimizations`, `_SVM_SPACE`, `_LR_SPACE` | Búsqueda Optuna en dos etapas. La etapa 1 tunea el espacio TF-IDF de 5 dimensiones por variante. La etapa 2 tunea cada modelo (XGB 9 dims, SVM 2 dims, LR 2 dims) sobre la ganadora. `optimize_model` es la función genérica que acepta cualquier search space y modelo sklearn-compatible. Todos los estudios usan `JournalStorage` con checkpoint resumible. |
| `utils` | `load_config` | Parsea un archivo YAML a un dict. |

> **Dos "pipelines" distintos** — `pipeline.build_pipeline()` devuelve el **Pipeline de sklearn** (el objeto modelo serializable). `training.train_pipeline()` es el **orquestador del workflow**. El primero vive dentro del segundo.
>
> **`predict` vs `evaluate`** — `predict` es para datos *sin etiquetar*: emite `text_id, prediction, probability`. `evaluate` es para datos *etiquetados*: calcula el dict completo de métricas contra el ground truth y dumpea un reporte markdown.

---

## 5. Variantes de preprocesamiento

Las cuatro funciones cleaner se seleccionan vía `config["preprocessing"]["variant"]`:

| Variante | Qué hace | Justificación |
|---|---|---|
| `base` | Repara encoding + remueve URLs/menciones/hashtags/emojis/caracteres especiales + lowercase + normaliza whitespace. | Baseline de pérdida mínima. Confía en que `max_df=0.95` filtra implícitamente las stopwords corpus-wide. |
| `stopwords_nltk` | `base` + remueve stopwords del inglés de NLTK **excepto** las negaciones (`no`, `not`, `never`, `nor`, `neither`, `without`, `nothing`, `nobody`, `nowhere`, `none`). | Filtrado estándar de stopwords con una excepción crítica: las negaciones invierten la polaridad en texto de ideación suicida y deben preservarse. |
| `stopwords_domain` | `base` + remueve una lista hand-curada de **únicamente** tokens puramente gramaticales (artículos, preposiciones, conjunciones, auxiliares no negados). El contenido emocional se mantiene explícitamente. | Más conservador que NLTK: nunca quita tokens como "alone", "die", "hopeless" que las listas genéricas a veces incluyen. |
| `stemming` | `base` + Porter stemming token a token. | Colapsa variantes morfológicas (`die`/`died`/`dying` → `die`) para reducir el vocabulario en un corpus pequeño. |

Las cuatro variantes están expuestas en `src.pipeline.CLEANERS`; agregar una nueva es una entrada de una línea en el registro.

> **Importante sobre negaciones:** las 10 negaciones del whitelist se preservan en TODAS las variantes (incluso en `stopwords_domain`, donde técnicamente no se quitarían igual). Esto es intencional para garantizar que ninguna variante destruya la polaridad de frases como "I do **not** want to live".

---

## 6. Configuración (YAML)

Cada variante tiene su propio YAML en `configs/`. Comparten schema pero difieren en `preprocessing.variant`, `paths.*` y los bloques `vectorizer`/`model` post-Optuna.

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

**¿Por qué hay tantos campos en `paths`?** Para que cada variante escriba a su propia subcarpeta (`reports/<variant>/`), evitando que se sobreescriban entre sí. El baseline (`default.yaml`) escribe directo a `reports/`.

**¿Cuáles son obligatorios y cuáles opcionales?** Los seis primeros campos del bloque `model` son obligatorios. Los seis siguientes (subsample, colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda) y `early_stopping_rounds` son opcionales: `build_model()` y `train()` los manejan con defaults razonables si no están presentes. Esto preserva back-compat con configs viejas.

---

## 7. Inicio rápido

### 7.1. Instalación

```powershell
# Activar venv
venv\Scripts\activate

# Instalar paquete + deps de runtime
pip install -e .

# (Opcional) deps de desarrollo para tests
pip install -e ".[dev]"
```

Requiere Python 3.11+. Dependencias de runtime: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `nltk`, `joblib`, `pyyaml`, `matplotlib`, `optuna`, `ftfy`.

> **Nota Windows:** el `JournalStorage` de Optuna usa `JournalFileOpenLock` (no symlinks) para que funcione sin permisos de administrador.

### 7.2. Comandos del día a día

Estos tres subcomandos cubren todo el ciclo operativo. Cada uno acepta `--config configs/<variant>.yaml` para cambiar de variante (default: `configs/default.yaml`).

```powershell
# Entrenar el modelo en la variante activa
python -m src train

# Evaluar el modelo guardado contra el fold de test etiquetado
python -m src evaluate --input data/data_test_fold1.csv

# Predecir sobre datos sin labels
python -m src predict --input <csv> --output predictions.csv
```

**Qué produce `train`:**
- `models/<model>.joblib` — el Pipeline entrenado, serializado.
- `reports/<variant>/roc_curve_val.png`, `confusion_matrix_val.png`
- `reports/<variant>/training_metrics.json` — todas las métricas del val set.
- `reports/<variant>/training_report.md` — reporte con plots embebidos.

**Qué produce `evaluate`:**
- `reports/<variant>/roc_curve_test.png`, `confusion_matrix_test.png`
- `reports/<variant>/test_metrics.json`
- `reports/<variant>/final_evaluation.md`

**Qué produce `predict`:**
- Un CSV con columnas `text_id, prediction, probability`. Sin métricas, sin plots — solo predicciones.

### 7.3. Optimización en dos etapas con Optuna

```powershell
# Etapa 1: tuning de TF-IDF para las 4 variantes (30 trials cada una)
python scripts/run_optimization_pipeline.py --n-trials 30 --timeout 1650
# → escribe configs optimizados a configs/optimized/ (originales intactos)
# → produce reports/<variant>/* y reports/optimization_results.json

# Etapa 2-A: tuning de XGB sobre la variante ganadora (auto-detectada)
python scripts/run_xgb_optimization.py --n-trials 30 --timeout 600

# Etapa 2-B: tuning de SVM (rbf kernel, probability=True)
python scripts/run_svm_optimization.py --n-trials 30 --timeout 900

# Etapa 2-C: tuning de Logistic Regression
python scripts/run_lr_optimization.py --n-trials 80 --timeout 500

# Renderiza el reporte final consolidado con tabla comparativa de modelos
python scripts/generate_final_report.py
# → reports/final_report.md (muestra comparación de los 3 modelos)
```

Para forzar manualmente la variante (sin auto-detección):
```powershell
python scripts/run_xgb_optimization.py --variant base
python scripts/run_svm_optimization.py --variant stopwords_nltk
```

Todos los scripts soportan `--timeout <seconds>` para limitar la duración del estudio.

### 7.4. Reanudar una optimización interrumpida

**Todos los estudios** (Etapa 1 TF-IDF, Etapa 2 XGB/SVM/LR) usan `JournalStorage`, que persiste cada trial completado a un archivo journal en disco. Si el proceso se mata a la mitad, re-ejecutar el mismo comando reanuda desde donde quedó.

**`--n-trials` es el total deseado**, no trials adicionales. Si ya hay 21 trials en el journal y pides `--n-trials 30`, solo correrá 9 más hasta llegar a 30.

```powershell
python scripts/run_svm_optimization.py --n-trials 30
# Ctrl+C después del trial 15...
python scripts/run_svm_optimization.py --n-trials 30
# Reanuda desde el trial 16. Total = 30 trials.
```

Para empezar de cero (ignorando el checkpoint):
```powershell
python scripts/run_svm_optimization.py --no-resume
```

Los journals viven en `reports/optuna_<model>_<variant>.journal`.

---

## 8. Espacios de búsqueda de Optuna

**Etapa 1 — TF-IDF (5 dimensiones, por variante):**

| Parámetro | Espacio |
|---|---|
| `ngram_range` | `{(1,1), (1,2), (1,3)}` |
| `min_df` | int en `[1, 5]` |
| `max_df` | float en `[0.70, 0.99]` |
| `sublinear_tf` | `{True, False}` |
| `max_features` | `{5000, 10000, 20000, 50000}` |

**Etapa 2-A — XGBoost (9 dimensiones):**

| Parámetro | Espacio | Qué controla |
|---|---|---|
| `n_estimators` | int en `[100, 1500]` | Número máximo de árboles |
| `max_depth` | int en `[3, 10]` | Profundidad máxima por árbol |
| `learning_rate` | log-uniforme en `[0.005, 0.3]` | Contribución de cada árbol |
| `subsample` | float en `[0.6, 1.0]` | Fracción de filas por árbol (regularización) |
| `colsample_bytree` | float en `[0.6, 1.0]` | Fracción de features por árbol |
| `gamma` | float en `[0.0, 5.0]` | Penalización por nuevos splits |
| `min_child_weight` | int en `[1, 10]` | Mínimo de muestras por hoja |
| `reg_alpha` | log-uniforme en `[1e-8, 10.0]` | Regularización L1 |
| `reg_lambda` | log-uniforme en `[1e-8, 10.0]` | Regularización L2 |

Fijos (no se tunean): `scale_pos_weight=0.93`, `eval_metric=auc`, `random_state=42`.

**Etapa 2-B — SVM (2 dimensiones):**

| Parámetro | Espacio | Qué controla |
|---|---|---|
| `C` | log-uniforme en `[0.01, 100]` | Regularización (trade-off margen/clasificación) |
| `gamma` | log-uniforme en `[1e-4, 1.0]` | Ancho del kernel RBF |

Fijos: `kernel=rbf`, `probability=True`, `random_state=42`.

**Etapa 2-C — Logistic Regression (2 dimensiones):**

| Parámetro | Espacio | Qué controla |
|---|---|---|
| `C` | log-uniforme en `[0.001, 100]` | Fuerza de regularización inversa |
| `max_iter` | int en `[200, 2000]` | Iteraciones máximas del solver |

Fijos: `solver=lbfgs`, `penalty=l2`, `random_state=42`.

Ambos estudios usan **TPE sampler** con `random_state=42`. Función objetivo: media de **5-fold Stratified ROC-AUC** sobre `data/data_train.csv`. El fold de prueba (`data_test_fold1.csv`) **nunca se toca** durante la búsqueda — solo al final, para el Test AUC reportado.

**Early stopping** **no** se aplica durante CV (el `n_estimators` propuesto por Optuna es el número exacto de árboles que se entrenan en cada fold, manteniendo los scores comparables entre folds). **Sí** se aplica en el entrenamiento final via un split interno 90/10 de la partición train del 80%, cuando `early_stopping_rounds` está presente en el YAML.

**¿Por qué 30 trials?** Es el balance estándar: suficiente para que el TPE sampler converja, sin volverse computacionalmente caro. Cada trial implica vectorizar el corpus + 5 fits del XGB. En la etapa 2 cada trial son ~5–10 segundos en una laptop moderna; los 30 trials toman 5–10 minutos. La etapa 1 (4 variantes × 30 trials) toma 15–30 min.

---

## 9. Resultados actuales

> Los resultados completos viven en [`reports/final_report.md`](reports/final_report.md), generado por `scripts/generate_final_report.py`. Esta sección es un resumen.

### Etapa 1 — Comparación de variantes de preprocesamiento

Sobre el conjunto de prueba (252 publicaciones nunca vistas durante CV ni Optuna):

| Variante | CV AUC | Test AUC | Recall | FPR | TP | FN |
|---|---|---|---|---|---|---|
| **Baseline** | 0.7590 | **0.7347** | 0.6923 | 0.3361 | 90 | 40 |
| Stopwords NLTK | 0.7638 | 0.7149 | 0.7000 | 0.3525 | 91 | 39 |
| Stopwords curadas | 0.7605 | 0.7228 | 0.6692 | 0.3689 | 87 | 43 |
| Stemming | 0.7607 | 0.7294 | 0.7231 | 0.3607 | 94 | 36 |

Las 4 variantes terminan en un rango estrecho de Test AUC (0.71–0.73). Baseline gana por Test AUC.

### Etapa 2 — Comparación de modelos (sobre variante ganadora)

| Modelo | CV AUC | Test AUC | F1 | Recall | FPR | TP | FN |
|---|---|---|---|---|---|---|---|
| XGBoost (tuneado) | 0.7666 | 0.7351 | 0.6822 | 0.6769 | 0.3279 | 88 | 42 |
| SVM (tuneado) | 0.7676 | 0.7658 | 0.7368 | 0.7538 | 0.3115 | 98 | 32 |
| **LR (tuneada)** | **0.7741** | **0.7721** | **0.7197** | 0.7308 | 0.3197 | 95 | 35 |

LR logra el mejor Test AUC (0.7721) y SVM el mejor Recall (0.7538). XGBoost queda detrás de ambos modelos lineales — en un corpus de ~1,500 filas con features TF-IDF sparse, modelos más simples generalizan mejor.

**Modelo ganador:**
- Tipo: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- Variante: `base` · Config: `configs/optimized/lr_base.yaml`
- Test AUC: 0.7721 · F1: 0.7197 · Recall: 73.1%

---

## 10. Pruebas (testing)

```powershell
pytest
```

La suite cubre cada módulo: ingestion, preprocessing (las 4 variantes), vectorizer, model, pipeline, evaluation (incluyendo `generate_report`), inference, evaluate_cli, y optimization (incluyendo `optimize_xgb` con `n_trials=1` como smoke test + un test de checkpoint-resume que verifica que el journal persiste trials entre llamadas).

**Cobertura total:** 215+ tests passing.

Las fixtures compartidas viven en `tests/conftest.py` (config toy del pipeline, pipeline entrenado-y-serializado en `tmp_path`, CSV sintético etiquetado, dict de config completo con todas las rutas anidadas en `tmp_path` para que los tests sean herméticos).

Para correr solo un módulo:
```powershell
pytest tests/test_optimization.py -v
```

Para ver coverage:
```powershell
pytest --cov=src --cov-report=term-missing
```

---

## 11. Decisiones de diseño

- **TF-IDF + modelos clásicos en lugar de deep learning.** Para ~1,500 filas de entrenamiento de texto en inglés tipo Reddit, TF-IDF con modelos supervisados (LR, SVM, XGBoost) alcanza AUC competitivo a una fracción del costo — y es más fácil de debuggear, versionar y desplegar. Los resultados confirman que modelos lineales (LR, SVM) generalizan mejor que XGBoost en este régimen de datos.
- **`scale_pos_weight = 0.93`.** Es la razón empírica `n_neg / n_pos` del set de entrenamiento; pasada a XGBoost en lugar de oversampling para preservar la distribución original al evaluar.
- **Cleaning dentro del Pipeline.** Envolver el `clean_text*` elegido en un `FunctionTransformer` garantiza que la inferencia aplique exactamente el mismo preprocesamiento que el entrenamiento — no hay un path de "serve" separado que pueda derivar (skew). Esto aplica para las cuatro variantes.
- **Tuning en dos etapas en lugar de conjunto.** TF-IDF (5 dims) + modelo (2–9 dims) = espacio conjunto intratable en 30 trials. Tunear primero el vectorizer y congelarlo antes de tunear cada modelo descompone el problema. Los TF-IDF params se comparten entre los 3 modelos (tuneados con XGBoost en Stage 1 y reutilizados por SVM/LR en Stage 2).
- **Early stopping solo en el fit final.** Dentro del CV, cada fold entrena exactamente el `n_estimators` que Optuna propone, así los scores fold-a-fold son directamente comparables. Después de que Optuna escoge la ganadora, el entrenamiento final usa un eval split interno + `early_stopping_rounds=30` para que el modelo desplegado deje de crecer árboles cuando el AUC de validación se estanca, evitando overfitting al deployment sin distorsionar la señal del CV.
- **`predict` y `evaluate` como subcomandos separados.** `predict` es para inferencia productiva ciega (sin labels). `evaluate` es para scoring offline contra un fold etiquetado. Separarlos mantiene a `run_inference` ignorante del ground truth y hace que `evaluate` sea el punto de entrada obvio para el reporte final del test.
- **JournalStorage para checkpoints de XGB.** Los estudios de XGB son 5–10× más lentos que los de TF-IDF (árboles más profundos, más rondas). Persistir el estudio a un journal file significa que un Ctrl+C o el laptop entrando en sleep no malgasta los trials ya hechos.
- **Una negación destruida = un FN potencial.** Por eso las cuatro variantes (incluso `stopwords_nltk`) preservan religiosamente las 10 negaciones del whitelist. "I do not want to live" tiene polaridad opuesta a "I do want to live"; perder el "not" sería catastrófico.

---

## 12. Roadmap

`[x]` ya implementado, `[ ]` pendiente.

### Implementado

- [x] Scaffolding del proyecto (`pyproject.toml`, `src/`, `tests/`, `configs/`, `notebooks/`, `models/`, `data/`, `reports/`, `scripts/`).
- [x] `data_ingestion`: validación de schema, loader CSV, manejo NaN, concatenación `title + text`, mapeo `yes/no → 1/0`, `split_dataset` estratificado.
- [x] `preprocessing`: `clean_text` baseline + 3 variantes (`stopwords_nltk`, `stopwords_domain`, `stemming`). Whitelist de negaciones preservada en todas.
- [x] `vectorizer`: `TfidfVectorizer` configurado por YAML.
- [x] `model`: `build_model` genérico vía `importlib` (`config["model"]["class"]`). Soporta XGBoost, SVM, LR, o cualquier clase sklearn-compatible.
- [x] `pipeline`: ensamble end-to-end `cleaner → tfidf → classifier` con registro `CLEANERS`. Variante elegida vía config.
- [x] `training`: split estratificado 80/20 + 5-fold Stratified CV (AUC); `early_stopping_rounds` condicional (solo XGBoost).
- [x] `evaluation`: TP/TN/FP/FN, TPR, FPR, AUC, precisión, recall, F1; PNGs de ROC y CM; `generate_report` markdown.
- [x] `inference`: CLI ciego (`python -m src predict`).
- [x] `evaluate_cli`: CLI con scoring (`python -m src evaluate`).
- [x] `optimization` etapa 1: `optimize_vectorizer` (TF-IDF 5 dims, 4 variantes, checkpoint con `JournalStorage`).
- [x] `optimization` etapa 2: `optimize_xgb` (9 dims), `optimize_model` genérico (SVM 2 dims, LR 2 dims), todos con `JournalStorage`.
- [x] `pick_winner_variant` para conectar etapa 1 → etapa 2 automáticamente.
- [x] Scripts orquestadores: `run_optimization_pipeline.py`, `run_xgb_optimization.py`, `run_svm_optimization.py`, `run_lr_optimization.py`, `generate_final_report.py`.
- [x] Reporte final con tabla comparativa de modelos (elige el mejor por Test AUC).
- [x] Semántica de `--n-trials`: total deseado (no aditivo al resumir).
- [x] `--timeout` en todos los scripts de optimización.
- [x] Notebook EDA con análisis de outliers.
- [x] Cobertura pytest (~228 tests passing).
- [x] Fix del lock de Windows en `JournalStorage` (`JournalFileOpenLock`).

### Próximos pasos (corto plazo)

- [ ] **Cambiar `pick_winner_variant` para usar CV AUC** en lugar de Test AUC. Usar el test set para selección de modelo es fuga de información — el test set solo debería verse al final para reportar métricas definitivas.
- [ ] **Reorganizar `reports/`** — actualmente los journals, PNGs, y JSONs están mezclados. Estructura propuesta: `reports/journals/`, `reports/tfidf/<variant>/`, `reports/<model>/<variant>/`.
- [ ] **Función maestra multi-modelo** — consolidar `run_xgb_optimization.py`, `run_svm_optimization.py`, `run_lr_optimization.py` en un solo `run_model_optimization.py --model xgb|svm|lr` para reducir duplicación.
- [ ] **Tuning de threshold**. Default 0.5; en este dominio un FN es más costoso que un FP. Evaluar thresholds en `{0.35, 0.40, 0.45}` por piso de recall.
- [ ] **Análisis de errores**. Top-K falsos negativos y positivos, buscar patrones (longitud, vocabulario, sarcasmo).

### Próximos pasos (mediano plazo)

- [ ] **Splits user-disjoint**: ~14% de filas del test vienen de usuarios también presentes en train. Estratificar por `user_id` para metodología más estricta.
- [ ] **`reports/algorithm_justification.md`** — por qué TF-IDF + modelos clásicos en lugar de deep learning, con referencias.
- [ ] **Documentar la procedencia del split train/test**: wrappear `split_dataset` para que sea reproducible desde fuente raw.

### Próximos pasos (stretch)

- [ ] **`TruncatedSVD` post-TF-IDF** — si la matriz sparse se vuelve cuello de botella.
- [ ] **Variantes combinadas**: e.g., `stemming + stopwords_domain`.
- [ ] **Pruning durante CV**: reportar AUCs por-fold para que Optuna pode trials no-prometedores.
- [ ] **Joint search TF-IDF + modelo** vía hierarchical search.

---


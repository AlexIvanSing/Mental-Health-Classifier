# Mental Health Classifier — Detección de Ideación Suicida

> Clasificador binario de texto que identifica ideación suicida en publicaciones tipo Reddit (título + cuerpo). Construido sobre un pipeline reproducible de scikit-learn con soporte multi-modelo (XGBoost, SVM, Logistic Regression), e incluye CLI end-to-end para entrenamiento, evaluación contra un fold de prueba etiquetado, e inferencia ciega — más optimización en dos etapas con Optuna (TF-IDF + modelo), estudios checkpoint-ables con journals resumibles, y reportes markdown auto-generados con tabla comparativa de modelos.

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Instalación](#3-instalación)
4. [Comandos del día a día](#4-comandos-del-día-a-día)
5. [Resultados actuales](#5-resultados-actuales)
6. [Roadmap](#6-roadmap)

**Documentación detallada por carpeta:**

| Carpeta | README |
|---|---|
| `src/` — pipeline, módulos, variantes, decisiones de diseño | [src/README.md](src/README.md) |
| `configs/` — estructura YAML y referencia de parámetros | [configs/README.md](configs/README.md) |
| `scripts/` — optimización con Optuna, espacios de búsqueda | [scripts/README.md](scripts/README.md) |
| `tests/` — cómo correr los tests, fixtures | [tests/README.md](tests/README.md) |
| `notebooks/` — descripción del EDA y pipeline test | [notebooks/README.md](notebooks/README.md) |

---

## 1. Visión general

Este repositorio implementa un pipeline de machine learning end-to-end para la tarea de clasificación binaria **`is_suicide ∈ {0, 1}`** sobre texto generado por usuarios. El sistema está diseñado para ser:

- **Reproducible** — todo hiperparámetro, ruta y split vive en archivos YAML bajo `configs/`.
- **Modular** — ingesta, preprocesamiento, vectorización, modelo, evaluación, reporte y optimización son módulos independientes, cada uno con tests unitarios.
- **Preprocesamiento configurable** — cuatro variantes de limpieza de texto intercambiables (`base`, `stopwords_nltk`, `stopwords_domain`, `stemming`) seleccionadas vía la config YAML. Cada una puede compararse A/B contra las demás.
- **Tuning en dos etapas con Optuna** — primero los hiperparámetros del TF-IDF se tunean independientemente por variante; luego los hiperparámetros de cada modelo (XGBoost, SVM, LR) se tunean sobre la variante ganadora. Todos los estudios persisten a disco con `JournalStorage` para que un proceso interrumpido se reanude desde donde se quedó.
- **CLI-driven** — tres subcomandos cubren el ciclo del día a día (`train`, `evaluate`, `predict`); dos scripts orquestan el ciclo de optimización.
- **Train/serve consistente** — la limpieza del texto vive **dentro** del `Pipeline` de sklearn como un `FunctionTransformer` serializable, así que la misma transformación corre en entrenamiento e inferencia. Sin desviaciones (skew), independientemente de la variante.
- **Runs auto-documentados** — cada invocación de entrenamiento y evaluación deja en disco JSON con métricas + PNG con curva ROC + PNG con matriz de confusión + un reporte `*.md` dentro de una carpeta específica de la variante bajo `reports/`.

El alcance es académico (TC3002B, Tecnológico de Monterrey) pero la estructura sigue las convenciones de un proyecto de ML productivo.

**Equipo:** Iván Alexander Ramos Ramírez · Miguel Ángel Galicia Sánchez · Aislinn Ruiz Sandoval · Víctor Alejandro Morales García.

---

## 2. Estructura del repositorio

```
Suicide_classifier/
├── configs/                               # Configuraciones YAML del pipeline
│   ├── default.yaml                       #   Variante baseline (variant: "base")
│   ├── variant_stopwords_nltk.yaml        #   Stopwords NLTK (negaciones preservadas)
│   ├── variant_stopwords_domain.yaml      #   Lista curada solo de palabras gramaticales
│   ├── variant_stemming.yaml              #   Porter stemmer
│   └── optimized/                         #   Generado por scripts de optimización
│       └── ...                            #   (nunca editar manualmente)
│   └── README.md                          # → Estructura YAML y referencia de parámetros
├── data/
│   ├── data_train.csv                     # Split de entrenamiento (1,516 filas)
│   └── data_test_fold1.csv                # Fold de prueba (252 filas)
├── models/                                # Artefactos .joblib entrenados
├── notebooks/
│   ├── EDA.ipynb                          # Análisis exploratorio
│   ├── pipeline_test.ipynb                # Smoke tests del pipeline
│   └── README.md                          # → Descripción de los notebooks
├── reports/                               # Auto-poblado por train/evaluate/optimization
│   ├── <variant>/                         #   Una subcarpeta por variante
│   ├── optimization_results.json          #   Resultados consolidados de Optuna
│   ├── final_report.md                    #   Reporte comparativo de modelos
│   └── optuna_*.journal                   #   Checkpoints de estudios Optuna
├── scripts/
│   ├── run_optimization_pipeline.py       # Etapa 1: tuning TF-IDF (4 variantes)
│   ├── run_xgb_optimization.py            # Etapa 2-A: tuning XGBoost
│   ├── run_svm_optimization.py            # Etapa 2-B: tuning SVM
│   ├── run_lr_optimization.py             # Etapa 2-C: tuning Logistic Regression
│   ├── generate_final_report.py           # Genera reports/final_report.md
│   ├── generate_optimization_report.py    # Genera reports/optimization.md
│   └── README.md                          # → Guía de optimización con Optuna
├── src/
│   ├── __main__.py                        # CLI: python -m src {train|evaluate|predict}
│   ├── data_ingestion.py                  # Carga, validación, manejo NA, mapeo labels
│   ├── preprocessing.py                   # clean_text + 3 variantes de preprocesamiento
│   ├── vectorizer.py                      # Factory de TfidfVectorizer
│   ├── model.py                           # Factory genérico de clasificador (importlib)
│   ├── pipeline.py                        # Ensamble Pipeline sklearn + registro CLEANERS
│   ├── training.py                        # Split 80/20 + CV + fit final
│   ├── inference.py                       # Predicción ciega (sin labels)
│   ├── evaluate_cli.py                    # Predicción con scoring (con labels)
│   ├── evaluation.py                      # Métricas, plots ROC/CM, reporte markdown
│   ├── optimization.py                    # Optuna 2 etapas + JournalStorage
│   ├── utils.py                           # Loader de YAML
│   └── README.md                          # → Pipeline, módulos, variantes, decisiones
├── tests/
│   ├── conftest.py                        # Fixtures compartidas
│   ├── test_*.py                          # Un archivo por módulo de src
│   └── README.md                          # → Cómo correr los tests
└── pyproject.toml
```

---

## 3. Instalación

```powershell
# Activar venv
venv\Scripts\activate

# Instalar paquete + deps de runtime
pip install -e .

# (Opcional) deps de desarrollo para tests
pip install -e ".[dev]"
```

Requiere Python 3.11+. Dependencias de runtime: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `nltk`, `joblib`, `pyyaml`, `matplotlib`, `optuna`, `ftfy`.

---

## 4. Comandos del día a día

Estos tres subcomandos cubren todo el ciclo operativo. Cada uno acepta `--config configs/<variant>.yaml` para cambiar de variante (default: `configs/default.yaml`).

```powershell
# Entrenar el modelo en la variante activa
python -m src train

# Evaluar el modelo guardado contra el fold de test etiquetado
python -m src evaluate --input data/data_test_fold1.csv

# Predecir sobre datos sin labels
python -m src predict --input <csv> --output predictions.csv
```

**Qué produce `train`:** `models/<model>.joblib` + PNGs ROC/CM + JSON métricas + reporte markdown en `reports/<variant>/`.

**Qué produce `evaluate`:** PNGs ROC/CM de test + JSON métricas de test + `final_evaluation.md` en `reports/<variant>/`.

**Qué produce `predict`:** CSV con columnas `text_id, prediction, probability`. Sin métricas, sin plots.

Para la guía completa de optimización con Optuna ver [scripts/README.md](scripts/README.md).

---

## 5. Resultados actuales

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

## 6. Roadmap

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


### IDEAS FUTURAS

### Próximos pasos (stretch)

- [ ] **`TruncatedSVD` post-TF-IDF** — si la matriz sparse se vuelve cuello de botella.
- [ ] **Variantes combinadas**: e.g., `stemming + stopwords_domain`.
- [ ] **Pruning durante CV**: reportar AUCs por-fold para que Optuna pode trials no-prometedores.
- [ ] **Joint search TF-IDF + modelo** vía hierarchical search.



- [ ] Creación de Atomatizacion de EDA segun el tipo de dato y el esquema de los datos + objetivo. Analizando el impacto del uso de diferentes LLMs u otro tipo de herramientas 

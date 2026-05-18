# src/ — Módulos del pipeline

Documentación técnica de los módulos en `src/`. Para la guía de uso del día a día ver el [README raíz](../README.md).

---

## Tabla de contenidos

1. [Cómo funciona el pipeline](#1-cómo-funciona-el-pipeline)
2. [División de datos](#2-división-de-datos)
3. [Referencia de módulos](#3-referencia-de-módulos)
4. [Variantes de preprocesamiento](#4-variantes-de-preprocesamiento)
5. [Decisiones de diseño](#5-decisiones-de-diseño)

---

## 1. Cómo funciona el pipeline

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

Los **tres subcomandos del CLI** (`train`, `evaluate`, `predict`) y los **dos scripts de orquestación** (`run_optimization_pipeline.py`, `run_xgb_optimization.py`) son las superficies que usarás día a día. Todo lo demás es interno.

---

## 2. División de datos

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

## 3. Referencia de módulos

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

## 4. Variantes de preprocesamiento

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

## 5. Decisiones de diseño

- **TF-IDF + modelos clásicos en lugar de deep learning.** Para ~1,500 filas de entrenamiento de texto en inglés tipo Reddit, TF-IDF con modelos supervisados (LR, SVM, XGBoost) alcanza AUC competitivo a una fracción del costo — y es más fácil de debuggear, versionar y desplegar. Los resultados confirman que modelos lineales (LR, SVM) generalizan mejor que XGBoost en este régimen de datos.
- **`scale_pos_weight = 0.93`.** Es la razón empírica `n_neg / n_pos` del set de entrenamiento; pasada a XGBoost en lugar de oversampling para preservar la distribución original al evaluar.
- **Cleaning dentro del Pipeline.** Envolver el `clean_text*` elegido en un `FunctionTransformer` garantiza que la inferencia aplique exactamente el mismo preprocesamiento que el entrenamiento — no hay un path de "serve" separado que pueda derivar (skew). Esto aplica para las cuatro variantes.
- **Tuning en dos etapas en lugar de conjunto.** TF-IDF (5 dims) + modelo (2–9 dims) = espacio conjunto intratable en 30 trials. Tunear primero el vectorizer y congelarlo antes de tunear cada modelo descompone el problema. Los TF-IDF params se comparten entre los 3 modelos (tuneados con XGBoost en Stage 1 y reutilizados por SVM/LR en Stage 2).
- **Early stopping solo en el fit final.** Dentro del CV, cada fold entrena exactamente el `n_estimators` que Optuna propone, así los scores fold-a-fold son directamente comparables. Después de que Optuna escoge la ganadora, el entrenamiento final usa un eval split interno + `early_stopping_rounds=30` para que el modelo desplegado deje de crecer árboles cuando el AUC de validación se estanca, evitando overfitting al deployment sin distorsionar la señal del CV.
- **`predict` y `evaluate` como subcomandos separados.** `predict` es para inferencia productiva ciega (sin labels). `evaluate` es para scoring offline contra un fold etiquetado. Separarlos mantiene a `run_inference` ignorante del ground truth y hace que `evaluate` sea el punto de entrada obvio para el reporte final del test.
- **JournalStorage para checkpoints de XGB.** Los estudios de XGB son 5–10× más lentos que los de TF-IDF (árboles más profundos, más rondas). Persistir el estudio a un journal file significa que un Ctrl+C o el laptop entrando en sleep no malgasta los trials ya hechos.
- **Una negación destruida = un FN potencial.** Por eso las cuatro variantes (incluso `stopwords_nltk`) preservan religiosamente las 10 negaciones del whitelist. "I do not want to live" tiene polaridad opuesta a "I do want to live"; perder el "not" sería catastrófico.

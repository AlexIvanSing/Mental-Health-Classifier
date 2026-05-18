# scripts/ — Optimización con Optuna

Scripts de orquestación para el ciclo de tuning de hiperparámetros. Todos son independientes del CLI principal (`python -m src`).

---

## Tabla de contenidos

1. [Scripts disponibles](#1-scripts-disponibles)
2. [Optimización en dos etapas](#2-optimización-en-dos-etapas)
3. [Reanudar una optimización interrumpida](#3-reanudar-una-optimización-interrumpida)
4. [Espacios de búsqueda de Optuna](#4-espacios-de-búsqueda-de-optuna)

---

## 1. Scripts disponibles

| Script | Etapa | Qué produce |
|---|---|---|
| `run_optimization_pipeline.py` | Etapa 1 — TF-IDF | `configs/optimized/<variant>.yaml` para las 4 variantes + reportes por variante + `reports/optimization_results.json` |
| `run_xgb_optimization.py` | Etapa 2-A — XGBoost | `configs/optimized/xgb_<winner>.yaml` + métricas en `optimization_results.json` |
| `run_svm_optimization.py` | Etapa 2-B — SVM | `configs/optimized/svm_<winner>.yaml` + métricas |
| `run_lr_optimization.py` | Etapa 2-C — Logistic Regression | `configs/optimized/lr_<winner>.yaml` + métricas |
| `generate_final_report.py` | Post-procesamiento | `reports/final_report.md` con tabla comparativa de los 3 modelos |
| `generate_optimization_report.py` | Post-procesamiento | `reports/optimization.md` — reporte narrativo en español |

---

## 2. Optimización en dos etapas

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

---

## 3. Reanudar una optimización interrumpida

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

> **Nota Windows:** el `JournalStorage` usa `JournalFileOpenLock` (no symlinks) para que funcione sin permisos de administrador.

---

## 4. Espacios de búsqueda de Optuna

### Etapa 1 — TF-IDF (5 dimensiones, por variante)

| Parámetro | Espacio |
|---|---|
| `ngram_range` | `{(1,1), (1,2), (1,3)}` |
| `min_df` | int en `[1, 5]` |
| `max_df` | float en `[0.70, 0.99]` |
| `sublinear_tf` | `{True, False}` |
| `max_features` | `{5000, 10000, 20000, 50000}` |

### Etapa 2-A — XGBoost (9 dimensiones)

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

### Etapa 2-B — SVM (2 dimensiones)

| Parámetro | Espacio | Qué controla |
|---|---|---|
| `C` | log-uniforme en `[0.01, 100]` | Regularización (trade-off margen/clasificación) |
| `gamma` | log-uniforme en `[1e-4, 1.0]` | Ancho del kernel RBF |

Fijos: `kernel=rbf`, `probability=True`, `random_state=42`.

### Etapa 2-C — Logistic Regression (2 dimensiones)

| Parámetro | Espacio | Qué controla |
|---|---|---|
| `C` | log-uniforme en `[0.001, 100]` | Fuerza de regularización inversa |
| `max_iter` | int en `[200, 2000]` | Iteraciones máximas del solver |

Fijos: `solver=lbfgs`, `penalty=l2`, `random_state=42`.

---

Ambos estudios usan **TPE sampler** con `random_state=42`. Función objetivo: media de **5-fold Stratified ROC-AUC** sobre `data/data_train.csv`. El fold de prueba (`data_test_fold1.csv`) **nunca se toca** durante la búsqueda — solo al final, para el Test AUC reportado.

**Early stopping** **no** se aplica durante CV (el `n_estimators` propuesto por Optuna es el número exacto de árboles que se entrenan en cada fold, manteniendo los scores comparables entre folds). **Sí** se aplica en el entrenamiento final via un split interno 90/10 de la partición train del 80%, cuando `early_stopping_rounds` está presente en el YAML.

**¿Por qué 30 trials?** Es el balance estándar: suficiente para que el TPE sampler converja, sin volverse computacionalmente caro. Cada trial implica vectorizar el corpus + 5 fits del XGB. En la etapa 2 cada trial son ~5–10 segundos en una laptop moderna; los 30 trials toman 5–10 minutos. La etapa 1 (4 variantes × 30 trials) toma 15–30 min.

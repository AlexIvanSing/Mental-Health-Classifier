# Detector de Ideacion Suicida — Reporte Final

> Equipo TC3002B · 2026-05-15
>
> Aislinn Ruiz Sandoval · Ivan Alexander Ramos Ramirez · Miguel Angel Galicia Sanchez · Victor Alejandro Morales Garcia

---

## 1. Pipeline

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 2. Preprocesamiento — Comparativa de variantes

| Variante | CV AUC | Test AUC | Recall | FPR | TP | FN |
|---|---|---|---|---|---|---|
| **Baseline** | 0.7590 | 0.7347 | 0.6923 | 0.3361 | 90 | 40 |
| Stopwords NLTK | 0.7638 | 0.7149 | 0.7000 | 0.3525 | 91 | 39 |
| Stopwords curadas | 0.7605 | 0.7228 | 0.6692 | 0.3689 | 87 | 43 |
| Stemming | 0.7607 | 0.7294 | 0.7231 | 0.3607 | 94 | 36 |

Ganadora: **Baseline** (Test AUC = 0.7347)

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 3. Optimizacion TF-IDF — Hiperparametros finales

| Parametro | Valor |
|---|---|
| `ngram_range` | `[1, 1]` |
| `min_df` | `4` |
| `max_df` | `0.8093` |
| `sublinear_tf` | `False` |
| `max_features` | `50000` |

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 4. Optimizacion XGBoost — Antes vs Despues

| Metrica | Stage 1 | Stage 2 | Delta |
|---|---|---|---|
| AUC | 0.7347 | 0.7351 | +0.0004 |
| Recall (TPR) | 0.6923 | 0.6769 | -0.0154 |
| Precision | 0.6870 | 0.6875 | +0.0005 |
| F1 | 0.6897 | 0.6822 | -0.0075 |
| FPR | 0.3361 | 0.3279 | -0.0082 |

**Hiperparametros XGBoost tuneados:**

| Parametro | Valor |
|---|---|
| `n_estimators` | `937` |
| `max_depth` | `10` |
| `learning_rate` | `0.007183` |
| `subsample` | `0.6784` |
| `colsample_bytree` | `0.6181` |
| `gamma` | `1.6267` |
| `min_child_weight` | `4` |
| `reg_alpha` | `2.77e-06` |
| `reg_lambda` | `0.28749982` |

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 5. Comparacion de Modelos

| Modelo | CV AUC | Test AUC | F1 | Recall | FPR | TP | FN |
|---|---|---|---|---|---|---|---|
| XGBoost (tuneado) | 0.7666 | 0.7351 | 0.6822 | 0.6769 | 0.3279 | 88 | 42 |
| SVM (tuneado) | 0.7676 | 0.7658 | 0.7368 | 0.7538 | 0.3115 | 98 | 32 |
| Logistic Regression (tuneada) | 0.7741 | 0.7721 | 0.7197 | 0.7308 | 0.3197 | 95 | 35 |

Ganador: **LR** (Test AUC = 0.7721)

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 6. Resultados finales sobre test (LR)

| Metrica | Valor |
|---|---|
| TP | 95 |
| TN | 83 |
| FP | 39 |
| FN | 35 |
| Recall (TPR) | 0.7308 |
| FPR | 0.3197 |
| Precision | 0.7090 |
| F1 | 0.7197 |
| AUC | 0.7721 |

![Curva ROC — Test](lr_base_roc_test.png)

![Matriz de Confusion — Test](lr_base_cm_test.png)

<!-- INSIGHT: escribe tu interpretacion aqui -->

## 7. Limitaciones

<!-- INSIGHT: escribe tu interpretacion aqui -->


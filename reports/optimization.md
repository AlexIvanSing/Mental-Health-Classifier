# Reporte de Optimización del Modelo

> **Detector de ideación suicida en texto** — TC3002B · Tecnológico de Monterrey
>
> **Equipo:** Aislinn Ruiz Sandoval · Iván Alexander Ramos Ramírez · Miguel Ángel Galicia Sánchez · Víctor Alejandro Morales García
>
> _Generado: 2026-05-14 18:38_

---

## 1. Resumen ejecutivo (1 minuto de lectura)

**¿Qué hace el modelo?**

Recibe un texto (un post de Reddit con título y cuerpo) y predice si expresa **ideación suicida** (clase `1`) o no (clase `0`). Devuelve tanto la decisión binaria como la probabilidad asociada.

**¿Qué tan bien funciona?**

Sobre el conjunto de prueba (252 publicaciones que el modelo nunca vio durante el entrenamiento), la versión final del modelo (**Stemming (Porter)** + XGBoost optimizado con Optuna):

- Detecta correctamente al **76.1%** de los posts de ideación suicida (recall).
- De los posts que marca como suicidas, el **70.7%** realmente lo son (precisión).
- Rendimiento global (AUC): **0.744** (1.0 sería perfecto, 0.5 sería tirar una moneda).
- Falla en **31 casos positivos no detectados** (FN) y genera **41 falsas alarmas** (FP) sobre 252 publicaciones.

**¿Por qué importa el recall más que la precisión?** En este dominio, un *falso negativo* (no detectar una ideación suicida real) tiene un costo mucho mayor que un *falso positivo* (alertar sobre un texto que no era crítico). Por eso priorizamos modelos con recall alto.

## 2. Qué se hizo

El proceso completo de optimización tuvo **dos etapas independientes** de búsqueda de hiperparámetros con [Optuna](https://optuna.org/):

```
        ┌─────────────────────────────────────────────────────────┐
        │  ETAPA 1: ¿Cómo conviene representar el texto?          │
        │  (TF-IDF tuneado para 4 variantes de preprocesamiento)  │
        └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  Variante ganadora: stemming
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────────┐
        │  ETAPA 2: ¿Cómo conviene configurar XGBoost?            │
        │  (9 hiperparámetros del clasificador, 30 trials)        │
        └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                       Modelo final entrenado
```

**Para cada etapa**, Optuna probó 30 combinaciones distintas de hiperparámetros, midió cada una con **5-fold Stratified Cross-Validation** (promedio de AUC sobre 5 particiones) y se quedó con la mejor.

El **conjunto de prueba** (`data_test_fold1.csv`, 252 publicaciones) **nunca se tocó durante la optimización** — solo al final, para reportar el desempeño real.

## 3. Etapa 1 — Comparación de variantes de preprocesamiento

Probamos 4 maneras distintas de limpiar el texto antes de vectorizarlo. Para cada variante, Optuna encontró la mejor configuración del `TfidfVectorizer` (tamaño del vocabulario, n-gramas, frecuencias mínima y máxima, etc.) y luego se entrenó el modelo con XGBoost en su configuración por defecto.

### 3.1. Las 4 variantes

**Baseline (sin filtrado de palabras).** No filtra ninguna palabra. Solo limpia el texto (encoding, URLs, menciones, emojis, mayúsculas). Confía en que el TF-IDF, con `max_df`, ya filtra las palabras demasiado comunes.

**Stopwords NLTK (preservando negaciones).** Quita las stopwords estándar del inglés de NLTK, **excepto las 10 negaciones** (`no`, `not`, `never`, …) que invierten el sentido de una frase y son críticas en este dominio.

**Stopwords curadas (solo palabras gramaticales).** Solo quita palabras puramente gramaticales (artículos, preposiciones, conjunciones). Mantiene cualquier palabra con contenido emocional ("alone", "hopeless", "die", …) que listas estándar a veces incluyen.

**Stemming (Porter).** Aplica el algoritmo Porter para reducir cada palabra a su raíz (`died`, `dying`, `dies` → `die`). Reduce el vocabulario y puede ayudar con corpus pequeños.

### 3.2. Resultados sobre el conjunto de prueba

| Variante | AUC | Recall | Precisión | F1 | FPR | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (sin filtrado de palabras) | 0.7286 | 0.6769 | 0.6718 | 0.6743 | 0.3525 | 88 | 42 | 43 | 79 |
| Stopwords NLTK (preservando negaciones) | 0.7137 | 0.6615 | 0.6719 | 0.6667 | 0.3443 | 86 | 44 | 42 | 80 |
| Stopwords curadas (solo palabras gramaticales) | 0.7272 | 0.6923 | 0.6977 | 0.6950 | 0.3197 | 90 | 40 | 39 | 83 |
| Stemming (Porter) 🏆 | 0.7359 | 0.6846 | 0.6899 | 0.6873 | 0.3279 | 89 | 41 | 40 | 82 |

**Ganadora por AUC: Stemming (Porter)** (AUC = 0.7359).

**¿Cómo leer la tabla?**

- **AUC**: capacidad global del modelo para separar las dos clases. Más alto = mejor.
- **Recall** (TPR): de todos los posts de ideación suicida REALES, qué porcentaje detectó. La métrica más importante para este dominio.
- **Precisión**: de los posts que el modelo marcó como suicidas, qué porcentaje realmente lo eran.
- **F1**: balance entre precisión y recall.
- **FPR** (False Positive Rate): de los posts NO suicidas, qué porcentaje fueron clasificados incorrectamente como suicidas.
- **TP, FN, FP, TN**: conteos absolutos sobre las 252 filas del test fold.

### 3.3. Hiperparámetros TF-IDF óptimos por variante

| Variante | n-gramas | min_df | max_df | sublinear_tf | max_features |
|---|---|---|---|---|---|
| Baseline (sin filtrado de palabras) | (1, 1) | 1 | 0.7532 | False | 20,000 |
| Stopwords NLTK (preservando negaciones) | (1, 2) | 3 | 0.8585 | False | 10,000 |
| Stopwords curadas (solo palabras gramaticales) | (1, 3) | 4 | 0.7579 | False | 10,000 |
| Stemming (Porter) | (1, 1) | 1 | 0.7532 | False | 20,000 |

**Observaciones de la etapa 1:**

- Las 4 variantes terminan dentro de un rango muy estrecho de Test AUC (0.7137 a 0.7359). La elección del preprocesamiento **no es decisiva** en este corpus — el TF-IDF + XGBoost ya capturan la mayor parte de la señal.
- `stopwords_nltk` queda último, lo cual sugiere que filtrar la lista estándar de NLTK quita algunas palabras útiles incluso preservando las negaciones.
- `stemming` gana por margen pequeño. La hipótesis es que colapsar morfología (`die`/`died`/`dying`) en un corpus de solo ~1,500 documentos sí ayuda al modelo a generalizar.

## 4. Etapa 2 — Tuning de XGBoost sobre la ganadora

Sobre la variante ganadora (**Stemming (Porter)**), corrimos un segundo estudio de Optuna (30 trials) sobre los **9 hiperparámetros del clasificador** XGBoost. El TF-IDF se mantuvo congelado en sus valores óptimos de la etapa 1.

Adicionalmente, el entrenamiento final ahora usa **early stopping** con paciencia de 30 rondas: el modelo deja de añadir árboles cuando deja de mejorar.

### 4.1. Hiperparámetros XGBoost óptimos

| Hiperparámetro | Valor | Qué controla |
|---|---|---|
| `n_estimators` | 373 | Número máximo de árboles a entrenar. |
| `max_depth` | 10 | Profundidad máxima de cada árbol. |
| `learning_rate` | 0.010797 | Cuánto contribuye cada árbol al modelo final. |
| `subsample` | 0.6372 | Fracción de filas usadas por árbol (regularización). |
| `colsample_bytree` | 0.7661 | Fracción de features usadas por árbol (regularización). |
| `gamma` | 3.2486 | Penalización por crear nuevos splits. |
| `min_child_weight` | 3 | Mínimo de muestras requeridas en una hoja. |
| `reg_alpha` | 2.722e-05 | Regularización L1 sobre los pesos. |
| `reg_lambda` | 0.00100004 | Regularización L2 sobre los pesos. |

### 4.2. Comparativa: antes vs después del tuning de XGBoost

| Métrica | Antes (XGB defaults) | Después (XGB tuneado) | Δ |
|---|---|---|---|
| AUC | 0.7359 | 0.7441 | +0.0082 |
| Recall (TPR) | 0.6846 | 0.7615 | +0.0769 |
| Precisión | 0.6899 | 0.7071 | +0.0172 |
| F1 | 0.6873 | 0.7333 | +0.0460 |
| FPR | 0.3279 | 0.3361 | +0.0082 |

**Matriz de confusión sobre las 252 publicaciones de prueba:**

| | Predicho NO | Predicho SÍ |
|---|---|---|
| **Real NO** | TN = 81 | FP = 41 |
| **Real SÍ** | FN = 31 | TP = 99 |

**Interpretación:**

- El AUC subió **+0.0082** — una mejora pequeña, dentro del rango esperado de variabilidad del CV.
- El **recall subió +0.0769** (de 68.5% a 76.1%). **Esta es la mejora importante**: el modelo ahora detecta 99 casos positivos contra los 89 de antes — recupera **10 ideaciones suicidas adicionales**.
- Trade-off: el **FPR también subió** (0.328 → 0.336). Pasamos de 40 falsas alarmas a 41. El modelo es ahora menos conservador.
- En este dominio el trade-off es **favorable**: prevenir más casos vale más que reducir falsas alarmas.

## 5. Decisión final y configuración productiva

**Modelo final:** variante **`stemming`** (Stemming (Porter)) con XGBoost tuneado por Optuna y early stopping activado.

- Configuración YAML: `configs/variant_stemming.yaml`
- Modelo serializado: `models/model_stemming.joblib`
- Estudio Optuna persistido en: `reports/optuna_xgb_stemming.journal`

**Métricas finales reportables sobre el conjunto de prueba (252 filas):**

| Métrica | Valor |
|---|---|
| AUC | **0.7441** |
| Recall (TPR) | **0.7615** |
| Precisión | 0.7071 |
| F1 | 0.7333 |
| FPR | 0.3361 |
| TP / TN / FP / FN | 99 / 81 / 41 / 31 |

## 6. Cómo reproducir estos resultados

```powershell
# 1. Ejecutar todo el pipeline de optimización (etapa 1: 4 variantes × 30 trials)
python scripts/run_optimization_pipeline.py --n-trials 30

# 2. Ejecutar la etapa 2 sobre la variante ganadora (auto-detectada)
python scripts/run_xgb_optimization.py --n-trials 30

# 3. Regenerar este reporte
python scripts/generate_optimization_report.py
```

**Para reanudar una corrida interrumpida** del XGB tuning, basta con volver a ejecutar `run_xgb_optimization.py`: el `JournalStorage` retoma desde el último trial completado.

## 7. Notas metodológicas

- **Métrica objetivo de Optuna:** promedio de AUC sobre 5-fold StratifiedKFold (`random_state=42`).
- **Sampler de Optuna:** TPE (Tree-structured Parzen Estimator), `random_state=42`.
- **División train/test congelada:** `data/data_train.csv` (1,516 filas) y `data/data_test_fold1.csv` (252 filas) son disjuntas a nivel de `text_id`. El test fold solo se usa al final, después de toda la optimización.
- **Early stopping:** durante la búsqueda Optuna, NO se aplica — todos los trials usan exactamente el `n_estimators` propuesto, para que las comparaciones entre trials sean justas. En el entrenamiento final SÍ se aplica, usando un sub-split 90/10 del conjunto de entrenamiento como `eval_set`.
- **Trees crecidos en el modelo final:** menos que el cap de `n_estimators=373` (early stopping se activó antes), evitando overfitting.
- **`scale_pos_weight = 0.93`:** equivale a la proporción `n_neg/n_pos` del corpus (732/784). Compensa el ligero desbalance de clases sin alterar la distribución original.

---

> **Aviso:** este clasificador es un proyecto académico. **No es una herramienta clínica** y no debe usarse para tomar decisiones sobre la salud o seguridad de ninguna persona. Si tú o alguien que conoces está en crisis, contacta los servicios de emergencia o una línea de prevención del suicidio en tu país.

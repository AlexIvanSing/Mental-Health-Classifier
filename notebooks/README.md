# notebooks/ — Análisis exploratorio y pruebas

---

## Notebooks disponibles

### `EDA.ipynb` — Análisis Exploratorio de Datos

Lee `data/data_train.csv` **únicamente** (el fold de test permanece intocado). Cubre:

1. **Setup y carga** — ingestion via `src.data_ingestion`
2. **Schema y valores faltantes** — dtypes, nulos por columna
3. **Distribución de clases** — balance 51.7% positivos / 48.3% negativos → `scale_pos_weight ≈ 0.93`
4. **Longitud de textos** — distribución en caracteres y tokens, histograma log, boxplot por clase
5. **"Outliers" de longitud** — análisis de posts cortos (≤10 tokens) y largos (top 5), y **decisión justificada de no filtrarlos**
6. **Marcadores `[removed]` / `[deleted]`** — sesgo de moderación hacia clase positiva; anotado como sesgo conocido
7. **Duplicados exactos** — <0.5% del corpus, mantenidos en baseline
8. **Encoding corrupto (mojibake)** — validación de `fix_encoding()` con `ftfy`
9. **Vocabulario y palabras frecuentes** — justificación de **no usar lista de stopwords explícita** (se confía en `max_df=0.95`)
10. **Palabras más discriminativas** — ratio Laplace-smoothed por clase; confirma vocabulario esperado en la clase positiva
11. **Resumen de decisiones de diseño** — tabla que mapea cada hallazgo del EDA a una decisión concreta del pipeline

> Leer el EDA antes de modificar la configuración del vectorizador o las variantes de preprocesamiento — las decisiones de diseño están directamente derivadas de los hallazgos numéricos de cada sección.

---

### `pipeline_test.ipynb` — Smoke test del pipeline

Prueba end-to-end rápida del flujo de ingesta → preprocesamiento sobre `data/data_test_fold1.csv`. Útil para verificar que los módulos de `src/` funcionan correctamente en un entorno interactivo antes de correr la suite pytest completa.

---

## Orden recomendado de lectura

1. `EDA.ipynb` — para entender el corpus y las decisiones de diseño
2. `pipeline_test.ipynb` — para ver el pipeline en acción
3. [`src/README.md`](../src/README.md) — para entender la arquitectura de módulos

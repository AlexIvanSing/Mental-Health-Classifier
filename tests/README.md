# tests/ — Suite de pruebas

Suite pytest con cobertura de todos los módulos de `src/`.

---

## Correr los tests

```powershell
# Correr toda la suite
pytest

# Correr solo un módulo
pytest tests/test_optimization.py -v

# Ver cobertura de código
pytest --cov=src --cov-report=term-missing
```

**Cobertura total:** 215+ tests passing.

---

## Qué cubre cada módulo de test

| Archivo | Módulo cubierto | Qué prueba |
|---|---|---|
| `test_data_ingestion.py` | `src/data_ingestion.py` | Carga CSV, validación de schema, manejo de nulos, mapeo de labels, concatenación de columnas |
| `test_preprocessing.py` | `src/preprocessing.py` | Cada función de limpieza individual (encoding, URLs, menciones, emojis, etc.) |
| `test_preprocessing_variants.py` | `src/preprocessing.py` | Contratos de las 4 variantes: negaciones preservadas, contenido emocional conservado, stems colapsados |
| `test_vectorizer.py` | `src/vectorizer.py` | Construcción del TfidfVectorizer desde config |
| `test_model.py` | `src/model.py` | Instanciación dinámica de XGBClassifier, SVC, LogisticRegression |
| `test_pipeline.py` | `src/pipeline.py` | Ensamble end-to-end del Pipeline sklearn, flujo fit/predict |
| `test_training.py` | `src/training.py` | Split estratificado, CV, fit final, early stopping condicional |
| `test_evaluation.py` | `src/evaluation.py` | Métricas (TP/TN/FP/FN/AUC/F1), generación de PNGs, `generate_report` |
| `test_inference.py` | `src/inference.py` | Carga de pipeline, predicción ciega, CLI wrapper |
| `test_evaluate_cli.py` | `src/evaluate_cli.py` | Predicción con scoring, CLI wrapper |
| `test_optimization.py` | `src/optimization.py` | `optimize_xgb` con `n_trials=1` (smoke test) + checkpoint-resume (verifica que el journal persiste trials entre llamadas) |

---

## Fixtures compartidas

Las fixtures viven en `tests/conftest.py` y están disponibles en todos los módulos sin importar:

- **`pipeline_config`** — dict de config mínimo para un pipeline toy (XGBoost, variante `base`)
- **`trained_pipeline`** — pipeline entrenado en datos sintéticos, listo para predict/evaluate
- **`model_on_disk`** — el mismo pipeline serializado a `tmp_path/model.joblib`
- **`input_csv`** — CSV sintético etiquetado escrito a `tmp_path`
- **`full_config`** — dict de config completo con todas las claves `paths.*` apuntando a `tmp_path`
- **`config_file`** — el `full_config` escrito a `tmp_path/config.yaml`

El uso de `tmp_path` (fixture nativa de pytest) garantiza que los tests sean herméticos: no leen ni escriben al directorio del proyecto.

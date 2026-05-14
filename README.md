# Mental Health Classifier

> **Estado:** Work in Progress

Repositorio destinado al análisis, entrenamiento y despliegue de un modelo de machine learning, con el objetivo de la clasificación de pensamientos / textos de problemas de salud mental, específicamente sobre el suicidio.

## Objetivo

Construir un clasificador binario (suicida / no suicida) sobre texto, usando un pipeline reproducible que cubra desde la ingesta del dato crudo hasta la inferencia, parametrizado vía `configs/default.yaml`.

## Estructura del proyecto

```
Suicide_classifier/
├── configs/         # Configuración (hiperparámetros, rutas, vectorizer)
├── data/            # Datasets (raw y procesados)
├── models/          # Artefactos de modelos entrenados
├── notebooks/       # EDA y pruebas de pipeline
├── src/             # Código fuente
└── tests/           # Pruebas unitarias (pytest)
```

## Estado de los módulos

| Módulo | Estado | Descripción |
|---|---|---|
| `src/data_ingestion.py` | Implementado | Validación de schema, carga, manejo de nulos, concatenación de columnas y mapeo de target |
| `src/preprocessing.py` | Implementado | Limpieza de texto (encoding, URLs, menciones, hashtags, emojis, caracteres especiales) y tokenización |
| `src/vectorizer.py` | Implementado | Construcción del `TfidfVectorizer` desde config |
| `src/model.py` | Implementado | Construcción del clasificador `XGBClassifier` desde config |
| `src/utils.py` | Implementado | Carga de archivos YAML de configuración |
| `src/pipeline.py` | **WIP** | Debería unir vectorizer + modelo en un `sklearn.Pipeline` reutilizable |
| `src/training.py` | **WIP** | Debería ejecutar el flujo completo: ingesta → preprocesamiento → vectorización → entrenamiento → guardado del modelo. Actualmente solo dispara la ingesta |
| `src/evaluation.py` | **WIP** (vacío) | Debería calcular métricas (AUC, F1, matriz de confusión) sobre el set de validación / test y generar reportes |
| `src/inference.py` | **WIP** (vacío) | Debería cargar el modelo serializado y exponer una función para predecir sobre texto nuevo |
| `src/__main__.py` | **WIP** | Punto de entrada del paquete, actualmente sin lógica real |

## Configuración

Los hiperparámetros del vectorizer, del modelo y del entrenamiento viven en `configs/default.yaml`. La idea es que ningún módulo tenga valores hardcodeados.

## Tests

| Test | Estado |
|---|---|
| `tests/test_data_ingestion.py` | Implementado |
| `tests/test_preprocessing.py` | Implementado |
| `tests/test_vectorizer.py` | Implementado |
| `tests/test_model.py` | Implementado |
| `tests/test_pipeline.py` | **WIP** (vacío) |
| `tests/test_evaluation.py` | **WIP** (vacío) |
| `tests/test_inference.py` | **WIP** (vacío) |

Ejecutar:
```bash
pytest
```

## Setup

```bash
# Activar entorno virtual
venv\Scripts\activate

# Instalar el paquete y sus dependencias
pip install -e .

# Dependencias de desarrollo (pytest)
pip install -e ".[dev]"
```

## Scripts (declarados en `pyproject.toml`, pendientes de implementación real)

```bash
train   # ejecuta src.training:main  (WIP)
infer   # ejecuta src.inference:main (WIP)
```

## Pendientes / próximos pasos

- Completar `pipeline.py` para encapsular vectorizer + modelo.
- Implementar `training.py` end-to-end con guardado del artefacto en `models/`.
- Implementar `evaluation.py` con métricas y reporte.
- Implementar `inference.py` para predicción sobre texto nuevo.
- Agregar tests para `pipeline`, `evaluation` e `inference`.
- Definir y agregar a `data/` los archivos `train.csv` y `test.csv` referenciados en la config.
- Integrar `optuna` (ya declarado en dependencias) para tuning de hiperparámetros.

# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-15 23:10_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7658** |
| F1 | 0.7368 |
| Precision | 0.7206 |
| Recall (TPR) | 0.7538 |
| FPR | 0.3115 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 84 | FP = 38 |
| **Real Positivo** | FN = 32 | TP = 98 |

## Curva ROC

![ROC Curve](svm_base_roc_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](svm_base_cm_test.png)

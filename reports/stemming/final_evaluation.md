# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-14 17:15_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7359** |
| F1 | 0.6873 |
| Precision | 0.6899 |
| Recall (TPR) | 0.6846 |
| FPR | 0.3279 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 82 | FP = 40 |
| **Real Positivo** | FN = 41 | TP = 89 |

## Curva ROC

![ROC Curve](roc_curve_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_test.png)

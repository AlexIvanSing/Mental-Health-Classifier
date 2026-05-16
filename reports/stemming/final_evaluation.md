# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-15 23:02_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7294** |
| F1 | 0.7015 |
| Precision | 0.6812 |
| Recall (TPR) | 0.7231 |
| FPR | 0.3607 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 78 | FP = 44 |
| **Real Positivo** | FN = 36 | TP = 94 |

## Curva ROC

![ROC Curve](roc_curve_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_test.png)

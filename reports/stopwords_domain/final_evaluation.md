# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-15 23:02_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7228** |
| F1 | 0.6641 |
| Precision | 0.6591 |
| Recall (TPR) | 0.6692 |
| FPR | 0.3689 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 77 | FP = 45 |
| **Real Positivo** | FN = 43 | TP = 87 |

## Curva ROC

![ROC Curve](roc_curve_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_test.png)

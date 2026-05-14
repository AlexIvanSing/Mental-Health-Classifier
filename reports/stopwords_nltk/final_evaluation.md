# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-14 17:12_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7137** |
| F1 | 0.6667 |
| Precision | 0.6719 |
| Recall (TPR) | 0.6615 |
| FPR | 0.3443 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 80 | FP = 42 |
| **Real Positivo** | FN = 44 | TP = 86 |

## Curva ROC

![ROC Curve](roc_curve_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_test.png)

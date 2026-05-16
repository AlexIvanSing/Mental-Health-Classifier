# Reporte de Entrenamiento — Detección de Ideación Suicida

_Generado: 2026-05-15 23:02_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7255** |
| F1 | 0.649 |
| Precision | 0.6759 |
| Recall (TPR) | 0.6242 |
| FPR | 0.3197 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 100 | FP = 47 |
| **Real Positivo** | FN = 59 | TP = 98 |

## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
| Fold 1 | 0.7899 |
| Fold 2 | 0.7728 |
| Fold 3 | 0.7573 |
| Fold 4 | 0.7871 |
| Fold 5 | 0.7661 |
| **Promedio** | **0.7746** |
| **Std** | 0.0124 |

## Curva ROC

![ROC Curve](roc_curve_val.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_val.png)

# Reporte de Entrenamiento — Detección de Ideación Suicida

_Generado: 2026-05-14 18:24_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7224** |
| F1 | 0.6512 |
| Precision | 0.6806 |
| Recall (TPR) | 0.6242 |
| FPR | 0.3129 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 101 | FP = 46 |
| **Real Positivo** | FN = 59 | TP = 98 |

## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
| Fold 1 | 0.8015 |
| Fold 2 | 0.7586 |
| Fold 3 | 0.7799 |
| Fold 4 | 0.7706 |
| Fold 5 | 0.7455 |
| **Promedio** | **0.7712** |
| **Std** | 0.019 |

## Curva ROC

![ROC Curve](roc_curve_val.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_val.png)

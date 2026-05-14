# Reporte de Entrenamiento — Detección de Ideación Suicida

_Generado: 2026-05-14 17:12_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7164** |
| F1 | 0.6579 |
| Precision | 0.6803 |
| Recall (TPR) | 0.6369 |
| FPR | 0.3197 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 100 | FP = 47 |
| **Real Positivo** | FN = 57 | TP = 100 |

## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
| Fold 1 | 0.7932 |
| Fold 2 | 0.7557 |
| Fold 3 | 0.7523 |
| Fold 4 | 0.7692 |
| Fold 5 | 0.7502 |
| **Promedio** | **0.7641** |
| **Std** | 0.016 |

## Curva ROC

![ROC Curve](roc_curve_val.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_val.png)

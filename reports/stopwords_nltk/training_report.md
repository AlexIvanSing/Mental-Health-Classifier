# Reporte de Entrenamiento — Detección de Ideación Suicida

_Generado: 2026-05-15 23:01_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7393** |
| F1 | 0.66 |
| Precision | 0.6923 |
| Recall (TPR) | 0.6306 |
| FPR | 0.2993 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 103 | FP = 44 |
| **Real Positivo** | FN = 58 | TP = 99 |

## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
| Fold 1 | 0.8080 |
| Fold 2 | 0.7568 |
| Fold 3 | 0.7538 |
| Fold 4 | 0.7487 |
| Fold 5 | 0.7268 |
| **Promedio** | **0.7588** |
| **Std** | 0.0267 |

## Curva ROC

![ROC Curve](roc_curve_val.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_val.png)

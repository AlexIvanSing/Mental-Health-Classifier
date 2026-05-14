# Reporte de Entrenamiento — Detección de Ideación Suicida

_Generado: 2026-05-14 17:14_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7233** |
| F1 | 0.6443 |
| Precision | 0.6809 |
| Recall (TPR) | 0.6115 |
| FPR | 0.3061 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 102 | FP = 45 |
| **Real Positivo** | FN = 61 | TP = 96 |

## Validación cruzada (K-Fold)

| Fold | AUC |
|------|-----|
| Fold 1 | 0.7991 |
| Fold 2 | 0.7843 |
| Fold 3 | 0.7370 |
| Fold 4 | 0.7539 |
| Fold 5 | 0.7752 |
| **Promedio** | **0.7699** |
| **Std** | 0.022 |

## Curva ROC

![ROC Curve](roc_curve_val.png)

## Matriz de confusión (visualización)

![Confusion Matrix](confusion_matrix_val.png)

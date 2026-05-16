# Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)

_Generado: 2026-05-15 23:14_

## Métricas sobre el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| AUC | **0.7721** |
| F1 | 0.7197 |
| Precision | 0.709 |
| Recall (TPR) | 0.7308 |
| FPR | 0.3197 |

## Matriz de confusión

| | Pred. Negativo | Pred. Positivo |
|--|--|--|
| **Real Negativo** | TN = 83 | FP = 39 |
| **Real Positivo** | FN = 35 | TP = 95 |

## Curva ROC

![ROC Curve](lr_base_roc_test.png)

## Matriz de confusión (visualización)

![Confusion Matrix](lr_base_cm_test.png)

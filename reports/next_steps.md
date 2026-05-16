# Next Steps / Consideraciones

## Selección de variante ganadora (pick_winner_variant)

**Comportamiento actual:** usa Test AUC para elegir la variante que pasa a Stage 2.

**Problema:** usar el test set para decisiones de selección de modelo es fuga de información
— el test set ya no es un estimado limpio del rendimiento final.

**Lo correcto:** usar CV AUC (viene del train set puro, nunca toca el test set).
Con CV AUC el ganador sería `stopwords_nltk` (0.7638) en vez de `base` (0.7347 por Test AUC).

**Cambio pendiente:** modificar `pick_winner_variant()` en `src/optimization.py` para
ordenar por `optuna.best_cv_auc` en vez de `test.AUC`.

---

## Función maestra de optimización multi-modelo

Actualmente hay un script separado por modelo (run_xgb_optimization.py,
run_svm_optimization.py, run_lr_optimization.py). Consolidar en un solo
`run_model_optimization.py --model xgb|svm|lr` para evitar duplicación.

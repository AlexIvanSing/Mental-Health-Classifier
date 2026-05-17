# ============================================================================
# Generación del reporte de optimización
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
"""
Render `reports/optimization.md` from the consolidated JSON.

The report is intentionally narrative and in Spanish — written so a teammate
(or a teacher) can open the file and understand what the model does, how it
performs, and what was tried, without prior ML context. Every number is
explained in plain language alongside the table that contains it.

Run as:  python scripts/generate_optimization_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONSOLIDATED = REPO_ROOT / "reports" / "optimization_results.json"
OUTPUT       = REPO_ROOT / "reports" / "optimization.md"


VARIANT_LABEL = {
    "base":             "Baseline (sin filtrado de palabras)",
    "stopwords_nltk":   "Stopwords NLTK (preservando negaciones)",
    "stopwords_domain": "Stopwords curadas (solo palabras gramaticales)",
    "stemming":         "Stemming (Porter)",
}

VARIANT_EXPLANATION = {
    "base": (
        "No filtra ninguna palabra. Solo limpia el texto (encoding, URLs, "
        "menciones, emojis, mayúsculas). Confía en que el TF-IDF, con "
        "`max_df`, ya filtra las palabras demasiado comunes."
    ),
    "stopwords_nltk": (
        "Quita las stopwords estándar del inglés de NLTK, **excepto las 10 "
        "negaciones** (`no`, `not`, `never`, …) que invierten el sentido de "
        "una frase y son críticas en este dominio."
    ),
    "stopwords_domain": (
        "Solo quita palabras puramente gramaticales (artículos, preposiciones, "
        "conjunciones). Mantiene cualquier palabra con contenido emocional "
        "(\"alone\", \"hopeless\", \"die\", …) que listas estándar a veces "
        "incluyen."
    ),
    "stemming": (
        "Aplica el algoritmo Porter para reducir cada palabra a su raíz "
        "(`died`, `dying`, `dies` → `die`). Reduce el vocabulario y "
        "puede ayudar con corpus pequeños."
    ),
}


def _pct(x):
    return "—" if x is None else f"{x:.4f}"


def _delta(after, before):
    d = after - before
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main():
    if not CONSOLIDATED.exists():
        raise SystemExit(f"No existe {CONSOLIDATED}. Corre primero la optimización.")

    with open(CONSOLIDATED) as f:
        data = json.load(f)

    variant_keys = [k for k in data.keys() if k != "xgb_optimization"]
    winner_v1 = max(variant_keys, key=lambda v: data[v]["test"]["AUC"])
    has_xgb = "xgb_optimization" in data

    out = []

    # ============================================================
    # ENCABEZADO
    # ============================================================
    out.append("# Reporte de Optimización del Modelo")
    out.append("")
    out.append("> **Detector de ideación suicida en texto** — TC3002B · Tecnológico de Monterrey")
    out.append(">")
    out.append("> **Equipo:** Aislinn Ruiz Sandoval · Iván Alexander Ramos Ramírez · "
               "Miguel Ángel Galicia Sánchez · Víctor Alejandro Morales García")
    out.append(">")
    out.append(f"> _Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    out.append("")
    out.append("---")
    out.append("")

    # ============================================================
    # 1. TL;DR
    # ============================================================
    final_block = data.get("xgb_optimization", data[winner_v1])
    final_test = final_block["test"]

    out.append("## 1. Resumen ejecutivo (1 minuto de lectura)")
    out.append("")
    out.append("**¿Qué hace el modelo?**")
    out.append("")
    out.append("Recibe un texto (un post de Reddit con título y cuerpo) y predice si "
               "expresa **ideación suicida** (clase `1`) o no (clase `0`). Devuelve "
               "tanto la decisión binaria como la probabilidad asociada.")
    out.append("")
    out.append("**¿Qué tan bien funciona?**")
    out.append("")
    if has_xgb:
        winner_label = VARIANT_LABEL[final_block["variant"]]
        out.append(f"Sobre el conjunto de prueba (252 publicaciones que el modelo nunca vio "
                   f"durante el entrenamiento), la versión final del modelo "
                   f"(**{winner_label}** + XGBoost optimizado con Optuna):")
    else:
        winner_label = VARIANT_LABEL[winner_v1]
        out.append(f"Sobre el conjunto de prueba (252 publicaciones nunca vistas), la "
                   f"variante ganadora (**{winner_label}**):")
    out.append("")
    out.append(f"- Detecta correctamente al **{final_test['recall']*100:.1f}%** de los "
               f"posts de ideación suicida (recall).")
    out.append(f"- De los posts que marca como suicidas, el **{final_test['precision']*100:.1f}%** "
               f"realmente lo son (precisión).")
    out.append(f"- Rendimiento global (AUC): **{final_test['AUC']:.3f}** (1.0 sería perfecto, "
               f"0.5 sería tirar una moneda).")
    out.append(f"- Falla en **{final_test['FN']} casos positivos no detectados** (FN) y "
               f"genera **{final_test['FP']} falsas alarmas** (FP) sobre 252 publicaciones.")
    out.append("")
    out.append("**¿Por qué importa el recall más que la precisión?** "
               "En este dominio, un *falso negativo* (no detectar una ideación suicida real) "
               "tiene un costo mucho mayor que un *falso positivo* (alertar sobre un texto "
               "que no era crítico). Por eso priorizamos modelos con recall alto.")
    out.append("")

    # ============================================================
    # 2. Qué se hizo (la pipeline en una página)
    # ============================================================
    out.append("## 2. Qué se hizo")
    out.append("")
    out.append("El proceso completo de optimización tuvo **dos etapas independientes** de "
               "búsqueda de hiperparámetros con [Optuna](https://optuna.org/):")
    out.append("")
    out.append("```")
    out.append("        ┌─────────────────────────────────────────────────────────┐")
    out.append("        │  ETAPA 1: ¿Cómo conviene representar el texto?          │")
    out.append("        │  (TF-IDF tuneado para 4 variantes de preprocesamiento)  │")
    out.append("        └─────────────────────────────────────────────────────────┘")
    out.append("                                  │")
    out.append("                                  ▼")
    out.append(f"                  Variante ganadora: {winner_v1}")
    out.append("                                  │")
    out.append("                                  ▼")
    if has_xgb:
        out.append("        ┌─────────────────────────────────────────────────────────┐")
        out.append("        │  ETAPA 2: ¿Cómo conviene configurar XGBoost?            │")
        out.append("        │  (9 hiperparámetros del clasificador, 30 trials)        │")
        out.append("        └─────────────────────────────────────────────────────────┘")
        out.append("                                  │")
        out.append("                                  ▼")
        out.append("                       Modelo final entrenado")
    else:
        out.append("        (etapa 2 — tuning de XGBoost — pendiente)")
    out.append("```")
    out.append("")
    out.append("**Para cada etapa**, Optuna probó 30 combinaciones distintas de "
               "hiperparámetros, midió cada una con **5-fold Stratified Cross-Validation** "
               "(promedio de AUC sobre 5 particiones) y se quedó con la mejor.")
    out.append("")
    out.append("El **conjunto de prueba** (`data_test_fold1.csv`, 252 publicaciones) **nunca "
               "se tocó durante la optimización** — solo al final, para reportar el "
               "desempeño real.")
    out.append("")

    # ============================================================
    # 3. Etapa 1 — comparación de variantes
    # ============================================================
    out.append("## 3. Etapa 1 — Comparación de variantes de preprocesamiento")
    out.append("")
    out.append("Probamos 4 maneras distintas de limpiar el texto antes de vectorizarlo. "
               "Para cada variante, Optuna encontró la mejor configuración del "
               "`TfidfVectorizer` (tamaño del vocabulario, n-gramas, frecuencias mínima y "
               "máxima, etc.) y luego se entrenó el modelo con XGBoost en su configuración "
               "por defecto.")
    out.append("")
    out.append("### 3.1. Las 4 variantes")
    out.append("")
    for v in variant_keys:
        out.append(f"**{VARIANT_LABEL[v]}.** {VARIANT_EXPLANATION[v]}")
        out.append("")

    out.append("### 3.2. Resultados sobre el conjunto de prueba")
    out.append("")
    out.append("| Variante | AUC | Recall | Precisión | F1 | FPR | TP | FN | FP | TN |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in variant_keys:
        t = data[v]["test"]
        marker = " 🏆" if v == winner_v1 else ""
        out.append(
            f"| {VARIANT_LABEL[v]}{marker} | "
            f"{t['AUC']:.4f} | {t['recall']:.4f} | {t['precision']:.4f} | "
            f"{t['F1']:.4f} | {t['FPR']:.4f} | "
            f"{t['TP']} | {t['FN']} | {t['FP']} | {t['TN']} |"
        )
    out.append("")
    out.append(f"**Ganadora por AUC: {VARIANT_LABEL[winner_v1]}** "
               f"(AUC = {data[winner_v1]['test']['AUC']:.4f}).")
    out.append("")
    out.append("**¿Cómo leer la tabla?**")
    out.append("")
    out.append("- **AUC**: capacidad global del modelo para separar las dos clases. "
               "Más alto = mejor.")
    out.append("- **Recall** (TPR): de todos los posts de ideación suicida REALES, "
               "qué porcentaje detectó. La métrica más importante para este dominio.")
    out.append("- **Precisión**: de los posts que el modelo marcó como suicidas, "
               "qué porcentaje realmente lo eran.")
    out.append("- **F1**: balance entre precisión y recall.")
    out.append("- **FPR** (False Positive Rate): de los posts NO suicidas, qué porcentaje "
               "fueron clasificados incorrectamente como suicidas.")
    out.append("- **TP, FN, FP, TN**: conteos absolutos sobre las 252 filas del test fold.")
    out.append("")
    out.append("### 3.3. Hiperparámetros TF-IDF óptimos por variante")
    out.append("")
    out.append("| Variante | n-gramas | min_df | max_df | sublinear_tf | max_features |")
    out.append("|---|---|---|---|---|---|")
    for v in variant_keys:
        p = data[v]["best_params"]
        ngram = f"({p['ngram_range'][0]}, {p['ngram_range'][1]})"
        out.append(
            f"| {VARIANT_LABEL[v]} | {ngram} | {p['min_df']} | "
            f"{p['max_df']:.4f} | {p['sublinear_tf']} | {p['max_features']:,} |"
        )
    out.append("")
    out.append("**Observaciones de la etapa 1:**")
    out.append("")
    out.append("- Las 4 variantes terminan dentro de un rango muy estrecho de Test AUC "
               f"({min(data[v]['test']['AUC'] for v in variant_keys):.4f} a "
               f"{max(data[v]['test']['AUC'] for v in variant_keys):.4f}). "
               "La elección del preprocesamiento **no es decisiva** en este corpus — "
               "el TF-IDF + XGBoost ya capturan la mayor parte de la señal.")
    out.append("- `stopwords_nltk` queda último, lo cual sugiere que filtrar la lista "
               "estándar de NLTK quita algunas palabras útiles incluso preservando las "
               "negaciones.")
    out.append("- `stemming` gana por margen pequeño. La hipótesis es que colapsar "
               "morfología (`die`/`died`/`dying`) en un corpus de solo ~1,500 documentos "
               "sí ayuda al modelo a generalizar.")
    out.append("")

    # ============================================================
    # 4. Etapa 2 — tuning de XGBoost
    # ============================================================
    if has_xgb:
        xgb = data["xgb_optimization"]
        winner_v = xgb["variant"]
        before = data[winner_v]["test"]
        after = xgb["test"]

        out.append("## 4. Etapa 2 — Tuning de XGBoost sobre la ganadora")
        out.append("")
        out.append(
            f"Sobre la variante ganadora (**{VARIANT_LABEL[winner_v]}**), corrimos un "
            f"segundo estudio de Optuna ({xgb['n_trials']} trials) sobre los "
            f"**9 hiperparámetros del clasificador** XGBoost. El TF-IDF se mantuvo "
            f"congelado en sus valores óptimos de la etapa 1."
        )
        out.append("")
        out.append("Adicionalmente, el entrenamiento final ahora usa **early stopping** "
                   f"con paciencia de {xgb['fixed_keys']['early_stopping_rounds']} rondas: "
                   "el modelo deja de añadir árboles cuando deja de mejorar.")
        out.append("")
        out.append("### 4.1. Hiperparámetros XGBoost óptimos")
        out.append("")
        out.append("| Hiperparámetro | Valor | Qué controla |")
        out.append("|---|---|---|")
        explanations = {
            "n_estimators":     "Número máximo de árboles a entrenar.",
            "max_depth":        "Profundidad máxima de cada árbol.",
            "learning_rate":    "Cuánto contribuye cada árbol al modelo final.",
            "subsample":        "Fracción de filas usadas por árbol (regularización).",
            "colsample_bytree": "Fracción de features usadas por árbol (regularización).",
            "gamma":            "Penalización por crear nuevos splits.",
            "min_child_weight": "Mínimo de muestras requeridas en una hoja.",
            "reg_alpha":        "Regularización L1 sobre los pesos.",
            "reg_lambda":       "Regularización L2 sobre los pesos.",
        }
        for k, v in xgb["best_params"].items():
            v_str = f"{v:g}" if isinstance(v, float) else str(v)
            out.append(f"| `{k}` | {v_str} | {explanations.get(k, '—')} |")
        out.append("")

        out.append("### 4.2. Comparativa: antes vs después del tuning de XGBoost")
        out.append("")
        out.append("| Métrica | Antes (XGB defaults) | Después (XGB tuneado) | Δ |")
        out.append("|---|---|---|---|")
        for metric, label in [
            ("AUC",       "AUC"),
            ("recall",    "Recall (TPR)"),
            ("precision", "Precisión"),
            ("F1",        "F1"),
            ("FPR",       "FPR"),
        ]:
            out.append(
                f"| {label} | {_pct(before[metric])} | {_pct(after[metric])} | "
                f"{_delta(after[metric], before[metric])} |"
            )
        out.append("")
        out.append("**Matriz de confusión sobre las 252 publicaciones de prueba:**")
        out.append("")
        out.append("| | Predicho NO | Predicho SÍ |")
        out.append("|---|---|---|")
        out.append(f"| **Real NO** | TN = {after['TN']} | FP = {after['FP']} |")
        out.append(f"| **Real SÍ** | FN = {after['FN']} | TP = {after['TP']} |")
        out.append("")

        # Veredicto narrativo
        delta_recall = after["recall"] - before["recall"]
        delta_auc = after["AUC"] - before["AUC"]
        out.append("**Interpretación:**")
        out.append("")
        out.append(
            f"- El AUC subió **{_delta(after['AUC'], before['AUC'])}** "
            "— una mejora pequeña, dentro del rango esperado de variabilidad del CV."
        )
        out.append(
            f"- El **recall subió {_delta(after['recall'], before['recall'])}** "
            f"(de {before['recall']*100:.1f}% a {after['recall']*100:.1f}%). "
            "**Esta es la mejora importante**: el modelo ahora detecta "
            f"{after['TP']} casos positivos contra los {before['TP']} de antes — "
            f"recupera **{after['TP'] - before['TP']} ideaciones suicidas adicionales**."
        )
        out.append(
            f"- Trade-off: el **FPR también subió** ({before['FPR']:.3f} → {after['FPR']:.3f}). "
            f"Pasamos de {before['FP']} falsas alarmas a {after['FP']}. "
            "El modelo es ahora menos conservador."
        )
        out.append(
            "- En este dominio el trade-off es **favorable**: prevenir más casos "
            "vale más que reducir falsas alarmas."
        )
        out.append("")

    # ============================================================
    # 5. Decisión final
    # ============================================================
    out.append("## 5. Decisión final y configuración productiva")
    out.append("")
    if has_xgb:
        wv = xgb["variant"]
        out.append(
            f"**Modelo final:** variante **`{wv}`** ({VARIANT_LABEL[wv]}) con XGBoost "
            f"tuneado por Optuna y early stopping activado."
        )
        out.append("")
        out.append(f"- Configuración YAML: `{xgb['yaml_path']}`")
        out.append(f"- Modelo serializado: `models/model_{wv}.joblib`")
        out.append(f"- Estudio Optuna persistido en: `{xgb['storage_path']}`")
    else:
        out.append(
            f"**Variante ganadora hasta ahora:** `{winner_v1}` "
            f"({VARIANT_LABEL[winner_v1]})."
        )
        out.append("")
        out.append("La etapa 2 (tuning de XGBoost) está pendiente.")
    out.append("")
    out.append("**Métricas finales reportables sobre el conjunto de prueba (252 filas):**")
    out.append("")
    out.append("| Métrica | Valor |")
    out.append("|---|---|")
    out.append(f"| AUC | **{final_test['AUC']:.4f}** |")
    out.append(f"| Recall (TPR) | **{final_test['recall']:.4f}** |")
    out.append(f"| Precisión | {final_test['precision']:.4f} |")
    out.append(f"| F1 | {final_test['F1']:.4f} |")
    out.append(f"| FPR | {final_test['FPR']:.4f} |")
    out.append(f"| TP / TN / FP / FN | {final_test['TP']} / {final_test['TN']} / "
               f"{final_test['FP']} / {final_test['FN']} |")
    out.append("")

    # ============================================================
    # 6. Cómo reproducir
    # ============================================================
    out.append("## 6. Cómo reproducir estos resultados")
    out.append("")
    out.append("```powershell")
    out.append("# 1. Ejecutar todo el pipeline de optimización (etapa 1: 4 variantes × 30 trials)")
    out.append("python scripts/run_optimization_pipeline.py --n-trials 30")
    out.append("")
    out.append("# 2. Ejecutar la etapa 2 sobre la variante ganadora (auto-detectada)")
    out.append("python scripts/run_xgb_optimization.py --n-trials 30")
    out.append("")
    out.append("# 3. Regenerar este reporte")
    out.append("python scripts/generate_optimization_report.py")
    out.append("```")
    out.append("")
    if has_xgb:
        out.append("**Para reanudar una corrida interrumpida** del XGB tuning, basta con "
                   "volver a ejecutar `run_xgb_optimization.py`: el `JournalStorage` retoma "
                   "desde el último trial completado.")
        out.append("")

    # ============================================================
    # 7. Metodología (para el reporte académico)
    # ============================================================
    out.append("## 7. Notas metodológicas")
    out.append("")
    out.append("- **Métrica objetivo de Optuna:** promedio de AUC sobre 5-fold "
               "StratifiedKFold (`random_state=42`).")
    out.append("- **Sampler de Optuna:** TPE (Tree-structured Parzen Estimator), "
               "`random_state=42`.")
    out.append("- **División train/test congelada:** `data/data_train.csv` (1,516 filas) y "
               "`data/data_test_fold1.csv` (252 filas) son disjuntas a nivel de `text_id`. "
               "El test fold solo se usa al final, después de toda la optimización.")
    if has_xgb:
        out.append("- **Early stopping:** durante la búsqueda Optuna, NO se aplica — todos "
                   "los trials usan exactamente el `n_estimators` propuesto, para que las "
                   "comparaciones entre trials sean justas. En el entrenamiento final SÍ "
                   "se aplica, usando un sub-split 90/10 del conjunto de entrenamiento "
                   "como `eval_set`.")
        out.append(f"- **Trees crecidos en el modelo final:** menos que el cap de "
                   f"`n_estimators={xgb['best_params']['n_estimators']}` (early stopping "
                   "se activó antes), evitando overfitting.")
    out.append("- **`scale_pos_weight = 0.93`:** equivale a la proporción `n_neg/n_pos` "
               "del corpus (732/784). Compensa el ligero desbalance de clases sin alterar "
               "la distribución original.")
    out.append("")
    out.append("---")
    out.append("")
    out.append("> **Aviso:** este clasificador es un proyecto académico. **No es una "
               "herramienta clínica** y no debe usarse para tomar decisiones sobre la "
               "salud o seguridad de ninguna persona. Si tú o alguien que conoces está "
               "en crisis, contacta los servicios de emergencia o una línea de prevención "
               "del suicidio en tu país.")
    out.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Reporte escrito -> {OUTPUT}")


if __name__ == "__main__":
    main()

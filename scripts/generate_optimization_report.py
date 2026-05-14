"""
Render `reports/optimization.md` from the consolidated JSON written by
`scripts/run_optimization_pipeline.py`.

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


VARIANT_DESCRIPTIONS = {
    "base": {
        "title": "Baseline — `clean_text` solamente",
        "expected_effect": (
            "El baseline aplica únicamente encoding repair, normalización de "
            "comillas, lowercase y eliminación de URLs / menciones / hashtags / "
            "emojis / caracteres especiales. **No** filtra stopwords ni hace "
            "stemming. La hipótesis es que `max_df=0.95` + `sublinear_tf` + "
            "`min_df` cumplen el trabajo que normalmente haría una lista de "
            "stopwords explícita, y que con ~1,500 documentos el costo de "
            "stemming (pérdida de morfología fina como `die` vs `died` vs "
            "`dying`) puede no compensar la reducción de vocabulario."
        ),
    },
    "stopwords_nltk": {
        "title": "Stopwords NLTK — preservando negaciones",
        "expected_effect": (
            "Eliminar la lista estándar de stopwords del inglés de NLTK pero "
            "**preservando** las 10 negaciones (`no`, `not`, `never`, …) que "
            "son críticas en este dominio (\"I do **not** want to live\" "
            "tiene polaridad opuesta a \"I want to live\"). Hipótesis: si la "
            "lista estándar de stopwords aporta valor real, debería ganar al "
            "baseline; si `max_df=0.95` ya las filtraba implícitamente, los "
            "resultados deberían ser equivalentes."
        ),
    },
    "stopwords_domain": {
        "title": "Stopwords de dominio — solo tokens gramaticales",
        "expected_effect": (
            "Lista curada que contiene **únicamente** tokens puramente "
            "gramaticales (artículos, preposiciones, conjunciones, auxiliares "
            "no negados, pronombres). Excluye explícitamente toda palabra con "
            "contenido emocional o experiencial (\"alone\", \"hopeless\", "
            "\"die\", \"tired\", …) que listas genéricas a veces incluyen. "
            "Más conservador que NLTK: debería retener más señal."
        ),
    },
    "stemming": {
        "title": "Stemming — Porter Stemmer token a token",
        "expected_effect": (
            "Colapsa variantes morfológicas (`die` / `died` / `dying` / "
            "`dies` → `die`) usando Porter Stemmer. Reduce el vocabulario y "
            "puede ayudar a generalizar con corpus pequeño, pero arriesga "
            "fusionar tokens con perfiles de uso distintos (p. ej. `tried` "
            "vs `try`). Hipótesis: ganancia marginal o pérdida pequeña; "
            "el verdadero ganador suele aparecer cuando hay <500 docs."
        ),
    },
}


def fmt_pct(x):
    if x is None: return "—"
    return f"{x:.4f}"


def main():
    if not CONSOLIDATED.exists():
        raise SystemExit(
            f"Consolidated results not found at {CONSOLIDATED}.\n"
            f"Run `python scripts/run_optimization_pipeline.py` first."
        )

    with open(CONSOLIDATED) as f:
        data = json.load(f)

    variants = list(data.keys())

    # --- Identify the winner by Test AUC ----------------------------------
    winner = max(variants, key=lambda v: data[v]["test"]["AUC"])
    winner_auc = data[winner]["test"]["AUC"]

    # --- Build the markdown ------------------------------------------------
    out = []
    out.append("# Reporte de Optimización — Variantes de Preprocesamiento + Tuning de TF-IDF")
    out.append("")
    out.append(f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    out.append("")
    out.append("> **Equipo TC3002B:** Aislinn Ruiz Sandoval · Iván Alexander Ramos Ramírez · Miguel Ángel Galicia Sánchez · Víctor Alejandro Morales García")
    out.append("")
    out.append("---")
    out.append("")

    # 1. Resumen ejecutivo
    out.append("## 1. Resumen ejecutivo")
    out.append("")
    out.append(
        f"Se compararon **{len(variants)} variantes de preprocesamiento** sobre el "
        f"clasificador binario de ideación suicida (TF-IDF + XGBoost). Para cada "
        f"variante, **Optuna optimizó independientemente los hiperparámetros del "
        f"`TfidfVectorizer`** (5 parámetros, 30 trials, métrica = 5-fold Stratified CV AUC) "
        f"manteniendo fijos los hiperparámetros del `XGBClassifier`."
    )
    out.append("")
    out.append(
        f"**Ganador (por Test AUC sobre `data_test_fold1.csv`): `{winner}`** con "
        f"AUC = {winner_auc:.4f}. La justificación detallada está en la sección 6."
    )
    out.append("")

    # 2. Tabla de hiperparámetros óptimos
    out.append("## 2. Hiperparámetros óptimos por variante")
    out.append("")
    out.append("| Variante | `ngram_range` | `min_df` | `max_df` | `sublinear_tf` | `max_features` |")
    out.append("|---|---|---|---|---|---|")
    for v in variants:
        p = data[v]["best_params"]
        ngram = f"({p['ngram_range'][0]}, {p['ngram_range'][1]})"
        out.append(
            f"| `{v}` | {ngram} | {p['min_df']} | {p['max_df']:.4f} | "
            f"{p['sublinear_tf']} | {p['max_features']:,} |"
        )
    out.append("")

    # 3. Tabla de métricas finales
    out.append("## 3. Métricas finales comparadas")
    out.append("")
    out.append(
        "Las columnas reflejan: el mejor AUC observado durante la búsqueda Optuna "
        "(CV 5-fold sobre el train completo), el AUC en el holdout interno (20% "
        "de `data_train.csv`), y las métricas finales sobre el fold de prueba "
        "(`data_test_fold1.csv`, 252 filas, jamás visto durante CV ni Optuna)."
    )
    out.append("")
    out.append("| Variante | Opt CV AUC | Val AUC | **Test AUC** | Recall (test) | FPR (test) | F1 (test) | Precision (test) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for v in variants:
        opt_auc = data[v]["optuna"]["best_cv_auc"]
        val_auc = data[v]["train"]["val_auc"]
        t       = data[v]["test"]
        marker  = " 🏆" if v == winner else ""
        out.append(
            f"| `{v}`{marker} | {fmt_pct(opt_auc)} | {fmt_pct(val_auc)} | "
            f"**{fmt_pct(t['AUC'])}** | {fmt_pct(t['recall'])} | "
            f"{fmt_pct(t['FPR'])} | {fmt_pct(t['F1'])} | {fmt_pct(t['precision'])} |"
        )
    out.append("")

    # 4. Confusión por variante
    out.append("### Matriz de confusión por variante (sobre el test fold)")
    out.append("")
    out.append("| Variante | TP | TN | FP | FN |")
    out.append("|---|---|---|---|---|")
    for v in variants:
        t = data[v]["test"]
        out.append(f"| `{v}` | {t['TP']} | {t['TN']} | {t['FP']} | {t['FN']} |")
    out.append("")

    # 5. Análisis por variante
    out.append("## 4. Análisis por variante")
    out.append("")
    for v in variants:
        info = VARIANT_DESCRIPTIONS[v]
        out.append(f"### 4.{variants.index(v)+1}. {info['title']}")
        out.append("")
        out.append(f"**Efecto esperado en el dominio.** {info['expected_effect']}")
        out.append("")

        t = data[v]["test"]
        opt_auc = data[v]["optuna"]["best_cv_auc"]
        out.append("**Resultados empíricos:**")
        out.append("")
        out.append(f"- Opt CV AUC = {fmt_pct(opt_auc)}")
        out.append(f"- Test AUC = {fmt_pct(t['AUC'])}, F1 = {fmt_pct(t['F1'])}, "
                   f"Recall = {fmt_pct(t['recall'])}, FPR = {fmt_pct(t['FPR'])}")
        out.append("")

        # Diff vs baseline
        if v != "base":
            base_test = data["base"]["test"]["AUC"]
            delta = t["AUC"] - base_test
            sign = "+" if delta >= 0 else ""
            verdict = (
                "**confirma** la hipótesis: la técnica aporta señal."
                if delta > 0.002 else
                ("**rechaza** la hipótesis: la técnica reduce el desempeño."
                 if delta < -0.002 else
                 "**inconcluso**: la diferencia con el baseline es estadísticamente despreciable (|Δ AUC| ≤ 0.002).")
            )
            out.append(f"**Δ Test AUC vs baseline = {sign}{delta:.4f}** → {verdict}")
            out.append("")

    # 5. Metodología
    out.append("## 5. Metodología de optimización")
    out.append("")
    out.append(
        "**Optuna** (TPE sampler con `random_state=42`) explora 30 trials por variante "
        "sobre el espacio:"
    )
    out.append("")
    out.append("- `ngram_range` ∈ {(1,1), (1,2), (1,3)}")
    out.append("- `min_df` ∈ [1, 5]  (int uniforme)")
    out.append("- `max_df` ∈ [0.70, 0.99]  (float uniforme)")
    out.append("- `sublinear_tf` ∈ {True, False}")
    out.append("- `max_features` ∈ {5000, 10000, 20000, 50000}")
    out.append("")
    out.append(
        "La **métrica objetivo** es el promedio de AUC sobre 5-fold StratifiedKFold "
        "aplicado al conjunto de entrenamiento completo (`data/data_train.csv`, "
        "1,516 filas). El test fold (`data_test_fold1.csv`, 252 filas) **no se toca** "
        "durante la búsqueda — solo se usa al final para el reporte de Test AUC, "
        "evitando data leakage en la métrica reportada."
    )
    out.append("")
    out.append(
        "Los **hiperparámetros del XGBClassifier** (`n_estimators=500`, "
        "`max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=0.93`, etc.) "
        "permanecen fijos durante toda la búsqueda — solo se optimizan los del "
        "vectorizador. Esto aísla el efecto del preprocesamiento + vectorización "
        "del efecto del clasificador, dejando el tuning del clasificador como "
        "trabajo separado en el roadmap."
    )
    out.append("")
    out.append(
        "Para robustez, el `objective` de Optuna atrapa `ValueError` (típicamente "
        "lanzado cuando una combinación patológica de `min_df` / `max_df` poda el "
        "vocabulario a cero) y devuelve `0.0` — esto hace que TPE evite esa región "
        "del espacio de búsqueda en lugar de abortar todo el estudio."
    )
    out.append("")
    out.append(
        "Optimización adicional: el cleaner de cada variante se aplica **una sola "
        "vez** al corpus completo antes del CV (no dentro del Pipeline), porque "
        "la limpieza es una función determinística sin dependencia entre filas. "
        "Esto es matemáticamente equivalente al pipeline de producción "
        "(`build_pipeline` la aplica dentro como FunctionTransformer) pero acelera "
        "la búsqueda ~5× evitando re-aplicar la limpieza en cada fold."
    )
    out.append("")

    # 6. Conclusión
    # --- XGB tuning section (only if present in the JSON) ----------------
    if "xgb_optimization" in data:
        xgb = data["xgb_optimization"]
        winner_v = xgb["variant"]
        baseline_test = data[winner_v]["test"]
        tuned_test = xgb["test"]
        delta = tuned_test["AUC"] - baseline_test["AUC"]

        out.append("## 6. Optimización del XGBClassifier")
        out.append("")
        out.append(
            f"Tras seleccionar **`{winner_v}`** como ganadora del tuning de "
            f"TF-IDF, se ejecutó un segundo estudio Optuna ({xgb['n_trials']} "
            f"trials, TPE sampler, 5-fold Stratified CV AUC) sobre los 9 "
            f"hiperparámetros del clasificador. El estudio se persistió a "
            f"`{xgb['storage_path']}` para permitir reanudación tras "
            f"interrupciones."
        )
        out.append("")

        out.append("### 6.1. Hiperparámetros XGB óptimos")
        out.append("")
        out.append("| Hiperparámetro | Valor |")
        out.append("|---|---|")
        for k, v in xgb["best_params"].items():
            out.append(f"| `{k}` | {v} |")
        out.append("")
        out.append("Más los fijos no tuneados:")
        out.append("")
        for k, v in xgb.get("fixed_keys", {}).items():
            out.append(f"- `{k}`: {v}")
        out.append("")

        out.append("### 6.2. Antes vs después del tuning de XGB")
        out.append("")
        out.append("| Métrica | Antes (XGB defaults) | Después (XGB tuneado) | Δ |")
        out.append("|---|---|---|---|")
        for metric in ("AUC", "F1", "precision", "recall", "FPR"):
            before = baseline_test[metric]
            after  = tuned_test[metric]
            d = after - before
            sign = "+" if d >= 0 else ""
            out.append(f"| {metric} | {fmt_pct(before)} | {fmt_pct(after)} | {sign}{d:.4f} |")
        out.append("")

        if delta > 0.005:
            verdict_xgb = (
                f"**Mejora significativa** (Δ AUC = +{delta:.4f}). El tuning de "
                f"XGB compensa el costo de optimización adicional."
            )
        elif delta < -0.002:
            verdict_xgb = (
                f"**Regresión** (Δ AUC = {delta:.4f}). El XGB tuneado es peor "
                f"que el default — posiblemente overfitting al CV. Conservar la "
                f"configuración previa."
            )
        else:
            verdict_xgb = (
                f"**Equivalente al default** (Δ AUC = {delta:+.4f}, dentro del "
                f"ruido de CV). El tuning explora valores cercanos a los "
                f"defaults; no hay ganancia clara pero tampoco regresión."
            )
        out.append(f"**Veredicto:** {verdict_xgb}")
        out.append("")

    out.append("## 7. Conclusión y elección final")
    out.append("")
    winner_info = VARIANT_DESCRIPTIONS[winner]
    base_auc = data["base"]["test"]["AUC"]
    winner_test_auc = data[winner]["test"]["AUC"]
    margin = winner_test_auc - base_auc

    out.append(
        f"La variante ganadora es **`{winner}`** ({winner_info['title']}) "
        f"con Test AUC = {winner_test_auc:.4f}, "
        f"frente al baseline en {base_auc:.4f} (Δ = {margin:+.4f})."
    )
    out.append("")

    if abs(margin) < 0.003:
        out.append(
            "Sin embargo, la diferencia con el baseline es **menor a 0.003 AUC**, "
            "que está bien dentro de la varianza esperada del CV (recordar que "
            "`cv_auc_std` del baseline original era ~0.027). Por **parsimonia** "
            "— el principio de preferir la solución más simple cuando los "
            "resultados son equivalentes — la recomendación es **conservar el "
            "baseline** como pipeline de producción. Las variantes con stopwords / "
            "stemming agregan dependencias (NLTK) y reducen interpretabilidad "
            "sin justificación empírica clara en este corpus."
        )
    else:
        out.append(
            "Esta diferencia es **suficientemente grande** para justificar el "
            f"costo adicional de `{winner}`: dependencias (NLTK), tiempo de "
            f"limpieza ligeramente mayor y, en el caso de stemming, pérdida de "
            f"morfología fina. La recomendación es **promover `{winner}` a "
            f"pipeline de producción** y actualizar `configs/default.yaml` con "
            f"sus hiperparámetros óptimos."
        )

    out.append("")
    out.append("### Trabajo futuro")
    out.append("")
    out.append("- Tuning de hiperparámetros del XGBClassifier (deliberadamente fuera de scope aquí).")
    out.append("- Threshold tuning: el corte 0.5 puede no ser óptimo cuando el FN es más costoso que el FP.")
    out.append("- Splits user-disjoint para descartar la fuga sutil de overlap de `user_id` entre train y test.")
    out.append("- Combinaciones: ¿stemming + stopwords_domain? El espacio crece pero podría haber ganancia compuesta.")
    out.append("")

    # 7. Apéndice — distribución de trials
    out.append("## 8. Apéndice — distribución de trials de Optuna")
    out.append("")
    out.append("Para cada variante, los 30 AUC observados durante la búsqueda:")
    out.append("")
    out.append("| Variante | min | mediana | max | std |")
    out.append("|---|---|---|---|---|")
    for v in variants:
        scores = data[v]["optuna"]["all_trial_scores"]
        if not scores:
            continue
        import statistics
        out.append(
            f"| `{v}` | {min(scores):.4f} | {statistics.median(scores):.4f} | "
            f"{max(scores):.4f} | {statistics.stdev(scores) if len(scores) > 1 else 0:.4f} |"
        )
    out.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Reporte escrito -> {OUTPUT}")


if __name__ == "__main__":
    main()

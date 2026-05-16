"""
Generate a single consolidated ``reports/final_report.md`` from the
JSONs produced by the optimization pipeline.

Reads ``reports/optimization_results.json`` (the one source of truth
written by stages 1 and 2) and renders a markdown report with:

  1. Pipeline overview (placeholder for manual insight).
  2. Variant comparison table.
  3. TF-IDF hyperparameters of the winner.
  4. XGBoost before-vs-after table (if stage 2 was run).
  5. Final test-set results with embedded ROC / confusion-matrix PNGs.
  6. Limitations (placeholder for manual insight).

Run as:  python scripts/generate_final_report.py
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_config

CONSOLIDATED = "reports/optimization_results.json"
OUTPUT = "reports/final_report.md"
INSIGHT = "<!-- INSIGHT: escribe tu interpretacion aqui -->"

VARIANT_ORDER = ("base", "stopwords_nltk", "stopwords_domain", "stemming")
VARIANT_DISPLAY = {
    "base":             "Baseline",
    "stopwords_nltk":   "Stopwords NLTK",
    "stopwords_domain": "Stopwords curadas",
    "stemming":         "Stemming",
}


def _fmt(v, decimals=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _resolve_image_paths(yaml_path: str) -> dict:
    """Return paths to ROC and CM PNGs, relative to reports/."""
    try:
        cfg = load_config(yaml_path)
        roc_test = cfg["paths"]["roc_test_output"]
        cm_test = cfg["paths"]["cm_test_output"]
        roc_val = cfg["paths"]["roc_val_output"]
        cm_val = cfg["paths"]["cm_val_output"]
    except (FileNotFoundError, KeyError):
        roc_test = cm_test = roc_val = cm_val = None

    def _rel(p):
        if p is None:
            return None
        return str(Path(p).relative_to("reports")) if p.startswith("reports") else p

    return {
        "roc_test": _rel(roc_test),
        "cm_test": _rel(cm_test),
        "roc_val": _rel(roc_val),
        "cm_val": _rel(cm_val),
    }


def main() -> None:
    with open(CONSOLIDATED) as f:
        data = json.load(f)

    _OPT_KEYS = {"xgb_optimization", "svm_optimization", "lr_optimization"}
    variants = {k: v for k, v in data.items() if k not in _OPT_KEYS}
    xgb_opt = data.get("xgb_optimization")
    svm_opt = data.get("svm_optimization")
    lr_opt  = data.get("lr_optimization")

    winner_stage1 = max(variants, key=lambda v: variants[v]["test"]["AUC"])

    # Pick best model across all tuned options by Test AUC
    model_candidates = {
        label: entry for label, entry in [
            ("XGBoost", xgb_opt),
            ("SVM",     svm_opt),
            ("LR",      lr_opt),
        ] if entry is not None
    }
    if model_candidates:
        best_model_label = max(model_candidates, key=lambda k: model_candidates[k]["test"]["AUC"])
        final = model_candidates[best_model_label]
    else:
        best_model_label = None
        final = variants[winner_stage1]

    final_yaml = final.get("yaml_path", f"configs/optimized/{final.get('variant', winner_stage1)}.yaml")
    imgs = _resolve_image_paths(final_yaml)

    out = []

    # ── Header ──────────────────────────────────────────────────────────
    out.append("# Detector de Ideacion Suicida — Reporte Final")
    out.append("")
    out.append(f"> Equipo TC3002B · {date.today().isoformat()}")
    out.append(">")
    out.append("> Aislinn Ruiz Sandoval · Ivan Alexander Ramos Ramirez "
               "· Miguel Angel Galicia Sanchez · Victor Alejandro Morales Garcia")
    out.append("")
    out.append("---")
    out.append("")

    # ── 1. Pipeline ─────────────────────────────────────────────────────
    out.append("## 1. Pipeline")
    out.append("")
    out.append(INSIGHT)
    out.append("")

    # ── 2. Comparativa de variantes ─────────────────────────────────────
    out.append("## 2. Preprocesamiento — Comparativa de variantes")
    out.append("")
    out.append("| Variante | CV AUC | Test AUC | Recall | FPR | TP | FN |")
    out.append("|---|---|---|---|---|---|---|")

    for v_name in VARIANT_ORDER:
        if v_name not in variants:
            continue
        v = variants[v_name]
        t = v["test"]
        display = VARIANT_DISPLAY.get(v_name, v_name)
        if v_name == winner_stage1:
            display = f"**{display}**"
        cv_auc = v.get("optuna", {}).get("best_cv_auc")
        out.append(
            f"| {display} "
            f"| {_fmt(cv_auc)} "
            f"| {_fmt(t['AUC'])} "
            f"| {_fmt(t.get('recall', t.get('TPR')))} "
            f"| {_fmt(t['FPR'])} "
            f"| {t['TP']} "
            f"| {t['FN']} |"
        )

    out.append("")
    winner_auc = variants[winner_stage1]["test"]["AUC"]
    out.append(
        f"Ganadora: **{VARIANT_DISPLAY.get(winner_stage1, winner_stage1)}** "
        f"(Test AUC = {_fmt(winner_auc)})"
    )
    out.append("")
    out.append(INSIGHT)
    out.append("")

    # ── 3. TF-IDF hiperparametros ──────────────────────────────────────
    out.append("## 3. Optimizacion TF-IDF — Hiperparametros finales")
    out.append("")

    winner_params = variants[winner_stage1].get("best_params", {})
    out.append("| Parametro | Valor |")
    out.append("|---|---|")
    for k, v in winner_params.items():
        out.append(f"| `{k}` | `{v}` |")

    out.append("")
    out.append(INSIGHT)
    out.append("")

    # ── 4. XGBoost antes vs despues ────────────────────────────────────
    if xgb_opt:
        out.append("## 4. Optimizacion XGBoost — Antes vs Despues")
        out.append("")

        before = variants[winner_stage1]["test"]
        after = xgb_opt["test"]

        metrics_to_compare = [
            ("AUC", "AUC"),
            ("Recall (TPR)", "recall"),
            ("Precision", "precision"),
            ("F1", "F1"),
            ("FPR", "FPR"),
        ]

        out.append("| Metrica | Stage 1 | Stage 2 | Delta |")
        out.append("|---|---|---|---|")
        for label, key in metrics_to_compare:
            b = before.get(key)
            a = after.get(key)
            if b is not None and a is not None:
                delta = a - b
                sign = "+" if delta >= 0 else ""
                out.append(
                    f"| {label} | {_fmt(b)} | {_fmt(a)} | {sign}{_fmt(delta)} |"
                )
            else:
                out.append(f"| {label} | {_fmt(b)} | {_fmt(a)} | — |")

        out.append("")

        xgb_params = xgb_opt.get("best_params", {})
        out.append("**Hiperparametros XGBoost tuneados:**")
        out.append("")
        out.append("| Parametro | Valor |")
        out.append("|---|---|")
        for k, v in xgb_params.items():
            out.append(f"| `{k}` | `{v}` |")

        out.append("")
        out.append(INSIGHT)
        out.append("")

    # ── 5. Comparacion de modelos ──────────────────────────────────────
    model_entries = []
    if xgb_opt:
        model_entries.append(("XGBoost (tuneado)", xgb_opt))
    if svm_opt:
        model_entries.append(("SVM (tuneado)", svm_opt))
    if lr_opt:
        model_entries.append(("Logistic Regression (tuneada)", lr_opt))

    if len(model_entries) >= 2:
        out.append("## 5. Comparacion de Modelos")
        out.append("")
        out.append("| Modelo | CV AUC | Test AUC | F1 | Recall | FPR | TP | FN |")
        out.append("|---|---|---|---|---|---|---|---|")
        for label, entry in model_entries:
            t = entry["test"]
            cv_auc = entry.get("best_cv_auc")
            out.append(
                f"| {label} "
                f"| {_fmt(cv_auc)} "
                f"| {_fmt(t['AUC'])} "
                f"| {_fmt(t['F1'])} "
                f"| {_fmt(t.get('recall', t.get('TPR')))} "
                f"| {_fmt(t['FPR'])} "
                f"| {t['TP']} "
                f"| {t['FN']} |"
            )
        out.append("")
        if best_model_label:
            best_auc = model_candidates[best_model_label]["test"]["AUC"]
            out.append(f"Ganador: **{best_model_label}** (Test AUC = {_fmt(best_auc)})")
            out.append("")
        out.append(INSIGHT)
        out.append("")

    # ── 6. Resultados finales sobre test ───────────────────────────────
    section6_label = f"## 6. Resultados finales sobre test ({best_model_label})" if best_model_label else "## 6. Resultados finales sobre test"
    out.append(section6_label)
    out.append("")

    ft = final["test"]
    out.append("| Metrica | Valor |")
    out.append("|---|---|")
    out.append(f"| TP | {ft['TP']} |")
    out.append(f"| TN | {ft['TN']} |")
    out.append(f"| FP | {ft['FP']} |")
    out.append(f"| FN | {ft['FN']} |")
    out.append(f"| Recall (TPR) | {_fmt(ft.get('recall', ft.get('TPR')))} |")
    out.append(f"| FPR | {_fmt(ft['FPR'])} |")
    out.append(f"| Precision | {_fmt(ft['precision'])} |")
    out.append(f"| F1 | {_fmt(ft['F1'])} |")
    out.append(f"| AUC | {_fmt(ft['AUC'])} |")
    out.append("")

    if imgs["roc_test"]:
        out.append(f"![Curva ROC — Test]({imgs['roc_test']})")
        out.append("")
    if imgs["cm_test"]:
        out.append(f"![Matriz de Confusion — Test]({imgs['cm_test']})")
        out.append("")

    out.append(INSIGHT)
    out.append("")

    # ── 7. Limitaciones ────────────────────────────────────────────────
    out.append("## 7. Limitaciones")
    out.append("")
    out.append(INSIGHT)
    out.append("")

    # ── Write ──────────────────────────────────────────────────────────
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"Reporte generado en {OUTPUT}")


if __name__ == "__main__":
    main()

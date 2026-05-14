import argparse
import json
import os

from src.data_ingestion import ingestion
from src.evaluation import evaluate, generate_report
from src.inference import load_pipeline, predict
from src.preprocessing import clean_text
from src.utils import load_config


def run_evaluation(input_path: str, config: dict) -> dict:
    """
    Score a trained pipeline against a labeled CSV and write reports to disk.

    Distinct from `inference.run_inference`: this function REQUIRES the input
    CSV to contain the ground-truth target column declared in the config, so
    it can compute the full metrics dict and ROC / confusion-matrix plots.
    Use this for the held-out test fold; use `run_inference` for blind
    prediction on unlabeled data.

    Parameters
    ----------
    input_path : str
        Path to a labeled CSV (must contain `config["data"]["target_column"]`).
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    dict
        Full metrics from `evaluation.evaluate()`:
        TP, TN, FP, FN, TPR, FPR, AUC, precision, recall, F1.
        Also persisted as JSON at `config["paths"]["metrics_test_output"]`,
        with ROC PNG at `roc_test_output` and confusion-matrix PNG at
        `cm_test_output`.

    Raises
    ------
    ValueError
        If the target column is missing from the input CSV.
    """
    target_col = config["data"]["target_column"]

    df = ingestion(
        data_url=input_path,
        expected_columns=config["data"]["expected_columns"],
        text_columns=config["data"]["text_columns"],
        target_column=target_col,
        value_true=config["data"]["value_true"],
        value_false=config["data"]["value_false"],
    )

    if df is None or target_col not in df.columns:
        raise ValueError(
            f"Input CSV at {input_path} is missing the target column "
            f"'{target_col}'. Use `predict` for unlabeled data."
        )

    text_col = "_".join(config["data"]["text_columns"])
    df[text_col] = df[text_col].apply(clean_text)

    pipeline = load_pipeline(config["paths"]["model_output"])
    texts = df[text_col].tolist()
    preds, probas = predict(pipeline, texts)

    y_true = df[target_col].tolist()

    os.makedirs(config["paths"]["reports_dir"], exist_ok=True)
    metrics = evaluate(
        y_true,
        preds,
        probas,
        roc_path=config["paths"]["roc_test_output"],
        cm_path=config["paths"]["cm_test_output"],
    )

    with open(config["paths"]["metrics_test_output"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test metrics saved -> {config['paths']['metrics_test_output']}")

    generate_report(
        metrics,
        roc_path=config["paths"]["roc_test_output"],
        cm_path=config["paths"]["cm_test_output"],
        output_path=config["paths"]["report_test_output"],
        title="Reporte de Evaluación — Detección de Ideación Suicida (Test Fold)",
    )

    print(metrics)

    return metrics


def run_evaluation_cli(input_path: str, config_path: str) -> dict:
    """
    CLI wrapper: loads the YAML config from disk and delegates to `run_evaluation`.
    """
    config = load_config(config_path)
    return run_evaluation(input_path, config)


def main():
    parser = argparse.ArgumentParser(
        description="Score a trained model against a labeled CSV and emit reports."
    )
    parser.add_argument("--input",  required=True, help="Path to labeled input CSV")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    run_evaluation_cli(args.input, args.config)


if __name__ == "__main__":
    main()

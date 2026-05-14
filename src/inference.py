import argparse
import pandas as pd
import joblib

from src.data_ingestion import ingestion
from src.preprocessing import clean_text
from src.utils import load_config


def load_pipeline(model_path: str):
    """
    Load a trained sklearn Pipeline from disk.

    Parameters
    ----------
    model_path : str
        Path to the .joblib file saved by training.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for inference.
    """
    return joblib.load(model_path)


def predict(pipeline, texts: list) -> tuple[list, list]:
    """
    Run inference on a list of preprocessed texts.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    texts    : list of str — already cleaned texts

    Returns
    -------
    tuple (predictions, probabilities)
        predictions  : list of int (0 or 1)
        probabilities: list of float (probability of class 1)
    """
    predictions  = pipeline.predict(texts).tolist()
    probabilities = pipeline.predict_proba(texts)[:, 1].tolist()
    return predictions, probabilities


def run_inference(
    input_path: str,
    output_path: str,
    config: dict,
) -> pd.DataFrame:
    """
    Full inference pipeline: load CSV → preprocess → predict → save results.

    Parameters
    ----------
    input_path  : str — path to input CSV (same schema as training data)
    output_path : str — path to write predictions CSV
    config      : dict — loaded YAML config

    Returns
    -------
    pd.DataFrame with columns: text_id, prediction, probability
    """
    # --- Ingest ---
    df = ingestion(
        data_url=input_path,
        expected_columns=config["data"]["expected_columns"],
        text_columns=config["data"]["text_columns"],
    )

    # --- Preprocess ---
    text_col = "_".join(config["data"]["text_columns"])
    df[text_col] = df[text_col].apply(clean_text)

    # --- Load model & predict ---
    pipeline = load_pipeline(config["paths"]["model_output"])
    texts = df[text_col].tolist()
    preds, probas = predict(pipeline, texts)

    # --- Build output ---
    results = pd.DataFrame({
        "text_id":     df["text_id"],
        "prediction":  preds,
        "probability": [round(p, 4) for p in probas],
    })

    results.to_csv(output_path, index=False)
    print(f"Predictions saved → {output_path}  ({len(results)} rows)")

    return results

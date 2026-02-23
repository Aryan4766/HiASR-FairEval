"""
Compute WER — JoshTalks ASR Research
======================================
Standalone WER/CER computation from predictions CSV.
Outputs structured metrics JSON files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import compute_corpus_metrics

logger = setup_logger("compute_wer", log_to_file=True)


def load_predictions(csv_path: str) -> pd.DataFrame:
    """
    Load predictions CSV with reference and hypothesis columns.

    Expected columns: reference, prediction (or hypothesis)

    Args:
        csv_path: Path to predictions CSV.

    Returns:
        DataFrame with 'reference' and 'prediction' columns.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    col_mapping = {
        "hypothesis": "prediction",
        "hyp": "prediction",
        "pred": "prediction",
        "ref": "reference",
        "ground_truth": "reference",
        "target": "reference",
    }

    df = df.rename(columns={
        col: col_mapping.get(col.lower(), col)
        for col in df.columns
    })

    required = {"reference", "prediction"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns: {required}. Found: {set(df.columns)}"
        )

    # Handle NaN values
    df["reference"] = df["reference"].fillna("").astype(str)
    df["prediction"] = df["prediction"].fillna("").astype(str)

    return df


def compute_and_save_metrics(
    predictions_csv: str,
    output_json: str,
    label: str = "evaluation",
) -> Dict:
    """
    Compute WER/CER from predictions CSV and save as JSON.

    Args:
        predictions_csv: Path to predictions CSV.
        output_json: Path to save metrics JSON.
        label: Label for this evaluation run.

    Returns:
        Metrics dictionary.
    """
    logger.info(f"Computing metrics for: {label}")
    logger.info(f"Loading predictions from: {predictions_csv}")

    df = load_predictions(predictions_csv)
    references = df["reference"].tolist()
    predictions = df["prediction"].tolist()

    logger.info(f"Loaded {len(references)} utterance pairs")

    # Compute corpus metrics
    metrics = compute_corpus_metrics(references, predictions)

    # Remove per-utterance details from summary (too large for JSON)
    per_utterance = metrics.pop("per_utterance", [])

    # Add label and metadata
    metrics["label"] = label
    metrics["predictions_file"] = predictions_csv

    # Save metrics
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Metrics saved to: {output_json}")
    logger.info(f"  WER: {metrics['corpus_wer']:.4f}")
    logger.info(f"  CER: {metrics['corpus_cer']:.4f}")
    logger.info(f"  Sub rate: {metrics['substitution_rate']:.4f}")
    logger.info(f"  Ins rate: {metrics['insertion_rate']:.4f}")
    logger.info(f"  Del rate: {metrics['deletion_rate']:.4f}")

    # Also save per-utterance metrics
    per_utt_df = pd.DataFrame(per_utterance)
    per_utt_csv = output_json.replace(".json", "_per_utterance.csv")
    per_utt_df.to_csv(per_utt_csv, index=False)
    logger.info(f"Per-utterance metrics saved to: {per_utt_csv}")

    return metrics


def print_wer_table(metrics_list: List[Dict]) -> None:
    """
    Print a formatted WER comparison table.

    Args:
        metrics_list: List of metrics dictionaries with 'label' key.
    """
    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'WER':>8} {'CER':>8} {'Sub%':>8} {'Ins%':>8} {'Del%':>8}")
    print(f"{'='*70}")
    for m in metrics_list:
        print(
            f"{m['label']:<25} "
            f"{m['corpus_wer']:>8.4f} "
            f"{m['corpus_cer']:>8.4f} "
            f"{m['substitution_rate']:>8.4f} "
            f"{m['insertion_rate']:>8.4f} "
            f"{m['deletion_rate']:>8.4f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute WER from predictions CSV")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--label", default="evaluation", help="Label for this run")
    args = parser.parse_args()

    metrics = compute_and_save_metrics(args.predictions, args.output, args.label)
    print_wer_table([metrics])

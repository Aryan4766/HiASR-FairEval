"""
Error Analysis — JoshTalks ASR Research
=========================================
Detailed error analysis for Whisper ASR predictions:
- Top substitution pairs
- Most frequent deletion words
- Length vs WER correlation
- Hardest utterances
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from src.utils.metrics import compute_wer_detailed, get_error_pairs
from src.q1_whisper.compute_wer import load_predictions

logger = setup_logger("error_analysis", log_to_file=True)


def analyze_substitutions(
    references: List[str],
    hypotheses: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find top-N most frequent substitution pairs.

    Args:
        references: Reference transcriptions.
        hypotheses: Model predictions.
        top_n: Number of top pairs to return.

    Returns:
        DataFrame with columns: ref_word, hyp_word, count.
    """
    substitutions, _, _ = get_error_pairs(references, hypotheses)
    counter = Counter(substitutions)
    top_pairs = counter.most_common(top_n)

    df = pd.DataFrame(top_pairs, columns=["pair", "count"])
    df["ref_word"] = df["pair"].apply(lambda x: x[0])
    df["hyp_word"] = df["pair"].apply(lambda x: x[1])
    df = df[["ref_word", "hyp_word", "count"]]

    return df


def analyze_deletions(
    references: List[str],
    hypotheses: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find most frequently deleted words.

    Args:
        references: Reference transcriptions.
        hypotheses: Model predictions.
        top_n: Number of top words to return.

    Returns:
        DataFrame with columns: word, count.
    """
    _, deleted_words, _ = get_error_pairs(references, hypotheses)
    counter = Counter(deleted_words)
    top_deletions = counter.most_common(top_n)

    return pd.DataFrame(top_deletions, columns=["word", "count"])


def compute_length_vs_wer(
    references: List[str],
    hypotheses: List[str],
) -> pd.DataFrame:
    """
    Compute per-utterance WER vs reference length correlation.

    Returns:
        DataFrame with columns: ref_length, wer, reference, prediction.
    """
    results = []
    for ref, hyp in zip(references, hypotheses):
        metrics = compute_wer_detailed(ref, hyp)
        results.append({
            "ref_length": len(ref.split()),
            "wer": metrics["wer"],
            "reference": ref[:100],
            "prediction": hyp[:100],
        })

    return pd.DataFrame(results)


def find_hardest_utterances(
    references: List[str],
    hypotheses: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find utterances with highest WER.

    Returns:
        DataFrame sorted by WER descending.
    """
    results = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        metrics = compute_wer_detailed(ref, hyp)
        results.append({
            "index": i,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "ref_length": len(ref.split()),
            "reference": ref,
            "prediction": hyp,
        })

    df = pd.DataFrame(results)
    return df.nlargest(top_n, "wer")


def plot_length_vs_wer(
    length_wer_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot scatter of reference length vs WER with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        length_wer_df["ref_length"],
        length_wer_df["wer"],
        alpha=0.5,
        color="#4A90D9",
        s=30,
    )

    # Add regression line
    z = np.polyfit(length_wer_df["ref_length"], length_wer_df["wer"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(
        length_wer_df["ref_length"].min(),
        length_wer_df["ref_length"].max(),
        100,
    )
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, label=f"Trend (slope={z[0]:.4f})")

    # Correlation
    corr = length_wer_df["ref_length"].corr(length_wer_df["wer"])
    ax.set_title(f"Reference Length vs WER (r={corr:.3f})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Reference Length (words)", fontsize=12)
    ax.set_ylabel("WER", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Length vs WER plot saved to {save_path}")


def plot_wer_comparison(
    metrics_list: List[Dict],
    save_path: str,
) -> None:
    """Plot WER comparison bar chart across models/experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [m["label"] for m in metrics_list]
    wers = [m["corpus_wer"] for m in metrics_list]
    cers = [m["corpus_cer"] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, wers, width, label="WER", color="#4A90D9")
    bars2 = ax.bar(x + width/2, cers, width, label="CER", color="#E8636F")

    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("WER / CER Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"WER comparison plot saved to {save_path}")


def run_full_analysis(
    predictions_csv: str,
    output_dir: str = "outputs/q1",
) -> None:
    """
    Run complete error analysis pipeline.

    Args:
        predictions_csv: Path to predictions CSV.
        output_dir: Directory to save analysis outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_predictions(predictions_csv)
    references = df["reference"].tolist()
    hypotheses = df["prediction"].tolist()

    logger.info(f"Running error analysis on {len(references)} utterances...")

    # 1. Substitution analysis
    sub_df = analyze_substitutions(references, hypotheses, top_n=20)
    sub_df.to_csv(os.path.join(output_dir, "top_substitutions.csv"), index=False)
    logger.info(f"Top 20 substitution pairs:\n{sub_df.to_string()}")

    # 2. Deletion analysis
    del_df = analyze_deletions(references, hypotheses, top_n=20)
    del_df.to_csv(os.path.join(output_dir, "top_deletions.csv"), index=False)
    logger.info(f"Top 20 deleted words:\n{del_df.to_string()}")

    # 3. Length vs WER
    length_wer_df = compute_length_vs_wer(references, hypotheses)
    length_wer_df.to_csv(os.path.join(output_dir, "length_vs_wer.csv"), index=False)
    plot_length_vs_wer(length_wer_df, os.path.join(output_dir, "length_vs_wer.png"))

    # 4. Hardest utterances
    hardest_df = find_hardest_utterances(references, hypotheses, top_n=10)
    hardest_df.to_csv(os.path.join(output_dir, "hardest_utterances.csv"), index=False)
    logger.info(f"Hardest 10 utterances:\n{hardest_df[['index', 'wer', 'ref_length']].to_string()}")

    # 5. Combined error analysis CSV
    combined = pd.concat([
        sub_df.assign(analysis_type="substitution"),
        del_df.assign(analysis_type="deletion"),
    ], ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    logger.info(f"Full error analysis saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Error Analysis")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--output-dir", default="outputs/q1", help="Output directory")
    args = parser.parse_args()

    run_full_analysis(args.predictions, args.output_dir)

"""
Disfluency Statistics & Visualization — JoshTalks ASR Research
================================================================
Generate distribution charts, frequency tables, and summary stats
for detected disfluencies.
"""

import os
import sys
from collections import Counter
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("disfluency_stats", log_to_file=True)


def compute_disfluency_stats(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for disfluency detections.

    Args:
        df: Disfluency report DataFrame.

    Returns:
        Dictionary of summary statistics.
    """
    stats = {
        "total_disfluencies": len(df),
        "unique_recordings": df["recording_id"].nunique(),
        "unique_speakers": df["speaker_id"].nunique() if "speaker_id" in df.columns else 0,
    }

    # Per type
    type_counts = df["type"].value_counts().to_dict()
    stats["per_type"] = type_counts

    # Duration stats (if segment timestamps available)
    if "segment_start" in df.columns and "segment_end" in df.columns:
        df_with_dur = df.copy()
        df_with_dur["duration"] = df_with_dur["segment_end"] - df_with_dur["segment_start"]
        df_with_dur = df_with_dur[df_with_dur["duration"] > 0]

        if len(df_with_dur) > 0:
            stats["avg_duration_sec"] = round(df_with_dur["duration"].mean(), 3)
            stats["per_type_avg_duration"] = (
                df_with_dur.groupby("type")["duration"]
                .mean()
                .round(3)
                .to_dict()
            )

    # Per speaker frequency
    if "speaker_id" in df.columns:
        stats["per_speaker_count"] = df["speaker_id"].value_counts().to_dict()

    return stats


def plot_disfluency_distribution(
    df: pd.DataFrame,
    save_path: str = "outputs/q2/disfluency_distribution.png",
) -> None:
    """
    Plot pie chart of disfluency type distribution.

    Args:
        df: Disfluency report DataFrame.
        save_path: Path to save plot.
    """
    type_counts = df["type"].value_counts()

    colors = ["#4A90D9", "#E8636F", "#F5A623", "#7ED321", "#BD10E0", "#50E3C2"]
    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        colors=colors[:len(type_counts)],
        startangle=90,
        textprops={"fontsize": 12},
        pctdistance=0.85,
    )

    # Style
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Disfluency Type Distribution", fontsize=16, fontweight="bold", pad=20)

    # Add count labels
    legend_labels = [f"{t}: {c}" for t, c in zip(type_counts.index, type_counts.values)]
    ax.legend(legend_labels, loc="lower right", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Disfluency distribution plot saved to {save_path}")


def plot_duration_per_type(
    df: pd.DataFrame,
    save_path: str = "outputs/q2/duration_per_type.png",
) -> None:
    """
    Plot average duration per disfluency type.

    Args:
        df: Disfluency report DataFrame with segment_start, segment_end.
        save_path: Path to save plot.
    """
    df_dur = df.copy()
    df_dur["duration"] = df_dur["segment_end"] - df_dur["segment_start"]
    df_dur = df_dur[df_dur["duration"] > 0]

    if len(df_dur) == 0:
        logger.warning("No duration data available for plotting.")
        return

    avg_dur = df_dur.groupby("type")["duration"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4A90D9", "#E8636F", "#F5A623", "#7ED321"]
    bars = ax.bar(avg_dur.index, avg_dur.values, color=colors[:len(avg_dur)], edgecolor="white")

    ax.set_xlabel("Disfluency Type", fontsize=12)
    ax.set_ylabel("Average Duration (seconds)", fontsize=12)
    ax.set_title("Average Duration per Disfluency Type", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, avg_dur.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}s", ha="center", fontweight="bold", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Duration per type plot saved to {save_path}")


def plot_speaker_frequency(
    df: pd.DataFrame,
    save_path: str = "outputs/q2/speaker_frequency.png",
    top_n: int = 15,
) -> None:
    """
    Plot disfluency frequency per speaker.

    Args:
        df: Disfluency report DataFrame.
        save_path: Path to save plot.
        top_n: Number of top speakers to show.
    """
    if "speaker_id" not in df.columns:
        logger.warning("No speaker_id column for frequency plot.")
        return

    speaker_counts = df["speaker_id"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        range(len(speaker_counts)),
        speaker_counts.values,
        color="#4A90D9",
        edgecolor="white",
    )
    ax.set_yticks(range(len(speaker_counts)))
    ax.set_yticklabels(speaker_counts.index, fontsize=10)
    ax.set_xlabel("Disfluency Count", fontsize=12)
    ax.set_title(f"Top {top_n} Speakers by Disfluency Count", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Speaker frequency plot saved to {save_path}")


def generate_full_report(
    report_csv: str,
    output_dir: str = "outputs/q2",
) -> None:
    """
    Generate all disfluency statistics and visualizations.

    Args:
        report_csv: Path to disfluency report CSV.
        output_dir: Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(report_csv)

    if len(df) == 0:
        logger.warning("Empty disfluency report. No stats to generate.")
        return

    # Compute stats
    stats = compute_disfluency_stats(df)
    logger.info(f"Disfluency Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")

    # Save stats
    import json
    with open(os.path.join(output_dir, "disfluency_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Generate plots
    plot_disfluency_distribution(df, os.path.join(output_dir, "disfluency_distribution.png"))
    plot_duration_per_type(df, os.path.join(output_dir, "duration_per_type.png"))
    plot_speaker_frequency(df, os.path.join(output_dir, "speaker_frequency.png"))

    # Summary table
    summary = df.groupby("type").agg(
        count=("type", "size"),
        unique_recordings=("recording_id", "nunique"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "disfluency_summary.csv"), index=False)

    logger.info(f"\nFull report generated in {output_dir}/")
    print(f"\nDisfluency Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disfluency Statistics")
    parser.add_argument("--report", required=True, help="Disfluency report CSV")
    parser.add_argument("--output-dir", default="outputs/q2", help="Output directory")
    args = parser.parse_args()

    generate_full_report(args.report, args.output_dir)

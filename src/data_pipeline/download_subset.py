"""
Download Subset — JoshTalks ASR Research
==========================================
Streaming download of Josh Talks Hindi ASR dataset.
Selects a configurable subset for training (~2-3 hours).
Memory-efficient: processes one sample at a time.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import setup_logger
from src.data_pipeline.fix_urls import build_urls, validate_url

logger = setup_logger("download_subset", log_to_file=True)


def download_file(url: str, save_path: str, timeout: int = 30) -> bool:
    """
    Download a file from URL with error handling.

    Args:
        url: Source URL.
        save_path: Local path to save.
        timeout: Request timeout.

    Returns:
        True if download successful.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.warning(f"Download failed for {url}: {e}")
        return False


def download_transcription(url: str, timeout: int = 30) -> Optional[dict]:
    """
    Download and parse a transcription JSON file.

    Args:
        url: Transcription JSON URL.
        timeout: Request timeout.

    Returns:
        Parsed JSON dict or None on failure.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning(f"Transcription download failed for {url}: {e}")
        return None


def load_dataset_index(index_path: str) -> pd.DataFrame:
    """
    Load the dataset index file (CSV or JSON).

    Args:
        index_path: Path to dataset index file.

    Returns:
        DataFrame with dataset records.
    """
    path = Path(index_path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported index file format: {path.suffix}")


def select_subset(
    df: pd.DataFrame,
    max_hours: float = 3.0,
    max_samples: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select a subset of data within duration and sample constraints.

    Args:
        df: Full dataset DataFrame.
        max_hours: Maximum total hours for subset.
        max_samples: Maximum number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Subset DataFrame.
    """
    # Shuffle reproducibly
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    max_seconds = max_hours * 3600
    cumulative_duration = 0
    selected_indices = []

    for idx, row in df_shuffled.iterrows():
        duration = float(row.get("duration", 0))
        if cumulative_duration + duration > max_seconds:
            continue
        if len(selected_indices) >= max_samples:
            break
        selected_indices.append(idx)
        cumulative_duration += duration

    subset = df_shuffled.loc[selected_indices].reset_index(drop=True)
    logger.info(
        f"Selected subset: {len(subset)} samples, "
        f"{cumulative_duration / 3600:.2f} hours"
    )
    return subset


def download_subset(
    dataset_df: pd.DataFrame,
    output_dir: str,
    download_audio: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Download audio and transcriptions for a dataset subset.

    Args:
        dataset_df: DataFrame with user_id, recording_id columns.
        output_dir: Directory to save downloaded files.
        download_audio: Whether to download audio files.

    Returns:
        Tuple of (updated DataFrame with local paths, summary stats).
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    trans_dir = output_path / "transcriptions"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)

    local_audio_paths = []
    transcription_texts = []
    download_success = []

    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Downloading"):
        user_id = str(row["user_id"])
        recording_id = str(row["recording_id"])
        urls = build_urls(user_id, recording_id)

        # Download audio
        audio_path = str(audio_dir / f"{recording_id}.wav")
        audio_ok = False
        if download_audio:
            audio_ok = download_file(urls["recording"], audio_path)

        # Download transcription
        trans_data = download_transcription(urls["transcription"])
        trans_text = ""
        if trans_data:
            # Extract text from transcription JSON
            # Handle both flat and nested formats
            if isinstance(trans_data, dict):
                if "transcription" in trans_data:
                    trans_text = trans_data["transcription"]
                elif "text" in trans_data:
                    trans_text = trans_data["text"]
                elif "segments" in trans_data:
                    # Concatenate segment texts
                    segments = trans_data["segments"]
                    if isinstance(segments, list):
                        trans_text = " ".join(
                            seg.get("text", "") for seg in segments if isinstance(seg, dict)
                        )

            # Save transcription
            trans_path = str(trans_dir / f"{recording_id}_transcription.json")
            with open(trans_path, "w", encoding="utf-8") as f:
                json.dump(trans_data, f, ensure_ascii=False, indent=2)

        local_audio_paths.append(audio_path if audio_ok else "")
        transcription_texts.append(trans_text)
        download_success.append(audio_ok)

    dataset_df = dataset_df.copy()
    dataset_df["local_audio_path"] = local_audio_paths
    dataset_df["transcription_text"] = transcription_texts
    dataset_df["download_success"] = download_success

    # Summary stats
    successful = dataset_df[dataset_df["download_success"]]
    durations = successful["duration"].astype(float).values
    summary = {
        "total_samples": len(dataset_df),
        "successful_downloads": int(successful.shape[0]),
        "failed_downloads": int((~dataset_df["download_success"]).sum()),
        "total_hours": round(float(durations.sum()) / 3600, 2),
        "mean_duration_sec": round(float(durations.mean()), 2) if len(durations) > 0 else 0,
        "median_duration_sec": round(float(np.median(durations)), 2) if len(durations) > 0 else 0,
        "min_duration_sec": round(float(durations.min()), 2) if len(durations) > 0 else 0,
        "max_duration_sec": round(float(durations.max()), 2) if len(durations) > 0 else 0,
    }

    logger.info(f"Download summary: {json.dumps(summary, indent=2)}")
    return dataset_df, summary


def plot_duration_histogram(
    durations: np.ndarray,
    save_path: str = "outputs/q1/duration_histogram.png",
) -> None:
    """
    Plot and save a duration distribution histogram.

    Args:
        durations: Array of audio durations in seconds.
        save_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(durations, bins=30, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Audio Duration Distribution", fontsize=14, fontweight="bold")
    ax.axvline(np.mean(durations), color="red", linestyle="--", label=f"Mean: {np.mean(durations):.1f}s")
    ax.axvline(np.median(durations), color="orange", linestyle="--", label=f"Median: {np.median(durations):.1f}s")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Duration histogram saved to {save_path}")


if __name__ == "__main__":
    config = load_config("configs/preprocessing.yaml")

    # Check if dataset index exists
    index_candidates = [
        "data/raw/dataset_index.csv",
        "data/raw/dataset_index.json",
        "data/raw/dataset.csv",
        "data/raw/dataset.json",
    ]

    index_path = None
    for candidate in index_candidates:
        if os.path.exists(candidate):
            index_path = candidate
            break

    if index_path is None:
        logger.error(
            "Dataset index file not found. Please place your dataset CSV/JSON in data/raw/. "
            "Expected columns: user_id, recording_id, language, duration, rec_url_gcp, "
            "transcription_url, metadata_url"
        )
        sys.exit(1)

    # Load and select subset
    df = load_dataset_index(index_path)
    logger.info(f"Loaded dataset index: {len(df)} records")

    subset = select_subset(
        df,
        max_hours=config["dataset"]["subset_max_hours"],
        max_samples=config["dataset"]["subset_max_samples"],
        seed=config["seed"],
    )

    # Download subset
    subset_df, summary = download_subset(subset, "data/subsets")

    # Save subset metadata
    subset_df.to_csv("data/subsets/subset_metadata.csv", index=False)

    # Save summary
    with open("data/subsets/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot duration histogram
    durations = subset_df[subset_df["download_success"]]["duration"].astype(float).values
    if len(durations) > 0:
        plot_duration_histogram(durations, "outputs/q1/duration_histogram.png")

    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    for key, value in summary.items():
        print(f"  {key}: {value}")

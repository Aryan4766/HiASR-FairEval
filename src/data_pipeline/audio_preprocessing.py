"""
Audio Preprocessing — JoshTalks ASR Research
==============================================
Convert audio files to 16kHz mono WAV format.
Memory-efficient: processes one file at a time.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("audio_preprocessing", log_to_file=True)


def preprocess_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Convert a single audio file to target format.

    Args:
        input_path: Path to input audio file.
        output_path: Path to save processed file.
        target_sr: Target sample rate in Hz.
        mono: Whether to convert to mono.
        max_duration: Maximum duration in seconds (clip if longer).
        min_duration: Minimum duration in seconds (skip if shorter).

    Returns:
        Tuple of (success, duration_seconds).
    """
    try:
        # Load audio with librosa (automatically resamples)
        audio, sr = librosa.load(
            input_path,
            sr=target_sr,
            mono=mono,
        )

        duration = len(audio) / sr

        # Skip too-short clips
        if min_duration and duration < min_duration:
            logger.debug(f"Skipping {input_path}: duration {duration:.2f}s < {min_duration}s")
            return False, duration

        # Clip too-long audio
        if max_duration and duration > max_duration:
            max_samples = int(max_duration * sr)
            audio = audio[:max_samples]
            duration = max_duration

        # Normalize amplitude to prevent clipping
        max_amp = np.abs(audio).max()
        if max_amp > 0:
            audio = audio / max_amp * 0.95

        # Save as WAV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, target_sr)

        return True, duration

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False, 0.0


def batch_preprocess(
    input_dir: str,
    output_dir: str,
    config: dict,
) -> dict:
    """
    Batch preprocess all audio files in a directory.

    Args:
        input_dir: Directory with raw audio files.
        output_dir: Directory for processed files.
        config: Audio preprocessing configuration.

    Returns:
        Summary statistics dictionary.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_config = config.get("audio", {})
    target_sr = audio_config.get("sample_rate", 16000)
    mono = audio_config.get("mono", True)
    max_dur = audio_config.get("max_duration_sec")
    min_dur = audio_config.get("min_duration_sec")

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    audio_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    logger.info(f"Processing {len(audio_files)} audio files...")

    processed = 0
    skipped = 0
    failed = 0
    total_duration = 0.0

    for audio_file in tqdm(audio_files, desc="Preprocessing audio"):
        out_file = output_path / f"{audio_file.stem}.wav"
        success, duration = preprocess_audio(
            str(audio_file),
            str(out_file),
            target_sr=target_sr,
            mono=mono,
            max_duration=max_dur,
            min_duration=min_dur,
        )
        if success:
            processed += 1
            total_duration += duration
        elif duration > 0:
            skipped += 1
        else:
            failed += 1

    summary = {
        "total": len(audio_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_duration_hours": round(total_duration / 3600, 2),
    }

    logger.info(f"Audio preprocessing complete: {summary}")
    return summary


if __name__ == "__main__":
    config = load_config("configs/preprocessing.yaml")

    summary = batch_preprocess(
        input_dir=config["paths"]["raw_data"],
        output_dir=config["paths"]["processed_data"],
        config=config,
    )

    print(f"\nAudio Preprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

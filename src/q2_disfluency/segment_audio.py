"""
Audio Segmentation — JoshTalks ASR Research
=============================================
Clip audio segments containing disfluencies.
Uses timestamps from transcription segments.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("segment_audio", log_to_file=True)


def clip_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    padding_sec: float = 0.2,
    normalize_audio: bool = True,
) -> bool:
    """
    Extract an audio segment and save as a clip.

    Args:
        audio_path: Path to full recording audio file.
        start_sec: Segment start time in seconds.
        end_sec: Segment end time in seconds.
        output_path: Path to save the clipped audio.
        padding_sec: Padding to add before/after the clip (seconds).
        normalize_audio: Whether to normalize clip amplitude.

    Returns:
        True if clip was successfully created.
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Convert to milliseconds with padding
        start_ms = max(0, int((start_sec - padding_sec) * 1000))
        end_ms = min(len(audio), int((end_sec + padding_sec) * 1000))

        # Validate segment
        if start_ms >= end_ms:
            logger.warning(f"Invalid segment: start={start_ms}ms >= end={end_ms}ms for {audio_path}")
            return False

        # Extract clip
        clip = audio[start_ms:end_ms]

        # Normalize amplitude
        if normalize_audio:
            clip = pydub_normalize(clip)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export as WAV
        clip.export(output_path, format="wav")
        return True

    except Exception as e:
        logger.error(f"Failed to clip {audio_path} [{start_sec}-{end_sec}]: {e}")
        return False


def batch_segment_disfluencies(
    disfluency_df: pd.DataFrame,
    audio_dir: str,
    output_dir: str = "outputs/q2/clips",
    audio_ext: str = ".wav",
) -> pd.DataFrame:
    """
    Clip audio for all detected disfluencies.

    Args:
        disfluency_df: DataFrame with disfluency records (from detect_rules).
            Must have: recording_id, segment_start, segment_end.
        audio_dir: Directory containing full recording audio files.
        output_dir: Directory to save clips.
        audio_ext: Audio file extension.

    Returns:
        Updated DataFrame with clip_path column populated.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(disfluency_df.iterrows(), total=len(disfluency_df), desc="Clipping audio"):
        recording_id = str(row["recording_id"])
        audio_path = os.path.join(audio_dir, f"{recording_id}{audio_ext}")

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            clip_paths.append("")
            fail_count += 1
            continue

        start = float(row.get("segment_start", 0))
        end = float(row.get("segment_end", 0))

        if start == 0 and end == 0:
            logger.warning(f"No timestamps for {recording_id} segment {row.get('segment_id', idx)}")
            clip_paths.append("")
            fail_count += 1
            continue

        # Build clip filename
        clip_name = f"{recording_id}_seg{row.get('segment_id', idx)}_{row['type']}.wav"
        clip_path = str(output_path / clip_name)

        success = clip_audio_segment(audio_path, start, end, clip_path)

        if success:
            clip_paths.append(clip_path)
            success_count += 1
        else:
            clip_paths.append("")
            fail_count += 1

    disfluency_df = disfluency_df.copy()
    disfluency_df["clip_path"] = clip_paths

    logger.info(f"Clipping complete: {success_count} success, {fail_count} failed")
    return disfluency_df


if __name__ == "__main__":
    # Demo: clip from a test audio file
    import argparse

    parser = argparse.ArgumentParser(description="Segment audio for disfluencies")
    parser.add_argument("--report", required=True, help="Disfluency report CSV path")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument("--output-dir", default="outputs/q2/clips", help="Output clips directory")
    args = parser.parse_args()

    df = pd.read_csv(args.report)
    updated_df = batch_segment_disfluencies(df, args.audio_dir, args.output_dir)
    updated_df.to_csv(args.report.replace(".csv", "_with_clips.csv"), index=False)
    print(f"Updated report saved with clip paths.")

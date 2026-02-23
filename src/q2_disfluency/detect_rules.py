"""
Disfluency Detection — JoshTalks ASR Research
================================================
Rule-based detection of speech disfluencies from transcriptions:
- Fillers ("uh", "umm", "अं", "हं", etc.)
- Consecutive word repetition
- Character elongation (≥3 same chars)
- False starts / hesitations
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("disfluency_detect", log_to_file=True)

# ---- Filler Patterns ----
# Hindi fillers
HINDI_FILLERS = [
    "अं", "हं", "उं", "एं", "ऐं", "ऊं",
    "अम्म", "हम्म", "उम्म",
    "मतलब", "बोले तो", "ये",
    "अच्छा", "हां", "ना",
    "वो", "तो", "और",
]

# English fillers (may appear in Hindi transcriptions)
ENGLISH_FILLERS = [
    "uh", "uhh", "uhhh",
    "um", "umm", "ummm",
    "hmm", "hmmm",
    "ah", "ahh",
    "er", "err",
    "like", "you know",
    "so", "okay", "ok",
]

# Build regex patterns
FILLER_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(f) for f in HINDI_FILLERS + ENGLISH_FILLERS) + r')\b',
    re.IGNORECASE | re.UNICODE
)

# Elongation pattern: same character repeated ≥3 times
ELONGATION_PATTERN = re.compile(r'(.)\1{2,}', re.UNICODE)

# Hesitation markers (common in transcriptions)
HESITATION_PATTERN = re.compile(r'[\.\-]{2,}|…', re.UNICODE)


def detect_fillers(text: str) -> List[Dict]:
    """
    Detect filler words in text.

    Args:
        text: Transcription text.

    Returns:
        List of disfluency records with type, text, position.
    """
    disfluencies = []
    for match in FILLER_PATTERN.finditer(text):
        disfluencies.append({
            "type": "filler",
            "text": match.group(),
            "start_char": match.start(),
            "end_char": match.end(),
        })
    return disfluencies


def detect_repetitions(text: str) -> List[Dict]:
    """
    Detect consecutive word repetitions.

    Looks for patterns like "I I was" or "the the cat".

    Args:
        text: Transcription text.

    Returns:
        List of repetition disfluency records.
    """
    disfluencies = []
    words = text.split()

    i = 0
    while i < len(words) - 1:
        # Check for 1-word repetition
        if words[i].lower() == words[i + 1].lower():
            rep_count = 1
            j = i + 1
            while j < len(words) and words[j].lower() == words[i].lower():
                rep_count += 1
                j += 1

            repeated_text = " ".join(words[i:j])
            # Find position in original text
            start_pos = text.find(repeated_text)

            disfluencies.append({
                "type": "repetition",
                "text": repeated_text,
                "start_char": start_pos if start_pos >= 0 else 0,
                "end_char": (start_pos + len(repeated_text)) if start_pos >= 0 else len(repeated_text),
                "repeat_count": rep_count,
            })
            i = j
        else:
            # Check 2-word repetition: "I was I was"
            if i < len(words) - 3:
                bigram1 = f"{words[i]} {words[i+1]}".lower()
                bigram2 = f"{words[i+2]} {words[i+3]}".lower() if i + 3 < len(words) else ""
                if bigram1 == bigram2:
                    rep_text = " ".join(words[i:i+4])
                    start_pos = text.find(rep_text)
                    disfluencies.append({
                        "type": "repetition",
                        "text": rep_text,
                        "start_char": start_pos if start_pos >= 0 else 0,
                        "end_char": (start_pos + len(rep_text)) if start_pos >= 0 else len(rep_text),
                        "repeat_count": 2,
                    })
                    i += 4
                    continue
            i += 1

    return disfluencies


def detect_elongations(text: str) -> List[Dict]:
    """
    Detect character elongation patterns (≥3 same consecutive chars).

    Examples: "soooo", "हांांां", "यार्ार्ार्"

    Args:
        text: Transcription text.

    Returns:
        List of elongation disfluency records.
    """
    disfluencies = []
    for match in ELONGATION_PATTERN.finditer(text):
        # Find the word containing this elongation
        word_start = text.rfind(' ', 0, match.start()) + 1
        word_end = text.find(' ', match.end())
        if word_end == -1:
            word_end = len(text)

        word = text[word_start:word_end]
        disfluencies.append({
            "type": "elongation",
            "text": word,
            "start_char": word_start,
            "end_char": word_end,
            "elongated_char": match.group(1),
            "repeat_length": len(match.group()),
        })
    return disfluencies


def detect_hesitations(text: str) -> List[Dict]:
    """
    Detect hesitation markers (ellipsis, dashes).

    Args:
        text: Transcription text.

    Returns:
        List of hesitation disfluency records.
    """
    disfluencies = []
    for match in HESITATION_PATTERN.finditer(text):
        disfluencies.append({
            "type": "hesitation",
            "text": match.group(),
            "start_char": match.start(),
            "end_char": match.end(),
        })
    return disfluencies


def detect_all_disfluencies(
    text: str,
    recording_id: str = "",
    speaker_id: str = "",
    segment_id: str = "",
    segment_start: float = 0.0,
    segment_end: float = 0.0,
) -> List[Dict]:
    """
    Run all disfluency detection rules on a text segment.

    Args:
        text: Transcription text for this segment.
        recording_id: Source recording identifier.
        speaker_id: Speaker/user identifier.
        segment_id: Segment identifier within recording.
        segment_start: Segment start time in seconds.
        segment_end: Segment end time in seconds.

    Returns:
        List of disfluency records with full metadata.
    """
    all_disfluencies = []

    detectors = [
        detect_fillers,
        detect_repetitions,
        detect_elongations,
        detect_hesitations,
    ]

    for detector in detectors:
        results = detector(text)
        for r in results:
            r["recording_id"] = recording_id
            r["speaker_id"] = speaker_id
            r["segment_id"] = segment_id
            r["segment_start"] = segment_start
            r["segment_end"] = segment_end
            all_disfluencies.append(r)

    return all_disfluencies


def process_transcription_file(
    transcription_data: dict,
    recording_id: str,
    speaker_id: str = "",
) -> List[Dict]:
    """
    Process a full transcription JSON to detect disfluencies.

    Handles both flat and segmented transcription formats.

    Args:
        transcription_data: Parsed transcription JSON.
        recording_id: Recording identifier.
        speaker_id: Speaker identifier.

    Returns:
        List of all detected disfluencies across segments.
    """
    all_disfluencies = []

    if "segments" in transcription_data:
        # Segmented format
        for i, segment in enumerate(transcription_data["segments"]):
            text = segment.get("text", "")
            start = segment.get("start", segment.get("start_time", 0.0))
            end = segment.get("end", segment.get("end_time", 0.0))
            segment_id = segment.get("id", str(i))

            disfluencies = detect_all_disfluencies(
                text=text,
                recording_id=recording_id,
                speaker_id=speaker_id,
                segment_id=segment_id,
                segment_start=float(start),
                segment_end=float(end),
            )
            all_disfluencies.extend(disfluencies)

    elif "transcription" in transcription_data:
        # Flat format — treat entire text as one segment
        text = transcription_data["transcription"]
        disfluencies = detect_all_disfluencies(
            text=text,
            recording_id=recording_id,
            speaker_id=speaker_id,
            segment_id="0",
        )
        all_disfluencies.extend(disfluencies)

    elif "text" in transcription_data:
        text = transcription_data["text"]
        disfluencies = detect_all_disfluencies(
            text=text,
            recording_id=recording_id,
            speaker_id=speaker_id,
            segment_id="0",
        )
        all_disfluencies.extend(disfluencies)

    return all_disfluencies


def save_disfluency_report(
    disfluencies: List[Dict],
    output_path: str = "outputs/q2/disfluency_report.csv",
) -> pd.DataFrame:
    """
    Save disfluency detections as structured CSV.

    Output schema:
        recording_id | speaker_id | segment_id | type | text |
        segment_start | segment_end | clip_path

    Args:
        disfluencies: List of disfluency records.
        output_path: Path to save CSV.

    Returns:
        DataFrame of disfluency report.
    """
    if not disfluencies:
        logger.warning("No disfluencies detected.")
        df = pd.DataFrame(columns=[
            "recording_id", "speaker_id", "segment_id", "type",
            "text", "segment_start", "segment_end", "clip_path",
        ])
        df.to_csv(output_path, index=False)
        return df

    df = pd.DataFrame(disfluencies)

    # Select and order columns
    columns = [
        "recording_id", "speaker_id", "segment_id", "type",
        "text", "segment_start", "segment_end",
    ]
    # Keep only existing columns
    columns = [c for c in columns if c in df.columns]
    df = df[columns]

    # Add clip_path placeholder
    df["clip_path"] = ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Disfluency report saved: {output_path} ({len(df)} detections)")

    return df


if __name__ == "__main__":
    # Demo with sample text
    test_texts = [
        ("rec_001", "अं मैं ये बोल रहा था कि कि वो sooo अच्छा था"),
        ("rec_002", "umm हां हां मतलब ये ये बात है..."),
        ("rec_003", "I was I was saying that uhh it's okaaay"),
    ]

    all_results = []
    for rec_id, text in test_texts:
        results = detect_all_disfluencies(
            text=text, recording_id=rec_id, speaker_id="test_speaker"
        )
        all_results.extend(results)
        print(f"\n{rec_id}: '{text}'")
        for r in results:
            print(f"  [{r['type']}] '{r['text']}'")

    df = save_disfluency_report(all_results)
    print(f"\nTotal disfluencies detected: {len(df)}")
    print(df.to_string(index=False))

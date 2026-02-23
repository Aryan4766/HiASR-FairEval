"""
Extract Unique Words — JoshTalks ASR Research
================================================
Extract and normalize unique words from transcription data.
Handles the provided ~1,77,000 unique word list.
"""

import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("extract_words", log_to_file=True)


def load_word_list(file_path: str) -> List[str]:
    """
    Load word list from file (CSV, TXT, or JSON).

    Args:
        file_path: Path to word list file.

    Returns:
        List of words.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        # Try common column names
        for col in ["word", "words", "text", df.columns[0]]:
            if col in df.columns:
                words = df[col].dropna().astype(str).tolist()
                break
        else:
            words = df.iloc[:, 0].dropna().astype(str).tolist()

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

    elif ext == ".json":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            words = [str(w) for w in data]
        elif isinstance(data, dict):
            words = list(data.keys())
        else:
            words = [str(data)]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info(f"Loaded {len(words)} words from {file_path}")
    return words


def normalize_word(word: str) -> str:
    """
    Normalize a single word: Unicode NFKC, strip punctuation, remove extra whitespace.

    Args:
        word: Input word.

    Returns:
        Normalized word with leading/trailing punctuation removed.
    """
    if not word or not isinstance(word, str):
        return ""

    # Unicode normalization
    word = unicodedata.normalize("NFKC", word)

    # Remove zero-width characters
    word = word.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')

    # Strip whitespace
    word = word.strip()

    # Strip leading/trailing punctuation (commas, quotes, periods, etc.)
    # Keep Devanagari, Latin alphanumeric, and Hindi numerals
    word = re.sub(r'^[^\w\u0900-\u097F\u0966-\u096F]+', '', word)
    word = re.sub(r'[^\w\u0900-\u097F\u0966-\u096F]+$', '', word)

    # Also strip Devanagari danda (।) and double danda (॥) from edges
    word = word.strip('।॥')

    return word


def extract_unique_words(
    words: List[str],
    normalize: bool = True,
) -> Tuple[List[str], Counter]:
    """
    Extract unique words with frequency counts.

    Args:
        words: Raw word list.
        normalize: Whether to apply Unicode normalization.

    Returns:
        Tuple of (unique_words sorted list, frequency counter).
    """
    if normalize:
        words = [normalize_word(w) for w in words]

    # Remove empty strings
    words = [w for w in words if w]

    # Count frequencies
    freq = Counter(words)

    # Unique words sorted by frequency
    unique_words = sorted(freq.keys())

    logger.info(f"Total words: {len(words)}")
    logger.info(f"Unique words: {len(unique_words)}")

    return unique_words, freq


def is_devanagari(text: str) -> bool:
    """
    Check if text is primarily in Devanagari script.

    Args:
        text: Input text.

    Returns:
        True if >50% of characters are Devanagari.
    """
    if not text:
        return False

    devanagari_count = sum(
        1 for c in text if '\u0900' <= c <= '\u097F' or '\uA8E0' <= c <= '\uA8FF'
    )
    return devanagari_count > len(text) * 0.5


def generate_word_stats(
    unique_words: List[str],
    freq: Counter,
    output_dir: str = "outputs/q3",
) -> Dict:
    """
    Generate statistics about the word list.

    Args:
        unique_words: List of unique words.
        freq: Word frequency counter.
        output_dir: Output directory.

    Returns:
        Statistics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    devanagari_words = [w for w in unique_words if is_devanagari(w)]
    latin_words = [w for w in unique_words if w.isascii()]
    mixed_words = [w for w in unique_words if not is_devanagari(w) and not w.isascii()]

    stats = {
        "total_unique_words": len(unique_words),
        "devanagari_words": len(devanagari_words),
        "latin_words": len(latin_words),
        "mixed_script_words": len(mixed_words),
        "avg_word_length": round(sum(len(w) for w in unique_words) / max(len(unique_words), 1), 2),
        "top_20_frequent": freq.most_common(20),
    }

    # Save frequency list
    freq_df = pd.DataFrame(
        freq.most_common(),
        columns=["word", "frequency"],
    )
    freq_df.to_csv(os.path.join(output_dir, "word_frequencies.csv"), index=False)

    logger.info(f"Word Statistics:")
    for key, value in stats.items():
        if key != "top_20_frequent":
            logger.info(f"  {key}: {value}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract unique words")
    parser.add_argument("--input", required=True, help="Path to word list file")
    parser.add_argument("--output-dir", default="outputs/q3", help="Output directory")
    args = parser.parse_args()

    words = load_word_list(args.input)
    unique_words, freq = extract_unique_words(words)
    stats = generate_word_stats(unique_words, freq, args.output_dir)

    print(f"\nWord Extraction Summary:")
    for key, value in stats.items():
        if key != "top_20_frequent":
            print(f"  {key}: {value}")

"""
Text Normalization — JoshTalks ASR Research
=============================================
Unicode normalization, punctuation handling, and whitespace cleanup.
Designed for Hindi/Devanagari text with mixed-script support.
"""

import os
import re
import sys
import unicodedata
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("text_normalization", log_to_file=False)

# Hindi/Devanagari punctuation and special characters
HINDI_PUNCTUATION = re.compile(r'[।॥,\.\?\!;:\-\—\–\(\)\[\]\{\}\"\'\'\'\"\"…\u200b\u200c\u200d\ufeff]')

# Multiple whitespace
MULTI_SPACE = re.compile(r'\s+')

# Numeric characters (optional removal)
NUMBERS = re.compile(r'[0-9०-९]')


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """
    Apply Unicode normalization.

    Args:
        text: Input text.
        form: Unicode normalization form (NFC, NFKC, NFD, NFKD).

    Returns:
        Normalized text.
    """
    return unicodedata.normalize(form, text)


def remove_punctuation(text: str, keep_devanagari_danda: bool = False) -> str:
    """
    Remove punctuation while preserving Devanagari text.

    Args:
        text: Input text.
        keep_devanagari_danda: Whether to keep ।  (purna viram).

    Returns:
        Text without punctuation.
    """
    if keep_devanagari_danda:
        # Remove all except Devanagari danda
        text = re.sub(r'[^\w\s।]', '', text)
    else:
        text = HINDI_PUNCTUATION.sub(' ', text)
        # Also remove any remaining standard punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces, strip edges.

    Args:
        text: Input text.

    Returns:
        Whitespace-normalized text.
    """
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


def remove_numbers(text: str) -> str:
    """
    Remove numeric characters (both ASCII and Devanagari).

    Args:
        text: Input text.

    Returns:
        Text without numbers.
    """
    return NUMBERS.sub('', text)


def normalize_text(
    text: str,
    unicode_form: str = "NFKC",
    remove_punct: bool = True,
    remove_nums: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Full text normalization pipeline.

    Args:
        text: Input text.
        unicode_form: Unicode normalization form.
        remove_punct: Whether to remove punctuation.
        remove_nums: Whether to remove numbers.
        lowercase: Whether to lowercase (mainly for English words).

    Returns:
        Normalized text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Unicode normalization
    text = normalize_unicode(text, unicode_form)

    # Step 2: Remove zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')

    # Step 3: Punctuation removal
    if remove_punct:
        text = remove_punctuation(text)

    # Step 4: Number removal (optional)
    if remove_nums:
        text = remove_numbers(text)

    # Step 5: Lowercase (optional — Hindi doesn't have case)
    if lowercase:
        text = text.lower()

    # Step 6: Whitespace normalization (always last)
    text = normalize_whitespace(text)

    return text


def normalize_batch(
    texts: List[str],
    config: Optional[dict] = None,
) -> List[str]:
    """
    Normalize a batch of texts using config settings.

    Args:
        texts: List of text strings.
        config: Text normalization config (from preprocessing.yaml).

    Returns:
        List of normalized texts.
    """
    if config is None:
        config = {}

    text_cfg = config.get("text", {})
    unicode_form = text_cfg.get("unicode_normalization", "NFKC")
    remove_punct = text_cfg.get("remove_punctuation", True)
    lowercase = text_cfg.get("lowercase", False)

    normalized = []
    for text in texts:
        normalized.append(normalize_text(
            text,
            unicode_form=unicode_form,
            remove_punct=remove_punct,
            lowercase=lowercase,
        ))

    return normalized


if __name__ == "__main__":
    # Demo
    test_texts = [
        "  हैलो,  World!  ",
        "यह एक  test  है।  कंप्यूटर ",
        '  "हिंदी"   में   transcription   है  ',
        "soooo  बहुत  अच्छा!!!",
    ]

    print("Text Normalization Demo:")
    print("=" * 50)
    for text in test_texts:
        normalized = normalize_text(text)
        print(f"  Input:  '{text}'")
        print(f"  Output: '{normalized}'")
        print()

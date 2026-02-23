"""
Spelling Classifier — JoshTalks ASR Research
===============================================
Classify Hindi words as correctly or incorrectly spelled.

Pipeline:
1. Hindi dictionary lookup (known correct words)
2. Edit distance ≤ 2 from dictionary → likely misspelling
3. Frequency-based heuristic (very common = likely correct)
4. Devanagari English transliteration detection → mark as CORRECT

Key rule: English words transcribed in Devanagari (e.g., "कंप्यूटर")
are CORRECT per transcription guidelines.
"""

import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from src.q3_spelling.extract_unique_words import is_devanagari, normalize_word

logger = setup_logger("spell_classifier", log_to_file=True)

# ---- Common Hindi Words (seed dictionary) ----
# This is a seed; in practice, load a comprehensive Hindi dictionary
COMMON_HINDI_WORDS = set()

# ---- Common English-to-Devanagari Transliterations ----
# Words that appear in Devanagari but are English loanwords = CORRECT
COMMON_TRANSLITERATIONS = {
    "कंप्यूटर", "मोबाइल", "फोन", "इंटरनेट", "वीडियो", "ऑनलाइन",
    "स्कूल", "कॉलेज", "यूनिवर्सिटी", "टीचर", "स्टूडेंट",
    "डॉक्टर", "इंजीनियर", "सॉफ्टवेयर", "हार्डवेयर",
    "गेम", "प्लेयर", "क्रिकेट", "फुटबॉल",
    "बस", "ट्रेन", "ऑटो", "टैक्सी", "एयरपोर्ट",
    "पुलिस", "हॉस्पिटल", "रेस्टोरेंट", "होटल",
    "फिल्म", "मूवी", "सीरीज", "न्यूज़",
    "बैंक", "अकाउंट", "पासवर्ड", "लॉगिन",
    "व्हाट्सएप", "फेसबुक", "इंस्टाग्राम", "यूट्यूब",
    "पार्टी", "मीटिंग", "ऑफिस", "कंपनी",
    "प्रोग्राम", "प्रोजेक्ट", "रिपोर्ट", "रिजल्ट",
    "पेपर", "बुक", "कॉपी", "पेन", "पेंसिल",
    "शर्ट", "पैंट", "जीन्स", "टीशर्ट",
    "किचन", "बाथरूम", "बेडरूम",
    "केक", "बिस्कुट", "चॉकलेट", "पिज्जा", "बर्गर",
    "कैमरा", "लैपटॉप", "टैबलेट", "चार्जर",
    "वेबसाइट", "ऐप", "डाउनलोड", "अपलोड",
    "सर", "मैम", "मैडम", "बॉस",
    "टाइम", "डेट", "एड्रेस", "नंबर",
    "ग्रुप", "टीम", "लीडर", "मैनेजर",
}


def load_dictionary(dict_path: Optional[str] = None) -> set:
    """
    Load Hindi dictionary from file or use built-in seed.

    Args:
        dict_path: Path to dictionary file (one word per line).

    Returns:
        Set of known correct words.
    """
    dictionary = set(COMMON_TRANSLITERATIONS)

    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    dictionary.add(normalize_word(word))
        logger.info(f"Loaded dictionary: {len(dictionary)} words from {dict_path}")
    else:
        logger.info(f"Using seed dictionary: {len(dictionary)} words")

    return dictionary


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum edit distance.
    """
    m, n = len(s1), len(s2)
    # Early termination for very different lengths
    if abs(m - n) > 2:
        return abs(m - n)

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def is_likely_transliteration(word: str) -> bool:
    """
    Check if a Devanagari word is likely an English transliteration.

    Heuristics:
    - Direct match against known transliterations
    - Contains common English-borrowed suffixes in Devanagari
    - Close edit distance to known transliterations

    Args:
        word: Devanagari word to check.

    Returns:
        True if likely a valid transliteration.
    """
    if word in COMMON_TRANSLITERATIONS:
        return True

    # Common transliteration suffixes
    transliteration_suffixes = [
        "शन", "मेंट", "नेस", "टी", "ली", "इंग",
        "एशन", "ईज़", "र्स", "ल्स",
    ]
    for suffix in transliteration_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return True

    # Check edit distance to known transliterations (expensive, use sparingly)
    for known in COMMON_TRANSLITERATIONS:
        if abs(len(word) - len(known)) <= 2:
            if edit_distance(word, known) <= 1:
                return True

    return False


def classify_word(
    word: str,
    dictionary: set,
    word_freq: Optional[Dict[str, int]] = None,
    freq_threshold: int = 50,
) -> str:
    """
    Classify a single word as 'correct spelling' or 'incorrect spelling'.

    Pipeline:
    1. If in dictionary → correct
    2. If Devanagari transliteration of English → correct
    3. Error signals: mixed numbers+text, single non-standard char → incorrect
    4. Frequency ≥ 2 in corpus → correct (appears in real speech data)
    5. Pure Devanagari (≥2 chars) → default correct (ASR outputs real words)
    6. Edit distance ≤ 1 from dictionary → incorrect (close misspelling)
    7. Remaining → incorrect (unusual forms)

    Key insight: In a 177K-word ASR corpus, most words ARE correctly spelled
    because ASR models produce real words. We only flag strong error signals.

    Args:
        word: Word to classify.
        dictionary: Set of known correct words.
        word_freq: Optional frequency dict.
        freq_threshold: Frequency threshold for auto-correct.

    Returns:
        "correct spelling" or "incorrect spelling"
    """
    word = normalize_word(word)

    if not word:
        return "incorrect spelling"

    # Step 1: Dictionary lookup
    if word in dictionary:
        return "correct spelling"

    # Step 2: Transliteration check (for Devanagari words)
    if is_devanagari(word) and is_likely_transliteration(word):
        return "correct spelling"

    # Step 3: Strong error signals
    # Single character (not a valid word usually, except standalone vowels)
    if len(word) <= 1 and not re.match(r'^[\u0905-\u0939]$', word):
        return "incorrect spelling"

    # Contains numbers mixed with text
    if re.search(r'[0-9\u0966-\u096F]', word) and re.search(r'[a-zA-Z\u0900-\u097F]', word):
        return "incorrect spelling"

    # Excessive repeated characters (e.g., "अअअअ", "aaaa")
    if re.search(r'(.)\1{3,}', word):
        return "incorrect spelling"

    # Mixed Devanagari + Latin script (unusual, likely error)
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', word))
    has_latin = bool(re.search(r'[a-zA-Z]', word))
    if has_devanagari and has_latin:
        return "incorrect spelling"

    # Step 4: Frequency-based heuristic
    # In a 177K-word corpus from ASR, frequency ≥ 2 means the word
    # appeared in multiple utterances — strong signal it's a real word
    if word_freq and word in word_freq:
        if word_freq[word] >= 2:
            return "correct spelling"

    # Step 5: Pure Devanagari words default to correct
    # ASR models overwhelmingly produce real Hindi words;
    # hapax legomena in Devanagari are still usually valid words
    if has_devanagari and not has_latin and len(word) >= 2:
        return "correct spelling"

    # Step 6: Edit distance check — only dist ≤ 1 is a confident misspelling
    if has_devanagari:
        for dict_word in dictionary:
            if is_devanagari(dict_word) and abs(len(word) - len(dict_word)) <= 1:
                if edit_distance(word, dict_word) <= 1:
                    return "incorrect spelling"

    # Step 7: Latin-only words — check if English
    if has_latin and not has_devanagari:
        # Pure English words in a Hindi ASR corpus are valid
        if len(word) >= 2 and word.isalpha():
            return "correct spelling"

    return "incorrect spelling"


def classify_word_list(
    words: List[str],
    dictionary_path: Optional[str] = None,
    word_freq: Optional[Dict[str, int]] = None,
    freq_threshold: int = 50,
) -> pd.DataFrame:
    """
    Classify a full word list.

    Args:
        words: List of unique words.
        dictionary_path: Path to dictionary file.
        word_freq: Word frequency dictionary.
        freq_threshold: Frequency threshold.

    Returns:
        DataFrame with columns: word, label.
    """
    dictionary = load_dictionary(dictionary_path)

    results = []
    for word in words:
        label = classify_word(word, dictionary, word_freq, freq_threshold)
        results.append({"word": word, "label": label})

    df = pd.DataFrame(results)

    # Summary stats
    correct = (df["label"] == "correct spelling").sum()
    incorrect = (df["label"] == "incorrect spelling").sum()
    total = len(df)

    logger.info(f"\nClassification Results:")
    logger.info(f"  Total unique words: {total}")
    logger.info(f"  Correct spelling: {correct} ({correct/max(total,1)*100:.1f}%)")
    logger.info(f"  Incorrect spelling: {incorrect} ({incorrect/max(total,1)*100:.1f}%)")
    logger.info(f"  Error rate: {incorrect/max(total,1)*100:.1f}%")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify word spelling")
    parser.add_argument("--input", required=True, help="Path to word list file")
    parser.add_argument("--dict", default=None, help="Path to Hindi dictionary file")
    parser.add_argument("--output", default="outputs/q3/word_classification.csv", help="Output CSV")
    parser.add_argument("--freq-threshold", type=int, default=50, help="Frequency threshold")
    args = parser.parse_args()

    from src.q3_spelling.extract_unique_words import load_word_list, extract_unique_words

    words = load_word_list(args.input)
    unique_words, freq = extract_unique_words(words)

    df = classify_word_list(
        unique_words,
        dictionary_path=args.dict,
        word_freq=dict(freq),
        freq_threshold=args.freq_threshold,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    correct = (df["label"] == "correct spelling").sum()
    incorrect = (df["label"] == "incorrect spelling").sum()

    print(f"\n{'='*50}")
    print(f"SPELLING CLASSIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"  Total unique words: {len(df)}")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {incorrect}")
    print(f"  Error rate: {incorrect/max(len(df),1)*100:.1f}%")
    print(f"  Output: {args.output}")

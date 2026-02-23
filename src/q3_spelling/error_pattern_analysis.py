"""
Error Pattern Analysis — JoshTalks ASR Research
==================================================
Analyze common spelling error patterns and character-level confusions.
"""

import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from src.q3_spelling.spell_classifier import edit_distance, load_dictionary
from src.q3_spelling.extract_unique_words import is_devanagari, normalize_word

logger = setup_logger("error_patterns", log_to_file=True)


def find_closest_correct(
    incorrect_word: str,
    dictionary: set,
    max_distance: int = 2,
) -> Optional[Tuple[str, int]]:
    """
    Find the closest correct word for a misspelling.

    Args:
        incorrect_word: Misspelled word.
        dictionary: Set of known correct words.
        max_distance: Maximum edit distance to consider.

    Returns:
        Tuple of (closest_word, distance) or None.
    """
    best_word = None
    best_dist = max_distance + 1

    for correct_word in dictionary:
        if abs(len(incorrect_word) - len(correct_word)) > max_distance:
            continue
        dist = edit_distance(incorrect_word, correct_word)
        if dist < best_dist:
            best_dist = dist
            best_word = correct_word

    if best_word and best_dist <= max_distance:
        return (best_word, best_dist)
    return None


def analyze_character_confusions(
    incorrect_words: List[str],
    dictionary: set,
) -> Dict[Tuple[str, str], int]:
    """
    Analyze character-level substitution patterns.

    For each misspelling, find the closest correct word and
    identify which characters were confused.

    Args:
        incorrect_words: List of misspelled words.
        dictionary: Set of known correct words.

    Returns:
        Counter of (correct_char, incorrect_char) confusion pairs.
    """
    confusions = Counter()
    analyzed = 0

    for word in incorrect_words[:500]:  # Limit for performance
        result = find_closest_correct(word, dictionary)
        if result is None:
            continue

        correct_word, dist = result
        if dist == 1:
            # Single substitution — find the differing character
            for c1, c2 in zip(correct_word, word):
                if c1 != c2:
                    confusions[(c1, c2)] += 1
                    break
            analyzed += 1

    logger.info(f"Analyzed {analyzed} single-substitution errors")
    return confusions


def analyze_error_patterns(
    classification_df: pd.DataFrame,
    dictionary_path: Optional[str] = None,
    output_dir: str = "outputs/q3",
) -> Dict:
    """
    Full error pattern analysis.

    Args:
        classification_df: DataFrame with 'word' and 'label' columns.
        dictionary_path: Path to dictionary file.
        output_dir: Output directory.

    Returns:
        Analysis results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    correct_words = classification_df[classification_df["label"] == "correct spelling"]["word"].tolist()
    incorrect_words = classification_df[classification_df["label"] == "incorrect spelling"]["word"].tolist()

    total = len(classification_df)
    correct_count = len(correct_words)
    incorrect_count = len(incorrect_words)

    # Summary
    summary = {
        "total_unique_words": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "error_rate_percent": round(incorrect_count / max(total, 1) * 100, 2),
    }

    print(f"\n{'='*50}")
    print("SPELLING ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"  Total unique words:  {total}")
    print(f"  Correct spelling:    {correct_count}")
    print(f"  Incorrect spelling:  {incorrect_count}")
    print(f"  Error rate:          {summary['error_rate_percent']:.2f}%")
    print(f"{'='*50}")

    # Error patterns by word length
    if incorrect_words:
        length_dist = Counter(len(w) for w in incorrect_words)
        length_df = pd.DataFrame(
            sorted(length_dist.items()),
            columns=["word_length", "count"],
        )
        length_df.to_csv(os.path.join(output_dir, "error_by_length.csv"), index=False)

    # Character confusion analysis
    dictionary = load_dictionary(dictionary_path)
    confusions = analyze_character_confusions(incorrect_words, dictionary)

    if confusions:
        confusion_df = pd.DataFrame(
            [(c[0], c[1], count) for (c, count) in confusions.most_common(30)],
            columns=["correct_char", "confused_char", "count"],
        )
        confusion_df.to_csv(os.path.join(output_dir, "character_confusions.csv"), index=False)
        logger.info(f"Top character confusions:\n{confusion_df.head(10).to_string()}")

    # Devanagari vs Latin script error distribution
    dev_errors = sum(1 for w in incorrect_words if is_devanagari(w))
    latin_errors = sum(1 for w in incorrect_words if w.isascii())
    mixed_errors = incorrect_count - dev_errors - latin_errors

    script_dist = {
        "devanagari": dev_errors,
        "latin": latin_errors,
        "mixed": mixed_errors,
    }
    summary["script_distribution"] = script_dist

    # === Visualizations ===

    # 1. Correct vs Incorrect pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels_pie = ["Correct", "Incorrect"]
    sizes = [correct_count, incorrect_count]
    colors = ["#7ED321", "#E8636F"]
    axes[0].pie(sizes, labels=labels_pie, autopct="%1.1f%%", colors=colors,
                startangle=90, textprops={"fontsize": 13})
    axes[0].set_title("Spelling Classification Distribution",
                      fontsize=14, fontweight="bold")

    # 2. Error by script type
    script_labels = list(script_dist.keys())
    script_values = list(script_dist.values())
    script_colors = ["#4A90D9", "#F5A623", "#BD10E0"]
    axes[1].bar(script_labels, script_values, color=script_colors,
                edgecolor="white")
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Errors by Script Type", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(script_values):
        axes[1].text(i, v + max(script_values) * 0.01, str(v),
                     ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spelling_error_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Save summary
    import json
    with open(os.path.join(output_dir, "spelling_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"Error pattern analysis saved to {output_dir}/")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spelling Error Pattern Analysis")
    parser.add_argument("--classification", required=True, help="Path to classification CSV")
    parser.add_argument("--dict", default=None, help="Path to dictionary file")
    parser.add_argument("--output-dir", default="outputs/q3", help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.classification)
    analyze_error_patterns(df, args.dict, args.output_dir)

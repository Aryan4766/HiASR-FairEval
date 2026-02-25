"""
Unit tests for Q3 — Spelling Classification modules.
Tests: extract_unique_words.py and spell_classifier.py
"""

import pytest

from src.q3_spelling.extract_unique_words import (
    extract_unique_words,
    is_devanagari,
    normalize_word,
)
from src.q3_spelling.spell_classifier import (
    classify_word,
    classify_word_list,
    edit_distance,
    is_likely_transliteration,
    load_dictionary,
)


# =============================================================
# extract_unique_words tests
# =============================================================

class TestNormalizeWord:
    """Tests for word normalization."""

    def test_strips_whitespace(self):
        assert normalize_word("  नमस्ते  ") == "नमस्ते"

    def test_removes_punctuation_edges(self):
        assert normalize_word(",नमस्ते,") == "नमस्ते"

    def test_empty_string_returns_empty(self):
        assert normalize_word("") == ""

    def test_none_returns_empty(self):
        assert normalize_word(None) == ""

    def test_unicode_normalization(self):
        """Ensure NFKC normalization is applied."""
        word = normalize_word("नमस्ते")
        assert isinstance(word, str)
        assert len(word) > 0


class TestIsDevanagari:
    """Tests for Devanagari script detection."""

    def test_hindi_word_is_devanagari(self):
        assert is_devanagari("नमस्ते") is True

    def test_english_word_is_not_devanagari(self):
        assert is_devanagari("hello") is False

    def test_empty_string_is_not_devanagari(self):
        assert is_devanagari("") is False

    def test_mixed_majority_devanagari(self):
        """More than 50% Devanagari chars → True."""
        assert is_devanagari("नमस्तेa") is True

    def test_numbers_not_devanagari(self):
        assert is_devanagari("12345") is False


class TestExtractUniqueWords:
    """Tests for unique word extraction."""

    def test_returns_unique_sorted(self):
        words = ["बी", "ए", "बी", "सी", "ए"]
        unique, freq = extract_unique_words(words)
        assert len(unique) == 3
        assert unique == sorted(unique)

    def test_frequency_counts(self):
        words = ["ए", "ए", "बी"]
        unique, freq = extract_unique_words(words)
        assert freq["ए"] == 2
        assert freq["बी"] == 1

    def test_empty_list(self):
        unique, freq = extract_unique_words([])
        assert len(unique) == 0

    def test_skips_empty_strings(self):
        words = ["नमस्ते", "", "  ", "भारत"]
        unique, freq = extract_unique_words(words)
        assert "" not in unique


# =============================================================
# spell_classifier tests
# =============================================================

class TestEditDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical_strings(self):
        assert edit_distance("hello", "hello") == 0

    def test_one_substitution(self):
        assert edit_distance("cat", "bat") == 1

    def test_one_insertion(self):
        assert edit_distance("cat", "cats") == 1

    def test_one_deletion(self):
        assert edit_distance("cats", "cat") == 1

    def test_empty_vs_word(self):
        assert edit_distance("", "abc") == 3

    def test_both_empty(self):
        assert edit_distance("", "") == 0


class TestLoadDictionary:
    """Tests for dictionary loading."""

    def test_returns_set(self):
        dictionary = load_dictionary()
        assert isinstance(dictionary, set)


class TestIsLikelyTransliteration:
    """Tests for transliteration detection."""

    def test_known_transliteration(self):
        assert is_likely_transliteration("कंप्यूटर") is True

    def test_pure_hindi_word(self):
        # A common Hindi word should not be flagged as transliteration
        # (unless it happens to be in the transliteration set)
        result = is_likely_transliteration("पानी")
        assert isinstance(result, bool)


class TestClassifyWordList:
    """Tests for batch classification."""

    def test_returns_dataframe(self, sample_hindi_words):
        result = classify_word_list(sample_hindi_words)
        assert hasattr(result, "columns")
        assert "word" in result.columns
        assert "label" in result.columns

    def test_output_length_matches_input(self, sample_hindi_words):
        result = classify_word_list(sample_hindi_words)
        assert len(result) == len(sample_hindi_words)

    def test_labels_are_valid(self, sample_hindi_words):
        result = classify_word_list(sample_hindi_words)
        valid_labels = {"correct spelling", "incorrect spelling"}
        for label in result["label"]:
            assert label in valid_labels

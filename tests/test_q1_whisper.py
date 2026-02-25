"""
Unit tests for Q1 — Whisper Evaluation modules.
Tests: compute_wer.py (the src module, not lattice WER)
"""

import pytest

from src.utils.metrics import compute_wer_detailed, compute_corpus_metrics, get_error_pairs


class TestComputeWERDetailed:
    """Tests for detailed WER computation."""

    def test_identical_strings(self):
        result = compute_wer_detailed("नमस्ते दुनिया", "नमस्ते दुनिया")
        assert result["wer"] == 0.0
        assert result["hits"] == 2

    def test_complete_mismatch(self):
        result = compute_wer_detailed("a b c", "x y z")
        assert result["wer"] == 1.0
        assert result["substitutions"] == 3

    def test_empty_reference_empty_hypothesis(self):
        result = compute_wer_detailed("", "")
        assert result["wer"] == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        result = compute_wer_detailed("", "hello world")
        assert result["wer"] == 1.0

    def test_returns_all_keys(self):
        result = compute_wer_detailed("a b c", "a b d")
        required = {"wer", "cer", "substitutions", "insertions", "deletions", "hits", "ref_length"}
        assert required.issubset(set(result.keys()))


class TestComputeCorpusMetrics:
    """Tests for corpus-level metrics."""

    def test_single_pair(self):
        result = compute_corpus_metrics(["hello world"], ["hello world"])
        assert result["corpus_wer"] == 0.0

    def test_multiple_pairs(self):
        refs = ["a b c", "x y z"]
        hyps = ["a b c", "x y z"]
        result = compute_corpus_metrics(refs, hyps)
        assert result["corpus_wer"] == 0.0
        assert result["num_utterances"] == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_corpus_metrics(["a"], ["b", "c"])


class TestGetErrorPairs:
    """Tests for error pair extraction."""

    def test_returns_three_lists(self):
        subs, dels, ins = get_error_pairs(["a b c"], ["a x c"])
        assert isinstance(subs, list)
        assert isinstance(dels, list)
        assert isinstance(ins, list)

    def test_substitution_detected(self):
        subs, _, _ = get_error_pairs(["a b c"], ["a x c"])
        assert len(subs) >= 1
        assert ("b", "x") in subs

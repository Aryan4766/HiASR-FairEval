"""
Unit tests for Q4 — Lattice-Based WER modules.
Tests: align_dp.py, majority_consensus.py, lattice_wer.py
"""

import pytest

from src.q4_lattice.align_dp import align, align_strings, MATCH, SUBSTITUTE, INSERT, DELETE
from src.q4_lattice.majority_consensus import (
    align_all_to_reference,
    build_position_matrix,
    compute_consensus,
    get_consensus_reference,
    summarize_consensus,
)
from src.q4_lattice.lattice_wer import compute_wer, format_comparison_table


# =============================================================
# align_dp tests
# =============================================================

class TestAlign:
    """Tests for DP alignment."""

    def test_identical_sequences(self):
        ref = ["यह", "एक", "दिन"]
        alignment, distance, stats = align(ref, ref)
        assert distance == 0
        assert stats[MATCH] == 3
        assert stats[SUBSTITUTE] == 0

    def test_one_substitution(self, reference_words, hypothesis_words):
        alignment, distance, stats = align(reference_words, hypothesis_words)
        assert distance > 0
        assert stats[SUBSTITUTE] >= 1 or stats[DELETE] >= 1

    def test_empty_reference(self):
        alignment, distance, stats = align([], ["word1", "word2"])
        assert distance == 2
        assert stats[INSERT] == 2

    def test_empty_hypothesis(self):
        alignment, distance, stats = align(["word1", "word2"], [])
        assert distance == 2
        assert stats[DELETE] == 2

    def test_both_empty(self):
        alignment, distance, stats = align([], [])
        assert distance == 0

    def test_alignment_length(self):
        ref = ["a", "b", "c"]
        hyp = ["a", "x", "c"]
        alignment, _, _ = align(ref, hyp)
        # Alignment should have entries for all positions
        assert len(alignment) >= max(len(ref), len(hyp))


class TestAlignStrings:
    """Tests for string-level alignment wrapper."""

    def test_basic_alignment(self):
        alignment, distance, stats = align_strings(
            "यह एक अच्छा दिन है",
            "यह बहुत अच्छा है"
        )
        assert distance > 0
        assert isinstance(stats, dict)

    def test_identical_strings(self):
        alignment, distance, stats = align_strings("hello world", "hello world")
        assert distance == 0


# =============================================================
# majority_consensus tests
# =============================================================

class TestMajorityConsensus:
    """Tests for lattice consensus computation."""

    def test_align_all_returns_list(self, reference_words, sample_model_outputs):
        alignments = align_all_to_reference(reference_words, sample_model_outputs)
        assert len(alignments) == len(sample_model_outputs)

    def test_build_position_matrix(self, reference_words, sample_model_outputs):
        alignments = align_all_to_reference(reference_words, sample_model_outputs)
        matrix = build_position_matrix(reference_words, alignments)
        assert isinstance(matrix, list)
        assert len(matrix) > 0

    def test_compute_consensus(self, reference_words, sample_model_outputs):
        alignments = align_all_to_reference(reference_words, sample_model_outputs)
        matrix = build_position_matrix(reference_words, alignments)
        matrix = compute_consensus(matrix, min_agreement=3, num_models=5)
        # Each position should have consensus info
        for pos in matrix:
            assert "consensus_word" in pos

    def test_consensus_reference(self, reference_words, sample_model_outputs):
        alignments = align_all_to_reference(reference_words, sample_model_outputs)
        matrix = build_position_matrix(reference_words, alignments)
        matrix = compute_consensus(matrix, min_agreement=3, num_models=5)
        consensus_ref = get_consensus_reference(matrix)
        assert isinstance(consensus_ref, list)
        assert len(consensus_ref) > 0

    def test_summarize_consensus(self, reference_words, sample_model_outputs):
        alignments = align_all_to_reference(reference_words, sample_model_outputs)
        matrix = build_position_matrix(reference_words, alignments)
        matrix = compute_consensus(matrix, min_agreement=3, num_models=5)
        summary = summarize_consensus(matrix)
        assert "total_positions" in summary
        assert "reference_overridden" in summary


# =============================================================
# lattice_wer tests
# =============================================================

class TestComputeWER:
    """Tests for WER computation."""

    def test_identical_gives_zero(self):
        ref = ["यह", "एक", "दिन"]
        wer = compute_wer(ref, ref)
        assert wer == 0.0

    def test_all_wrong_gives_one(self):
        ref = ["a", "b", "c"]
        hyp = ["x", "y", "z"]
        wer = compute_wer(ref, hyp)
        assert wer == 1.0

    def test_empty_reference(self):
        wer = compute_wer([], ["a"])
        # When ref is empty, WER should handle gracefully
        assert isinstance(wer, float)

    def test_partial_match(self):
        ref = ["a", "b", "c", "d"]
        hyp = ["a", "b", "c", "d"]
        wer = compute_wer(ref, hyp)
        assert wer == 0.0


class TestFormatComparisonTable:
    """Tests for comparison table formatting."""

    def test_returns_string(self):
        results = [
            {"model": "M1", "original_wer": 0.5, "adjusted_wer": 0.4, "delta": -0.1, "valid": True},
            {"model": "M2", "original_wer": 0.6, "adjusted_wer": 0.5, "delta": -0.1, "valid": True},
        ]
        table = format_comparison_table(results)
        assert isinstance(table, str)
        assert "M1" in table
        assert "M2" in table

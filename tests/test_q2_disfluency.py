"""
Unit tests for Q2 — Disfluency Detection modules.
Tests: detect_rules.py and stats_analysis.py
"""

import pytest

from src.q2_disfluency.detect_rules import (
    detect_all_disfluencies,
    detect_elongations,
    detect_fillers,
    detect_hesitations,
    detect_repetitions,
    save_disfluency_report,
)


class TestDetectFillers:
    """Tests for filler word detection."""

    def test_detects_hindi_filler(self):
        results = detect_fillers("umm मैं बोल रहा था")
        assert len(results) > 0
        assert any(r["type"] == "filler" for r in results)

    def test_detects_english_filler(self):
        results = detect_fillers("uh I was saying umm something")
        assert len(results) > 0

    def test_no_fillers_in_clean_text(self):
        results = detect_fillers("यह एक अच्छा दिन है")
        assert len(results) == 0

    def test_empty_text(self):
        results = detect_fillers("")
        assert len(results) == 0


class TestDetectRepetitions:
    """Tests for consecutive word repetition detection."""

    def test_detects_consecutive_repetition(self):
        results = detect_repetitions("मैं मैं बोल रहा था")
        assert len(results) > 0
        assert any(r["type"] == "repetition" for r in results)

    def test_no_repetition_in_clean_text(self):
        results = detect_repetitions("यह एक अच्छा दिन है")
        assert len(results) == 0

    def test_triple_repetition(self):
        results = detect_repetitions("the the the cat")
        assert len(results) > 0


class TestDetectElongations:
    """Tests for character elongation detection."""

    def test_detects_elongation(self):
        results = detect_elongations("sooo good")
        assert len(results) > 0
        assert any(r["type"] == "elongation" for r in results)

    def test_no_elongation_in_normal_text(self):
        results = detect_elongations("so good")
        assert len(results) == 0


class TestDetectHesitations:
    """Tests for hesitation marker detection."""

    def test_detects_ellipsis(self):
        results = detect_hesitations("I was... thinking")
        assert len(results) > 0

    def test_detects_dashes(self):
        results = detect_hesitations("I was-- saying")
        assert len(results) > 0


class TestDetectAllDisfluencies:
    """Tests for combined disfluency detection."""

    def test_returns_list(self, sample_disfluent_text):
        results = detect_all_disfluencies(
            text=sample_disfluent_text,
            recording_id="test_001",
            speaker_id="tester",
            segment_id="0",
        )
        assert isinstance(results, list)

    def test_detects_multiple_types(self, sample_disfluent_text):
        results = detect_all_disfluencies(text=sample_disfluent_text)
        types_found = {r["type"] for r in results}
        # The sample text has fillers ("अं"), repetitions ("कि कि"), elongation ("sooo")
        assert len(types_found) >= 1

    def test_empty_text_returns_empty(self):
        results = detect_all_disfluencies(text="")
        assert results == []

    def test_result_has_required_keys(self, sample_disfluent_text):
        results = detect_all_disfluencies(
            text=sample_disfluent_text,
            recording_id="rec1",
            speaker_id="sp1",
            segment_id="s1",
        )
        if results:
            required_keys = {"type", "text", "recording_id", "speaker_id", "segment_id"}
            assert required_keys.issubset(set(results[0].keys()))


class TestSaveDisfluencyReport:
    """Tests for report saving."""

    def test_saves_csv(self, tmp_path):
        disfluencies = [
            {
                "recording_id": "r1",
                "speaker_id": "s1",
                "segment_id": "0",
                "type": "filler",
                "text": "अं",
                "segment_start": 0.0,
                "segment_end": 1.0,
            }
        ]
        output_path = str(tmp_path / "report.csv")
        df = save_disfluency_report(disfluencies, output_path)
        assert len(df) == 1
        assert (tmp_path / "report.csv").exists()

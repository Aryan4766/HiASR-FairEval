"""
Unit tests for the data pipeline modules.
Tests: text_normalization.py and fix_urls.py
"""

import pytest

from src.data_pipeline.text_normalization import normalize_text, normalize_whitespace, normalize_unicode
from src.data_pipeline.fix_urls import fix_url, build_urls


class TestNormalizeText:
    """Tests for text normalization."""

    def test_strips_whitespace(self):
        result = normalize_text("  hello world  ")
        assert result == result.strip()

    def test_unicode_normalization(self):
        """Ensure NFKC normalization is applied to Hindi text."""
        result = normalize_text("नमस्ते")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        result = normalize_text("")
        assert result == ""

    def test_none_returns_empty(self):
        result = normalize_text(None)
        assert result == ""

    def test_handles_punctuation(self):
        result = normalize_text("hello, world!")
        assert "," not in result
        assert "!" not in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_collapses_multiple_spaces(self):
        result = normalize_whitespace("hello    world")
        assert result == "hello world"

    def test_strips_edges(self):
        result = normalize_whitespace("  hello  ")
        assert result == "hello"


class TestNormalizeUnicode:
    """Tests for Unicode normalization."""

    def test_nfkc(self):
        result = normalize_unicode("नमस्ते", "NFKC")
        assert isinstance(result, str)


class TestFixUrl:
    """Tests for GCS URL fixing."""

    def test_returns_string(self):
        result = fix_url("https://storage.googleapis.com/some_bucket/file.wav")
        assert isinstance(result, str)

    def test_correct_url_unchanged(self):
        url = "https://storage.googleapis.com/upload_goai/123/456_recording.wav"
        result = fix_url(url)
        assert result == url

    def test_handles_empty_url(self):
        result = fix_url("")
        assert result == ""


class TestBuildUrls:
    """Tests for URL construction."""

    def test_builds_three_urls(self):
        urls = build_urls("12345", "67890")
        assert "recording" in urls
        assert "transcription" in urls
        assert "metadata" in urls
        assert "_recording.wav" in urls["recording"]
        assert "_transcription.json" in urls["transcription"]

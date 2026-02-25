"""
Shared pytest fixtures for HiASR-FairEval test suite.
"""

import os
import sys

import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_hindi_words():
    """Sample Hindi (Devanagari) words for testing."""
    return ["नमस्ते", "शुभकामनाएं", "भारत", "हिंदी", "कंप्यूटर", "विद्यालय"]


@pytest.fixture
def sample_latin_words():
    """Sample Latin-script words for testing."""
    return ["hello", "world", "computer", "test"]


@pytest.fixture
def sample_mixed_words():
    """Mix of Devanagari, Latin, and edge-case words."""
    return ["नमस्ते", "hello", "123", "", "  ", "कंप्यूटर", "abc"]


@pytest.fixture
def reference_words():
    """Sample reference word list for WER/alignment tests."""
    return ["यह", "एक", "अच्छा", "दिन", "है"]


@pytest.fixture
def hypothesis_words():
    """Sample hypothesis with errors for WER/alignment tests."""
    return ["यह", "बहुत", "अच्छा", "है"]


@pytest.fixture
def sample_disfluent_text():
    """Sample text with speech disfluencies."""
    return "अं मैं ये बोल रहा था कि कि वो sooo अच्छा था"


@pytest.fixture
def sample_model_outputs(reference_words):
    """5 model outputs for lattice consensus testing."""
    return [
        ["यह", "एक", "अच्छा", "दिन", "है"],       # Model 1: perfect
        ["यह", "एक", "अच्छा", "है"],                # Model 2: deletion
        ["यह", "बहुत", "अच्छा", "दिन", "है"],       # Model 3: substitution
        ["यह", "एक", "अच्छा", "दिन", "है"],         # Model 4: perfect
        ["यह", "एक", "अच्छी", "दिन", "है"],         # Model 5: substitution
    ]


@pytest.fixture
def configs_dir():
    """Path to configs directory."""
    return os.path.join(PROJECT_ROOT, "configs")

"""
Unit tests for shared utility modules.
Tests: config_loader.py, logger.py, metrics.py
"""

import logging
import os

import pytest

from src.utils.config_loader import (
    ensure_output_dirs,
    get_config_value,
    load_config,
    resolve_path,
)
from src.utils.logger import get_hardware_info, setup_logger


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_preprocessing_config(self, configs_dir):
        config = load_config(os.path.join(configs_dir, "preprocessing.yaml"))
        assert "audio" in config
        assert config["audio"]["sample_rate"] == 16000

    def test_load_training_config(self, configs_dir):
        config = load_config(os.path.join(configs_dir, "training.yaml"))
        assert "model" in config
        assert config["model"]["name"] == "openai/whisper-small"

    def test_load_lattice_config(self, configs_dir):
        config = load_config(os.path.join(configs_dir, "lattice.yaml"))
        assert "alignment" in config
        assert "consensus" in config

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestGetConfigValue:
    """Tests for dot-notation config access."""

    def test_nested_access(self):
        config = {"audio": {"sample_rate": 16000, "mono": True}}
        assert get_config_value(config, "audio.sample_rate") == 16000

    def test_missing_key_returns_default(self):
        config = {"audio": {"sample_rate": 16000}}
        assert get_config_value(config, "audio.nonexistent", default=42) == 42

    def test_top_level_access(self):
        config = {"seed": 42}
        assert get_config_value(config, "seed") == 42


class TestResolvePath:
    """Tests for path resolution."""

    def test_relative_path(self):
        path = resolve_path("configs/training.yaml")
        assert path.is_absolute()

    def test_absolute_path_unchanged(self):
        abs_path = os.path.abspath("configs/training.yaml")
        path = resolve_path(abs_path)
        assert str(path) == abs_path


class TestLogger:
    """Tests for logger setup."""

    def test_setup_logger_returns_logger(self):
        logger = setup_logger("test_logger", log_to_file=False)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_hardware_info_has_keys(self):
        info = get_hardware_info()
        assert "platform" in info
        assert "python_version" in info
        assert "timestamp" in info

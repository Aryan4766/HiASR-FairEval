"""
Config Loader — JoshTalks ASR Research
=======================================
YAML configuration loader with path resolution and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Project root (two levels up from src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file (relative to project root or absolute).

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    # Resolve relative paths from project root
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path relative to the project root.

    Args:
        path_str: Path string (relative or absolute).

    Returns:
        Resolved absolute Path object.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., 'audio.sample_rate').
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> cfg = {'audio': {'sample_rate': 16000}}
        >>> get_config_value(cfg, 'audio.sample_rate')
        16000
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def ensure_output_dirs(config: Dict) -> None:
    """Create all output directories specified in config paths."""
    if "paths" in config:
        for key, path_str in config["paths"].items():
            path = resolve_path(path_str)
            path.mkdir(parents=True, exist_ok=True)

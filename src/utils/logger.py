"""
Logger Module — JoshTalks ASR Research
=======================================
Structured logging with console + file output.
Captures hardware info, timestamps, and experiment metadata.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_hardware_info() -> dict:
    """Collect hardware information for experiment reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
    }

    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 2)
        info["ram_available_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        info["ram_total_gb"] = "psutil not installed"

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
            )
        else:
            info["gpu"] = "None (CPU only)"
    except ImportError:
        info["gpu"] = "torch not installed"

    return info


def setup_logger(
    name: str,
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Set up a structured logger with console and optional file output.

    Args:
        name: Logger name (typically module name).
        log_dir: Directory for log files.
        level: Logging level.
        log_to_file: Whether to also write logs to a file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def log_experiment_start(
    logger: logging.Logger,
    experiment_name: str,
    config: dict,
    seed: int = 42,
) -> None:
    """Log experiment metadata at the start of a run."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")

    hw_info = get_hardware_info()
    for key, value in hw_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 60)

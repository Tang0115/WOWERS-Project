"""Configure structured logging for the pipeline."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.common import config


def setup(name: str = "wowers", level: int = logging.INFO) -> logging.Logger:
    logs_dir = config.logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(logs_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get(name: str = "wowers") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup(name)
    return logger


def setup_run_log(phase_name: str) -> Path:
    """Add a timestamped run log file handler to the root wowers logger.

    Creates logs/runs/<phase_name>_YYYY-MM-DD_HH-MM-SS.log.
    All wowers.* child loggers propagate to the root logger, so every
    log line from every module in the run is captured in this single file.

    Returns the path to the run log file.
    """
    runs_dir = config.logs_dir() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = runs_dir / f"{phase_name}_{timestamp}.log"

    root_logger = logging.getLogger("wowers")
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    root_logger.info(f"Run log: {log_path}")
    return log_path

"""Load and expose project configuration from settings.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/common/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get(key_path: str, default: Any = None) -> Any:
    """Dot-separated key path lookup, e.g. get('physics.gravity_m_s2')."""
    settings = load_settings()
    keys = key_path.split(".")
    node = settings
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def project_root() -> Path:
    return PROJECT_ROOT


def raw_data_dir() -> Path:
    return PROJECT_ROOT / get("paths.raw_data", "data/raw")


def processed_dir() -> Path:
    return PROJECT_ROOT / get("paths.processed_data", "data/processed")


def checkpoints_dir() -> Path:
    return PROJECT_ROOT / get("paths.checkpoints", "data/checkpoints")


def logs_dir() -> Path:
    return PROJECT_ROOT / get("paths.logs", "logs")

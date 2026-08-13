"""Parquet read/write helpers and checkpointing utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import config, logging_setup

log = logging_setup.get("wowers.io")

MANIFEST_PATH = config.checkpoints_dir() / "manifest.json"


def write_parquet(df: pl.DataFrame, path: Path, partition_by: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_by:
        table = df.to_arrow()
        pq.write_to_dataset(table, root_path=str(path), partition_cols=[partition_by])
    else:
        df.write_parquet(path, compression="snappy")
    log.info(f"Wrote {len(df):,} rows → {path}")


def read_parquet(path: Path) -> pl.DataFrame:
    if path.is_dir():
        return pl.read_parquet(path / "**/*.parquet")
    return pl.read_parquet(path)


def checkpoint(df: pl.DataFrame, phase: str, step: str) -> Path:
    """Write a versioned checkpoint and update the manifest."""
    checkpoints_dir = config.checkpoints_dir()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment version
    existing = sorted(checkpoints_dir.glob(f"{phase}_{step}_v*.parquet"))
    version = len(existing) + 1
    path = checkpoints_dir / f"{phase}_{step}_v{version:03d}.parquet"

    df.write_parquet(path, compression="snappy")
    log.info(f"Checkpoint → {path.name} ({len(df):,} rows)")

    _update_manifest(phase, step, str(path), version)
    return path


def _update_manifest(phase: str, step: str, path: str, version: int) -> None:
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    manifest.setdefault(phase, {})[step] = {
        "path": path,
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_latest_checkpoint(phase: str, step: str) -> Optional[Path]:
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    entry = manifest.get(phase, {}).get(step)
    if entry:
        p = Path(entry["path"])
        if p.exists():
            return p
    return None

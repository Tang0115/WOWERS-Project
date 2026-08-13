"""Robust file downloader with streaming, resume support, and retry logic."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.common import config, logging_setup

log = logging_setup.get("wowers.download")


def download_file(
    url: str,
    dest_dir: Path,
    filename: Optional[str] = None,
    skip_if_exists: bool = True,
) -> Path:
    """Download a file to dest_dir. Skip if already present and size matches.

    Returns the local path to the downloaded file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or url.split("/")[-1].split("?")[0]
    dest_path = dest_dir / fname

    # Check remote size for skip logic
    remote_size = _get_remote_size(url)

    if skip_if_exists and dest_path.exists():
        local_size = dest_path.stat().st_size
        if remote_size and abs(local_size - remote_size) < 1024:
            log.info(f"Skipping {fname} — already downloaded ({local_size:,} bytes)")
            return dest_path
        elif not remote_size and dest_path.stat().st_size > 0:
            log.info(f"Skipping {fname} — already exists (size unverifiable)")
            return dest_path
        else:
            log.info(f"Re-downloading {fname} — size mismatch")

    retries = config.get("processing.download_retry_attempts", 3)
    backoff = config.get("processing.download_retry_backoff_s", 5)

    for attempt in range(1, retries + 1):
        try:
            log.info(f"Downloading {fname} (attempt {attempt}/{retries})")
            _stream_download(url, dest_path, remote_size)
            log.info(f"Downloaded {fname} → {dest_path}")
            return dest_path
        except Exception as exc:
            log.warning(f"Attempt {attempt} failed: {exc}")
            if attempt < retries:
                sleep_s = backoff * attempt
                log.info(f"Retrying in {sleep_s}s …")
                time.sleep(sleep_s)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts"
                ) from exc

    raise RuntimeError("Unreachable")


def _stream_download(url: str, dest_path: Path, total_size: Optional[int]) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with (
            open(dest_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
                leave=False,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                f.write(chunk)
                bar.update(len(chunk))


def _get_remote_size(url: str) -> Optional[int]:
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception:
        return None


def extract_zip(zip_path: Path, extract_to: Optional[Path] = None) -> Path:
    """Extract a ZIP file. Returns the extraction directory."""
    out_dir = extract_to or zip_path.parent / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    log.info(f"Extracted {len(list(out_dir.rglob('*')))} files")
    return out_dir

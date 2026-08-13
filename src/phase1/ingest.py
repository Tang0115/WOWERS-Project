"""Download and extract EPA ECHO bulk files for Phase 1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.common import config, download, logging_setup

log = logging_setup.get("wowers.phase1.ingest")

# ── Expected CSV files inside npdes_downloads.zip ────────────────────────────
REQUIRED_NPDES_FILES = ["ICIS_FACILITIES.CSV", "ICIS_PERMITS.CSV"]


def ingest_npdes_facilities(raw_dir: Optional[Path] = None) -> dict[str, Path]:
    """Download npdes_downloads.zip and return paths to required CSVs.

    Returns:
        dict with keys 'facilities' and 'permits' pointing to CSV paths.
    """
    raw_dir = raw_dir or config.raw_data_dir()
    url = config.get("epa.npdes_downloads_url")

    zip_path = download.download_file(url, raw_dir / "npdes_downloads")
    extract_dir = download.extract_zip(zip_path, raw_dir / "npdes_downloads" / "extracted")

    return _locate_npdes_csvs(extract_dir)


def ingest_dmr_files(
    fiscal_years: Optional[list[int]] = None,
    raw_dir: Optional[Path] = None,
    max_workers: int = 4,
) -> dict[int, Path]:
    """Download DMR ZIP files for each fiscal year, return {year: zip_path}.

    Downloads up to max_workers files in parallel.
    """
    raw_dir = raw_dir or config.raw_data_dir()
    fiscal_years = fiscal_years or config.get("epa.dmr_fiscal_years", [])
    url_template = config.get("epa.dmr_url_template")
    max_workers = min(max_workers, config.get("processing.max_parallel_downloads", 4))

    results: dict[int, Path] = {}

    def _download_year(year: int) -> tuple[int, Path]:
        url = url_template.format(year=year)  # fy2009, fy2010, ...
        dest_dir = raw_dir / "dmr" / f"fy{year}"
        path = download.download_file(url, dest_dir)
        return year, path

    log.info(f"Downloading DMR files for {len(fiscal_years)} fiscal years "
             f"(max {max_workers} parallel)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_year, yr): yr for yr in fiscal_years}
        for future in as_completed(futures):
            year = futures[future]
            try:
                yr, path = future.result()
                results[yr] = path
                log.info(f"FY{yr} DMR ready: {path.name}")
            except Exception as exc:
                log.error(f"FY{year} DMR download failed: {exc}")

    log.info(f"DMR downloads complete: {len(results)}/{len(fiscal_years)} years")
    return results


def ingest_outfalls(raw_dir: Optional[Path] = None) -> Path:
    """Download the NPDES outfalls layer ZIP. Returns the ZIP path."""
    raw_dir = raw_dir or config.raw_data_dir()
    url = config.get("epa.npdes_outfalls_url")
    zip_path = download.download_file(url, raw_dir / "npdes_outfalls")
    extract_dir = download.extract_zip(zip_path, raw_dir / "npdes_outfalls" / "extracted")
    return extract_dir


# ── Internal helpers ──────────────────────────────────────────────────────────

def _locate_npdes_csvs(extract_dir: Path) -> dict[str, Path]:
    """Find ICIS_FACILITIES.CSV and ICIS_PERMITS.CSV anywhere in extract_dir."""
    found: dict[str, Path] = {}

    for csv_path in extract_dir.rglob("*.CSV"):
        name_upper = csv_path.name.upper()
        if name_upper == "ICIS_FACILITIES.CSV":
            found["facilities"] = csv_path
        elif name_upper == "ICIS_PERMITS.CSV":
            found["permits"] = csv_path

    # Also check lowercase extensions
    if "facilities" not in found or "permits" not in found:
        for csv_path in extract_dir.rglob("*.csv"):
            name_upper = csv_path.name.upper()
            if name_upper == "ICIS_FACILITIES.CSV" and "facilities" not in found:
                found["facilities"] = csv_path
            elif name_upper == "ICIS_PERMITS.CSV" and "permits" not in found:
                found["permits"] = csv_path

    missing = [k for k in ("facilities", "permits") if k not in found]
    if missing:
        available = [p.name for p in extract_dir.rglob("*.CSV")]
        available += [p.name for p in extract_dir.rglob("*.csv")]
        raise FileNotFoundError(
            f"Could not find required CSVs: {missing}. "
            f"Available files: {sorted(set(available))[:20]}"
        )

    log.info(f"Located facilities CSV: {found['facilities']}")
    log.info(f"Located permits CSV:    {found['permits']}")
    return found

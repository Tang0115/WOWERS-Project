"""Phase 1 orchestrator — Rank Candidate Plants.

Usage:
    python -m src.phase1.run                          # full run
    python -m src.phase1.run --years 2022 2023 2024   # limit DMR years
    python -m src.phase1.run --skip-download          # use already-downloaded files
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.common import config, io, logging_setup
from src.phase1 import (
    dmr_timeseries,
    filter_potw,
    flow_features,
    ingest,
    ranking,
)

logging_setup.setup_run_log("phase1")
log = logging_setup.get("wowers.phase1")

OUTPUT_DIR = config.processed_dir() / "phase1"


def run(
    fiscal_years: list[int] | None = None,
    skip_download: bool = False,
    raw_dir: Path | None = None,
) -> Path:
    """Run the full Phase 1 pipeline.

    Args:
        fiscal_years: DMR years to process. None = all years from settings.yaml.
        skip_download: If True, assume files are already in raw_dir.
        raw_dir: Override for data/raw directory.

    Returns:
        Path to ranked_candidates.parquet.
    """
    t0 = time.time()
    raw_dir = raw_dir or config.raw_data_dir()
    fiscal_years = fiscal_years or config.get("epa.dmr_fiscal_years", [])

    log.info("=" * 60)
    log.info("WOWERS Phase 1 — Rank Candidate Plants")
    log.info(f"  DMR fiscal years: {fiscal_years}")
    log.info(f"  Raw data dir:     {raw_dir}")
    log.info(f"  Output dir:       {OUTPUT_DIR}")
    log.info("=" * 60)

    # ── Step 1: Download & extract ────────────────────────────────────────────
    log.info("[1/5] Downloading EPA ECHO bulk files …")
    if skip_download:
        csv_paths = _locate_existing_csvs(raw_dir)
        dmr_zips = _locate_existing_dmr_zips(raw_dir, fiscal_years)
    else:
        csv_paths = ingest.ingest_npdes_facilities(raw_dir)
        dmr_zips = ingest.ingest_dmr_files(fiscal_years, raw_dir)

    # ── Step 2: Filter to POTWs ───────────────────────────────────────────────
    log.info("[2/5] Filtering to POTW facilities …")
    potw_df = filter_potw.load_potw_facilities(
        facilities_csv=csv_paths["facilities"],
        permits_csv=csv_paths["permits"],
    )
    io.write_parquet(potw_df, OUTPUT_DIR / "potw_facilities.parquet")
    io.checkpoint(potw_df, "phase1", "potw_facilities")

    potw_ids: set[str] = set(potw_df["npdes_id"].to_list())
    log.info(f"  {len(potw_ids):,} active POTW permit IDs")

    # ── Step 3: Parse DMR flow time series ────────────────────────────────────
    log.info("[3/5] Parsing DMR flow time series …")
    timeseries_df = dmr_timeseries.parse_all_dmr_years(dmr_zips, potw_ids)
    io.write_parquet(
        timeseries_df,
        OUTPUT_DIR / "dmr_flow_timeseries.parquet",
        partition_by="fiscal_year",
    )
    io.checkpoint(timeseries_df, "phase1", "dmr_flow_timeseries")

    # ── Step 4: Compute flow features ─────────────────────────────────────────
    log.info("[4/5] Computing flow features …")
    features_df = flow_features.compute_flow_features(timeseries_df, potw_df)
    io.write_parquet(features_df, OUTPUT_DIR / "flow_features.parquet")
    io.checkpoint(features_df, "phase1", "flow_features")

    # ── Step 5: Compute ranking score ─────────────────────────────────────────
    log.info("[5/5] Computing ranking scores …")
    ranked_df = ranking.compute_ranking(features_df)
    out_path = OUTPUT_DIR / "ranked_candidates.parquet"
    io.write_parquet(ranked_df, out_path)
    io.checkpoint(ranked_df, "phase1", "ranked_candidates")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_summary(ranked_df, elapsed)

    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(df, elapsed: float) -> None:
    total = len(df)
    with_dmr = df.filter(
        ~df["data_quality"].is_in(["design_only", "actual_avg_only"])
        if "data_quality" in df.columns else True
    ).shape[0]

    log.info("")
    log.info("=" * 60)
    log.info("Phase 1 Complete")
    log.info(f"  Total POTW facilities:     {total:,}")
    log.info(f"  With DMR flow data:        {with_dmr:,} ({with_dmr/total*100:.1f}%)")
    log.info(f"  Without DMR (fallback):    {total - with_dmr:,}")
    log.info(f"  States covered:            {df['state_code'].n_unique()}")
    if "mean_flow_mgd" in df.columns:
        valid_flow = df["mean_flow_mgd"].drop_nulls()
        log.info(f"  Mean flow (national avg):  {valid_flow.mean():.2f} MGD")
        log.info(f"  Median flow:               {valid_flow.median():.2f} MGD")
    log.info(f"  Runtime:                   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"  Output: {OUTPUT_DIR / 'ranked_candidates.parquet'}")
    log.info("=" * 60)


# ── Helpers for skip-download mode ───────────────────────────────────────────

def _locate_existing_csvs(raw_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in (raw_dir / "npdes_downloads").rglob("*.CSV"):
        name = p.name.upper()
        if name == "ICIS_FACILITIES.CSV":
            found["facilities"] = p
        elif name == "ICIS_PERMITS.CSV":
            found["permits"] = p
    if "facilities" not in found or "permits" not in found:
        # Try lowercase
        for p in (raw_dir / "npdes_downloads").rglob("*.csv"):
            name = p.name.upper()
            if name == "ICIS_FACILITIES.CSV" and "facilities" not in found:
                found["facilities"] = p
            elif name == "ICIS_PERMITS.CSV" and "permits" not in found:
                found["permits"] = p
    missing = [k for k in ("facilities", "permits") if k not in found]
    if missing:
        raise FileNotFoundError(
            f"--skip-download set but could not locate: {missing} under {raw_dir}/npdes_downloads"
        )
    return found


def _locate_existing_dmr_zips(raw_dir: Path, fiscal_years: list[int]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for year in fiscal_years:
        pattern = f"*fy{year}*.zip"
        matches = list((raw_dir / "dmr" / f"fy{year}").glob(pattern))
        if not matches:
            # Try parent dmr dir
            matches = list((raw_dir / "dmr").glob(pattern))
        if matches:
            result[year] = matches[0]
        else:
            log.warning(f"--skip-download: could not find DMR ZIP for FY{year}")
    if not result:
        raise FileNotFoundError(
            f"--skip-download set but no DMR ZIPs found under {raw_dir}/dmr"
        )
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOWERS Phase 1: Rank Candidate Plants")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="DMR fiscal years to process (e.g. --years 2022 2023 2024)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads and use already-downloaded files",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw data directory",
    )
    args = parser.parse_args()

    run(
        fiscal_years=args.years,
        skip_download=args.skip_download,
        raw_dir=args.raw_dir,
    )

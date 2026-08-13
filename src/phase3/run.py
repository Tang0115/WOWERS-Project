"""Phase 3 orchestrator — Turbine Sizing via USGS 3DEP.

Usage:
    python -m src.phase3.run                        # full run
    python -m src.phase3.run --skip-elevation       # reuse cached elevation_data.parquet
    python -m src.phase3.run --top-n 500            # only process top-N ranked candidates
    python -m src.phase3.run --phase2-input PATH    # custom Phase 2 output path

Pipeline steps
--------------
[1/4]  Load Phase 2 ranked candidates (monte_carlo_results.parquet or
       ranked_candidates.parquet if Phase 2 is not yet run).
[2/4]  Query USGS 3DEP elevation for each facility lat/lon; cache results.
[3/4]  Estimate net head (H_gross × loss factor; validate vs literature bounds).
[4/4]  Select turbine type and optimise rated power (MW) + annual energy (MWh/yr).

Outputs (data/processed/phase3/)
---------------------------------
elevation_data.parquet         — raw 3DEP elevations
head_estimates.parquet         — facilities with head_net_m, head_source, head_valid
turbine_sizing.parquet         — final Phase 3 output (one row per facility)

All output files also get a versioned checkpoint via io.checkpoint().
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.common import config, io, logging_setup
from src.phase3 import elevation, head_estimation, outfall_coords, turbine_selection

logging_setup.setup_run_log("phase3")
log = logging_setup.get("wowers.phase3")

OUTPUT_DIR: Path = config.processed_dir() / "phase3"

# ── Input paths ───────────────────────────────────────────────────────────────
# Phase 3 always uses Phase 1 ranked_candidates as its primary input because
# Phase 1 contains the columns Phase 3 needs: lat/lon, mean_flow_mgd,
# design_flow_mgd, flow_duration_curve.  Phase 2 energy_yield_estimates is
# used only to pre-filter Phase 2-excluded facilities (no usable flow data).
_PHASE1_CANDIDATES = config.processed_dir() / "phase1" / "ranked_candidates.parquet"
_PHASE2_ENERGY     = config.processed_dir() / "phase2" / "energy_yield_estimates.parquet"


def _find_input_parquet(override: Path | None) -> Path:
    """Locate Phase 1 ranked_candidates.parquet (or a CLI override)."""
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--phase2-input path not found: {override}")
        return override
    if _PHASE1_CANDIDATES.exists():
        return _PHASE1_CANDIDATES
    raise FileNotFoundError(
        "Phase 3 requires Phase 1 output. "
        f"Expected at: {_PHASE1_CANDIDATES}\n"
        "Run Phase 1 first: python -m src.phase1.run"
    )


def run(
    phase2_input: Path | None = None,
    skip_elevation: bool = False,
    top_n: int | None = None,
) -> Path:
    """Run the full Phase 3 pipeline.

    Args:
        phase2_input:    Path to Phase 1/2 output parquet. Auto-detected if None.
        skip_elevation:  If True, load elevation_data.parquet from OUTPUT_DIR
                         instead of querying the USGS API.
        top_n:           Limit to top-N facilities by rank (useful for testing).

    Returns:
        Path to turbine_sizing.parquet.
    """
    t0 = time.time()

    log.info("=" * 60)
    log.info("WOWERS Phase 3 — Turbine Sizing via USGS 3DEP")
    log.info(f"  Output dir: {OUTPUT_DIR}")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load candidates ───────────────────────────────────────────────
    log.info("[1/4] Loading candidate facilities …")
    input_path = _find_input_parquet(phase2_input)
    candidates = io.read_parquet(input_path)

    if top_n is not None:
        if "rank" not in candidates.columns:
            raise ValueError(
                f"--top-n requires a 'rank' column in Phase 1 ranked_candidates.parquet; "
                f"available columns: {candidates.columns}"
            )
        candidates = candidates.sort("rank").head(top_n)
        log.info(f"  Limited to top {top_n} facilities (--top-n)")

    log.info(f"  Loaded {len(candidates):,} facilities from {input_path.name}")

    # Pre-filter: remove facilities that Phase 2 excluded (no usable flow data).
    # Also join head_m_p50 from Phase 2 so head_estimation.py can use literature
    # head as a plausibility bound for the USGS 3DEP-derived head.
    if _PHASE2_ENERGY.exists():
        p2_df = io.read_parquet(_PHASE2_ENERGY)
        p2_excluded = p2_df.filter(pl.col("excluded")).select("npdes_id")
        before = len(candidates)
        candidates = candidates.join(p2_excluded, on="npdes_id", how="anti")
        removed = before - len(candidates)
        if removed:
            log.info(f"  Pre-filtered {removed:,} Phase 2-excluded facilities (no usable flow)")

        # Join Phase 2 head percentile columns onto Phase 1 candidates so that
        # head_estimation.estimate_head() can use head_m_p50 as a validation bound
        # for 3DEP-derived head (instead of falling back to design_fallback for all).
        head_cols = [c for c in ("head_m_p10", "head_m_p50", "head_m_p90")
                     if c in p2_df.columns]
        if head_cols:
            p2_head = p2_df.select(["npdes_id"] + head_cols)
            candidates = candidates.join(p2_head, on="npdes_id", how="left")
            n_with_head = candidates["head_m_p50"].drop_nulls().len()
            log.info(f"  Joined Phase 2 head estimates: {n_with_head:,} facilities have head_m_p50")
        else:
            log.warning(
                "  Phase 2 output lacks head_m_p50 column — head estimation will use "
                "design_fallback (5 m) for all sites. Re-run Phase 2 to fix."
            )
    else:
        log.warning(
            "  Phase 2 output not found — cannot pre-filter excluded facilities. "
            f"Expected at: {_PHASE2_ENERGY}"
        )

    # ── Step 1b: Load outfall coordinates from NPDES_PERM_FEATURE_COORDS.csv ──
    log.info("[1b/4] Loading NPDES outfall coordinates …")
    outfall_df = outfall_coords.load_primary_outfall_coords(
        candidates["npdes_id"].to_list()
    )
    if len(outfall_df) > 0:
        candidates = candidates.join(outfall_df, on="npdes_id", how="left")
        n_with_outfall = candidates["lat_outfall"].drop_nulls().len()
        log.info(
            f"  Outfall coords: {n_with_outfall:,}/{len(candidates):,} facilities "
            "have outfall lat/lon (enables real 3DEP head calculation)"
        )
    else:
        log.warning(
            "  No outfall coordinates loaded — Phase 3 will use literature-based "
            "head for all sites.  Place NPDES_PERM_FEATURE_COORDS.csv at "
            f"{outfall_coords.OUTFALL_COORDS_PATH} to enable 3DEP head."
        )
        for col in ("lat_outfall", "lon_outfall"):
            if col not in candidates.columns:
                candidates = candidates.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(col)
                )

    # Ensure facility lat/lon columns exist with consistent names
    candidates = _normalise_coordinates(candidates)

    # ── Step 2: Elevation queries ─────────────────────────────────────────────
    log.info("[2/4] Querying USGS 3DEP elevations …")
    elev_out_path        = OUTPUT_DIR / "elevation_data.parquet"
    outfall_elev_path    = OUTPUT_DIR / "outfall_elevation_data.parquet"

    if skip_elevation and elev_out_path.exists():
        log.info("  --skip-elevation: loading cached elevation_data.parquet")
        elev_df = pl.read_parquet(elev_out_path)
        if "elevation_m" not in candidates.columns:
            candidates = candidates.join(
                elev_df.select(["npdes_id", "elevation_m", "elev_source"]),
                on="npdes_id", how="left",
            )
    else:
        if skip_elevation and not elev_out_path.exists():
            log.warning(
                "  --skip-elevation requested but elevation_data.parquet not found at "
                f"{elev_out_path}; running USGS 3DEP API queries."
            )
        candidates = elevation.fetch_elevations(candidates)
        io.write_parquet(candidates, elev_out_path)
        io.checkpoint(candidates, "phase3", "elevation_data")

    # Outfall elevations — separate API batch for discharge-point coordinates
    if skip_elevation and outfall_elev_path.exists():
        log.info("  --skip-elevation: loading cached outfall_elevation_data.parquet")
        outfall_elev_df = pl.read_parquet(outfall_elev_path)
        if "elev_outfall_m" not in candidates.columns:
            candidates = candidates.join(
                outfall_elev_df.select(["npdes_id", "elev_outfall_m"]),
                on="npdes_id", how="left",
            )
    else:
        outfall_points = (
            candidates
            .filter(
                pl.col("lat_outfall").is_not_null()
                & pl.col("lon_outfall").is_not_null()
            )
            .select(["npdes_id", "lat_outfall", "lon_outfall"])
            .rename({"lat_outfall": "latitude", "lon_outfall": "longitude"})
        )
        if len(outfall_points) > 0:
            log.info(
                f"  Querying outfall elevations for {len(outfall_points):,} sites "
                "(this unlocks real 3DEP head calculations) …"
            )
            outfall_elev_raw = elevation.fetch_elevations(outfall_points)
            outfall_elev_df = (
                outfall_elev_raw
                .select(["npdes_id", "elevation_m"])
                .rename({"elevation_m": "elev_outfall_m"})
            )
            io.write_parquet(outfall_elev_df, outfall_elev_path)
        else:
            log.info("  No outfall coordinates available — skipping outfall elevation queries")
            outfall_elev_df = pl.DataFrame({
                "npdes_id":      pl.Series([], dtype=pl.Utf8),
                "elev_outfall_m": pl.Series([], dtype=pl.Float64),
            })

        if len(outfall_elev_df) > 0 and "elev_outfall_m" not in candidates.columns:
            candidates = candidates.join(
                outfall_elev_df, on="npdes_id", how="left"
            )
            n_outfall_elev = candidates["elev_outfall_m"].drop_nulls().len()
            log.info(
                f"  Outfall elevations obtained for {n_outfall_elev:,} sites → "
                "3DEP head calculation now active for those sites"
            )

    # ── Step 3: Head estimation ───────────────────────────────────────────────
    log.info("[3/4] Estimating net head …")
    candidates = head_estimation.estimate_head(candidates)
    io.write_parquet(candidates, OUTPUT_DIR / "head_estimates.parquet")
    io.checkpoint(candidates, "phase3", "head_estimates")

    # ── Step 4: Turbine selection and sizing ──────────────────────────────────
    log.info("[4/4] Selecting turbine types and sizing rated power …")
    candidates = turbine_selection.select_and_size_turbines(candidates)

    out_path = OUTPUT_DIR / "turbine_sizing.parquet"
    io.write_parquet(candidates, out_path)
    io.checkpoint(candidates, "phase3", "turbine_sizing")

    elapsed = time.time() - t0
    _print_summary(candidates, elapsed)

    return out_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure lat/lon columns are named 'latitude' and 'longitude'."""
    rename_map: dict[str, str] = {}
    for src, dst in [
        ("GEOCODE_LATITUDE",  "latitude"),
        ("GEOCODE_LONGITUDE", "longitude"),
        ("lat",               "latitude"),
        ("lon",               "longitude"),
        ("lng",               "longitude"),
    ]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(rename_map)
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return df


def _print_summary(df: pl.DataFrame, elapsed: float) -> None:
    total = len(df)
    viable = df.filter(pl.col("turbine_viable")).shape[0] if "turbine_viable" in df.columns else 0
    head_3dep = (df["head_source"] == "usgs_3dep").sum() if "head_source" in df.columns else 0
    head_lit  = (df["head_source"] == "phase2_literature").sum() if "head_source" in df.columns else 0
    total_mwh = (
        df.filter(pl.col("turbine_viable"))["annual_energy_mwh"].drop_nulls().sum()
        if "annual_energy_mwh" in df.columns else 0.0
    )
    avg_kw = (
        df.filter(pl.col("turbine_viable"))["rated_power_kw"].drop_nulls().mean()
        if "rated_power_kw" in df.columns else 0.0
    )

    log.info("")
    log.info("=" * 60)
    log.info("Phase 3 Complete")
    log.info(f"  Total facilities processed:  {total:,}")
    log.info(f"  Viable turbine sites:         {viable:,}  ({viable/total*100:.1f}%)")
    log.info(f"  Head from USGS 3DEP:          {head_3dep:,}")
    log.info(f"  Head from literature:         {head_lit:,}")
    log.info(f"  Estimated total energy:       {total_mwh:,.0f} MWh/yr")
    log.info(f"  Average rated power:          {avg_kw:.0f} kW")
    log.info(f"  Runtime:                      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"  Output: {OUTPUT_DIR / 'turbine_sizing.parquet'}")
    log.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOWERS Phase 3: Turbine Sizing")
    parser.add_argument(
        "--phase2-input",
        type=Path,
        default=None,
        help="Path to Phase 1/2 output parquet (auto-detected if omitted)",
    )
    parser.add_argument(
        "--skip-elevation",
        action="store_true",
        help="Skip USGS API queries; reuse elevation_data.parquet from output dir",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only process top-N ranked facilities (for testing)",
    )
    args = parser.parse_args()

    run(
        phase2_input=args.phase2_input,
        skip_elevation=args.skip_elevation,
        top_n=args.top_n,
    )

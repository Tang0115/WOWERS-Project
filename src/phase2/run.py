"""Phase 2 orchestrator — Monte Carlo Energy Yield Estimation.

Usage:
    python -m src.phase2.run                          # full run
    python -m src.phase2.run --top-n 500              # limit to top-N ranked facilities
    python -m src.phase2.run --iterations 1000        # fewer MC iterations (fast test)
    python -m src.phase2.run --workers 4              # parallel processing
    python -m src.phase2.run --seed 99                # override random seed

Pipeline steps
--------------
[1/3]  Load Phase 1 ranked_candidates.parquet.
[2/3]  Run Monte Carlo energy estimation (10,000 iterations per facility).
[3/3]  Save energy_yield_estimates.parquet.

Output: data/processed/phase2/energy_yield_estimates.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from src.common import config, io, logging_setup
from src.phase2.monte_carlo import estimate_all_facilities

logging_setup.setup_run_log("phase2")
log = logging_setup.get("wowers.phase2")

OUTPUT_DIR: Path = config.processed_dir() / "phase2"
_PHASE1_OUTPUT:  Path = config.processed_dir() / "phase1" / "ranked_candidates.parquet"


def run(
    phase1_input:  Path | None = None,
    top_n:         int | None = None,
    n_iterations:  int = 10_000,
    n_workers:     int = 1,
    seed:          int = 42,
) -> Path:
    """Run the full Phase 2 pipeline.

    Args:
        phase1_input:  Path to ranked_candidates.parquet.  Auto-detects if None.
        top_n:         Limit to top-N ranked facilities (useful for testing).
        n_iterations:  Monte Carlo samples per facility.
        n_workers:     Parallel worker processes (1 = single-threaded).
        seed:          Base random seed.  Per-facility seeds are derived from
                       (seed, npdes_id) via site-keyed seeding, so removing or
                       inserting rows never changes draws for other facilities.

    Returns:
        Path to energy_yield_estimates.parquet.
    """
    t0 = time.time()
    input_path = phase1_input or _PHASE1_OUTPUT

    log.info("=" * 60)
    log.info("WOWERS Phase 2 — Monte Carlo Energy Yield Estimation")
    log.info(f"  Input:       {input_path}")
    log.info(f"  Output dir:  {OUTPUT_DIR}")
    log.info(f"  Iterations:  {n_iterations:,}")
    log.info(f"  Workers:     {n_workers}")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Phase 1 output ───────────────────────────────────────────
    log.info("[1/3] Loading Phase 1 candidates …")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Phase 1 output not found: {input_path}\n"
            "Run Phase 1 first: python -m src.phase1.run"
        )

    candidates = io.read_parquet(input_path)
    log.info(f"  Loaded {len(candidates):,} facilities")

    if top_n is not None:
        sort_col = "rank" if "rank" in candidates.columns else candidates.columns[0]
        candidates = candidates.sort(sort_col).head(top_n)
        log.info(f"  Limited to top {top_n} facilities (--top-n)")

    # ── Step 2: Monte Carlo estimation ───────────────────────────────────────
    log.info("[2/3] Running Monte Carlo energy estimation …")
    results_df = estimate_all_facilities(
        candidates,
        n_iterations=n_iterations,
        n_workers=n_workers,
        seed=seed,
    )

    # ── Step 3: Save output ───────────────────────────────────────────────────
    log.info("[3/3] Saving energy_yield_estimates.parquet …")
    out_path = OUTPUT_DIR / "energy_yield_estimates.parquet"
    io.write_parquet(results_df, out_path)
    io.checkpoint(results_df, "phase2", "energy_yield_estimates")

    elapsed = time.time() - t0
    _print_summary(results_df, elapsed)

    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pl.DataFrame, elapsed: float) -> None:
    total = len(df)
    estimated = df.filter(~pl.col("excluded")).shape[0] if "excluded" in df.columns else total
    excluded  = total - estimated

    if "energy_p50_kwh_yr" in df.columns:
        valid_energy = df.filter(
            pl.col("energy_p50_kwh_yr").is_not_null()
        )["energy_p50_kwh_yr"]
        national_gwh = valid_energy.sum() / 1e6  # kWh → GWh
        median_kwh   = valid_energy.median()
        top10_gwh    = (
            df.filter(pl.col("energy_p50_kwh_yr").is_not_null())
            .sort("energy_p50_kwh_yr", descending=True)
            .head(10)["energy_p50_kwh_yr"]
            .sum() / 1e6
        )
    else:
        national_gwh = median_kwh = top10_gwh = 0.0

    log.info("")
    log.info("=" * 60)
    log.info("Phase 2 Complete")
    log.info(f"  Total facilities:         {total:,}")
    log.info(f"  Estimated (not excluded): {estimated:,}")
    log.info(f"  Excluded (no usable data):{excluded:,}")
    log.info(f"  National P50 energy:      {national_gwh:,.1f} GWh/yr")
    log.info(f"  Median facility P50:      {median_kwh:,.0f} kWh/yr" if median_kwh else "")
    log.info(f"  Top-10 facilities:        {top10_gwh:,.2f} GWh/yr")
    log.info(f"  Runtime:                  {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log.info(f"  Output: {OUTPUT_DIR / 'energy_yield_estimates.parquet'}")
    log.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOWERS Phase 2: Energy Yield Estimation")
    parser.add_argument("--phase1-input", type=Path, default=None,
                        help="Path to Phase 1 ranked_candidates.parquet")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only process top-N ranked facilities")
    parser.add_argument("--iterations", type=int, default=10_000,
                        help="Monte Carlo iterations per facility (default 10,000)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default 42)")
    args = parser.parse_args()

    run(
        phase1_input=args.phase1_input,
        top_n=args.top_n,
        n_iterations=args.iterations,
        n_workers=args.workers,
        seed=args.seed,
    )

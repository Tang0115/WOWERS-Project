"""Phase 2 — Per-facility Monte Carlo runner.

Processes all facilities from Phase 1 ``ranked_candidates.parquet``, applies
the pre-Phase-2 exclusion filter (zero/null mean flow, sparse DMR data), and
runs the Monte Carlo energy estimation for each.

Performance
-----------
With vectorised NumPy, 10,000 iterations per facility completes in ~10 ms per
facility.  For 15,000 facilities, the total wall-clock time is roughly 2–3
minutes single-threaded, or under 30 seconds with parallelisation.

Parallelisation uses ``concurrent.futures.ProcessPoolExecutor`` (process-based
to bypass the GIL for NumPy workloads).  Each worker receives a batch of
facilities and processes them sequentially, returning a list of result dicts.
"""

from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import polars as pl

from src.common import config
from src.phase2.energy_physics import run_monte_carlo
from src.phase2.head_assumptions import classify_archetype, get_head_distribution

log = logging.getLogger("wowers.phase2.monte_carlo")

# W13: minimum mean flow threshold — sites below this are not viable for micro-hydro
_MIN_VIABLE_FLOW_MGD: float = float(
    config.get("phase2.min_viable_mean_flow_mgd", 0.5)
)

# Schema key list — exclusion rows are built from this to prevent silent drift
# if the success-path dict gains new fields.
_OUTPUT_KEYS: tuple[str, ...] = (
    "npdes_id", "archetype",
    "head_assumed_low_m", "head_assumed_mode_m", "head_assumed_high_m",
    "energy_p10_kwh_yr", "energy_p50_kwh_yr", "energy_p90_kwh_yr",
    "energy_mean_kwh_yr", "energy_std_kwh_yr",
    "power_p50_kw", "capacity_factor_p50",
    "head_m_p10", "head_m_p50", "head_m_p90",
    "equivalent_homes_p50", "excluded", "exclusion_reason",
)

# ── Site-keyed seeding ───────────────────────────────────────────────────────

def _site_seed_sequence(base_seed: int, npdes_id: str) -> np.random.SeedSequence:
    """Per-facility seed independent of row order and row count.

    Uses sha256 (not builtin hash(), which is process-salted and changes between
    runs) so the draw for a given (base_seed, npdes_id) pair is reproducible
    across runs, processes, and Python versions.  Removing or inserting rows in
    the input does not change the seed — or the Monte Carlo draws — for any other
    facility.

    Background: positional seeding (``seed + row_index``) meant that removing
    WI0025194 from 17,158 → 17,148 facilities on 2026-07-06 shifted seeds for
    all facilities after WI0025194's old position, re-drawing 1,090 scorecard
    rows and flipping 3 viability calls.  Site-keyed seeding eliminates this
    class of bug entirely.

    Args:
        base_seed: CLI ``--seed`` (default 42).  Changing this deliberately
            re-draws all facilities (one-time re-baseline).
        npdes_id:  NPDES permit number identifying the facility.

    Returns:
        ``np.random.SeedSequence`` that can be passed to ``np.random.default_rng``.
    """
    site_key = int.from_bytes(
        hashlib.sha256(npdes_id.encode("utf-8")).digest()[:8], "big"
    )
    return np.random.SeedSequence([base_seed, site_key])


# ── Exclusion filter (pre-Phase-2) ────────────────────────────────────────────

def _exclude(row: dict) -> str | None:
    """Return an exclusion reason string, or None if the row is usable.

    Checks in priority order:
      1. no_usable_flow  — None, NaN, zero, or negative mean flow
      2. small_potw      — mean flow below W13 viability threshold
      3. sparse_dmr_artifact — dmr_limited with < 3 months of data
    """
    mean_flow = row.get("mean_flow_mgd")
    # isinstance guard handles None and non-numeric; `> 0` rejects NaN (NaN > 0 == False)
    if not (isinstance(mean_flow, (int, float)) and mean_flow > 0):
        return "no_usable_flow"
    if mean_flow < _MIN_VIABLE_FLOW_MGD:
        return "small_potw"
    dq = row.get("data_quality", "design_only")
    n_months = row.get("n_months_data", 99)
    if dq == "dmr_limited" and (n_months is None or n_months < 3):
        return "sparse_dmr_artifact"
    return None


# ── Single-facility processing (called in worker processes) ───────────────────

def _process_one(row: dict, n_iterations: int, base_seed: int) -> dict:
    """Run Monte Carlo for one facility row.  Returns output dict.

    The RNG seed is derived from ``(base_seed, npdes_id)`` via
    ``_site_seed_sequence`` so that draws are independent of row position.
    Removing or inserting other facilities never changes this facility's draws.
    """
    exclusion = _exclude(row)
    npdes_id = row["npdes_id"]

    if exclusion is not None:
        return {k: None for k in _OUTPUT_KEYS} | {
            "npdes_id":         npdes_id,
            "excluded":         True,
            "exclusion_reason": exclusion,
        }

    design_flow = row.get("design_flow_mgd")
    archetype = classify_archetype(design_flow)
    head_dist = get_head_distribution(archetype)

    # Build FDC array (20-point, MGD)
    raw_fdc = row.get("flow_duration_curve")
    if raw_fdc is not None and len(raw_fdc) >= 2:
        fdc_flows = np.asarray(raw_fdc, dtype=np.float64)
    else:
        # Fall back to mean flow as a flat FDC
        mean_flow = float(row["mean_flow_mgd"])
        fdc_flows = np.full(20, mean_flow, dtype=np.float64)

    rng = np.random.default_rng(_site_seed_sequence(base_seed, npdes_id))
    mc = run_monte_carlo(
        fdc_flows_mgd=fdc_flows,
        head_low_m=head_dist.low_m,
        head_mode_m=head_dist.mode_m,
        head_high_m=head_dist.high_m,
        n_iterations=n_iterations,
        rng=rng,
    )

    energy_p50_kwh = mc["energy_p50_kwh_yr"]
    avg_us_household_kwh = 10_500  # EIA 2023 average
    equiv_homes = int(round(energy_p50_kwh / avg_us_household_kwh)) if energy_p50_kwh > 0 else 0

    return {
        "npdes_id":             npdes_id,
        "archetype":            archetype,
        "head_assumed_low_m":   head_dist.low_m,
        "head_assumed_mode_m":  head_dist.mode_m,
        "head_assumed_high_m":  head_dist.high_m,
        **mc,
        "equivalent_homes_p50": equiv_homes,
        "excluded":             False,
        "exclusion_reason":     None,
    }


def _process_batch(rows: list[dict], n_iterations: int, base_seed: int) -> list[dict]:
    """Process a batch of facility rows.  Used in worker processes.

    Each facility's seed is derived from ``(base_seed, npdes_id)`` — independent
    of batch offset or row index.  All batches use the same ``base_seed`` so
    sequential and parallel paths produce identical outputs.
    """
    results = []
    for row in rows:
        results.append(_process_one(row, n_iterations, base_seed))
    return results


# ── Public batch runner ───────────────────────────────────────────────────────

def estimate_all_facilities(
    candidates: pl.DataFrame,
    n_iterations: int = 10_000,
    n_workers: int = 1,
    seed: int = 42,
    log_interval: int = 1_000,
) -> pl.DataFrame:
    """Run Monte Carlo for all facilities.

    Args:
        candidates:   Phase 1 ``ranked_candidates.parquet`` loaded as a Polars
                      DataFrame.  Must contain columns: ``npdes_id``,
                      ``mean_flow_mgd``, ``design_flow_mgd``,
                      ``flow_duration_curve``, optionally ``data_quality`` and
                      ``n_months_data``.
        n_iterations: MC samples per facility (default 10,000).
        n_workers:    Parallel worker processes.  1 = single-threaded.
        seed:         Base random seed.  Each facility's RNG is derived from
                      ``(seed, npdes_id)`` via ``_site_seed_sequence`` — independent
                      of row position.  Removing or inserting rows never changes
                      draws for other facilities.  Change ``seed`` deliberately to
                      trigger a one-time fleet-wide re-baseline.
        log_interval: Log a progress message every N facilities.

    Returns:
        Polars DataFrame with one row per facility (all 15,000+, including
        excluded ones flagged with ``excluded=True``).
    """
    rows: list[dict[str, Any]] = candidates.to_dicts()
    n_total = len(rows)
    log.info(f"Phase 2 Monte Carlo: {n_total:,} facilities × {n_iterations:,} iterations")

    if n_workers <= 1:
        results: list[dict] = []
        for i, row in enumerate(rows):
            results.append(_process_one(row, n_iterations, seed))
            if (i + 1) % log_interval == 0 or (i + 1) == n_total:
                n_exc = sum(1 for r in results if r["excluded"])
                log.info(f"  Processed {i + 1:,}/{n_total:,} | excluded: {n_exc:,}")
    else:
        # Split into batches, one per worker. All batches share the same base_seed;
        # per-facility seeding is done inside _process_one via _site_seed_sequence.
        batch_size = math.ceil(n_total / n_workers)
        batches = [
            (rows[i: i + batch_size], n_iterations, seed)
            for i in range(0, n_total, batch_size)
        ]
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {
                exe.submit(_process_batch, *b): idx
                for idx, b in enumerate(batches)
            }
            for fut in as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                log.info(f"  Batch complete: {len(results):,}/{n_total:,} processed")

    log.info(f"  Done. {sum(1 for r in results if not r['excluded']):,} facilities estimated.")

    return pl.DataFrame(results)

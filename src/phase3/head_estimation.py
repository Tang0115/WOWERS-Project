"""Phase 3 — Net Head Estimation from USGS 3DEP Elevation Data.

Head is the vertical drop that drives the turbine.  For a wastewater
treatment plant the relevant measurement is the difference between the
facility (treatment-plant) elevation and the outfall (discharge point)
elevation, reduced by pipeline friction and fitting losses.

Gross head   H_gross = facility_elevation_m - outfall_elevation_m

Net head     H_net   = H_gross × (1 - head_loss_fraction)

where head_loss_fraction = 0.15 (configurable) accounts for pipe friction,
valves, bends, and screen losses in the penstock/effluent conduit.

For this pipeline the "facility elevation" comes from the USGS 3DEP API
query using the NPDES permit lat/lon (treatment plant centroid).  We do
NOT yet have a separate outfall coordinate so we substitute two proxies
ranked by data availability:

Priority 1  Outfall elevation from NPDES Outfalls Layer (future data).
Priority 2  Head archetype from Phase 2 (literature-derived range used as
            validation bound and fallback for failed API queries).
Priority 3  Facility elevation alone via terrain slope proxy (∆z = 0 → use
            Phase 2 literature head).

In practice for Phase 3, Phase 1/2 already computed `head_m_p50` from a
Monte Carlo over literature ranges.  We override that with the 3DEP-derived
head when it is available and physically plausible (H_net ≥ min_net_head_m
AND |H_3dep - H_literature| < 2× H_literature, i.e. not wildly inconsistent).

Output schema added to the input DataFrame
------------------------------------------
head_gross_m       float   raw elevation difference (or None)
head_net_m         float   after loss fraction
head_source        str     'usgs_3dep' | 'phase2_literature' | 'design_fallback'
head_valid         bool    True if head_net_m ≥ min_net_head_m
head_confidence    str     'high' | 'medium' | 'low'
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase3.head_estimation")

# ── Config ────────────────────────────────────────────────────────────────────

def _cfg(key: str, default=None):
    return config.get(f"phase3.{key}", default)


HEAD_LOSS_FRACTION: float = _cfg("head_loss_fraction", 0.15)
MIN_NET_HEAD_M: float = _cfg("min_net_head_m", 1.0)
# Gross head assumed when neither 3DEP elevation nor literature head is
# available.  Represents the median gravity-fed POTW in the US literature.
DESIGN_FALLBACK_GROSS_M: float = _cfg("design_fallback_head_gross_m", 5.0)

# Sanity: reject 3DEP-derived head if it differs from Phase-2 literature head
# by more than this multiplier (catches cases where API returned wrong point)
_MAX_DIVERGENCE_RATIO: float = 4.0


# ── Head source classifiers ───────────────────────────────────────────────────

def _classify_confidence(source: str, head_net_m: float | None) -> str:
    if head_net_m is None:
        return "low"
    if source == "usgs_3dep":
        return "high"
    if source == "phase2_literature":
        return "medium"
    return "low"


# ── Core computation (operates row-by-row via Polars struct map) ──────────────

def _compute_head_row(
    elev_facility_m: float | None,
    elev_outfall_m: float | None,
    head_p50_literature_m: float | None,
) -> tuple[float | None, float | None, str, bool, str]:
    """Return (head_gross_m, head_net_m, source, valid, confidence)."""
    gross: float | None = None
    net: float | None = None
    source = "phase2_literature"

    # ── Path 1: Both elevations available ──────────────────────────────────
    have_both_elevations = (
        elev_facility_m is not None and elev_outfall_m is not None
    )
    if have_both_elevations:
        candidate_gross = elev_facility_m - elev_outfall_m
        candidate_net = candidate_gross * (1.0 - HEAD_LOSS_FRACTION)

        # Negative head means the outfall is higher than the facility.
        # This is physically impossible for a gravity outfall — the API
        # returned a bad point.  Fall through to literature / fallback.
        if candidate_net <= 0:
            pass  # fall through below

        # Sub-minimum but positive head: the data is valid and tells us this
        # site genuinely lacks head.  Trust it and mark invalid immediately —
        # do NOT replace it with a 5 m design assumption.
        elif candidate_net < MIN_NET_HEAD_M:
            gross = candidate_gross
            net = candidate_net
            source = "usgs_3dep"
            return gross, net, source, False, "high"

        else:
            # Positive head above minimum: apply plausibility gate only here.
            # No need to check divergence for negative-head or sub-minimum
            # paths — those are already handled above.
            plausible = True
            if head_p50_literature_m is not None and head_p50_literature_m > 0:
                ratio = abs(candidate_net - head_p50_literature_m) / head_p50_literature_m
                if ratio > _MAX_DIVERGENCE_RATIO:
                    plausible = False  # wildly inconsistent — fall back below

            if plausible:
                gross = candidate_gross
                net = candidate_net
                source = "usgs_3dep"

    # ── Path 2: Only facility elevation — no outfall elevation yet ──────────
    # In Phase 3 we don't yet have separate outfall coordinates from NPDES
    # Outfalls Layer, so fall through to literature head.

    # ── Path 3: Use Phase-2 literature head ────────────────────────────────
    if net is None and head_p50_literature_m is not None:
        net = head_p50_literature_m * (1.0 - HEAD_LOSS_FRACTION)
        gross = head_p50_literature_m  # treat literature value as gross proxy
        source = "phase2_literature"

    # ── Path 4: Design-flow-based static default ───────────────────────────
    # Only reached when we have NO elevation data and NO literature estimate.
    if net is None:
        net = DESIGN_FALLBACK_GROSS_M * (1.0 - HEAD_LOSS_FRACTION)
        gross = DESIGN_FALLBACK_GROSS_M
        source = "design_fallback"

    valid = net is not None and net >= MIN_NET_HEAD_M
    confidence = _classify_confidence(source, net)
    return gross, net, source, valid, confidence


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_head(facilities: pl.DataFrame) -> pl.DataFrame:
    """Compute net head for each facility.

    Args:
        facilities: DataFrame from elevation.fetch_elevations() joined with
                    Phase 2 output.  Expected columns:
                      - elevation_m          (facility 3DEP elevation, may be null)
                      - elev_outfall_m       (outfall elevation, may be null / absent)
                      - head_m_p50           (Phase-2 Monte Carlo P50 head, may be null)

    Returns:
        Input DataFrame with additional columns:
            head_gross_m, head_net_m, head_source, head_valid, head_confidence
    """
    required = {"npdes_id", "elevation_m"}
    missing = required - set(facilities.columns)
    if missing:
        raise ValueError(f"estimate_head: missing columns {missing}")

    # Ensure optional columns exist with nulls if absent
    df = facilities
    for col, dtype in [
        ("elev_outfall_m", pl.Float64),
        ("head_m_p50", pl.Float64),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

    # ── Map row-by-row (Python loop is fine at 15k rows) ───────────────────
    head_gross_list: list[float | None] = []
    head_net_list: list[float | None] = []
    source_list: list[str] = []
    valid_list: list[bool] = []
    confidence_list: list[str] = []

    for row in df.select(["elevation_m", "elev_outfall_m", "head_m_p50"]).to_dicts():
        gross, net, src, valid, conf = _compute_head_row(
            row["elevation_m"],
            row["elev_outfall_m"],
            row["head_m_p50"],
        )
        head_gross_list.append(gross)
        head_net_list.append(net)
        source_list.append(src)
        valid_list.append(valid)
        confidence_list.append(conf)

    result = df.with_columns([
        pl.Series("head_gross_m",    head_gross_list,    dtype=pl.Float64),
        pl.Series("head_net_m",      head_net_list,      dtype=pl.Float64),
        pl.Series("head_source",     source_list,        dtype=pl.Utf8),
        pl.Series("head_valid",      valid_list,         dtype=pl.Boolean),
        pl.Series("head_confidence", confidence_list,    dtype=pl.Utf8),
    ])

    n_usgs     = (result["head_source"] == "usgs_3dep").sum()
    n_lit      = (result["head_source"] == "phase2_literature").sum()
    n_fallback = (result["head_source"] == "design_fallback").sum()
    n_viable   = result["head_valid"].sum()
    log.info(
        f"Head estimation: {n_usgs:,} 3DEP | {n_lit:,} literature | "
        f"{n_fallback:,} fallback | {n_viable:,} viable sites"
    )
    return result


def head_summary_stats(df: pl.DataFrame) -> dict:
    """Return a dict of summary statistics for logging / reporting."""
    valid = df.filter(pl.col("head_valid"))
    h = valid["head_net_m"].drop_nulls()
    return {
        "n_viable": len(valid),
        "head_p10_m": float(h.quantile(0.10)) if len(h) else None,
        "head_p50_m": float(h.quantile(0.50)) if len(h) else None,
        "head_p90_m": float(h.quantile(0.90)) if len(h) else None,
        "head_mean_m": float(h.mean()) if len(h) else None,
    }

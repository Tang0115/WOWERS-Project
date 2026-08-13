"""Compute per-facility statistical flow features from DMR time series."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase1.flow_features")

# 20-point standard exceedance probabilities for the FDC
FDC_PROBS: list[float] = config.get("ranking.fdc_exceedance_probs") or [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]

MIN_MONTHS = config.get("processing.min_months_for_features", 12)
MAX_FLOW_MGD = config.get("processing.max_flow_mgd_sanity", 2000.0)
_DMR_DESIGN_RATIO_CAP: float = config.get("processing.dmr_design_ratio_cap", 5.0)


def compute_flow_features(
    timeseries: pl.DataFrame,
    potw_facilities: pl.DataFrame,
) -> pl.DataFrame:
    """Compute flow features for every POTW that has DMR data.

    For POTWs with no DMR data, fall back to design_flow_mgd.

    Args:
        timeseries: Output of dmr_timeseries.parse_all_dmr_years()
        potw_facilities: Output of filter_potw.load_potw_facilities()

    Returns:
        One row per NPDES ID with all flow feature columns.
    """
    log.info("Computing flow features …")

    # Aggregate DMR data per facility (use primary outfall = max avg flow outfall)
    primary = _select_primary_outfall(timeseries)

    # Join design_flow_mgd into the timeseries so we can filter per-reading outliers
    # before computing statistics.  This prevents a single bogus month (e.g. 300 MGD
    # on a 0.19 MGD plant) from inflating the mean and triggering the ratio guard
    # against all 70+ legitimate months.
    design_lookup = potw_facilities.select(["npdes_id", "design_flow_mgd"])
    primary = primary.join(design_lookup, on="npdes_id", how="left")

    # Compute features via Polars + numpy (per group)
    feature_rows: list[dict] = []
    npdes_ids_with_data: set[str] = set()
    n_outlier_readings_dropped: int = 0

    for (npdes_id,), group in primary.group_by(["npdes_id"]):
        flows = group["avg_flow_mgd"].drop_nulls().to_numpy()
        if len(flows) == 0:
            continue

        # Per-reading outlier filter: drop individual months that exceed cap × design.
        # This handles the "single bad reading inflates mean" failure mode.
        # Sites where ALL readings fail (true unit-error like TN0056545) produce
        # len(flows)==0 after filtering → skipped → fall back to design_flow later.
        design_val = group["design_flow_mgd"][0]
        if design_val is not None and design_val > 0:
            cap = float(design_val) * _DMR_DESIGN_RATIO_CAP
            n_before = len(flows)
            flows = flows[flows <= cap]
            dropped = n_before - len(flows)
            if dropped:
                log.debug(
                    "  %s: dropped %d reading(s) > %.1f× design (%.3f MGD cap)",
                    npdes_id, dropped, _DMR_DESIGN_RATIO_CAP, cap,
                )
                n_outlier_readings_dropped += dropped

        if len(flows) == 0:
            continue

        npdes_ids_with_data.add(npdes_id)
        feature_rows.append(_compute_for_facility(npdes_id, flows, group))

    if n_outlier_readings_dropped:
        log.info(
            "  Dropped %d individual DMR readings exceeding %.0f× design flow "
            "(single-month unit errors)",
            n_outlier_readings_dropped, _DMR_DESIGN_RATIO_CAP,
        )

    dmr_features = pl.DataFrame(feature_rows) if feature_rows else _empty_features_df()
    log.info(f"  DMR features computed for {len(dmr_features):,} facilities")

    # Merge with facility metadata (design_flow, location, etc.)
    result = potw_facilities.join(dmr_features, on="npdes_id", how="left")

    # For facilities with NO DMR data: fall back to design_flow as the only flow signal
    result = _fill_missing_with_design_flow(result)

    log.info(
        f"  Total facilities with features: {len(result):,} "
        f"({len(dmr_features):,} from DMR, "
        f"{len(result) - len(dmr_features):,} design-flow fallback)"
    )
    return result


# ── Per-facility computation ──────────────────────────────────────────────────

def _compute_for_facility(
    npdes_id: str,
    flows: np.ndarray,
    group: pl.DataFrame,
) -> dict:
    # Sanity cap: DMR records sometimes contain unit errors (e.g. GPD filed as MGD).
    # Largest known US POTW (MWRDGC Stickney) is ~1,200 MGD; cap at MAX_FLOW_MGD.
    n_capped = int(np.sum(flows > MAX_FLOW_MGD))
    if n_capped:
        log.warning(
            f"  {npdes_id}: capping {n_capped} flow reading(s) above {MAX_FLOW_MGD} MGD "
            f"(max was {float(flows.max()):,.0f} MGD) — likely unit error in DMR data"
        )
        flows = np.clip(flows, 0.0, MAX_FLOW_MGD)

    n = len(flows)
    mean_f = float(np.mean(flows))
    std_f = float(np.std(flows, ddof=1)) if n > 1 else 0.0
    cv = std_f / mean_f if mean_f > 0 else 0.0

    # Fiscal years with data
    fiscal_years = group["fiscal_year"].drop_nulls().unique().to_list()
    n_years = len(fiscal_years)

    # Expected months: from earliest to latest period_end
    dates = group["period_end"].drop_nulls().sort()
    if len(dates) >= 2:
        from_date = dates[0]
        to_date = dates[-1]
        expected_months = max(
            1,
            (to_date.year - from_date.year) * 12 + (to_date.month - from_date.month) + 1,
        )
    else:
        expected_months = n
    pct_missing = max(0.0, (expected_months - n) / expected_months)

    # Seasonal amplitude: mean flow per calendar month, then max-min
    if n >= MIN_MONTHS:
        monthly_means = (
            group.with_columns(
                pl.col("period_end").dt.month().alias("month")
            )
            .group_by("month")
            .agg(pl.col("avg_flow_mgd").mean())
            .filter(pl.col("avg_flow_mgd").is_not_null())
        )
        if len(monthly_means) >= 2:
            month_vals = monthly_means["avg_flow_mgd"].to_numpy()
            seasonal_amplitude = float(np.max(month_vals) - np.min(month_vals))
        else:
            seasonal_amplitude = 0.0
    else:
        seasonal_amplitude = 0.0

    # Linear trend (MGD per year)
    flow_trend = 0.0
    if n >= 6:
        # Time index in fractional years
        dates_list = group["period_end"].drop_nulls().to_list()
        flows_for_trend = group["avg_flow_mgd"].drop_nulls().to_numpy()
        if len(dates_list) == len(flows_for_trend) >= 6:
            t = np.array([d.year + d.month / 12.0 for d in dates_list])
            # Guard: scipy.stats.linregress raises ValueError when all x-values are
            # identical (e.g. a facility whose 6+ DMR records all share the same
            # period_end — duplicate submissions or a single batch-reported period).
            # In that case we cannot estimate a trend; leave flow_trend = 0.0.
            if np.unique(t).size >= 2:
                slope, _, _, _, _ = stats.linregress(t, flows_for_trend)
                flow_trend = float(slope)

    # Flow duration curve
    fdc = _compute_fdc(flows)

    return {
        "npdes_id": npdes_id,
        "mean_flow_mgd": mean_f,
        "median_flow_mgd": float(np.median(flows)),
        "std_flow_mgd": std_f,
        "cv_flow": cv,
        "p10_flow_mgd": float(np.percentile(flows, 10)),
        "p25_flow_mgd": float(np.percentile(flows, 25)),
        "p75_flow_mgd": float(np.percentile(flows, 75)),
        "p90_flow_mgd": float(np.percentile(flows, 90)),
        "min_flow_mgd": float(np.min(flows)),
        "max_flow_mgd": float(np.max(flows)),
        "n_months_data": n,
        "n_years_data": n_years,
        "pct_missing": pct_missing,
        "flow_trend_mgd_per_year": flow_trend,
        "seasonal_amplitude_mgd": seasonal_amplitude,
        "flow_duration_curve": fdc,
        "data_quality": "dmr" if n >= MIN_MONTHS else "dmr_limited",
    }


def _compute_fdc(flows: np.ndarray) -> list[float]:
    """Compute flow duration curve at standard exceedance probabilities.

    Returns flow values at each exceedance probability in FDC_PROBS.
    A value at exceedance probability p means flow is exceeded p*100% of the time.
    """
    if len(flows) == 0:
        return [0.0] * len(FDC_PROBS)
    sorted_desc = np.sort(flows)[::-1]
    n = len(sorted_desc)
    # Weibull plotting positions
    exceedance = np.arange(1, n + 1) / (n + 1)
    # Interpolate at our standard probabilities
    fdc = np.interp(FDC_PROBS, exceedance, sorted_desc)
    return [float(v) for v in fdc]


# ── Outfall selection ─────────────────────────────────────────────────────────

def _select_primary_outfall(timeseries: pl.DataFrame) -> pl.DataFrame:
    """For facilities with multiple outfalls, keep only the primary one.

    Primary = outfall with the most non-null monthly avg_flow records
    (non-storm/CSO outfalls preferred). Mean is used only as tiebreaker.

    This prevents a single corrupted high-value row on a rarely-reported
    outfall from beating dozens of clean records on the real treatment outfall
    (the old max-mean rule's failure mode — fixed April 2026).
    """
    if "outfall" not in timeseries.columns:
        return timeseries

    # CSO/storm outfall codes to deprioritise (push to back of sort)
    cso_prefixes = ("C", "S", "E")  # CSO, storm, emergency overflow

    outfall_stats = (
        timeseries
        .group_by(["npdes_id", "outfall"])
        .agg([
            pl.col("avg_flow_mgd").drop_nulls().len().alias("n_nonnull"),
            pl.col("avg_flow_mgd").mean().alias("outfall_mean"),
        ])
        .with_columns(
            # Flag probable CSO/storm outfalls so they sort last
            pl.col("outfall")
              .str.to_uppercase()
              .str.slice(0, 1)
              .is_in(list(cso_prefixes))
              .cast(pl.Int8)
              .alias("is_cso")
        )
        # Sort: non-CSO first, then by most non-null records, then by mean as tiebreaker
        .sort(["is_cso", "n_nonnull", "outfall_mean"], descending=[False, True, True])
        .unique(subset=["npdes_id"], keep="first")
        .select(["npdes_id", "outfall"])
    )

    return timeseries.join(outfall_stats, on=["npdes_id", "outfall"], how="inner")


# ── Fallback for missing DMR data ─────────────────────────────────────────────

def _fill_missing_with_design_flow(df: pl.DataFrame) -> pl.DataFrame:
    """For rows with no DMR data, fill mean_flow from design_flow."""
    has_dmr = pl.col("mean_flow_mgd").is_not_null()

    df = df.with_columns([
        # mean_flow: use design_flow or actual_avg_flow as proxy
        pl.when(has_dmr)
          .then(pl.col("mean_flow_mgd"))
          .when(pl.col("actual_avg_flow_mgd").is_not_null())
          .then(pl.col("actual_avg_flow_mgd"))
          .otherwise(pl.col("design_flow_mgd") * 0.75)  # assume 75% utilization
          .alias("mean_flow_mgd"),

        # data_quality flag
        pl.when(has_dmr)
          .then(pl.col("data_quality"))
          .when(pl.col("actual_avg_flow_mgd").is_not_null())
          .then(pl.lit("actual_avg_only"))
          .otherwise(pl.lit("design_only"))
          .alias("data_quality"),

        # n_months / n_years: 0 if no DMR
        pl.when(pl.col("n_months_data").is_not_null())
          .then(pl.col("n_months_data"))
          .otherwise(pl.lit(0))
          .alias("n_months_data"),

        pl.when(pl.col("n_years_data").is_not_null())
          .then(pl.col("n_years_data"))
          .otherwise(pl.lit(0))
          .alias("n_years_data"),

        pl.when(pl.col("pct_missing").is_not_null())
          .then(pl.col("pct_missing"))
          .otherwise(pl.lit(1.0))
          .alias("pct_missing"),
    ])

    # Safety-net: null all flow stats if mean_flow still exceeds ratio cap after
    # per-reading filtering (catches sites where design_flow_mgd was null at
    # filter time, or where ALL readings were bogus — TN0056545-style unit errors).
    # After the upstream per-reading filter this should fire for very few sites.
    suspicious_dmr = (
        pl.col("mean_flow_mgd").is_not_null()
        & pl.col("design_flow_mgd").is_not_null()
        & (pl.col("design_flow_mgd") > 0)
        & (pl.col("mean_flow_mgd") > pl.col("design_flow_mgd") * _DMR_DESIGN_RATIO_CAP)
    )
    n_suspicious = df.filter(suspicious_dmr).height
    if n_suspicious > 0:
        log.warning(
            "Nulling mean_flow_mgd for %d rows where mean_flow > %.0f× design_flow "
            "(residual DMR reporting artifacts after per-reading filter)",
            n_suspicious,
            _DMR_DESIGN_RATIO_CAP,
        )
    # Null out all flow stats derived from the same bogus DMR readings so the
    # Phase 1 parquet is not misleading for excluded sites (e.g. TN0056545 with
    # median=807 MGD, FDC=[939, …] on a 0.023 MGD plant).
    _null_cols = [
        "mean_flow_mgd", "median_flow_mgd", "std_flow_mgd", "cv_flow",
        "p10_flow_mgd", "p25_flow_mgd", "p75_flow_mgd", "p90_flow_mgd",
        "min_flow_mgd", "max_flow_mgd", "flow_duration_curve",
    ]
    df = df.with_columns([
        pl.when(suspicious_dmr)
        .then(pl.lit(None))
        .otherwise(pl.col(c))
        .alias(c)
        for c in _null_cols
        if c in df.columns
    ])

    # Compute utilization ratio where possible
    df = df.with_columns([
        pl.when(
            pl.col("mean_flow_mgd").is_not_null()
            & pl.col("design_flow_mgd").is_not_null()
            & (pl.col("design_flow_mgd") > 0)
        )
        .then(pl.col("mean_flow_mgd") / pl.col("design_flow_mgd"))
        .otherwise(pl.lit(None))
        .alias("utilization_ratio"),
    ])

    return df


# ── Empty schema ──────────────────────────────────────────────────────────────

def _empty_features_df() -> pl.DataFrame:
    return pl.DataFrame({
        "npdes_id": pl.Series([], dtype=pl.Utf8),
        "mean_flow_mgd": pl.Series([], dtype=pl.Float64),
        "median_flow_mgd": pl.Series([], dtype=pl.Float64),
        "std_flow_mgd": pl.Series([], dtype=pl.Float64),
        "cv_flow": pl.Series([], dtype=pl.Float64),
        "p10_flow_mgd": pl.Series([], dtype=pl.Float64),
        "p25_flow_mgd": pl.Series([], dtype=pl.Float64),
        "p75_flow_mgd": pl.Series([], dtype=pl.Float64),
        "p90_flow_mgd": pl.Series([], dtype=pl.Float64),
        "min_flow_mgd": pl.Series([], dtype=pl.Float64),
        "max_flow_mgd": pl.Series([], dtype=pl.Float64),
        "n_months_data": pl.Series([], dtype=pl.Int32),
        "n_years_data": pl.Series([], dtype=pl.Int32),
        "pct_missing": pl.Series([], dtype=pl.Float64),
        "flow_trend_mgd_per_year": pl.Series([], dtype=pl.Float64),
        "seasonal_amplitude_mgd": pl.Series([], dtype=pl.Float64),
        "flow_duration_curve": pl.Series([], dtype=pl.List(pl.Float64)),
        "data_quality": pl.Series([], dtype=pl.Utf8),
    })

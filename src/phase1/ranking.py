"""Compute composite ranking score for POTW turbine installation candidates."""

from __future__ import annotations

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase1.ranking")

# Default weights — overrideable via settings.yaml
_DEFAULT_WEIGHTS = {
    "mean_flow": 0.35,
    "flow_consistency": 0.20,   # normalised (1 - cv_flow)
    "utilization": 0.15,
    "data_quality": 0.10,       # proxy via n_years_data
    "p10_flow": 0.20,
}

MAX_UTIL = config.get("processing.max_utilization_ratio", 2.0)


def compute_ranking(df: pl.DataFrame) -> pl.DataFrame:
    """Add `ranking_score` [0,1] and `rank` (1 = best) columns to df.

    Modifies nothing else. Returns a new DataFrame.
    """
    weights = _load_weights()
    log.info(f"Ranking {len(df):,} facilities with weights: {weights}")

    df = _add_normalised_components(df)
    df = _add_composite_score(df, weights)
    df = _add_rank(df)

    top5 = df.sort("rank").head(5)
    for row in top5.iter_rows(named=True):
        name = row.get("facility_name") or row.get("npdes_id") or ""
        log.info(
            f"  #{row['rank'] or 0:4d} {name[:50]:<50} "
            f"score={row['ranking_score'] or 0:.4f}  "
            f"mean_flow={row.get('mean_flow_mgd') or 0:.1f} MGD"
        )

    return df


# ── Internal ──────────────────────────────────────────────────────────────────

def _load_weights() -> dict[str, float]:
    cfg = config.get("ranking.weights") or {}
    w = dict(_DEFAULT_WEIGHTS)
    w.update({k: v for k, v in cfg.items() if k in w})
    # Re-normalise so weights always sum to 1.0
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _add_normalised_components(df: pl.DataFrame) -> pl.DataFrame:
    """Add _norm_ columns for each ranking component, scaled [0,1]."""

    def minmax(col: str, invert: bool = False) -> pl.Expr:
        """Min-max normalise a column. If invert=True, higher raw = lower score."""
        c = pl.col(col).cast(pl.Float64, strict=False)
        min_val = c.min()
        max_val = c.max()
        normalised = (c - min_val) / (max_val - min_val + 1e-12)
        return (1.0 - normalised) if invert else normalised

    # 1. mean_flow — higher is better.  Use a temporary ranking column so the
    #    F4 null signal in mean_flow_mgd is preserved in the output parquet.
    #    Null sites score 0 (correctly ranked last for flow).
    df = df.with_columns(
        pl.col("mean_flow_mgd").fill_null(0.0).alias("_mean_flow_for_ranking")
    )
    df = df.with_columns(minmax("_mean_flow_for_ranking").alias("_norm_mean_flow"))

    # 2. flow consistency — lower CV is better (invert cv_flow)
    #    Cap CV at 2.0 before normalising so extreme outliers don't dominate
    df = df.with_columns(
        pl.col("cv_flow").fill_null(1.0).clip(0.0, 2.0).alias("_cv_capped")
    ).with_columns(
        minmax("_cv_capped", invert=True).alias("_norm_consistency")
    )

    # 3. utilisation — higher is better, but cap at MAX_UTIL.
    #    fill_null(0.0) so sites with design_flow=0 or null don't get
    #    a neutral 0.5 mid-score — they should rank last on this dimension.
    df = df.with_columns(
        pl.col("utilization_ratio").fill_null(0.0).clip(0.0, MAX_UTIL).alias("_util_capped")
    ).with_columns(
        minmax("_util_capped").alias("_norm_utilization")
    )

    # 4. data quality — more years = better
    df = df.with_columns(
        pl.col("n_years_data").fill_null(0).cast(pl.Float64).alias("_years_f")
    ).with_columns(
        minmax("_years_f").alias("_norm_data_quality")
    )

    # 5. p10 flow — higher is better (reliable baseline)
    df = df.with_columns(
        pl.col("p10_flow_mgd").fill_null(0.0).clip(lower_bound=0.0).alias("_p10_capped")
    ).with_columns(
        minmax("_p10_capped").alias("_norm_p10")
    )

    return df


def _add_composite_score(df: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    score_expr = (
        pl.col("_norm_mean_flow")       * weights["mean_flow"]
        + pl.col("_norm_consistency")   * weights["flow_consistency"]
        + pl.col("_norm_utilization")   * weights["utilization"]
        + pl.col("_norm_data_quality")  * weights["data_quality"]
        + pl.col("_norm_p10")           * weights["p10_flow"]
    )
    df = df.with_columns(score_expr.alias("ranking_score"))

    # Zero ranking_score for facilities with design_flow_mgd = 0 or null.
    # These are misclassified industrial / data-incomplete sites that happen
    # to have high DMR flow (e.g. Boyd County KYP000044).  They are correctly
    # excluded at Phase 2 (sparse_dmr_artifact), but a zero ranking_score
    # prevents them from appearing at the top of any P1 visualisation or ML
    # feature table.
    if "design_flow_mgd" in df.columns:
        # Zero only explicit design_flow == 0 (industrial misclassification signal).
        # Null design_flow means missing permit data — may still be a real POTW,
        # so leave those in normal ranking rather than collateral-zeroing ~2,106 sites.
        df = df.with_columns(
            pl.when(
                pl.col("design_flow_mgd").is_not_null()
                & (pl.col("design_flow_mgd") <= 0.0)
            )
              .then(0.0)
              .otherwise(pl.col("ranking_score"))
              .alias("ranking_score")
        )

    # Drop the intermediate _norm_ and _capped columns
    temp_cols = [c for c in df.columns if c.startswith("_norm_") or c.endswith("_capped") or c in ("_cv_capped", "_util_capped", "_years_f", "_p10_capped", "_mean_flow_for_ranking")]
    return df.drop(temp_cols)


def _add_rank(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ranking_score")
          .rank(method="ordinal", descending=True)
          .cast(pl.Int32)
          .alias("rank")
    )

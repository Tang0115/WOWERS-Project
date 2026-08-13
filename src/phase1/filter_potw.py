"""Filter ICIS facilities and permits to active POTWs only."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase1.filter_potw")

# Hard upper bound on permit flow fields.  Values above this are almost
# certainly unit errors in EPA ECHO (e.g. GPD filed as MGD).
_MAX_FLOW_MGD: float = config.get("processing.max_flow_mgd_sanity", 2000.0)

# EPA ECHO uses 999 (or 999.0) as a sentinel for "no data" in numeric flow
# fields (TOTAL_DESIGN_FLOW_NMBR, ACTUAL_AVERAGE_FLOW_NMBR).  These must be
# treated as null rather than as real flow values.
_EPA_999_SENTINEL: float = 999.0

# If actual_avg_flow_mgd > N × design_flow_mgd it is almost certainly a
# units mismatch (e.g. DMR flow reported in GPD while design flow is MGD).
# Cap at configurable ratio; default 5×.
_DMR_DESIGN_RATIO_CAP: float = config.get("processing.dmr_design_ratio_cap", 5.0)

# P1-COORD-GUARD (2026-07-06): inclusive lat/lon validity bands for all US
# NPDES territories.  A row is valid only when BOTH coordinates fall within
# at least one of their respective bands.
#
# Design decisions (do not change without updating the journal):
#   REJECT, don't fix — longitude sign-flips and digit errors are not
#   auto-corrected even when the fix is obvious.  Auto-rescue without an
#   authoritative source creates hidden data divergence.  The 10 known
#   bad rows are simply removed; the probe-verified drop count is 10.
#
#   Naïve "lat < 15°" rejection is WRONG: Guam (~13.4°N), CNMI, and
#   American Samoa (~-14.3°S) all have active NPDES permits.  Use the
#   inclusive territory bands below instead.
#
# Latitude bands:
#   [-14.8, -10.8]  American Samoa (AS)
#   [ 13.0,  71.5]  Guam/CNMI → USVI/PR → Hawaii → CONUS → Alaska
# Longitude bands:
#   [-180.0, -64.4]  Western-hemisphere US (incl. USVI east tip ~-64.56)
#   [ 144.5, 146.2]  Guam / CNMI (Northern Mariana Islands)
_COORD_LAT_VALID_BANDS: list[list[float]] = config.get(
    "processing.coord_lat_valid_bands", [[-14.8, -10.8], [13.0, 71.5]]
)
_COORD_LON_VALID_BANDS: list[list[float]] = config.get(
    "processing.coord_lon_valid_bands", [[-180.0, -64.4], [144.5, 146.2]]
)

# Output schema — enforced after join
POTW_SCHEMA = {
    "npdes_id": pl.Utf8,
    "facility_name": pl.Utf8,
    "city": pl.Utf8,
    "state_code": pl.Utf8,
    "zip": pl.Utf8,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "facility_type_indicator": pl.Utf8,
    "facility_type_code": pl.Utf8,
    "major_minor": pl.Utf8,
    "design_flow_mgd": pl.Float64,
    "actual_avg_flow_mgd": pl.Float64,
    "permit_status_code": pl.Utf8,
}


def load_potw_facilities(
    facilities_csv: Path,
    permits_csv: Path,
) -> pl.DataFrame:
    """Read, filter, and join ICIS CSVs to produce the POTW facility table.

    Returns a DataFrame conforming to POTW_SCHEMA (~16,000 rows expected).
    """
    log.info("Loading ICIS_FACILITIES.CSV …")
    facilities = _load_facilities(facilities_csv)

    log.info("Loading ICIS_PERMITS.CSV …")
    permits = _load_permits(permits_csv)

    log.info(f"Raw facilities: {len(facilities):,} | Raw permits: {len(permits):,}")

    potw_permits = _filter_potw_permits(permits)
    log.info(f"POTW permits after filter: {len(potw_permits):,}")

    joined = _join(facilities, potw_permits)
    log.info(f"POTW facilities after join: {len(joined):,}")

    joined = _drop_invalid_coords(joined)   # P1-COORD-GUARD
    log.info(f"POTW facilities after coord guard: {len(joined):,}")

    result = _cast_schema(joined)
    _log_summary(result)
    return result


# ── Loading ───────────────────────────────────────────────────────────────────

def _load_facilities(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NULL", "null", "N/A"],
        ignore_errors=True,
        encoding="utf8-lossy",
    )
    # Normalise column names to uppercase
    df = df.rename({c: c.upper().strip() for c in df.columns})

    keep = {
        "NPDES_ID": "npdes_id",
        "FACILITY_TYPE_CODE": "facility_type_code",
        "FACILITY_NAME": "facility_name",
        "LOCATION_ADDRESS": "location_address",
        "CITY": "city",
        "STATE_CODE": "state_code",
        "ZIP": "zip",
        "GEOCODE_LATITUDE": "latitude",
        "GEOCODE_LONGITUDE": "longitude",
    }
    # Only keep columns that exist in the actual file
    rename_map = {k: v for k, v in keep.items() if k in df.columns}
    df = df.select(list(rename_map.keys())).rename(rename_map)

    return df.with_columns([
        pl.col("npdes_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    ])


def _load_permits(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NULL", "null", "N/A"],
        ignore_errors=True,
        encoding="utf8-lossy",
    )
    df = df.rename({c: c.upper().strip() for c in df.columns})

    keep = {
        "EXTERNAL_PERMIT_NMBR": "npdes_id",
        "FACILITY_TYPE_INDICATOR": "facility_type_indicator",
        "TOTAL_DESIGN_FLOW_NMBR": "design_flow_mgd",
        "ACTUAL_AVERAGE_FLOW_NMBR": "actual_avg_flow_mgd",
        "MAJOR_MINOR_STATUS_FLAG": "major_minor",
        "PERMIT_STATUS_CODE": "permit_status_code",
    }
    rename_map = {k: v for k, v in keep.items() if k in df.columns}
    df = df.select(list(rename_map.keys())).rename(rename_map)

    df = df.with_columns([
        pl.col("npdes_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("design_flow_mgd").cast(pl.Float64, strict=False),
        pl.col("actual_avg_flow_mgd").cast(pl.Float64, strict=False),
    ])

    # Null out EPA 999 sentinel values first.  EPA ECHO stores "no data"
    # as exactly 999.0 in numeric flow columns — these are NOT real flows.
    for col in ("design_flow_mgd", "actual_avg_flow_mgd"):
        if col in df.columns:
            n_sentinel = int((df[col] == _EPA_999_SENTINEL).sum())
            if n_sentinel:
                log.warning(
                    f"  Nulling {n_sentinel} permit row(s) where {col} == {_EPA_999_SENTINEL} "
                    f"(EPA ECHO sentinel for missing data)"
                )
                df = df.with_columns(
                    pl.when(pl.col(col) == _EPA_999_SENTINEL)
                    .then(pl.lit(None, dtype=pl.Float64))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Null out any flow values that exceed the sanity cap.
    # Values this large (e.g. 750,000 "MGD") are unit errors in EPA ECHO —
    # the field was likely populated in GPD or MLD instead of MGD.
    for col in ("design_flow_mgd", "actual_avg_flow_mgd"):
        if col in df.columns:
            n_bad = int((df[col] > _MAX_FLOW_MGD).sum())
            if n_bad:
                log.warning(
                    f"  Nulling {n_bad} permit row(s) where {col} > {_MAX_FLOW_MGD} MGD "
                    f"(max was {float(df[col].max()):,.0f} MGD) — probable unit error in EPA ECHO"
                )
                df = df.with_columns(
                    pl.when(pl.col(col) > _MAX_FLOW_MGD)
                    .then(pl.lit(None, dtype=pl.Float64))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Null out actual_avg_flow_mgd when it exceeds N × design_flow_mgd.
    # A ratio this extreme almost always means the two fields used different
    # units (e.g. DMR in GPD, design flow in MGD).  We null the suspicious
    # value rather than the design flow because design flow is more reliable.
    if "actual_avg_flow_mgd" in df.columns and "design_flow_mgd" in df.columns:
        cap_expr = pl.col("design_flow_mgd") * _DMR_DESIGN_RATIO_CAP
        suspicious = (
            pl.col("actual_avg_flow_mgd").is_not_null()
            & pl.col("design_flow_mgd").is_not_null()
            & (pl.col("design_flow_mgd") > 0)
            & (pl.col("actual_avg_flow_mgd") > cap_expr)
        )
        n_ratio_bad = int(df.filter(suspicious).shape[0])
        if n_ratio_bad:
            log.warning(
                f"  Nulling {n_ratio_bad} row(s) where actual_avg_flow_mgd > "
                f"{_DMR_DESIGN_RATIO_CAP}× design_flow_mgd "
                f"(probable units mismatch in DMR/ICIS data)"
            )
            df = df.with_columns(
                pl.when(suspicious)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col("actual_avg_flow_mgd"))
                .alias("actual_avg_flow_mgd")
            )

    return df


# ── Filtering ─────────────────────────────────────────────────────────────────

def _filter_potw_permits(permits: pl.DataFrame) -> pl.DataFrame:
    cfg_types = config.get("potw_filter.facility_type_indicators", ["POTW"])
    cfg_statuses = config.get("potw_filter.active_permit_status_codes", ["EFF", "NON"])
    cfg_exclude_codes = config.get("potw_filter.exclude_facility_type_codes", ["FDF"])

    # Normalise to uppercase
    permits = permits.with_columns(
        pl.col("facility_type_indicator").str.strip_chars().str.to_uppercase(),
        pl.col("permit_status_code").str.strip_chars().str.to_uppercase()
        if "permit_status_code" in permits.columns else pl.lit(None).alias("permit_status_code"),
    )

    # Filter: type must be in allowed list
    type_filter = pl.col("facility_type_indicator").is_in(
        [t.upper() for t in cfg_types]
    )

    # Filter: permit must be active (if column exists and is non-null)
    if "permit_status_code" in permits.columns:
        status_filter = (
            pl.col("permit_status_code").is_null()
            | pl.col("permit_status_code").is_in([s.upper() for s in cfg_statuses])
        )
        permits = permits.filter(type_filter & status_filter)
    else:
        permits = permits.filter(type_filter)

    # Deduplicate: keep one row per NPDES ID (largest design flow wins)
    if "design_flow_mgd" in permits.columns:
        permits = (
            permits.sort("design_flow_mgd", descending=True, nulls_last=True)
            .unique(subset=["npdes_id"], keep="first")
        )
    else:
        permits = permits.unique(subset=["npdes_id"], keep="first")

    return permits


# ── Join ──────────────────────────────────────────────────────────────────────

def _join(facilities: pl.DataFrame, permits: pl.DataFrame) -> pl.DataFrame:
    # Left join: facilities is the left table so we keep all POTW permits
    # even if facility record is missing (will have null location fields)
    joined = permits.join(facilities, on="npdes_id", how="left")

    # Drop rows with no location at all (can't be used for mapping or elevation)
    before = len(joined)
    joined = joined.filter(
        pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
    )
    dropped = before - len(joined)
    if dropped > 0:
        log.warning(f"Dropped {dropped:,} facilities with no coordinates")

    return joined


# ── Coordinate validity guard ─────────────────────────────────────────────────

def _in_any_band(value: pl.Expr, bands: list[list[float]]) -> pl.Expr:
    """Return a boolean Expr: True if ``value`` falls in ANY of ``bands``."""
    exprs = [
        (value >= lo) & (value <= hi)
        for lo, hi in bands
    ]
    result = exprs[0]
    for e in exprs[1:]:
        result = result | e
    return result


def _drop_invalid_coords(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows whose lat/lon fall outside any recognised US NPDES territory.

    US territories covered by the validity bands (see ``_COORD_LAT_VALID_BANDS`` /
    ``_COORD_LON_VALID_BANDS``):
        American Samoa (AS), Guam (GU), CNMI (MP), Puerto Rico (PR),
        US Virgin Islands (VI), Hawaii (HI), contiguous US, Alaska (AK).

    Rows are **rejected, not corrected**, even when the error is visually
    obvious (e.g. positive longitude = likely sign flip).  Auto-correction
    without an authoritative source creates hidden data divergence; the
    correct fix is to obtain a verified coordinate from EPA ECHO or a GIS
    source and re-ingest (see WOWERS_PROJECT_JOURNAL.md, 2026-07-06).

    Rows with null latitude or longitude are left for the existing null-drop
    in ``_join`` (called before this function in ``load_potw_facilities``).

    Args:
        df: Post-join, pre-schema-cast DataFrame containing ``latitude``,
            ``longitude``, ``npdes_id``, and ``state_code`` columns.

    Returns:
        df with invalid-coordinate rows removed.
    """
    lat_ok = _in_any_band(pl.col("latitude"),  _COORD_LAT_VALID_BANDS)
    lon_ok = _in_any_band(pl.col("longitude"), _COORD_LON_VALID_BANDS)

    # Treat null coords as "ok here" — null-drop handled upstream in _join
    valid_mask = (
        pl.col("latitude").is_null()
        | pl.col("longitude").is_null()
        | (lat_ok & lon_ok)
    )

    bad = df.filter(~valid_mask)
    n_bad = bad.height

    if n_bad > 0:
        log.warning(
            "P1-COORD-GUARD: dropping %d facilit%s with coordinates outside "
            "all US NPDES territory bands (reject-not-fix; see module docstring):",
            n_bad, "ies" if n_bad != 1 else "y",
        )
        detail_rows = bad.select(
            ["npdes_id", "state_code", "latitude", "longitude"]
        ).to_dicts()
        for row in detail_rows[:20]:   # cap at 20 lines of detail
            log.warning(
                "  %s  state=%s  lat=%.6f  lon=%.6f",
                row["npdes_id"], row["state_code"],
                row["latitude"], row["longitude"],
            )

    return df.filter(valid_mask)


# ── Schema cast ───────────────────────────────────────────────────────────────

def _cast_schema(df: pl.DataFrame) -> pl.DataFrame:
    casts = []
    for col, dtype in POTW_SCHEMA.items():
        if col in df.columns:
            casts.append(pl.col(col).cast(dtype, strict=False))
        else:
            casts.append(pl.lit(None).cast(dtype).alias(col))
    return df.with_columns(casts).select(list(POTW_SCHEMA.keys()))


# ── Summary ───────────────────────────────────────────────────────────────────

def _log_summary(df: pl.DataFrame) -> None:
    total = len(df)
    with_design_flow = df.filter(pl.col("design_flow_mgd").is_not_null()).shape[0]
    major = df.filter(pl.col("major_minor") == "M").shape[0]
    by_state = df.group_by("state_code").len().sort("len", descending=True)

    log.info(f"POTW facilities: {total:,} total")
    log.info(f"  With design flow: {with_design_flow:,} ({with_design_flow/total*100:.1f}%)")
    log.info(f"  Major facilities: {major:,}")
    log.info(f"  States covered:   {df['state_code'].n_unique()}")
    top5 = by_state.head(5)
    for row in top5.iter_rows(named=True):
        log.info(f"  {row['state_code']}: {row['len']:,} plants")

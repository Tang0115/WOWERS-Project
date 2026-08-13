"""Parse DMR ZIP files and extract monthly flow time series (parameter 50050)."""

from __future__ import annotations

import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Set

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase1.dmr")

FLOW_PARAMETER_CODE = "50050"

# Statistical base descriptors that map to avg / max / min
AVG_DESCS = {
    "MO AVG", "MONTHLY AVERAGE", "AVG", "AVERAGE", "MONTH AVG",
    # EPA STATISTICAL_BASE_CODE values
    "MK",  # Monthly Average
    "WA",  # Weekly Average
    "QA",  # Quarterly Average
    "YA",  # Yearly Average
    "VA",  # Annual Average
    "3C",  # 30-day average
    "MB",  # Monthly Average (alternate)
}
MAX_DESCS = {
    "MO MAX", "MONTHLY MAXIMUM", "MAX", "MAXIMUM", "DAILY MAX", "1 DAY MAX",
    # EPA STATISTICAL_BASE_CODE values
    "MX",  # Monthly Maximum
    "MP",  # Monthly Peak
    "DC",  # Daily Maximum
    "DD",  # Daily Maximum (alternate)
    "TB",  # Two-week Maximum
}
MIN_DESCS = {
    "MO MIN", "MONTHLY MINIMUM", "MIN", "MINIMUM", "DAILY MIN", "1 DAY MIN",
    # EPA STATISTICAL_BASE_CODE values
    "MN",  # Monthly Minimum
    "MS",  # Monthly Minimum (alternate)
}

# NODI codes
NODI_ZERO = set(config.get("nodi_codes.treat_as_zero", ["C", "B"]))
NODI_ESTIMATED = set(config.get("nodi_codes.treat_as_estimated", ["E"]))
# Everything else → None (missing)

# Known column name variants across fiscal years
_COL_ALIASES: dict[str, list[str]] = {
    "npdes_id": [
        "EXTERNAL_PERMIT_NMBR",
        "NPDES_ID",
        "PERMIT_ID",
    ],
    "outfall": [
        "PERM_FEATURE_NMBR",
        "OUTFALL_CODE",
        "FEATURE_NMBR",
    ],
    "period_end": [
        "MONITORING_PERIOD_END_DATE",
        "MONITORING_END_DATE",
        "PERIOD_END_DATE",
    ],
    "stat_base": [
        "STATISTICAL_BASE_SHORT_DESC",
        "STAT_BASE_SHORT_DESC",
        "STATISTICAL_BASE_CODE",
    ],
    "param_code": [
        "PARAMETER_CODE",
        "PARAM_CODE",
    ],
    "value": [
        # EPA pre-converts to standard units (always MGD for param 50050).
        # Prefer this over DMR_VALUE_NMBR, which is the raw reported value
        # in whatever unit the facility chose (gal/d, gal/min, Mgal/mo, etc.).
        "DMR_VALUE_STANDARD_UNITS",
        "DMR_VALUE_NMBR",
        "DMR_VALUE",
        "REPORTED_VALUE",
    ],
    "nodi": [
        "NODI_CODE",
        "NO_DATA_INDICATOR_CODE",
    ],
}


def parse_dmr_year(
    zip_path: Path,
    fiscal_year: int,
    potw_ids: Set[str],
    chunk_size: int = 500_000,
) -> pl.DataFrame:
    """Parse one DMR fiscal year ZIP. Returns monthly flow rows for POTW plants.

    Output schema:
        npdes_id       Utf8
        outfall        Utf8
        period_end     Date
        fiscal_year    Int32
        avg_flow_mgd   Float64   (may be null)
        max_flow_mgd   Float64
        min_flow_mgd   Float64
        is_estimated   Boolean
    """
    log.info(f"Parsing FY{fiscal_year} DMR: {zip_path.name}")

    chunks: list[pl.DataFrame] = []
    total_rows_read = 0
    flow_rows_kept = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = _find_csv_in_zip(zf, fiscal_year)
        log.info(f"  CSV inside ZIP: {csv_name}")

        with zf.open(csv_name) as raw_file:
            text_file = TextIOWrapper(raw_file, encoding="utf-8", errors="replace")

            reader = _chunked_reader(text_file, chunk_size)
            col_map: Optional[dict[str, str]] = None

            for chunk_df in reader:
                total_rows_read += len(chunk_df)

                if col_map is None:
                    col_map = _build_col_map(chunk_df.columns)
                    log.info(f"  Column mapping: {col_map}")

                processed = _process_chunk(chunk_df, col_map, potw_ids, fiscal_year)
                if len(processed) > 0:
                    chunks.append(processed)
                    flow_rows_kept += len(processed)

    log.info(
        f"  FY{fiscal_year}: read {total_rows_read:,} rows, "
        f"kept {flow_rows_kept:,} POTW flow rows"
    )

    if not chunks:
        log.warning(f"  FY{fiscal_year}: no flow rows found for any POTW")
        return _empty_timeseries_df()

    raw = pl.concat(chunks)
    pivoted = _pivot_to_monthly(raw, fiscal_year)
    log.info(f"  FY{fiscal_year}: {len(pivoted):,} facility-month records")
    return pivoted


def parse_all_dmr_years(
    dmr_zip_paths: dict[int, Path],
    potw_ids: Set[str],
) -> pl.DataFrame:
    """Parse all fiscal year DMR ZIPs. Concatenate and return full time series."""
    all_years: list[pl.DataFrame] = []

    for year in sorted(dmr_zip_paths.keys()):
        path = dmr_zip_paths[year]
        try:
            df = parse_dmr_year(path, year, potw_ids)
            if len(df) > 0:
                all_years.append(df)
        except Exception as exc:
            log.error(f"FY{year} parsing failed: {exc}")

    if not all_years:
        log.warning("No DMR data parsed across any fiscal year")
        return _empty_timeseries_df()

    combined = pl.concat(all_years)

    # Deduplicate: same (npdes_id, outfall, period_end) — keep row with non-null avg
    combined = (
        combined
        .sort("avg_flow_mgd", nulls_last=True)
        .unique(subset=["npdes_id", "outfall", "period_end"], keep="first")
        .sort(["npdes_id", "period_end"])
    )

    log.info(
        f"Full DMR time series: {len(combined):,} facility-month records "
        f"for {combined['npdes_id'].n_unique():,} facilities"
    )
    return combined


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_csv_in_zip(zf: zipfile.ZipFile, fiscal_year: int) -> str:
    names = zf.namelist()
    # Prefer the largest file (usually the main DMR CSV)
    csv_names = [n for n in names if n.upper().endswith(".CSV")]
    if not csv_names:
        raise FileNotFoundError(
            f"No CSV found in DMR ZIP for FY{fiscal_year}. Files: {names[:10]}"
        )
    # Sort by file size descending — the DMR data file is the biggest
    sizes = {n: zf.getinfo(n).file_size for n in csv_names}
    return max(sizes, key=lambda n: sizes[n])


def _chunked_reader(text_file: TextIOWrapper, chunk_size: int):
    """Yield polars DataFrames of chunk_size rows from a text file object."""
    import csv
    from io import StringIO

    reader_obj = csv.reader(text_file)
    header = next(reader_obj)

    buffer_rows: list[list[str]] = []
    for row in reader_obj:
        buffer_rows.append(row)
        if len(buffer_rows) >= chunk_size:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(header)
            writer.writerows(buffer_rows)
            buf.seek(0)
            yield pl.read_csv(
                buf.read().encode(),
                infer_schema_length=1000,
                null_values=["", "NULL", "null"],
                ignore_errors=True,
            )
            buffer_rows = []

    if buffer_rows:
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        writer.writerows(buffer_rows)
        buf.seek(0)
        yield pl.read_csv(
            buf.read().encode(),
            infer_schema_length=1000,
            null_values=["", "NULL", "null"],
            ignore_errors=True,
        )


def _build_col_map(columns: list[str]) -> dict[str, str]:
    """Map our canonical field names to actual column names in this file."""
    upper_cols = {c.upper().strip(): c for c in columns}
    result: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias.upper() in upper_cols:
                result[canonical] = upper_cols[alias.upper()]
                break
    return result


def _process_chunk(
    df: pl.DataFrame,
    col_map: dict[str, str],
    potw_ids: Set[str],
    fiscal_year: int,
) -> pl.DataFrame:
    """Filter chunk to POTW flow rows and normalise columns."""
    # We need at minimum: npdes_id, param_code, value
    required = ["npdes_id", "param_code", "value"]
    if not all(k in col_map for k in required):
        log.debug(f"  Missing required columns in col_map: {required}")
        return pl.DataFrame()

    # Rename to canonical names
    rename = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(rename)

    # Filter: parameter code = 50050 (flow)
    df = df.filter(
        pl.col("param_code").cast(pl.Utf8).str.strip_chars() == FLOW_PARAMETER_CODE
    )
    if len(df) == 0:
        return pl.DataFrame()

    # Filter: POTW facilities only
    df = df.filter(
        pl.col("npdes_id").cast(pl.Utf8).str.strip_chars().is_in(potw_ids)
    )
    if len(df) == 0:
        return pl.DataFrame()

    # Resolve value with NODI codes
    df = df.with_columns([
        pl.col("npdes_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("value").cast(pl.Float64, strict=False).alias("raw_value"),
        (pl.col("nodi").cast(pl.Utf8).str.strip_chars()
         if "nodi" in df.columns else pl.lit(None).cast(pl.Utf8))
        .alias("nodi_code"),
        (pl.col("stat_base").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
         if "stat_base" in df.columns else pl.lit("AVG").cast(pl.Utf8))
        .alias("stat_base"),
        (pl.col("outfall").cast(pl.Utf8).str.strip_chars()
         if "outfall" in df.columns else pl.lit("001").cast(pl.Utf8))
        .alias("outfall"),
        (pl.col("period_end").cast(pl.Utf8).str.strip_chars()
         if "period_end" in df.columns else pl.lit(None).cast(pl.Utf8))
        .alias("period_end_str"),
    ])

    # Apply NODI logic
    # Max plausible POTW flow: Stickney (Chicago) ≈ 1,440 MGD; cap at 2,000 MGD
    MAX_FLOW_MGD = 2_000.0

    df = df.with_columns([
        pl.when(pl.col("nodi_code").is_in(list(NODI_ZERO)))
          .then(pl.lit(0.0))
          .when(pl.col("nodi_code").is_in(list(NODI_ESTIMATED)))
          .then(pl.col("raw_value"))
          .otherwise(pl.col("raw_value"))
          .alias("flow_value"),
        pl.col("nodi_code").is_in(list(NODI_ESTIMATED)).alias("is_estimated"),
    ])

    # Null out physically impossible values (data entry errors in EPA system)
    df = df.with_columns([
        pl.when(pl.col("flow_value") > MAX_FLOW_MGD)
          .then(pl.lit(None))
          .otherwise(pl.col("flow_value"))
          .alias("flow_value"),
    ])

    # Parse period_end to date
    df = df.with_columns([
        pl.col("period_end_str")
          .str.to_date(format="%m/%d/%Y", strict=False)
          .alias("period_end"),
    ])

    return df.select([
        "npdes_id",
        "outfall",
        "period_end",
        "stat_base",
        "flow_value",
        "is_estimated",
    ]).filter(pl.col("period_end").is_not_null())


def _pivot_to_monthly(df: pl.DataFrame, fiscal_year: int) -> pl.DataFrame:
    """Pivot stat_base rows into avg/max/min columns per (facility, outfall, month)."""

    def _extract_stat(descs: set[str]) -> pl.Expr:
        return (
            pl.when(pl.col("stat_base").is_in(descs))
              .then(pl.col("flow_value"))
              .otherwise(pl.lit(None))
        )

    df = df.with_columns([
        _extract_stat(AVG_DESCS).alias("avg_val"),
        _extract_stat(MAX_DESCS).alias("max_val"),
        _extract_stat(MIN_DESCS).alias("min_val"),
        pl.col("is_estimated").cast(pl.Boolean),
    ])

    pivoted = (
        df.group_by(["npdes_id", "outfall", "period_end"])
          .agg([
              pl.col("avg_val").drop_nulls().first().alias("avg_flow_mgd"),
              pl.col("max_val").drop_nulls().first().alias("max_flow_mgd"),
              pl.col("min_val").drop_nulls().first().alias("min_flow_mgd"),
              pl.col("is_estimated").any().alias("is_estimated"),
          ])
          .with_columns(pl.lit(fiscal_year).cast(pl.Int32).alias("fiscal_year"))
          .sort(["npdes_id", "period_end"])
    )

    # If only non-avg rows exist for a facility-month, use any available value as avg
    pivoted = pivoted.with_columns([
        pl.when(pl.col("avg_flow_mgd").is_null())
          .then(
              pl.coalesce(["max_flow_mgd", "min_flow_mgd"])
          )
          .otherwise(pl.col("avg_flow_mgd"))
          .alias("avg_flow_mgd"),
    ])

    return pivoted


def _empty_timeseries_df() -> pl.DataFrame:
    return pl.DataFrame({
        "npdes_id": pl.Series([], dtype=pl.Utf8),
        "outfall": pl.Series([], dtype=pl.Utf8),
        "period_end": pl.Series([], dtype=pl.Date),
        "fiscal_year": pl.Series([], dtype=pl.Int32),
        "avg_flow_mgd": pl.Series([], dtype=pl.Float64),
        "max_flow_mgd": pl.Series([], dtype=pl.Float64),
        "min_flow_mgd": pl.Series([], dtype=pl.Float64),
        "is_estimated": pl.Series([], dtype=pl.Boolean),
    })

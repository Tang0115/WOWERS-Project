"""Phase 5 ground-truth ingestion — EIA-860 + EIA-923 + DOE HydroSource EHA.

Builds the labeled dataset the Phase 5 ML model trains against: real
US hydropower plants with known installed capacity and annual energy output.
See ``ARCHITECTURE.md`` §5.2–5.3 for the schema and the (later) spatial/fuzzy
matching strategy to ECHO POTW plants.

Sources
-------
**EIA** (``ingest_eia_year``):
  EIA Form 860 (generator inventory) + EIA Form 923 (generation), on disk
  under ``phase5.eia_data_dir``:
    EIA860_<year>/2___Plant_Y<year>.xlsx       plant lat/lon/name/state
    EIA860_<year>/3_1_Generator_Y<year>.xlsx   per-generator nameplate + prime mover
    EIA923_<year>/EIA923_Schedules_...xlsx     per-plant annual net generation

  Three EIA gotchas handled:
  1. Generator workbook first sheet is "Proposed"; we read "Operable".
  2. EIA workbooks have title rows: Form-860 header is row 2, Form-923 row 6.
  3. Form-923 has multiple rows per plant; filter to ``HY`` and sum.

**DOE HydroSource EHA** (``ingest_eha``):
  ORNL Existing Hydropower Assets (EHA) on disk under ``phase5.eha_data_dir``:
    ORNL_EHAHydroPlant_PublicFY2024.xlsx   plant inventory (sheet "Operational")
    EHA_Annual_CapacityFactor.xlsx          per-plant annual CF + net gen
                                            (sheet "AnnualCapacityFactor")

  EHA ingest notes:
  - Pumped storage excluded via ``CH_MW > 0`` (conventional hydro capacity
    column only; ``PS_MW`` rows are already absent from the HY-only CF file).
  - Energy label: ``Net_Generation_MWh`` from the CF workbook, converted to kWh.
    This is actual measured annual generation — preferred over the
    ``CH_MWh`` plant-file column (stored as String, 15% nulls) and over
    the CF × cap × 8760 proxy.
  - Default year: 2022 (latest in the CF file, range 2005–2022).
  - ``source_plant_code`` = ``EIA_PtID`` (Int64) from the plant file, enabling
    later cross-source deduplication with EIA labels. EHA's native string key
    ``EHA_PtID`` is preserved in the intermediate frame before schema projection;
    it is NOT added to CANONICAL_SCHEMA.

KNOWN BIAS — large hydro (both sources)
----------------------------------------
EIA-860 inventories generators ≥ ~1 MW; EHA covers plants ≥ 1 MW in the CF
file. Both ground-truth sets skew to utility-scale conventional hydro and are
**not representative of the WWTP micro-scale** sites WOWERS targets (< 1 MW).
Use these to train the physics relationship energy = f(capacity, head, flow);
supplement with FERC conduit exemption labels before trusting micro-scale
predictions. Future sources append to the same canonical schema via additional
ingest functions — do not change the schema to fit one source.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from src.common import config, io, logging_setup

logging_setup.setup_run_log("phase5")
log = logging_setup.get("wowers.phase5")

# ── Canonical ground-truth schema (source-agnostic) ──────────────────────────
# Defined once. FERC / DOE HydroSource ingests must emit these same columns.
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "ground_truth_source": pl.Utf8,        # "EIA", later "FERC", "EHA", ...
    "facility_name": pl.Utf8,              # for later fuzzy match to ECHO
    "state_code": pl.Utf8,                 # 2-letter, for later spatial match
    "latitude": pl.Float64,                # decimal degrees (nullable)
    "longitude": pl.Float64,               # decimal degrees (nullable)
    "actual_annual_energy_kwh": pl.Float64,  # TARGET label
    "actual_installed_kw": pl.Float64,     # secondary label
    "actual_head_m": pl.Float64,           # null for EIA (not reported)
    "actual_flow_m3s": pl.Float64,         # null for EIA (not reported)
    "source_plant_code": pl.Int64,         # EIA plant code — provenance / join key
    "source_year": pl.Int64,
}

HYDRO_PRIME_MOVER = "HY"  # conventional hydro; PS / HPS (pumped storage) excluded
MW_TO_KW = 1_000.0
MWH_TO_KWH = 1_000.0

OUTPUT_PATH: Path = config.raw_data_dir() / "ground_truth" / "eia_hydro_ground_truth.parquet"


# ── Column-name helpers (EIA names drift slightly across years) ───────────────
def _norm(name: str) -> str:
    """Lowercase, collapse whitespace/newlines, strip — for tolerant matching."""
    return re.sub(r"\s+", " ", str(name).replace("\n", " ")).strip().lower()


def _find_col(columns: list[str], *substrings: str) -> str:
    """Return the first column whose normalized name contains ALL substrings.

    Raises KeyError if none match, so a silent column rename in a future EIA
    vintage fails loudly rather than producing a wrong label.
    """
    subs = [s.lower() for s in substrings]
    for c in columns:
        n = _norm(c)
        if all(s in n for s in subs):
            return c
    raise KeyError(f"no column matching {substrings!r} in {columns!r}")


# ── Pure transforms (no IO — unit-tested on synthetic frames) ─────────────────
# These operate on STANDARDIZED column names produced by the IO layer:
#   generators: plant_code, prime_mover, nameplate_mw
#   generation: plant_code, prime_mover, net_gen_mwh
#   plants:     plant_code, plant_name, state_code, latitude, longitude
def aggregate_capacity_kw(generators: pl.DataFrame) -> pl.DataFrame:
    """Filter to hydro generators, sum nameplate MW to plant level, convert kW.

    Returns columns: plant_code, actual_installed_kw.
    """
    return (
        generators.filter(pl.col("prime_mover") == HYDRO_PRIME_MOVER)
        .group_by("plant_code")
        .agg((pl.col("nameplate_mw").cast(pl.Float64).sum() * MW_TO_KW).alias("actual_installed_kw"))
    )


def aggregate_generation_kwh(generation: pl.DataFrame) -> pl.DataFrame:
    """Filter to hydro rows, sum annual net generation MWh to plant level, to kWh.

    Returns columns: plant_code, actual_annual_energy_kwh.
    """
    return (
        generation.filter(pl.col("prime_mover") == HYDRO_PRIME_MOVER)
        .group_by("plant_code")
        .agg((pl.col("net_gen_mwh").cast(pl.Float64).sum() * MWH_TO_KWH).alias("actual_annual_energy_kwh"))
    )


def assemble_ground_truth(
    generators: pl.DataFrame,
    generation: pl.DataFrame,
    plants: pl.DataFrame,
    year: int,
) -> pl.DataFrame:
    """Join hydro capacity + generation + plant metadata into canonical schema.

    Drops plants missing a positive capacity OR a positive generation (cannot be
    a usable label). Returns a DataFrame with exactly CANONICAL_SCHEMA columns.
    """
    cap = aggregate_capacity_kw(generators)
    gen = aggregate_generation_kwh(generation)

    df = (
        cap.join(gen, on="plant_code", how="inner")
        .join(
            plants.select(
                "plant_code", "plant_name", "state_code", "latitude", "longitude"
            ),
            on="plant_code",
            how="left",
        )
        .filter(
            (pl.col("actual_installed_kw") > 0) & (pl.col("actual_annual_energy_kwh") > 0)
        )
        .with_columns(
            pl.lit("EIA").alias("ground_truth_source"),
            pl.col("plant_name").alias("facility_name"),
            pl.lit(None, dtype=pl.Float64).alias("actual_head_m"),
            pl.lit(None, dtype=pl.Float64).alias("actual_flow_m3s"),
            pl.col("plant_code").cast(pl.Int64).alias("source_plant_code"),
            pl.lit(year, dtype=pl.Int64).alias("source_year"),
        )
    )
    # Project to canonical column order + dtypes.
    return df.select(
        [
            pl.col(name).cast(dtype).alias(name)
            for name, dtype in CANONICAL_SCHEMA.items()
        ]
    )


# ── IO layer ──────────────────────────────────────────────────────────────────
def _resolve_eia_dir() -> Path:
    raw = config.get("phase5.eia_data_dir")
    if not raw:
        raise FileNotFoundError(
            "config phase5.eia_data_dir is not set — cannot locate EIA-860/923 data."
        )
    d = Path(raw)
    if not d.is_dir():
        raise FileNotFoundError(
            f"EIA data dir not mounted/found: {d}. Connect the external drive or "
            "override phase5.eia_data_dir in config/settings.yaml."
        )
    return d


def _latest_year(eia_dir: Path) -> int:
    """Newest year with BOTH an EIA860_<y> and EIA923_<y> folder present."""
    y860 = {
        int(m.group(1))
        for p in eia_dir.glob("EIA860_*")
        if (m := re.search(r"EIA860_(\d{4})", p.name))
    }
    y923 = {
        int(m.group(1))
        for p in eia_dir.glob("EIA923_*")
        if (m := re.search(r"EIA923_(\d{4})", p.name))
    }
    common = sorted(y860 & y923)
    if not common:
        raise FileNotFoundError(f"no matching EIA860_/EIA923_ year pair under {eia_dir}")
    return common[-1]


def _read_excel(path: Path, sheet: str, header_row: int) -> pl.DataFrame:
    """Read one EIA sheet (header_row is 0-based)."""
    return pl.read_excel(path, sheet_name=sheet, read_options={"header_row": header_row})


def _load_generators(eia_dir: Path, year: int) -> pl.DataFrame:
    path = next(eia_dir.glob(f"EIA860_{year}/3_1_Generator_Y{year}.xlsx"))
    raw = _read_excel(path, sheet="Operable", header_row=1)  # NB: "Operable", not sheet 0
    cols = raw.columns
    return raw.select(
        pl.col(_find_col(cols, "plant", "code")).cast(pl.Int64).alias("plant_code"),
        pl.col(_find_col(cols, "prime", "mover")).cast(pl.Utf8).str.strip_chars().alias("prime_mover"),
        pl.col(_find_col(cols, "nameplate", "capacity")).cast(pl.Float64, strict=False).alias("nameplate_mw"),
    )


def _load_plants(eia_dir: Path, year: int) -> pl.DataFrame:
    path = next(eia_dir.glob(f"EIA860_{year}/2___Plant_Y{year}.xlsx"))
    raw = _read_excel(path, sheet="Plant", header_row=1)
    cols = raw.columns
    return raw.select(
        pl.col(_find_col(cols, "plant", "code")).cast(pl.Int64).alias("plant_code"),
        pl.col(_find_col(cols, "plant", "name")).cast(pl.Utf8).alias("plant_name"),
        pl.col(_find_col(cols, "state")).cast(pl.Utf8).alias("state_code"),
        pl.col(_find_col(cols, "latitude")).cast(pl.Float64, strict=False).alias("latitude"),
        pl.col(_find_col(cols, "longitude")).cast(pl.Float64, strict=False).alias("longitude"),
    )


def _load_generation(eia_dir: Path, year: int) -> pl.DataFrame:
    path = next(eia_dir.glob(f"EIA923_{year}/EIA923_Schedules_2_3_4_5*{year}*.xlsx"))
    raw = _read_excel(path, sheet="Page 1 Generation and Fuel Data", header_row=5)
    cols = raw.columns
    return raw.select(
        pl.col(_find_col(cols, "plant", "id")).cast(pl.Int64).alias("plant_code"),
        pl.col(_find_col(cols, "reported", "prime", "mover")).cast(pl.Utf8).str.strip_chars().alias("prime_mover"),
        pl.col(_find_col(cols, "net generation", "megawatthours")).cast(pl.Float64, strict=False).alias("net_gen_mwh"),
    )


def ingest_eia_year(year: int | None = None) -> pl.DataFrame:
    """Load EIA-860 + EIA-923 for ``year`` (default: latest present) and return
    the canonical hydro ground-truth DataFrame. Reads the external drive."""
    eia_dir = _resolve_eia_dir()
    if year is None:
        year = config.get("phase5.eia_year") or _latest_year(eia_dir)
    log.info("Ingesting EIA hydro ground truth for year %d from %s", year, eia_dir)
    generators = _load_generators(eia_dir, year)
    plants = _load_plants(eia_dir, year)
    generation = _load_generation(eia_dir, year)
    return assemble_ground_truth(generators, generation, plants, year)


# ── Runner ─────────────────────────────────────────────────────────────────────
def _summary(df: pl.DataFrame) -> None:
    n = df.height
    log.info("EIA hydro ground-truth plants: %d", n)
    if n:
        cap = df["actual_installed_kw"]
        en = df["actual_annual_energy_kwh"]
        log.info("  installed_kw  range: %.0f – %.0f (median %.0f)", cap.min(), cap.max(), cap.median())
        log.info("  annual_kwh    range: %.3g – %.3g (median %.3g)", en.min(), en.max(), en.median())
        log.info("  total fleet GWh/yr: %.1f", en.sum() / 1e6)


def run(year: int | None = None) -> Path:
    df = ingest_eia_year(year)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    io.write_parquet(df, OUTPUT_PATH)
    _summary(df)
    log.info("Wrote %s", OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — ingest EIA hydro ground truth.")
    ap.add_argument("--year", type=int, default=None, help="EIA data year (default: latest present)")
    args = ap.parse_args()
    run(args.year)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# P5-EHA: DOE HydroSource Existing Hydropower Assets ground-truth ingest
# ══════════════════════════════════════════════════════════════════════════════

EHA_DEFAULT_YEAR: int = 2022   # latest year in the AnnualCapacityFactor workbook
EHA_OUTPUT_PATH: Path = config.raw_data_dir() / "ground_truth" / "eha_hydro_ground_truth.parquet"

# Workbook / sheet names (inspected 2026-06-21; stable across EHA vintages)
_EHA_PLANT_FILE  = "ORNL_EHAHydroPlant_PublicFY2024.xlsx"
_EHA_CF_FILE     = "EHA_Annual_CapacityFactor.xlsx"
_EHA_PLANT_SHEET = "Operational"
_EHA_CF_SHEET    = "AnnualCapacityFactor"
# Both workbooks use row 0 as the header (no title rows above the column names).
_EHA_HEADER_ROW  = 0

CONVENTIONAL_HYDRO_TYPE = "HY"   # CF workbook Type column; all 2022 rows are HY


# ── IO layer (EHA) ────────────────────────────────────────────────────────────

def _resolve_eha_dir() -> Path:
    """Return the EHA data directory, raising FileNotFoundError if absent/unmounted."""
    raw = config.get("phase5.eha_data_dir")
    if not raw:
        raise FileNotFoundError(
            "config phase5.eha_data_dir is not set — cannot locate EHA ORNL data."
        )
    d = Path(raw)
    if not d.is_dir():
        raise FileNotFoundError(
            f"EHA data dir not mounted/found: {d}. Connect the external drive or "
            "override phase5.eha_data_dir in config/settings.yaml."
        )
    return d


def _load_eha_plants(eha_dir: Path) -> pl.DataFrame:
    """Read the EHA plant inventory (Operational sheet) into standardised columns.

    Returns columns:
        eha_pt_id        — str, EHA native plant identifier (e.g. "hc0124_p02")
        eia_pt_id        — Int64, EIA plant code (→ source_plant_code in schema)
        pt_name          — str, plant name
        state_code       — str, 2-letter state
        latitude         — Float64
        longitude        — Float64
        ch_mw            — Float64, conventional-hydro MW (null/0 = pumped-storage only)
    """
    path = eha_dir / _EHA_PLANT_FILE
    if not path.exists():
        raise FileNotFoundError(f"EHA plant file not found: {path}")
    raw = _read_excel(path, sheet=_EHA_PLANT_SHEET, header_row=_EHA_HEADER_ROW)
    cols = raw.columns
    return raw.select(
        pl.col(_find_col(cols, "eha_pt")).cast(pl.Utf8).alias("eha_pt_id"),
        pl.col(_find_col(cols, "eia_pt")).cast(pl.Int64, strict=False).alias("eia_pt_id"),
        pl.col(_find_col(cols, "ptname")).cast(pl.Utf8).alias("pt_name"),
        pl.col(_find_col(cols, "state")).cast(pl.Utf8).alias("state_code"),
        pl.col(_find_col(cols, "lat")).cast(pl.Float64, strict=False).alias("latitude"),
        pl.col(_find_col(cols, "lon")).cast(pl.Float64, strict=False).alias("longitude"),
        # CH_MW = conventional-hydro nameplate MW only (PS_MW is the pumped-storage column)
        pl.col(_find_col(cols, "ch_mw")).cast(pl.Float64, strict=False).alias("ch_mw"),
    )


def _load_eha_cf(eha_dir: Path) -> pl.DataFrame:
    """Read the EHA annual capacity-factor workbook (all years) into standardised columns.

    Returns columns:
        eha_pt_id         — str, EHA plant identifier (join key)
        type              — str, turbine type ("HY" = conventional hydro)
        year              — Int64
        net_gen_mwh       — Float64, measured annual net generation (MWh)
    """
    path = eha_dir / _EHA_CF_FILE
    if not path.exists():
        raise FileNotFoundError(f"EHA capacity-factor file not found: {path}")
    raw = _read_excel(path, sheet=_EHA_CF_SHEET, header_row=_EHA_HEADER_ROW)
    cols = raw.columns
    return raw.select(
        pl.col(_find_col(cols, "eha_pt")).cast(pl.Utf8).alias("eha_pt_id"),
        pl.col(_find_col(cols, "type")).cast(pl.Utf8).alias("type"),
        pl.col(_find_col(cols, "year")).cast(pl.Int64, strict=False).alias("year"),
        pl.col(_find_col(cols, "net_generation")).cast(pl.Float64, strict=False).alias("net_gen_mwh"),
    )


# ── Pure transforms (EHA, no IO — unit-tested on synthetic frames) ────────────
# Operate on standardised columns produced by the IO layer above.

def eha_filter_capacity(plants: pl.DataFrame) -> pl.DataFrame:
    """Filter to conventional-hydro plants by keeping only ch_mw > 0.

    Drops rows with null or zero CH_MW (pumped-storage-only sites have no
    conventional-hydro capacity and must be excluded from the energy label).

    Returns the input frame filtered; column set unchanged.
    """
    return plants.filter(pl.col("ch_mw").is_not_null() & (pl.col("ch_mw") > 0))


def eha_capacity_kw(plants: pl.DataFrame) -> pl.DataFrame:
    """Compute actual_installed_kw from ch_mw (conventional-hydro MW → kW).

    Returns columns: eha_pt_id, eia_pt_id, pt_name, state_code, latitude,
    longitude, actual_installed_kw.
    """
    return eha_filter_capacity(plants).with_columns(
        (pl.col("ch_mw") * MW_TO_KW).alias("actual_installed_kw")
    ).drop("ch_mw")


def eha_energy_kwh(cf: pl.DataFrame, year: int) -> pl.DataFrame:
    """Filter CF to conventional hydro for ``year`` and sum net generation → kWh.

    Energy derivation: ``Net_Generation_MWh`` from the CF workbook (actual
    measured annual generation), converted to kWh. This is the preferred path
    over the CF × cap × 8760 proxy — the CF file has direct measurements.

    Returns columns: eha_pt_id, actual_annual_energy_kwh.
    """
    return (
        cf.filter(
            (pl.col("year") == year)
            & (pl.col("type") == CONVENTIONAL_HYDRO_TYPE)
        )
        .group_by("eha_pt_id")
        .agg(
            (pl.col("net_gen_mwh").cast(pl.Float64).sum() * MWH_TO_KWH)
            .alias("actual_annual_energy_kwh")
        )
    )


def assemble_eha_ground_truth(
    plants: pl.DataFrame,
    cf: pl.DataFrame,
    year: int,
) -> pl.DataFrame:
    """Join EHA capacity + CF energy + metadata into CANONICAL_SCHEMA.

    Join strategy:
    - ``eha_capacity_kw(plants)`` gives installed_kw per EHA_PtID.
    - ``eha_energy_kwh(cf, year)`` gives annual_energy_kwh per EHA_PtID
      (inner join ⇒ plants absent from the CF file are dropped).
    - Drops plants with zero/null capacity or zero/null energy.

    source_plant_code = EIA_PtID (Int64) from the plant file, enabling later
    cross-source joins with the EIA ground-truth set. EHA's native string key
    ``eha_pt_id`` is used as the join key but is NOT projected into the canonical
    schema (not in CANONICAL_SCHEMA); it can be recovered from the EHA parquet
    via the EIA_PtID → EHA_PtID mapping in the source files if needed.

    Returns a DataFrame with exactly CANONICAL_SCHEMA columns.
    """
    cap = eha_capacity_kw(plants)
    gen = eha_energy_kwh(cf, year)

    df = (
        cap.join(gen, on="eha_pt_id", how="inner")
        .filter(
            (pl.col("actual_installed_kw") > 0)
            & (pl.col("actual_annual_energy_kwh") > 0)
        )
        .with_columns(
            pl.lit("EHA").alias("ground_truth_source"),
            pl.col("pt_name").alias("facility_name"),
            pl.lit(None, dtype=pl.Float64).alias("actual_head_m"),
            pl.lit(None, dtype=pl.Float64).alias("actual_flow_m3s"),
            pl.col("eia_pt_id").cast(pl.Int64, strict=False).alias("source_plant_code"),
            pl.lit(year, dtype=pl.Int64).alias("source_year"),
        )
    )
    # Project to canonical column order + dtypes (drops eha_pt_id in the process).
    return df.select(
        [
            pl.col(name).cast(dtype).alias(name)
            for name, dtype in CANONICAL_SCHEMA.items()
        ]
    )


# ── EHA orchestrator ──────────────────────────────────────────────────────────

def ingest_eha(year: int | None = None) -> pl.DataFrame:
    """Load EHA plant + CF workbooks for ``year`` and return canonical ground truth.

    Args:
        year: CF year to use (default: ``EHA_DEFAULT_YEAR`` = 2022, the latest
              available year in the EHA_Annual_CapacityFactor workbook).

    Returns:
        DataFrame with exactly CANONICAL_SCHEMA columns, ground_truth_source="EHA".
    """
    eha_dir = _resolve_eha_dir()
    if year is None:
        year = int(config.get("phase5.eha_year") or EHA_DEFAULT_YEAR)
    log.info("Ingesting EHA hydro ground truth for year %d from %s", year, eha_dir)
    plants = _load_eha_plants(eha_dir)
    cf = _load_eha_cf(eha_dir)
    return assemble_eha_ground_truth(plants, cf, year)


# ── EHA runner ────────────────────────────────────────────────────────────────

def _eha_summary(df: pl.DataFrame, n_plant_dropped: int, n_neg_energy: int) -> None:
    """Log EHA ingest summary stats."""
    n = df.height
    log.info("EHA hydro ground-truth plants: %d", n)
    if n_plant_dropped:
        log.info("  Dropped (null/zero CH_MW, pumped-storage-only): %d", n_plant_dropped)
    if n_neg_energy:
        log.info("  Dropped (zero/negative net generation 2022): %d", n_neg_energy)
    if n:
        cap = df["actual_installed_kw"]
        en  = df["actual_annual_energy_kwh"]
        log.info("  installed_kw  range: %.0f – %.0f (median %.0f)",
                 cap.min(), cap.max(), cap.median())
        log.info("  annual_kwh    range: %.3g – %.3g (median %.3g)",
                 en.min(), en.max(), en.median())
        log.info("  total fleet GWh/yr: %.1f", en.sum() / 1e6)
        log.info("  states covered: %d", df["state_code"].n_unique())
    log.info("  NOTE: EHA/CF file covers plants >=1 MW — skews large-hydro. "
             "Supplement with FERC conduit labels for micro-scale.")


def run_eha(year: int | None = None) -> Path:
    """Ingest EHA ground truth and write to ``EHA_OUTPUT_PATH``.

    Returns:
        Path to the written parquet file.
    """
    eha_dir = _resolve_eha_dir()
    if year is None:
        year = int(config.get("phase5.eha_year") or EHA_DEFAULT_YEAR)

    plants_raw = _load_eha_plants(eha_dir)
    cf_raw     = _load_eha_cf(eha_dir)

    # Count drops for the summary log
    n_total = plants_raw.height
    plants_filtered = eha_filter_capacity(plants_raw)
    n_plant_dropped = n_total - plants_filtered.height

    gen = eha_energy_kwh(cf_raw, year)
    # Negative/zero energy rows (drop happens inside assemble)
    n_neg_energy = int(
        gen.filter(pl.col("actual_annual_energy_kwh") <= 0).height
    )

    df = assemble_eha_ground_truth(plants_raw, cf_raw, year)

    EHA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    io.write_parquet(df, EHA_OUTPUT_PATH)
    _eha_summary(df, n_plant_dropped, n_neg_energy)
    log.info("Wrote %s", EHA_OUTPUT_PATH)
    return EHA_OUTPUT_PATH


def main_eha() -> None:
    """CLI entry for EHA ingest: ``python -m src.phase5.ground_truth --source eha``."""
    ap = argparse.ArgumentParser(description="Phase 5 — ingest DOE HydroSource EHA ground truth.")
    ap.add_argument("--year", type=int, default=None,
                    help=f"EHA CF year (default: {EHA_DEFAULT_YEAR}, the latest available)")
    args = ap.parse_args()
    run_eha(args.year)


# ══════════════════════════════════════════════════════════════════════════════
# D1 — combine_ground_truth(): merge EIA + EHA with code-level dedup
# ══════════════════════════════════════════════════════════════════════════════

COMBINED_OUTPUT_PATH: Path = (
    config.raw_data_dir() / "ground_truth" / "combined_ground_truth.parquet"
)


def combine_ground_truth(eia: pl.DataFrame, eha: pl.DataFrame) -> pl.DataFrame:
    """Merge EIA and EHA labeled frames into one deduped ground-truth table.

    Both inputs must conform to ``CANONICAL_SCHEMA`` (same columns, same dtypes).

    Deduplication key: ``source_plant_code`` (Int64 EIA plant code).
    On collision (same code in both sources) the **EHA row is kept** — EHA uses
    measured annual net generation from the ORNL CF workbook, whereas EIA uses
    Form-923 net generation; EHA is considered the hydro-curated authoritative
    source.  Rows with null ``source_plant_code`` cannot be deduplicated and are
    kept as-is (they appear only in EHA, n=1 per real-data run).

    EHA-internal dedup: EHA itself may contain multiple entries per
    ``source_plant_code`` (distinct EHA sub-sites matched to the same EIA plant
    ID).  Verified example: EIA code 61217 (U Canal hydro, ID) appears as both
    "U Canal Hydro 2" and "Head of U Canal Hydro Project" in the EHA workbook,
    both reporting identical 4,267 MWh — the EIA plant record captures combined
    output and was matched twice, so both rows report the full generation and
    summing would double-count.  **Keep-rule: keep the row with the highest
    ``actual_annual_energy_kwh`` (ties broken by first occurrence)** — this is
    correct whether the two rows are true duplicates (identical energy → no loss)
    or partial-generation splits (higher row = larger share).  In every verified
    real case the energy values are equal, so keep-max == keep-first.

    Overlap statistics (verified 2026-06-30 on real parquets, post-dedup):
        EIA non-null codes : 1,308
        EHA non-null codes : 1,267  (1,268 rows → 1 internal dup removed)
        EHA null-code rows :     1  → keep (cannot dedup)
        Overlap (EIA∩EHA)  : 1,216  → keep EHA row for each
        EIA-only codes     :    92  → keep as-is
        EHA-only codes     :    51  → keep as-is
        Expected combined  :    92 + 51 + 1 + 1,216 = 1,360  (0 dup codes)

    Returns:
        DataFrame with exactly ``CANONICAL_SCHEMA`` columns.  No new columns.
    """
    # --- validate inputs conform to schema ---
    for name, src in (("eia", eia), ("eha", eha)):
        missing = set(CANONICAL_SCHEMA) - set(src.columns)
        if missing:
            raise ValueError(f"{name} frame is missing CANONICAL_SCHEMA columns: {missing}")

    # --- null-code EHA rows: keep always (cannot participate in dedup) ---
    eha_null_code = eha.filter(pl.col("source_plant_code").is_null())
    eha_has_code  = eha.filter(pl.col("source_plant_code").is_not_null())

    # --- EHA-internal dedup: one row per source_plant_code, keep-max energy ---
    if eha_has_code.height > 0:
        eha_has_code = (
            eha_has_code
            .sort("actual_annual_energy_kwh", descending=True, nulls_last=True)
            .unique(subset=["source_plant_code"], keep="first")
        )

    # --- EHA codes take priority: remove EIA rows whose code appears in EHA ---
    # Guard: if eha_has_code is empty, the anti-join key type may be null → cast
    if eha_has_code.height == 0:
        eia_not_in_eha = eia
    else:
        eha_codes = eha_has_code.select(
            pl.col("source_plant_code").cast(pl.Int64)
        ).unique()
        eia_not_in_eha = eia.join(eha_codes, on="source_plant_code", how="anti")

    # --- stack: EIA-only rows, deduped EHA rows with code, null-code EHA rows ---
    combined = pl.concat([eia_not_in_eha, eha_has_code, eha_null_code], how="vertical")

    # --- project to canonical schema + dtypes (no new columns) ---
    combined = combined.select(
        [pl.col(name).cast(dtype).alias(name) for name, dtype in CANONICAL_SCHEMA.items()]
    )

    n = combined.height
    overlap = int(
        eia.filter(pl.col("source_plant_code").is_not_null())
        .join(eha_has_code.select("source_plant_code"), on="source_plant_code", how="semi")
        .height
    )
    log.info(
        "combine_ground_truth: %d EIA + %d EHA → %d combined rows "
        "(overlap=%d, EHA row kept per collision)",
        eia.height, eha.height, n, overlap,
    )
    return combined


def run_combine(
    eia_path: Path | None = None,
    eha_path: Path | None = None,
) -> Path:
    """Load EIA + EHA parquets, combine, write ``combined_ground_truth.parquet``.

    If the source parquets are absent, regenerates them from the external drive
    via the existing ``run()`` / ``run_eha()`` wrappers (requires SANDISK mount).

    Args:
        eia_path: Override path for the EIA parquet (default: ``OUTPUT_PATH``).
        eha_path: Override path for the EHA parquet (default: ``EHA_OUTPUT_PATH``).

    Returns:
        Path to combined_ground_truth.parquet.
    """
    eia_p = eia_path or OUTPUT_PATH
    eha_p = eha_path or EHA_OUTPUT_PATH

    if not eia_p.exists():
        log.info("EIA parquet absent — regenerating via run() …")
        run()
    if not eha_p.exists():
        log.info("EHA parquet absent — regenerating via run_eha() …")
        run_eha()

    eia = pl.read_parquet(eia_p)
    eha = pl.read_parquet(eha_p)
    combined = combine_ground_truth(eia, eha)

    COMBINED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    io.write_parquet(combined, COMBINED_OUTPUT_PATH)

    n = combined.height
    cap = combined["actual_installed_kw"]
    en  = combined["actual_annual_energy_kwh"]
    log.info("Combined ground truth: %d plants", n)
    log.info("  installed_kw  range: %.0f – %.0f (median %.0f)", cap.min(), cap.max(), cap.median())
    log.info("  annual_kwh    range: %.3g – %.3g (median %.3g)", en.min(), en.max(), en.median())
    log.info("  fleet GWh/yr: %.1f", en.sum() / 1e6)
    log.info("  sources: %s", combined.group_by("ground_truth_source").len().to_dicts())
    log.info("Wrote %s", COMBINED_OUTPUT_PATH)
    return COMBINED_OUTPUT_PATH

"""GeoJSON export for the WOWERS frontend map demo.

Reads Phase 1–4 parquets and exports GeoJSON FeatureCollections
(post P2-SEED re-baseline). The default run writes BOTH git-tracked files:

  exports/viable_sites.geojson  — 1,138 project-viable sites (reports/paper)
  exports/scored_sites.geojson  — all 3,778 scored sites; the *single static
                                  data source* for the vite+react+maplibre
                                  frontend (non-viable sites render as the
                                  grey "> 20 yr" band on the map)

The frontend derives national KPIs, per-state portfolios, and per-plant
detail views entirely client-side from scored_sites.geojson, filtering to
``project_viable`` for the KPI aggregates.

COORDINATE ORDER: [longitude, latitude] per GeoJSON spec (RFC 7946 §3.1.1).

ROUNDING:
  USD + kWh + int counts  → integer (round to 0 d.p.)
  Ratio columns           → 4 d.p.  (irr, lcoe, CF, energy_offset_pct, etc.)
  1 d.p. columns          → 1 d.p.  (MGD flows, metres, m³/s, efficiency %)
  Coordinates             → 5 d.p.  (≈ 1 m precision)
  NaN / inf               → JSON null

META FOREIGN MEMBER (RFC 7946 §6.1):
  "meta": {"plants_analyzed": N, "scored_sites": N, "baseline": "..."}
  Computed from parquet row counts — NOT hardcoded — so output is
  byte-deterministic across runs as long as parquets don't change.

USAGE:
    python scripts/export_geojson.py                 # writes BOTH default files
    python scripts/export_geojson.py --all           # all sites → --out (single file)
    python scripts/export_geojson.py --scorecard path --p1 path --p2 path --p3 path
    python scripts/export_geojson.py --out path.geojson   # viable only → custom path
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("wowers.export_geojson")

# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_SCORECARD = ROOT / "data/processed/phase4/financial_scorecards.parquet"
_DEFAULT_P1        = ROOT / "data/processed/phase1/ranked_candidates.parquet"
_DEFAULT_P2        = ROOT / "data/processed/phase2/energy_yield_estimates.parquet"
_DEFAULT_P3        = ROOT / "data/processed/phase3/turbine_sizing.parquet"
_DEFAULT_OUT       = ROOT / "exports/viable_sites.geojson"
_DEFAULT_OUT_ALL   = ROOT / "exports/scored_sites.geojson"

# ── Property list: original 24 + 34 new + 1 measured floor = 59 total ─────────
# P4-MEASURED-FLOOR (2026-08-06): energy_kwh_calib_measured_point_loma added,
# taking the contract from 58 to 59. It carries the reported band floor
# (89.8 GWh/yr fleet-wide); see src/phase4/financials.add_calibrated_energy_cols.

PROPERTIES: list[str] = [
    # ── original 24 ──────────────────────────────────────────────────────────
    "npdes_id",
    "facility_name",
    "city",
    "state_code",
    "turbine_type",
    "rated_power_kw",
    "annual_energy_kwh",
    "energy_kwh_calib_floor_p25",
    "energy_kwh_calib_floor_p50",
    "energy_kwh_calib_central",
    "energy_kwh_calib_measured_point_loma",   # P4-MEASURED-FLOOR: reported band floor
    "capacity_factor",
    "total_capex_usd",
    "npv_usd",
    "irr",
    "payback_years",
    "lcoe_per_kwh",
    "annual_revenue_usd",
    "energy_offset_pct",
    "site_tier",
    "econ_cat_payback",
    "econ_cat_npv",
    "econ_cat_irr",
    "data_quality_tier",
    "project_viable",
    # ── new: P1 flow stats ────────────────────────────────────────────────────
    "mean_flow_mgd",
    "p10_flow_mgd",
    "p90_flow_mgd",
    "n_months_data",
    "utilization_ratio",
    # ── new: P2 Monte-Carlo energy band ──────────────────────────────────────
    "energy_p10_kwh_yr",
    "energy_p50_kwh_yr",
    "energy_p90_kwh_yr",
    "equivalent_homes_p50",
    # ── new: P3 elevation + turbine detail ───────────────────────────────────
    "head_net_m",
    "head_gross_m",
    "elevation_m",
    "elev_outfall_m",
    "head_source",
    "head_confidence",
    "q_rated_m3s",
    "peak_efficiency_pct",
    # ── new: P4 capex breakdown, grant, opex, rate ────────────────────────────
    "npv_with_50pct_grant_usd",
    "annual_opex_usd",
    "elec_rate_per_kwh",
    "equipment_capex_usd",
    "installation_capex_usd",
    "interconnection_capex_usd",
    "permitting_capex_usd",
    "permitting_tier",
    # ── new: P4 sensitivity + provenance ─────────────────────────────────────
    "sensitivity_head_npv_low",
    "sensitivity_head_npv_high",
    "sensitivity_flow_npv_low",
    "sensitivity_flow_npv_high",
    "sensitivity_rate_npv_low",
    "sensitivity_rate_npv_high",
    "dominant_sensitivity",
    "data_quality",
    "project_viable_high_confidence",
]

# ── Rounding sets ─────────────────────────────────────────────────────────────

# Integer (USD magnitudes, kWh magnitudes, integer counts)
_INT_COLS: frozenset[str] = frozenset({
    # original
    "rated_power_kw",
    "annual_energy_kwh",
    "energy_kwh_calib_floor_p25",
    "energy_kwh_calib_floor_p50",
    "energy_kwh_calib_central",
    "energy_kwh_calib_measured_point_loma",
    "total_capex_usd",
    "npv_usd",
    "annual_revenue_usd",
    # new P2
    "energy_p10_kwh_yr",
    "energy_p50_kwh_yr",
    "energy_p90_kwh_yr",
    "equivalent_homes_p50",
    # new P4 USD
    "npv_with_50pct_grant_usd",
    "annual_opex_usd",
    "equipment_capex_usd",
    "installation_capex_usd",
    "interconnection_capex_usd",
    "permitting_capex_usd",
    # new P4 sensitivity (NPV impact → USD)
    "sensitivity_head_npv_low",
    "sensitivity_head_npv_high",
    "sensitivity_flow_npv_low",
    "sensitivity_flow_npv_high",
    "sensitivity_rate_npv_low",
    "sensitivity_rate_npv_high",
    # new P1 integer count
    "n_months_data",
})

# Ratio (4 d.p.)
_RATIO_COLS: frozenset[str] = frozenset({
    # original
    "irr",
    "lcoe_per_kwh",
    "capacity_factor",
    "energy_offset_pct",
    # new
    "utilization_ratio",
    "elec_rate_per_kwh",
})

# 1 decimal place (metres, MGD, m³/s, efficiency %)
_DP1_COLS: frozenset[str] = frozenset({
    # P1 flows
    "mean_flow_mgd",
    "p10_flow_mgd",
    "p90_flow_mgd",
    # P3 elevation
    "head_net_m",
    "head_gross_m",
    "elevation_m",
    "elev_outfall_m",
    # P3 turbine
    "q_rated_m3s",
    "peak_efficiency_pct",
})

_COORD_PRECISION: int = 5

# Baseline string embedded in meta (deterministic — no timestamp)
_BASELINE: str = "P2-SEED re-baseline 2026-07-06"


# ── Pure functions (unit-testable) ────────────────────────────────────────────

def _to_python(val: Any) -> Any:
    """Convert polars/numpy scalar to a Python primitive."""
    if hasattr(val, "item"):
        return val.item()
    return val


def round_property(col: str, val: Any) -> Any:
    """Apply per-column rounding rules; return JSON-safe Python value.

    Priority order:
      bool        → preserved as-is (bool is subclass of int, check first)
      _INT_COLS   → int  (None if not finite)
      _RATIO_COLS → float, 4 d.p. (None if not finite)
      _DP1_COLS   → float, 1 d.p. (None if not finite)
      int         → int as-is
      float       → pass-through  (None if NaN/inf)
      other       → pass-through unchanged
    """
    val = _to_python(val)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if col in _INT_COLS:
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else int(round(f))
        except (TypeError, ValueError):
            return val
    if col in _RATIO_COLS:
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        except (TypeError, ValueError):
            return val
    if col in _DP1_COLS:
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 1)
        except (TypeError, ValueError):
            return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else val
    return val


def build_feature(row: dict) -> dict | None:
    """Build one GeoJSON Feature dict from a joined row.

    Returns None if latitude or longitude is missing/invalid.
    Coordinates: [longitude, latitude] per GeoJSON spec.
    """
    lat = _to_python(row.get("latitude"))
    lon = _to_python(row.get("longitude"))
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat_f) or math.isnan(lon_f):
        return None

    props: dict[str, Any] = {
        col: round_property(col, row.get(col))
        for col in PROPERTIES
    }

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [
                round(lon_f, _COORD_PRECISION),
                round(lat_f, _COORD_PRECISION),
            ],
        },
        "properties": props,
    }


def build_feature_collection(
    df: pl.DataFrame,
    *,
    viable_only: bool = True,
    meta: dict | None = None,
) -> tuple[dict, list[str]]:
    """Build a GeoJSON FeatureCollection from the joined frame.

    Args:
        df:          Joined DataFrame with all PROPERTIES columns + lat/lon.
        viable_only: Include only project_viable == True rows when True.
        meta:        Optional ``meta`` foreign member (RFC 7946 §6.1).

    Returns:
        (feature_collection_dict, dropped_npdes_ids)
    """
    if viable_only:
        df = df.filter(pl.col("project_viable"))

    features: list[dict] = []
    dropped: list[str] = []

    for row in df.to_dicts():
        feat = build_feature(row)
        if feat is None:
            dropped.append(str(row.get("npdes_id", "?")))
        else:
            features.append(feat)

    if dropped:
        log.warning(
            "Excluded %d site(s) with null/invalid lat-lon: %s",
            len(dropped), dropped[:20],
        )

    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if meta is not None:
        fc["meta"] = meta

    return fc, dropped


def load_and_join(
    scorecard_path: Path,
    p1_path: Path,
    p2_path: Path | None = None,
    p3_path: Path | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Load parquets and left-join all phases on npdes_id.

    Returns:
        (joined_df, meta_dict)
        meta contains plants_analyzed (P1 row count) and scored_sites (P4 row count).

    STOP-on-surprise: any column listed in §1.1 that is missing from the
    parquet will cause a KeyError, which propagates loudly. Do not rename
    or guess alternative column names.
    """
    p4 = pl.read_parquet(scorecard_path)
    p1_full = pl.read_parquet(p1_path)

    plants_analyzed: int = p1_full.height
    scored_sites: int    = p4.height

    meta: dict = {
        "plants_analyzed": plants_analyzed,
        "scored_sites":    scored_sites,
        "baseline":        _BASELINE,
    }

    # P1: coords + name/city + flow stats
    p1_sub = p1_full.select([
        "npdes_id", "facility_name", "city", "latitude", "longitude",
        "mean_flow_mgd", "p10_flow_mgd", "p90_flow_mgd",
        "n_months_data", "utilization_ratio",
    ])
    df = p4.join(p1_sub, on="npdes_id", how="left")

    # P2: Monte-Carlo energy band
    if p2_path is not None:
        p2_sub = pl.read_parquet(p2_path).select([
            "npdes_id",
            "energy_p10_kwh_yr", "energy_p50_kwh_yr", "energy_p90_kwh_yr",
            "equivalent_homes_p50",
        ])
        df = df.join(p2_sub, on="npdes_id", how="left")

    # P3: elevation + turbine detail
    if p3_path is not None:
        p3_sub = pl.read_parquet(p3_path).select([
            "npdes_id",
            "head_net_m", "head_gross_m", "elevation_m", "elev_outfall_m",
            "head_source", "head_confidence",
            "q_rated_m3s", "peak_efficiency_pct",
        ])
        df = df.join(p3_sub, on="npdes_id", how="left")

    return df, meta


def validate_geojson(fc: dict, expected_features: int) -> None:
    """Assert GeoJSON structure, feature count, coordinates, and meta.

    If ``meta`` is present in fc, also asserts that ``plants_analyzed``
    and ``scored_sites`` are positive integers and ``baseline`` is a string.
    """
    assert fc.get("type") == "FeatureCollection", "type must be FeatureCollection"
    feats = fc.get("features", [])
    assert len(feats) == expected_features, (
        f"Expected {expected_features} features, got {len(feats)}"
    )
    for feat in feats[:5]:
        assert feat["type"] == "Feature"
        coords = feat["geometry"]["coordinates"]
        assert len(coords) == 2, "coordinates must be [lon, lat]"
        lon, lat = coords
        assert -180.0 <= lon <= 180.0, f"longitude {lon} out of range"
        assert  -90.0 <= lat <=  90.0, f"latitude {lat} out of range"

    meta = fc.get("meta")
    if meta is not None:
        assert isinstance(meta.get("plants_analyzed"), int), (
            "meta.plants_analyzed must be int"
        )
        assert meta["plants_analyzed"] > 0, "meta.plants_analyzed must be positive"
        assert isinstance(meta.get("scored_sites"), int), (
            "meta.scored_sites must be int"
        )
        assert meta["scored_sites"] > 0, "meta.scored_sites must be positive"
        assert isinstance(meta.get("baseline"), str), (
            "meta.baseline must be str"
        )


# ── IO orchestrator ───────────────────────────────────────────────────────────

def export(
    scorecard_path: Path = _DEFAULT_SCORECARD,
    p1_path: Path = _DEFAULT_P1,
    p2_path: Path | None = _DEFAULT_P2,
    p3_path: Path | None = _DEFAULT_P3,
    out_path: Path = _DEFAULT_OUT,
    *,
    viable_only: bool = True,
) -> tuple[Path, int, int]:
    """End-to-end: load → join → build → validate → write.

    Returns:
        (out_path, n_features_written, n_dropped_for_null_coords)
    """
    df, meta = load_and_join(scorecard_path, p1_path, p2_path, p3_path)
    fc, dropped = build_feature_collection(df, viable_only=viable_only, meta=meta)
    n_features = len(fc["features"])
    validate_geojson(fc, n_features)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(fc, fh, separators=(",", ":"))

    log.info(
        "Wrote %d features → %s  (%d dropped for null coords)",
        n_features, out_path, len(dropped),
    )
    return out_path, n_features, len(dropped)


# ── CLI entry ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export WOWERS viable sites as GeoJSON (single-file frontend data source)."
    )
    ap.add_argument("--scorecard", type=Path, default=_DEFAULT_SCORECARD)
    ap.add_argument("--p1", type=Path, default=_DEFAULT_P1,
                    help="Phase 1 ranked_candidates.parquet")
    ap.add_argument("--p2", type=Path, default=_DEFAULT_P2,
                    help="Phase 2 energy_yield_estimates.parquet")
    ap.add_argument("--p3", type=Path, default=_DEFAULT_P3,
                    help="Phase 3 turbine_sizing.parquet")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--all", dest="all_sites", action="store_true",
                    help="Export all scored sites, not just project_viable==True")
    args = ap.parse_args()

    # Default invocation (no --out, no --all): write BOTH tracked files so
    # viable_sites.geojson and scored_sites.geojson can never drift apart.
    default_run = args.out == _DEFAULT_OUT and not args.all_sites
    runs = (
        [(True, _DEFAULT_OUT), (False, _DEFAULT_OUT_ALL)]
        if default_run
        else [(not args.all_sites, args.out)]
    )
    for viable_only, out in runs:
        out_path, n_features, n_dropped = export(
            scorecard_path=args.scorecard,
            p1_path=args.p1,
            p2_path=args.p2,
            p3_path=args.p3,
            out_path=out,
            viable_only=viable_only,
        )
        print(f"Features written  : {n_features:,}")
        print(f"Dropped (no coord): {n_dropped}")
        print(f"Output            : {out_path}")


if __name__ == "__main__":
    main()

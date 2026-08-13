"""Tests for scripts/export_geojson.py.

All unit tests use small synthetic polars frames built in-memory.
No real parquets required for unit tests.

Coverage:
  round_property   : USD/kWh → int; ratio → 4 d.p.; bool passthrough;
                     NaN/inf → None; unknown float passthrough
  build_feature    : coordinate order [lon, lat]; property list; null-coord → None
  build_feature_collection: viable_only filter; null-coord exclusion; dropped list
  validate_geojson : type/count/coord-range assertions
  export (pure fn) : integration stub with synthetic frames

One real-data integration test gated on parquets + exports/ presence.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl
import pytest

# Add scripts/ to path for direct import (mirrors test_cf_calibration.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from export_geojson import (
    PROPERTIES,
    _INT_COLS,
    _RATIO_COLS,
    _DP1_COLS,
    build_feature,
    build_feature_collection,
    round_property,
    validate_geojson,
    load_and_join,
    export,
    _DEFAULT_SCORECARD,
    _DEFAULT_P1,
    _DEFAULT_P2,
    _DEFAULT_P3,
    _DEFAULT_OUT,
)

# ── Paths for integration tests ────────────────────────────────────────────────

_SCORECARD_PATH = Path("data/processed/phase4/financial_scorecards.parquet")
_P1_PATH        = Path("data/processed/phase1/ranked_candidates.parquet")
_EXPORT_PATH    = Path("exports/viable_sites.geojson")
_EXPORT_ALL_PATH = Path("exports/scored_sites.geojson")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_row(
    npdes_id: str = "TX0001234",
    lat: float | None = 30.1234,
    lon: float | None = -97.5678,
    energy: float = 1_500_000.0,
    viable: bool = True,
    irr: float = 0.12345,
    city: str = "Austin",
) -> dict:
    return {
        "npdes_id":                    npdes_id,
        "facility_name":               "Austin WWTP",
        "city":                        city,
        "state_code":                  "TX",
        "turbine_type":                "Kaplan",
        "rated_power_kw":              185.7,
        "annual_energy_kwh":           energy,
        "energy_kwh_calib_floor_p25":  energy * 0.291,
        "energy_kwh_calib_floor_p50":  energy * 0.447,
        "energy_kwh_calib_central":    energy * 0.688,
        "energy_kwh_calib_measured_point_loma": energy * 0.2195,
        "capacity_factor":             0.8716,
        "total_capex_usd":             412_500.0,
        "npv_usd":                     75_200.0,
        "irr":                         irr,
        "payback_years":               9.8,
        "lcoe_per_kwh":                0.0523,
        "annual_revenue_usd":          87_300.0,
        "energy_offset_pct":           14.22,
        "site_tier":                   "A",
        "econ_cat_payback":            "good",
        "econ_cat_npv":                "low",
        "econ_cat_irr":                "moderate",
        "data_quality_tier":           0,
        "project_viable":              viable,
        "latitude":                    lat,
        "longitude":                   lon,
    }


def _make_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# round_property
# ══════════════════════════════════════════════════════════════════════════════

class TestRoundProperty:

    def test_int_col_rounds_to_int(self):
        assert round_property("annual_energy_kwh", 1_234_567.89) == 1_234_568
        assert isinstance(round_property("total_capex_usd", 412_500.0), int)

    def test_int_col_nan_returns_none(self):
        assert round_property("annual_energy_kwh", float("nan")) is None

    def test_int_col_inf_returns_none(self):
        assert round_property("npv_usd", float("inf")) is None

    def test_ratio_col_rounds_to_4dp(self):
        result = round_property("irr", 0.123456789)
        assert result == pytest.approx(0.1235)
        assert isinstance(result, float)

    def test_ratio_col_nan_returns_none(self):
        assert round_property("capacity_factor", float("nan")) is None

    def test_ratio_col_inf_returns_none(self):
        assert round_property("lcoe_per_kwh", float("inf")) is None

    def test_bool_preserved_true(self):
        result = round_property("project_viable", True)
        assert result is True
        assert isinstance(result, bool)

    def test_bool_preserved_false(self):
        result = round_property("project_viable", False)
        assert result is False
        assert isinstance(result, bool)

    def test_int_passthrough(self):
        result = round_property("data_quality_tier", 0)
        assert result == 0
        assert isinstance(result, int)

    def test_string_passthrough(self):
        assert round_property("site_tier", "A") == "A"
        assert round_property("turbine_type", "Kaplan") == "Kaplan"

    def test_none_returns_none(self):
        assert round_property("npv_usd", None) is None

    def test_unknown_float_nan_returns_none(self):
        assert round_property("payback_years", float("nan")) is None

    def test_unknown_float_finite_passthrough(self):
        result = round_property("payback_years", 9.876)
        assert result == pytest.approx(9.876)

    def test_energy_calib_cols_are_int_rounded(self):
        for col in ("energy_kwh_calib_floor_p25",
                    "energy_kwh_calib_floor_p50",
                    "energy_kwh_calib_central",
                    "energy_kwh_calib_measured_point_loma"):
            result = round_property(col, 500_000.7)
            assert isinstance(result, int)
            assert result == 500_001

    # ── new rounding classes ────────────────────────────────────────────────

    def test_dp1_col_rounds_to_1dp(self):
        """mean_flow_mgd, head_net_m etc. → 1 decimal place."""
        for col in ("mean_flow_mgd", "p10_flow_mgd", "p90_flow_mgd",
                    "head_net_m", "head_gross_m", "elevation_m", "elev_outfall_m",
                    "q_rated_m3s", "peak_efficiency_pct"):
            result = round_property(col, 12.3456)
            assert result == pytest.approx(12.3), f"{col}: expected 12.3, got {result}"
            assert isinstance(result, float)

    def test_dp1_col_nan_returns_none(self):
        assert round_property("mean_flow_mgd", float("nan")) is None

    def test_dp1_col_inf_returns_none(self):
        assert round_property("head_net_m", float("inf")) is None

    def test_new_int_cols_energy_kwh_yr(self):
        """energy_p10/50/90_kwh_yr, equivalent_homes_p50 → int."""
        for col in ("energy_p10_kwh_yr", "energy_p50_kwh_yr", "energy_p90_kwh_yr",
                    "equivalent_homes_p50"):
            result = round_property(col, 1_234_567.8)
            assert isinstance(result, int)
            assert result == 1_234_568

    def test_new_int_cols_capex_breakdown(self):
        """All capex breakdown columns → int."""
        for col in ("equipment_capex_usd", "installation_capex_usd",
                    "interconnection_capex_usd", "permitting_capex_usd",
                    "npv_with_50pct_grant_usd", "annual_opex_usd"):
            result = round_property(col, 99_999.6)
            assert isinstance(result, int)
            assert result == 100_000

    def test_new_int_cols_sensitivity(self):
        """sensitivity_*_npv_* columns → int."""
        for col in ("sensitivity_head_npv_low", "sensitivity_head_npv_high",
                    "sensitivity_flow_npv_low", "sensitivity_flow_npv_high",
                    "sensitivity_rate_npv_low", "sensitivity_rate_npv_high"):
            result = round_property(col, -50_000.5)
            assert isinstance(result, int)

    def test_new_ratio_cols(self):
        """utilization_ratio, elec_rate_per_kwh → 4 d.p."""
        for col in ("utilization_ratio", "elec_rate_per_kwh"):
            result = round_property(col, 0.123456789)
            assert result == pytest.approx(0.1235)

    def test_bool_passthrough_project_viable_high_confidence(self):
        assert round_property("project_viable_high_confidence", True) is True
        assert round_property("project_viable_high_confidence", False) is False

    def test_string_passthrough_new_cols(self):
        for col in ("head_source", "head_confidence", "permitting_tier",
                    "dominant_sensitivity", "data_quality"):
            assert round_property(col, "usgs_3dep") == "usgs_3dep"


# ══════════════════════════════════════════════════════════════════════════════
# build_feature — coordinate order and property list
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildFeature:

    def test_coordinate_order_is_lon_lat(self):
        """GeoJSON spec: coordinates = [longitude, latitude]."""
        row = _make_row(lat=30.1234, lon=-97.5678)
        feat = build_feature(row)
        lon, lat = feat["geometry"]["coordinates"]
        assert lon == pytest.approx(-97.56780, abs=1e-6)
        assert lat == pytest.approx( 30.12340, abs=1e-6)

    def test_coordinates_rounded_to_5dp(self):
        row = _make_row(lat=30.123456789, lon=-97.567891234)
        feat = build_feature(row)
        lon, lat = feat["geometry"]["coordinates"]
        assert lon == pytest.approx(-97.56789, abs=1e-6)
        assert lat == pytest.approx( 30.12346, abs=1e-6)

    def test_geometry_type_is_point(self):
        feat = build_feature(_make_row())
        assert feat["geometry"]["type"] == "Point"

    def test_feature_type_is_feature(self):
        feat = build_feature(_make_row())
        assert feat["type"] == "Feature"

    def test_all_properties_present(self):
        feat = build_feature(_make_row())
        for prop in PROPERTIES:
            assert prop in feat["properties"], f"Missing property: {prop}"

    def test_no_extra_properties(self):
        feat = build_feature(_make_row())
        assert set(feat["properties"].keys()) == set(PROPERTIES)

    def test_null_lat_returns_none(self):
        assert build_feature(_make_row(lat=None)) is None

    def test_null_lon_returns_none(self):
        assert build_feature(_make_row(lon=None)) is None

    def test_nan_lat_returns_none(self):
        assert build_feature(_make_row(lat=float("nan"))) is None

    def test_nan_lon_returns_none(self):
        assert build_feature(_make_row(lon=float("nan"))) is None

    def test_energy_kwh_rounded_to_int_in_properties(self):
        row = _make_row(energy=1_234_567.89)
        feat = build_feature(row)
        val = feat["properties"]["annual_energy_kwh"]
        assert isinstance(val, int)
        assert val == 1_234_568

    def test_irr_rounded_to_4dp_in_properties(self):
        row = _make_row(irr=0.123456789)
        feat = build_feature(row)
        irr_val = feat["properties"]["irr"]
        assert irr_val == pytest.approx(0.1235)

    def test_project_viable_is_bool(self):
        feat = build_feature(_make_row(viable=True))
        assert feat["properties"]["project_viable"] is True

    def test_calib_central_is_int(self):
        row = _make_row(energy=1_000_000.0)
        feat = build_feature(row)
        cen = feat["properties"]["energy_kwh_calib_central"]
        assert isinstance(cen, int)
        assert cen == round(1_000_000.0 * 0.688)

    def test_property_count_is_59(self):
        """Total property count must be exactly 59 (24 + 34 + measured floor)."""
        feat = build_feature(_make_row())
        assert len(feat["properties"]) == 59, (
            f"Expected 59 properties, got {len(feat['properties'])}"
        )
        assert len(PROPERTIES) == 59


# ══════════════════════════════════════════════════════════════════════════════
# build_feature_collection
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatureCollection:

    def test_viable_only_filters_non_viable(self):
        rows = [
            _make_row(npdes_id="A", viable=True),
            _make_row(npdes_id="B", viable=False),
            _make_row(npdes_id="C", viable=True),
        ]
        df = _make_df(rows)
        fc, dropped = build_feature_collection(df, viable_only=True)
        assert len(fc["features"]) == 2
        ids = [f["properties"]["npdes_id"] for f in fc["features"]]
        assert "B" not in ids

    def test_all_sites_includes_non_viable(self):
        rows = [
            _make_row(npdes_id="A", viable=True),
            _make_row(npdes_id="B", viable=False),
        ]
        fc, dropped = build_feature_collection(_make_df(rows), viable_only=False)
        assert len(fc["features"]) == 2

    def test_null_coord_excluded_and_counted(self):
        rows = [
            _make_row(npdes_id="GOOD", lat=30.0, lon=-97.0),
            _make_row(npdes_id="BAD",  lat=None, lon=-97.0),
        ]
        fc, dropped = build_feature_collection(_make_df(rows), viable_only=False)
        assert len(fc["features"]) == 1
        assert len(dropped) == 1
        assert "BAD" in dropped

    def test_feature_collection_type(self):
        fc, _ = build_feature_collection(_make_df([_make_row()]), viable_only=False)
        assert fc["type"] == "FeatureCollection"

    def test_empty_after_filter(self):
        rows = [_make_row(npdes_id="X", viable=False)]
        fc, dropped = build_feature_collection(_make_df(rows), viable_only=True)
        assert len(fc["features"]) == 0
        assert len(dropped) == 0

    def test_multiple_null_coords_all_dropped(self):
        rows = [
            _make_row(npdes_id="A", lat=None),
            _make_row(npdes_id="B", lon=None),
            _make_row(npdes_id="C", lat=10.0, lon=-90.0),
        ]
        fc, dropped = build_feature_collection(_make_df(rows), viable_only=False)
        assert len(fc["features"]) == 1
        assert len(dropped) == 2


# ══════════════════════════════════════════════════════════════════════════════
# validate_geojson
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateGeojson:

    def _make_fc(self, n: int) -> dict:
        feats = []
        for i in range(n):
            feats.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-97.5 - i * 0.01, 30.0 + i * 0.01],
                },
                "properties": {"npdes_id": f"ID{i}"},
            })
        return {"type": "FeatureCollection", "features": feats}

    def test_passes_valid_fc(self):
        validate_geojson(self._make_fc(3), 3)   # should not raise

    def test_fc_with_meta_passes(self):
        fc = self._make_fc(2)
        fc["meta"] = {"plants_analyzed": 17148, "scored_sites": 3778, "baseline": "test"}
        validate_geojson(fc, 2)   # should not raise

    def test_meta_missing_plants_analyzed_raises(self):
        fc = self._make_fc(1)
        fc["meta"] = {"scored_sites": 3778, "baseline": "test"}
        with pytest.raises((AssertionError, KeyError)):
            validate_geojson(fc, 1)

    def test_meta_zero_plants_analyzed_raises(self):
        fc = self._make_fc(1)
        fc["meta"] = {"plants_analyzed": 0, "scored_sites": 3778, "baseline": "test"}
        with pytest.raises(AssertionError):
            validate_geojson(fc, 1)

    def test_meta_non_int_scored_sites_raises(self):
        fc = self._make_fc(1)
        fc["meta"] = {"plants_analyzed": 17148, "scored_sites": "3778", "baseline": "test"}
        with pytest.raises(AssertionError):
            validate_geojson(fc, 1)

    def test_wrong_count_raises(self):
        with pytest.raises(AssertionError):
            validate_geojson(self._make_fc(3), 5)

    def test_wrong_type_raises(self):
        fc = self._make_fc(1)
        fc["type"] = "Feature"
        with pytest.raises(AssertionError):
            validate_geojson(fc, 1)

    def test_out_of_range_longitude_raises(self):
        fc = self._make_fc(1)
        fc["features"][0]["geometry"]["coordinates"] = [200.0, 30.0]
        with pytest.raises(AssertionError):
            validate_geojson(fc, 1)

    def test_out_of_range_latitude_raises(self):
        fc = self._make_fc(1)
        fc["features"][0]["geometry"]["coordinates"] = [-97.0, 95.0]
        with pytest.raises(AssertionError):
            validate_geojson(fc, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Integration test (real data — gated on parquets + export file)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (_SCORECARD_PATH.exists() and _P1_PATH.exists()),
    reason="Real parquets not available — integration test skipped",
)
class TestExportIntegration:

    def test_geojson_exists_after_export(self):
        assert _EXPORT_PATH.exists(), (
            f"{_EXPORT_PATH} not found — run: python scripts/export_geojson.py"
        )

    def test_geojson_valid_feature_collection(self):
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        assert fc["type"] == "FeatureCollection"

    def test_geojson_1138_features(self):
        # P2-SEED: 1,140 → 1,138 (FL0A00002, NY0026328 lost viability after
        # site-keyed MC re-baseline; WI0025194 still absent from bad-coord removal)
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        assert len(fc["features"]) == 1138, (
            f"Expected 1138 features, got {len(fc['features'])}"
        )

    def test_geojson_has_59_properties(self):
        """Each feature must have exactly 59 properties (P4-MEASURED-FLOOR)."""
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        for feat in fc["features"][:5]:
            n = len(feat["properties"])
            assert n == 59, f"Expected 59 properties, got {n}"

    def test_geojson_meta_present_and_correct(self):
        """meta foreign member must have plants_analyzed=17148, scored_sites=3778."""
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        meta = fc.get("meta")
        assert meta is not None, "meta foreign member missing from GeoJSON"
        assert meta["plants_analyzed"] == 17148
        assert meta["scored_sites"] == 3778
        assert isinstance(meta["baseline"], str)

    def test_scored_geojson_has_3778_features_59_props_and_meta(self):
        """exports/scored_sites.geojson (frontend data source) — all scored sites."""
        assert _EXPORT_ALL_PATH.exists(), (
            f"{_EXPORT_ALL_PATH} not found — run: python scripts/export_geojson.py"
        )
        with open(_EXPORT_ALL_PATH) as f:
            fc = json.load(f)
        assert len(fc["features"]) == 3778, (
            f"Expected 3778 features, got {len(fc['features'])}"
        )
        for feat in fc["features"][:5]:
            assert len(feat["properties"]) == 59
        meta = fc.get("meta")
        assert meta is not None
        assert meta["plants_analyzed"] == 17148
        assert meta["scored_sites"] == 3778
        n_viable = sum(1 for feat in fc["features"] if feat["properties"]["project_viable"])
        assert n_viable == 1138, f"Expected 1138 viable among scored, got {n_viable}"

    def test_all_features_have_required_properties(self):
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        for feat in fc["features"]:
            for prop in PROPERTIES:
                assert prop in feat["properties"], (
                    f"Feature {feat['properties'].get('npdes_id')} "
                    f"missing property '{prop}'"
                )

    def test_coordinate_order_lon_lat(self):
        """Spot-check first feature: coordinates[0] is longitude (negative for US)."""
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        lon, lat = fc["features"][0]["geometry"]["coordinates"]
        assert -130.0 <= lon <= -60.0, f"Longitude {lon} outside US range — may be swapped"
        assert   20.0 <= lat <=  50.0, f"Latitude {lat} outside US range — may be swapped"

    def test_no_null_coordinates(self):
        """All exported features must have valid coordinates (0 dropped expected)."""
        with open(_EXPORT_PATH) as f:
            fc = json.load(f)
        for feat in fc["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            assert lon is not None and lat is not None

    def test_spot_check_parquet_vs_geojson(self):
        """Pick first viable site from parquet; verify GeoJSON properties match."""
        scorecard = pl.read_parquet(_SCORECARD_PATH)
        p1 = pl.read_parquet(_P1_PATH).select(
            ["npdes_id", "facility_name", "city", "latitude", "longitude"]
        )
        joined = scorecard.filter(pl.col("project_viable")).join(p1, on="npdes_id", how="left")
        first = joined.row(0, named=True)
        first_id = first["npdes_id"]

        with open(_EXPORT_PATH) as f:
            fc = json.load(f)

        match = next(
            (feat for feat in fc["features"]
             if feat["properties"]["npdes_id"] == first_id),
            None,
        )
        assert match is not None, f"npdes_id {first_id} not found in GeoJSON"

        # Verify integer-rounded kWh matches
        parquet_energy = first["annual_energy_kwh"]
        geojson_energy = match["properties"]["annual_energy_kwh"]
        assert geojson_energy == int(round(parquet_energy))

        # Verify coordinate precision and order
        lon_gj, lat_gj = match["geometry"]["coordinates"]
        assert lon_gj == pytest.approx(round(first["longitude"], 5), abs=1e-5)
        assert lat_gj == pytest.approx(round(first["latitude"],  5), abs=1e-5)

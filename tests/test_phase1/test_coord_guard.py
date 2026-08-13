"""Unit tests for P1-COORD-GUARD: _drop_invalid_coords in filter_potw.py.

All tests use small synthetic polars frames; no IO, no CSV files required.

Coverage:
  - Legitimate US territories pass (CONUS, AK, HI, GU, AS, PR, VI)
  - Each of the 5 real error classes from §1 is correctly rejected
  - Boundary values are inclusive (band endpoints are kept)
  - Null coordinates are left for the upstream null-drop in _join
  - Empty frame is a no-op
  - All-valid frame is returned unchanged
  - Row count: only invalid rows removed
  - Logged npdes_ids from the 10 known bad rows are correctly rejected
"""

from __future__ import annotations

import polars as pl
import pytest

from src.phase1.filter_potw import _drop_invalid_coords


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(npdes_id: str, state: str, lat: float | None, lon: float | None) -> dict:
    return {
        "npdes_id":   npdes_id,
        "state_code": state,
        "latitude":   lat,
        "longitude":  lon,
        "facility_name": "Test WWTP",
    }


def _df(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


# ── Legitimate US territories — must PASS ─────────────────────────────────────

class TestLegitimateTerritoriesPass:

    def test_conus_typical(self):
        df = _df(_row("OH0001", "OH", 39.9, -83.0))
        assert _drop_invalid_coords(df).height == 1

    def test_alaska(self):
        df = _df(_row("AK0001", "AK", 71.3, -156.8))
        assert _drop_invalid_coords(df).height == 1

    def test_alaska_southeast(self):
        df = _df(_row("AK0002", "AK", 55.357196, -131.696532))
        assert _drop_invalid_coords(df).height == 1

    def test_hawaii(self):
        df = _df(_row("HI0001", "HI", 21.33, -158.04))
        assert _drop_invalid_coords(df).height == 1

    def test_guam(self):
        """Guam ~13.5°N, 144.8°E — must NOT be rejected by a naïve lat<15 rule."""
        df = _df(_row("GU0001", "GU", 13.55, 144.8))
        assert _drop_invalid_coords(df).height == 1

    def test_guam_lower_bound(self):
        df = _df(_row("GU0002", "GU", 13.0, 144.5))   # band lower bounds
        assert _drop_invalid_coords(df).height == 1

    def test_american_samoa(self):
        """AS ~-14.3°S, -170°W — in the negative-lat band [-14.8, -10.8]."""
        df = _df(_row("AS0001", "AS", -14.3, -170.7))
        assert _drop_invalid_coords(df).height == 1

    def test_puerto_rico(self):
        df = _df(_row("PR0001", "PR", 18.4, -66.1))
        assert _drop_invalid_coords(df).height == 1

    def test_usvi(self):
        df = _df(_row("VI0001", "VI", 18.0, -64.8))
        assert _drop_invalid_coords(df).height == 1

    def test_key_west_fla(self):
        df = _df(_row("FL0001", "FL", 24.55, -81.8))
        assert _drop_invalid_coords(df).height == 1

    def test_cnmi(self):
        df = _df(_row("MP0001", "MP", 15.2, 145.7))
        assert _drop_invalid_coords(df).height == 1


# ── Band boundary values are inclusive ────────────────────────────────────────

class TestBandBoundaries:

    def test_lat_band1_lower_bound(self):
        """Lat = -14.8 is valid (AS band lower bound)."""
        df = _df(_row("AS0001", "AS", -14.8, -170.0))
        assert _drop_invalid_coords(df).height == 1

    def test_lat_band1_upper_bound(self):
        """Lat = -10.8 is valid (AS band upper bound)."""
        df = _df(_row("AS0002", "AS", -10.8, -171.0))
        assert _drop_invalid_coords(df).height == 1

    def test_lat_band2_lower_bound(self):
        """Lat = 13.0 is valid (GU/CNMI lower bound)."""
        df = _df(_row("GU0001", "GU", 13.0, 144.7))
        assert _drop_invalid_coords(df).height == 1

    def test_lat_band2_upper_bound(self):
        """Lat = 71.5 is valid (AK upper bound)."""
        df = _df(_row("AK0001", "AK", 71.5, -160.0))
        assert _drop_invalid_coords(df).height == 1

    def test_lon_band1_lower_bound(self):
        """Lon = -180.0 is valid (AK/HI/AS western edge)."""
        df = _df(_row("AK0001", "AK", 51.9, -180.0))
        assert _drop_invalid_coords(df).height == 1

    def test_lon_band1_upper_bound(self):
        """Lon = -64.4 is valid (USVI eastern tip)."""
        df = _df(_row("VI0001", "VI", 18.0, -64.4))
        assert _drop_invalid_coords(df).height == 1

    def test_lon_band2_lower_bound(self):
        """Lon = 144.5 is valid (GU western edge)."""
        df = _df(_row("GU0001", "GU", 13.5, 144.5))
        assert _drop_invalid_coords(df).height == 1

    def test_lon_band2_upper_bound(self):
        """Lon = 146.2 is valid (CNMI eastern edge)."""
        df = _df(_row("MP0001", "MP", 15.1, 146.2))
        assert _drop_invalid_coords(df).height == 1


# ── Each real error class is rejected ─────────────────────────────────────────

class TestRealErrorClassesRejected:
    """Each of the 10 known bad rows (5 error classes from §1) is rejected."""

    def test_positive_lon_sign_flip_wy(self):
        """WYG589102 GREAT PLAINS LAGOON WY: lon = +108.477 (should be -108)."""
        df = _df(_row("WYG589102", "WY", 42.9741, 108.477))
        assert _drop_invalid_coords(df).height == 0

    def test_positive_lon_sign_flip_ms1(self):
        """MS0061671 ABBEVILLE POTW MS: lon = +89.50 (should be -89)."""
        df = _df(_row("MS0061671", "MS", 34.504633, 89.504902))
        assert _drop_invalid_coords(df).height == 0

    def test_positive_lon_sign_flip_tx(self):
        """TX0137146 CANUTILLO MIDDLE SCHOOL WWTP TX: lon = +106.6 (should be -106)."""
        df = _df(_row("TX0137146", "TX", 31.926388, 106.608888))
        assert _drop_invalid_coords(df).height == 0

    def test_truncated_lat_nj(self):
        """NJ0020371 CAPE MAY REG WTF NJ: lat = 8.94 (should be ~38.9)."""
        df = _df(_row("NJ0020371", "NJ", 8.943708, -74.961818))
        assert _drop_invalid_coords(df).height == 0

    def test_corrupt_lat_wi(self):
        """WI0025194 RACINE WASTEWATER UTILITY WI: lat = 4.4 (should be ~42.7)."""
        df = _df(_row("WI0025194", "WI", 4.4, -87.766667))
        assert _drop_invalid_coords(df).height == 0

    def test_positive_lon_sign_flip_sc(self):
        """SC0047457 BOARD OF PUBLIC WORKS CANOE CR SC: lon = +81.5 (should be -81)."""
        df = _df(_row("SC0047457", "SC", 35.116885, 81.533247))
        assert _drop_invalid_coords(df).height == 0

    def test_corrupt_lat_ms1(self):
        """MS0024589 QUITMAN POTW MS: lat = 3.34 (should be ~32.0)."""
        df = _df(_row("MS0024589", "MS", 3.339083, -88.737083))
        assert _drop_invalid_coords(df).height == 0

    def test_corrupt_lon_pr(self):
        """PR0026042 VILLAS DEL GIGANTE, CAROLINA PR: lon = -56.17 (should be ~-66)."""
        df = _df(_row("PR0026042", "PR", 18.0, -56.166667))
        assert _drop_invalid_coords(df).height == 0

    def test_lat_sign_flip_ms(self):
        """MS0052477 BYHALIA POTW MS: lat = -34.9 (should be +34.9)."""
        df = _df(_row("MS0052477", "MS", -34.904056, -89.699667))
        assert _drop_invalid_coords(df).height == 0

    def test_positive_lon_sign_flip_ms2(self):
        """MS0020575 ACKERMAN POTW MS: lon = +89.19 (should be -89)."""
        df = _df(_row("MS0020575", "MS", 33.307778, 89.189139))
        assert _drop_invalid_coords(df).height == 0


# ── Gap between bands is rejected ─────────────────────────────────────────────

class TestBandGapRejected:

    def test_lat_between_as_and_gu_bands(self):
        """lat = 0.0 is in the gap between AS band and GU band — rejected."""
        df = _df(_row("XX0001", "XX", 0.0, -80.0))
        assert _drop_invalid_coords(df).height == 0

    def test_lat_just_below_as_band(self):
        """lat = -15.0 is below the AS band lower bound -14.8 — rejected."""
        df = _df(_row("XX0001", "XX", -15.0, -170.0))
        assert _drop_invalid_coords(df).height == 0

    def test_lat_just_above_ak_band(self):
        """lat = 72.0 exceeds AK upper bound 71.5 — rejected."""
        df = _df(_row("XX0001", "XX", 72.0, -150.0))
        assert _drop_invalid_coords(df).height == 0

    def test_lon_between_western_and_gu_bands(self):
        """lon = 50.0 falls between -64.4 and 144.5 — rejected."""
        df = _df(_row("XX0001", "XX", 30.0, 50.0))
        assert _drop_invalid_coords(df).height == 0


# ── Null coordinates ──────────────────────────────────────────────────────────

class TestNullCoords:

    def test_null_lat_passes_guard(self):
        """Null lat is left for upstream null-drop in _join; not rejected here."""
        df = _df(_row("XX0001", "XX", None, -80.0))
        assert _drop_invalid_coords(df).height == 1

    def test_null_lon_passes_guard(self):
        df = _df(_row("XX0001", "XX", 40.0, None))
        assert _drop_invalid_coords(df).height == 1

    def test_both_null_passes_guard(self):
        df = _df(_row("XX0001", "XX", None, None))
        assert _drop_invalid_coords(df).height == 1


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_frame_no_op(self):
        df = pl.DataFrame({
            "npdes_id": pl.Series([], dtype=pl.Utf8),
            "state_code": pl.Series([], dtype=pl.Utf8),
            "latitude":  pl.Series([], dtype=pl.Float64),
            "longitude": pl.Series([], dtype=pl.Float64),
            "facility_name": pl.Series([], dtype=pl.Utf8),
        })
        result = _drop_invalid_coords(df)
        assert result.height == 0

    def test_all_valid_unchanged(self):
        df = _df(
            _row("OH0001", "OH", 41.0, -83.0),
            _row("CA0001", "CA", 37.5, -122.0),
            _row("AK0001", "AK", 61.2, -149.9),
        )
        result = _drop_invalid_coords(df)
        assert result.height == 3

    def test_mixed_valid_invalid(self):
        """Valid rows survive; invalid rows are dropped."""
        df = _df(
            _row("GOOD1", "OH", 41.0,  -83.0),   # valid CONUS
            _row("BAD1",  "WY", 42.97, 108.48),   # lon sign flip
            _row("GOOD2", "GU", 13.5,  144.8),    # valid Guam
            _row("BAD2",  "WI", 4.4,   -87.77),   # corrupt lat
        )
        result = _drop_invalid_coords(df)
        assert result.height == 2
        kept_ids = set(result["npdes_id"].to_list())
        assert kept_ids == {"GOOD1", "GOOD2"}

    def test_only_invalid_rows_all_dropped(self):
        df = _df(
            _row("BAD1", "WY", 42.97, 108.48),
            _row("BAD2", "WI", 4.4,   -87.77),
        )
        assert _drop_invalid_coords(df).height == 0

    def test_row_count_reduced_by_n_invalid(self):
        """Output row count = input count minus number of invalid rows."""
        valid = [_row(f"V{i}", "OH", 41.0 + i * 0.01, -83.0) for i in range(5)]
        invalid = [_row("BAD", "WY", 42.97, 108.48)]
        df = _df(*valid, *invalid)
        result = _drop_invalid_coords(df)
        assert result.height == 5

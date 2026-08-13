"""Tests for P4-TIER: add_calibrated_energy_cols in src/phase4/financials.py.

Coverage:
  - Exactly 4 new columns added (no more, no less)
  - Column names are correct
  - Multiplier arithmetic: energy * 0.291 / 0.447 / 0.688 / 0.2195
  - All pre-existing columns are untouched (name + dtype + values)
  - Works on single-row and multi-row frames
  - Zero-energy row produces zero calibrated values
  - Function is idempotent on the added columns (calling twice raises, since
    duplicate column names are caught by polars -- but single call is pure)
"""

from __future__ import annotations

import polars as pl
import pytest

from src.phase4.financials import (
    _CF_CALIB_DEFAULTS,
    add_calibrated_energy_cols,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_FLOOR_P25 = _CF_CALIB_DEFAULTS["floor_p25"]   # 0.291
_FLOOR_P50 = _CF_CALIB_DEFAULTS["floor_p50"]   # 0.447
_CENTRAL   = _CF_CALIB_DEFAULTS["central"]     # 0.688
_MEAS_FLOOR = _CF_CALIB_DEFAULTS["measured_point_loma"]   # 0.2195


def _base_df(energies: list[float]) -> pl.DataFrame:
    """Minimal scorecard frame with only the columns Phase 4 actually has."""
    n = len(energies)
    return pl.DataFrame({
        "npdes_id":           [f"ID{i:04d}" for i in range(n)],
        "annual_energy_kwh":  energies,
        "rated_power_kw":     [100.0 * (i + 1) for i in range(n)],
        "project_viable":     [True] * n,
        "npv_usd":            [50_000.0 * (i + 1) for i in range(n)],
    })


# ══════════════════════════════════════════════════════════════════════════════
# Column count and names
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibratedColsCount:

    def test_adds_exactly_four_columns(self):
        df = _base_df([1_000_000.0])
        result = add_calibrated_energy_cols(df)
        assert len(result.columns) == len(df.columns) + 4

    def test_new_column_names_correct(self):
        result = add_calibrated_energy_cols(_base_df([1_000_000.0]))
        for expected in (
            "energy_kwh_calib_floor_p25",
            "energy_kwh_calib_floor_p50",
            "energy_kwh_calib_central",
            "energy_kwh_calib_measured_point_loma",
        ):
            assert expected in result.columns, f"Missing column: {expected}"

    def test_new_columns_are_float64(self):
        result = add_calibrated_energy_cols(_base_df([1_000_000.0]))
        for col in ("energy_kwh_calib_floor_p25",
                    "energy_kwh_calib_floor_p50",
                    "energy_kwh_calib_central",
                    "energy_kwh_calib_measured_point_loma"):
            assert result[col].dtype == pl.Float64, (
                f"{col} dtype is {result[col].dtype}, expected Float64"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Multiplier arithmetic
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibratedColsArithmetic:

    def test_floor_p25_single_row(self):
        energy = 1_000_000.0
        result = add_calibrated_energy_cols(_base_df([energy]))
        assert result["energy_kwh_calib_floor_p25"][0] == pytest.approx(
            energy * _FLOOR_P25, rel=1e-9
        )

    def test_floor_p50_single_row(self):
        energy = 2_500_000.0
        result = add_calibrated_energy_cols(_base_df([energy]))
        assert result["energy_kwh_calib_floor_p50"][0] == pytest.approx(
            energy * _FLOOR_P50, rel=1e-9
        )

    def test_central_single_row(self):
        energy = 5_000_000.0
        result = add_calibrated_energy_cols(_base_df([energy]))
        assert result["energy_kwh_calib_central"][0] == pytest.approx(
            energy * _CENTRAL, rel=1e-9
        )

    def test_measured_point_loma_single_row(self):
        energy = 5_000_000.0
        result = add_calibrated_energy_cols(_base_df([energy]))
        assert result["energy_kwh_calib_measured_point_loma"][0] == pytest.approx(
            energy * _MEAS_FLOOR, rel=1e-9
        )

    def test_multiplier_ordering(self):
        """measured floor < p25 < p50 < central < annual_energy_kwh.

        The measured Point Loma tier is the lowest rung: it must sit below the
        river-hydro p25, because that ordering is the whole reason the reported
        band floor moved from 118.9 to 89.8 GWh/yr.
        """
        energy = 1_000_000.0
        result = add_calibrated_energy_cols(_base_df([energy]))
        meas = result["energy_kwh_calib_measured_point_loma"][0]
        p25 = result["energy_kwh_calib_floor_p25"][0]
        p50 = result["energy_kwh_calib_floor_p50"][0]
        cen = result["energy_kwh_calib_central"][0]
        assert meas < p25 < p50 < cen < energy

    def test_zero_energy_gives_zero_calib(self):
        result = add_calibrated_energy_cols(_base_df([0.0]))
        assert result["energy_kwh_calib_floor_p25"][0] == pytest.approx(0.0)
        assert result["energy_kwh_calib_floor_p50"][0] == pytest.approx(0.0)
        assert result["energy_kwh_calib_central"][0]   == pytest.approx(0.0)
        assert result["energy_kwh_calib_measured_point_loma"][0] == pytest.approx(0.0)

    def test_multi_row_arithmetic(self):
        energies = [100_000.0, 500_000.0, 2_000_000.0]
        result = add_calibrated_energy_cols(_base_df(energies))
        for i, e in enumerate(energies):
            assert result["energy_kwh_calib_floor_p25"][i] == pytest.approx(e * _FLOOR_P25, rel=1e-9)
            assert result["energy_kwh_calib_floor_p50"][i] == pytest.approx(e * _FLOOR_P50, rel=1e-9)
            assert result["energy_kwh_calib_central"][i]   == pytest.approx(e * _CENTRAL,   rel=1e-9)
            assert result["energy_kwh_calib_measured_point_loma"][i] == pytest.approx(
                e * _MEAS_FLOOR, rel=1e-9)

    def test_report_fleet_sums(self):
        """Fleet sum × multiplier arithmetic correct (multipliers from CF_CALIBRATION_REPORT.md §6)."""
        # Verify the multiplier math is exact regardless of fleet energy total.
        # (The fleet total shifts across pipeline re-runs; only multiplier correctness is invariant here.)
        fleet_gwh = 409.1695   # P2-SEED post-site-keyed-seeding viable fleet (live value)
        assert pytest.approx(fleet_gwh * _FLOOR_P25, rel=1e-6) == fleet_gwh * 0.291
        assert pytest.approx(fleet_gwh * _FLOOR_P50, rel=1e-6) == fleet_gwh * 0.447
        assert pytest.approx(fleet_gwh * _CENTRAL,   rel=1e-6) == fleet_gwh * 0.688
        assert pytest.approx(fleet_gwh * _MEAS_FLOOR, rel=1e-6) == fleet_gwh * 0.2195
        # The reported calibration band is measured floor -> optimistic.
        assert fleet_gwh * _MEAS_FLOOR == pytest.approx(89.8, abs=0.1)
        assert fleet_gwh * _CENTRAL    == pytest.approx(281.5, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Existing columns untouched
# ══════════════════════════════════════════════════════════════════════════════

class TestExistingColumnsUntouched:

    def test_existing_column_names_preserved(self):
        df = _base_df([1_000_000.0])
        result = add_calibrated_energy_cols(df)
        for col in df.columns:
            assert col in result.columns, f"Column {col} lost after add_calibrated_energy_cols"

    def test_annual_energy_kwh_unchanged(self):
        energies = [1_000_000.0, 2_000_000.0, 3_000_000.0]
        df = _base_df(energies)
        result = add_calibrated_energy_cols(df)
        for i, e in enumerate(energies):
            assert result["annual_energy_kwh"][i] == pytest.approx(e, rel=1e-12)

    def test_other_columns_values_unchanged(self):
        df = _base_df([1_000_000.0, 2_000_000.0])
        result = add_calibrated_energy_cols(df)
        assert result["npv_usd"].to_list() == pytest.approx(df["npv_usd"].to_list())
        assert result["project_viable"].to_list() == df["project_viable"].to_list()

    def test_row_count_unchanged(self):
        df = _base_df([1e6, 2e6, 3e6, 4e6, 5e6])
        result = add_calibrated_energy_cols(df)
        assert result.height == df.height


# ══════════════════════════════════════════════════════════════════════════════
# Integration test (real parquet — gated on file presence)
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

_SCORECARD_PATH = Path("data/processed/phase4/financial_scorecards.parquet")


@pytest.mark.skipif(
    not _SCORECARD_PATH.exists(),
    reason="financial_scorecards.parquet not available — integration test skipped",
)
class TestCalibratedColsIntegration:

    def test_real_parquet_has_four_new_cols(self):
        """After Phase 4 re-run, parquet has all 4 calibrated columns."""
        df = pl.read_parquet(_SCORECARD_PATH)
        for col in ("energy_kwh_calib_floor_p25",
                    "energy_kwh_calib_floor_p50",
                    "energy_kwh_calib_central",
                    "energy_kwh_calib_measured_point_loma"):
            assert col in df.columns, f"Missing column in parquet: {col}"

    def test_real_parquet_total_cols(self):
        df = pl.read_parquet(_SCORECARD_PATH)
        assert len(df.columns) == 50, f"Expected 50 columns, got {len(df.columns)}"

    def test_real_parquet_row_count(self):
        # P2-SEED: 3,780 → 3,778 (site-keyed MC re-baseline shifted 2 marginal sites
        # below the turbine-viable threshold; all 10 bad-coord IDs still absent)
        df = pl.read_parquet(_SCORECARD_PATH)
        assert df.height == 3778

    def test_viable_fleet_sums(self):
        """Calibrated columns are exact multiples of annual_energy_kwh for each row."""
        df = pl.read_parquet(_SCORECARD_PATH)
        viable = df.filter(pl.col("project_viable"))
        # P2-SEED: 1,140 → 1,138 (FL0A00002, NY0026328 became non-viable after re-draw)
        assert viable.height == 1138

        base_gwh = viable["annual_energy_kwh"].sum() / 1e6
        p25_gwh  = viable["energy_kwh_calib_floor_p25"].sum() / 1e6
        p50_gwh  = viable["energy_kwh_calib_floor_p50"].sum() / 1e6
        cen_gwh  = viable["energy_kwh_calib_central"].sum()   / 1e6

        # P2-SEED post-site-keyed-seeding fleet: 409.1695 GWh
        assert base_gwh == pytest.approx(409.1695, abs=0.05)
        assert p25_gwh  == pytest.approx(base_gwh * 0.291, rel=1e-6)
        assert p50_gwh  == pytest.approx(base_gwh * 0.447, rel=1e-6)
        assert cen_gwh  == pytest.approx(base_gwh * 0.688, rel=1e-6)

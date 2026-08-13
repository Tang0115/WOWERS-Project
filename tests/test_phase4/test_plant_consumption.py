"""Tests for src/phase4/plant_consumption.py (F4-OFFSET).

Coverage:
  * observed_intensity() at band boundaries
  * intensity() log-linear interpolation + edge clamping
  * consumption arithmetic (flow × 365 × intensity)
  * band ordering invariants (achievable):
      est_consumption_low (TF) <= est_consumption_high (advanced+N)
      energy_offset_pct_low <= energy_offset_pct_high
    Note: the point estimate (Table 5-1 observed) is NOT bracketed by the
    Table 5-4 treatment-type band — observed exceeds advanced+N at every
    flow.  See TestBandOrderingInvariants for the documented explanation.
  * null/zero flow guard — all 6 keys return None, no divide-by-zero
"""

from __future__ import annotations

import math

import pytest

from src.phase4.plant_consumption import (
    consumption_and_offset,
    intensity,
    observed_intensity,
    _SENS_HIGH,
    _SENS_LOW,
)


# ── observed_intensity band tests ─────────────────────────────────────────────

class TestObservedIntensity:
    """Band lookup uses strict ``flow < max_mgd`` (matching validate script)."""

    def test_band_1_low_end(self):
        # flow < 2 → 3300
        assert observed_intensity(1.0) == pytest.approx(3300.0)

    def test_band_1_interior(self):
        assert observed_intensity(0.5) == pytest.approx(3300.0)

    def test_band_boundary_2_goes_to_next_band(self):
        # flow == 2: NOT < 2, so first band does NOT match → falls to band [<4] → 3000
        assert observed_intensity(2.0) == pytest.approx(3000.0)

    def test_band_2_interior(self):
        # 2 <= 3 < 4 → 3000
        assert observed_intensity(3.0) == pytest.approx(3000.0)

    def test_band_3_interior(self):
        # 4 <= 5 < 7 → 2400
        assert observed_intensity(5.0) == pytest.approx(2400.0)

    def test_band_4_interior(self):
        # 7 <= 10 < 16 → 2000
        assert observed_intensity(10.0) == pytest.approx(2000.0)

    def test_band_5_interior(self):
        # 16 <= 25 < 46 → 1700
        assert observed_intensity(25.0) == pytest.approx(1700.0)

    def test_band_6_interior(self):
        # 46 <= 50 < 100 → 1700
        assert observed_intensity(50.0) == pytest.approx(1700.0)

    def test_band_catch_all_exactly_100(self):
        # flow == 100: NOT < 100 for band[<100], falls to null band → 1600
        assert observed_intensity(100.0) == pytest.approx(1600.0)

    def test_band_catch_all_large(self):
        # flow >= 100 → 1600
        assert observed_intensity(150.0) == pytest.approx(1600.0)
        assert observed_intensity(500.0) == pytest.approx(1600.0)

    def test_monotone_decreasing_with_size(self):
        # Larger plants consume fewer kWh/MG (economies of scale).
        flows = [0.5, 1.5, 3.0, 5.0, 12.0, 30.0, 75.0, 200.0]
        intensities = [observed_intensity(f) for f in flows]
        for i in range(len(intensities) - 1):
            assert intensities[i] >= intensities[i + 1], (
                f"Not non-increasing at {flows[i]} → {flows[i+1]}: "
                f"{intensities[i]} vs {intensities[i+1]}"
            )


# ── intensity() log-linear interpolation + edge clamping ─────────────────────

class TestIntensityLogLinear:
    """Test against the Table 5-4 row for trickling_filter:
    size_points = [1, 5, 10, 20, 50, 100]
    kwh_per_mg  = [1811, 978, 852, 750, 687, 673]
    """

    _TYPE = "trickling_filter"

    def test_at_lower_boundary_returns_first_value(self):
        assert intensity(self._TYPE, 0.5) == pytest.approx(1811.0)

    def test_at_first_table_point_exactly(self):
        assert intensity(self._TYPE, 1.0) == pytest.approx(1811.0)

    def test_at_upper_boundary_returns_last_value(self):
        assert intensity(self._TYPE, 200.0) == pytest.approx(673.0)

    def test_at_last_table_point_exactly(self):
        assert intensity(self._TYPE, 100.0) == pytest.approx(673.0)

    def test_midpoint_interpolation_between_1_and_5(self):
        # log-linear midpoint at flow = sqrt(1*5) = ~2.236
        mid = math.sqrt(1.0 * 5.0)
        result = intensity(self._TYPE, mid)
        # Should be geometric midpoint of 1811 and 978: sqrt(1811*978)
        expected = math.sqrt(1811.0 * 978.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_monotone_decreasing_with_flow(self):
        flows = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        vals  = [intensity(self._TYPE, f) for f in flows]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6, (
                f"Not non-increasing at {flows[i]} → {flows[i+1]}"
            )

    def test_advanced_with_nitrification_higher_than_trickling(self):
        # advanced+N always consumes more energy than trickling filter
        for flow in (1.0, 5.0, 20.0, 100.0):
            low_val  = intensity("trickling_filter", flow)
            high_val = intensity("advanced_with_nitrification", flow)
            assert high_val > low_val, (
                f"advanced+N not > TF at {flow} MGD: {high_val} vs {low_val}"
            )


# ── consumption_and_offset arithmetic ────────────────────────────────────────

class TestConsumptionAndOffset:
    def test_point_consumption_formula(self):
        """est_plant_consumption_kwh_yr == flow × 365 × observed_intensity(flow)."""
        flow = 10.0
        result = consumption_and_offset(flow, annual_energy_kwh=1_000.0)
        expected = flow * 365.0 * observed_intensity(flow)
        assert result["est_plant_consumption_kwh_yr"] == pytest.approx(expected, rel=1e-9)

    def test_low_consumption_formula(self):
        flow = 10.0
        result = consumption_and_offset(flow, annual_energy_kwh=1_000.0)
        expected = flow * 365.0 * intensity(_SENS_LOW, flow)
        assert result["est_plant_consumption_low_kwh_yr"] == pytest.approx(expected, rel=1e-9)

    def test_high_consumption_formula(self):
        flow = 10.0
        result = consumption_and_offset(flow, annual_energy_kwh=1_000.0)
        expected = flow * 365.0 * intensity(_SENS_HIGH, flow)
        assert result["est_plant_consumption_high_kwh_yr"] == pytest.approx(expected, rel=1e-9)

    def test_offset_pct_formula(self):
        """energy_offset_pct == annual_energy_kwh / point_consumption × 100."""
        flow   = 5.0
        energy = 50_000.0
        result = consumption_and_offset(flow, energy)
        point_cons = flow * 365.0 * observed_intensity(flow)
        expected = energy / point_cons * 100.0
        assert result["energy_offset_pct"] == pytest.approx(expected, rel=1e-9)

    def test_offset_low_uses_high_consumption_denominator(self):
        """offset_low = energy / HIGH_consumption × 100."""
        flow   = 20.0
        energy = 100_000.0
        result = consumption_and_offset(flow, energy)
        high_cons = flow * 365.0 * intensity(_SENS_HIGH, flow)
        expected  = energy / high_cons * 100.0
        assert result["energy_offset_pct_low"] == pytest.approx(expected, rel=1e-9)

    def test_offset_high_uses_low_consumption_denominator(self):
        """offset_high = energy / LOW_consumption × 100."""
        flow   = 20.0
        energy = 100_000.0
        result = consumption_and_offset(flow, energy)
        low_cons = flow * 365.0 * intensity(_SENS_LOW, flow)
        expected = energy / low_cons * 100.0
        assert result["energy_offset_pct_high"] == pytest.approx(expected, rel=1e-9)


# ── Band-ordering invariants ──────────────────────────────────────────────────

class TestBandOrderingInvariants:
    """Invariants that must hold for all valid (flow, energy) inputs.

    Note on why the POINT estimate is NOT between cons_low and cons_high:
    The point estimate uses EPRI Table 5-1 OBSERVED flow-band averages
    (treatment-type-agnostic), while the sensitivity band uses Table 5-4
    theoretical treatment-type curves.  The YAML itself documents that
    "WEF Table 5-4 values run lower than observed values, especially above
    10 MGD." Empirically Table 5-1 values exceed the advanced_with_nitrification
    (high) curve at every flow, so cons_point > cons_high universally.
    The binding invariants are (a) cons_low <= cons_high and
    (b) offset_pct_low <= offset_pct_high — both always hold because
    trickling_filter is always cheaper than advanced_with_nitrification.
    """

    @pytest.mark.parametrize("flow, energy", [
        (0.5, 100.0),
        (3.0, 50_000.0),
        (10.0, 200_000.0),
        (50.0, 1_000_000.0),
        (150.0, 5_000_000.0),
    ])
    def test_consumption_low_le_high(self, flow, energy):
        """TF (low) always cheaper than advanced+N (high)."""
        r = consumption_and_offset(flow, energy)
        assert r["est_plant_consumption_low_kwh_yr"] <= r["est_plant_consumption_high_kwh_yr"] + 1e-6

    @pytest.mark.parametrize("flow, energy", [
        (0.5, 100.0),
        (3.0, 50_000.0),
        (10.0, 200_000.0),
        (50.0, 1_000_000.0),
        (150.0, 5_000_000.0),
    ])
    def test_offset_pct_low_le_pct_high(self, flow, energy):
        """offset_pct_low (energy/cons_HIGH) <= offset_pct_high (energy/cons_LOW)."""
        r = consumption_and_offset(flow, energy)
        assert r["energy_offset_pct_low"] <= r["energy_offset_pct_high"] + 1e-9, (
            f"offset_low > offset_high at flow={flow}: "
            f"{r['energy_offset_pct_low']} > {r['energy_offset_pct_high']}"
        )

    @pytest.mark.parametrize("flow, energy", [
        (0.5, 100.0),
        (3.0, 50_000.0),
        (10.0, 200_000.0),
        (50.0, 1_000_000.0),
        (150.0, 5_000_000.0),
    ])
    def test_all_offsets_positive(self, flow, energy):
        r = consumption_and_offset(flow, energy)
        assert r["energy_offset_pct"]      > 0
        assert r["energy_offset_pct_low"]  > 0
        assert r["energy_offset_pct_high"] > 0

    def test_six_keys_always_returned(self):
        expected_keys = {
            "est_plant_consumption_kwh_yr",
            "est_plant_consumption_low_kwh_yr",
            "est_plant_consumption_high_kwh_yr",
            "energy_offset_pct",
            "energy_offset_pct_low",
            "energy_offset_pct_high",
        }
        result = consumption_and_offset(10.0, 1_000.0)
        assert set(result.keys()) == expected_keys


# ── Null / zero guard ─────────────────────────────────────────────────────────

class TestNullAndZeroGuard:
    """All 6 keys must be None; no ZeroDivisionError or crash."""

    def test_none_flow_returns_all_none(self):
        result = consumption_and_offset(None, annual_energy_kwh=50_000.0)
        for k, v in result.items():
            assert v is None, f"Expected None for {k}, got {v}"

    def test_zero_flow_returns_all_none(self):
        result = consumption_and_offset(0.0, annual_energy_kwh=50_000.0)
        for k, v in result.items():
            assert v is None, f"Expected None for {k} at flow=0, got {v}"

    def test_negative_flow_returns_all_none(self):
        result = consumption_and_offset(-1.5, annual_energy_kwh=50_000.0)
        for k, v in result.items():
            assert v is None, f"Expected None for {k} at flow<0, got {v}"

    def test_no_divide_by_zero_on_zero_flow(self):
        # Must not raise
        result = consumption_and_offset(0.0, 0.0)
        assert result["energy_offset_pct"] is None

    def test_six_keys_present_even_when_null(self):
        expected_keys = {
            "est_plant_consumption_kwh_yr",
            "est_plant_consumption_low_kwh_yr",
            "est_plant_consumption_high_kwh_yr",
            "energy_offset_pct",
            "energy_offset_pct_low",
            "energy_offset_pct_high",
        }
        result = consumption_and_offset(None, 0.0)
        assert set(result.keys()) == expected_keys


# ── Sensitivity-low/high label check ─────────────────────────────────────────

class TestSensitivityLabels:
    def test_sens_low_is_trickling_filter(self):
        assert _SENS_LOW == "trickling_filter"

    def test_sens_high_is_advanced_with_nitrification(self):
        assert _SENS_HIGH == "advanced_with_nitrification"

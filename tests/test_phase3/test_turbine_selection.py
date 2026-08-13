"""Tests for phase3/turbine_selection.py — H-Q selection, efficiency curves, sizing."""

from __future__ import annotations

import polars as pl
import pytest

from src.phase3.turbine_selection import (
    MIN_CF,
    _PHASE1_FDC_EXCEEDANCES,
    _compute_annual_energy,
    efficiency_at_part_load,
    optimize_rated_power,
    peak_efficiency,
    select_and_size_turbines,
    select_turbine_type,
)

# ── Constants used in physics checks ─────────────────────────────────────────
RHO = 998.2
G = 9.81


# ── Turbine type selection ────────────────────────────────────────────────────

class TestSelectTurbineType:
    def test_pelton_high_head_low_flow(self):
        assert select_turbine_type(h_net_m=100.0, q_m3s=0.5) == "Pelton"

    def test_pelton_boundary(self):
        assert select_turbine_type(h_net_m=51.0, q_m3s=1.9) == "Pelton"

    def test_francis_medium_head(self):
        assert select_turbine_type(h_net_m=30.0, q_m3s=5.0) == "Francis"

    def test_francis_high_head_high_flow(self):
        # H > 50 but Q >= 2 → Francis (Pelton can't handle high Q)
        assert select_turbine_type(h_net_m=80.0, q_m3s=5.0) == "Francis"

    def test_kaplan_low_head_high_flow(self):
        assert select_turbine_type(h_net_m=3.0, q_m3s=2.0) == "Kaplan"

    def test_kaplan_boundary(self):
        assert select_turbine_type(h_net_m=9.9, q_m3s=0.5) == "Kaplan"

    def test_in_conduit_low_flow(self):
        # h<2m and q<0.5 → in_conduit_micro (Crossflow starts at 2m)
        assert select_turbine_type(h_net_m=1.5, q_m3s=0.1) == "in_conduit_micro"

    def test_crossflow_medium_low_head(self):
        # h in [2, 10) and q<0.5 → Crossflow (Ossberger/CINK range)
        assert select_turbine_type(h_net_m=2.0, q_m3s=0.1) == "Crossflow"
        assert select_turbine_type(h_net_m=6.0, q_m3s=0.3) == "Crossflow"

    def test_in_conduit_very_low_head(self):
        assert select_turbine_type(h_net_m=1.5, q_m3s=0.3) == "in_conduit_micro"


# ── Efficiency curves ─────────────────────────────────────────────────────────

class TestEfficiencyCurves:
    @pytest.mark.parametrize("turbine", ["Kaplan", "Francis", "Pelton", "in_conduit_micro"])
    def test_efficiency_in_unit_interval(self, turbine):
        for q in [0.0, 0.1, 0.5, 0.8, 1.0]:
            eta = efficiency_at_part_load(turbine, q)
            assert 0.0 <= eta <= 1.0, (
                f"{turbine} at q={q}: efficiency {eta} out of [0,1]"
            )

    def test_pelton_minimum_nozzle_cutoff(self):
        """Pelton should produce zero power below 15% nozzle opening."""
        assert efficiency_at_part_load("Pelton", 0.14) == 0.0
        assert efficiency_at_part_load("Pelton", 0.15) > 0.0

    def test_in_conduit_minimum_flow_cutoff(self):
        assert efficiency_at_part_load("in_conduit_micro", 0.09) == 0.0
        assert efficiency_at_part_load("in_conduit_micro", 0.10) > 0.0

    def test_kaplan_peak_near_rated(self):
        """Kaplan efficiency should be higher near rated (q=1.0) than at overload."""
        eta_rated  = efficiency_at_part_load("Kaplan", 1.0)
        eta_half   = efficiency_at_part_load("Kaplan", 0.5)
        # Kaplan maintains efficiency well, but rated > half-load
        assert eta_rated >= eta_half

    def test_peak_efficiency_matches_q1(self):
        """peak_efficiency() should equal efficiency_at_part_load(type, 1.0)."""
        for t in ["Kaplan", "Francis", "Pelton", "in_conduit_micro"]:
            assert peak_efficiency(t) == efficiency_at_part_load(t, 1.0)


# ── Rated-power optimizer ─────────────────────────────────────────────────────

class TestOptimizeRatedPower:
    def _flat_fdc(self, q: float, n: int = 10):
        """Flat FDC at constant flow q (q never varies)."""
        exceedances = [i / (n - 1) for i in range(n)]
        flows       = [q] * n
        return flows, exceedances

    def test_returns_four_values(self):
        result = optimize_rated_power(
            "Kaplan", q_design_m3s=1.0, h_net_m=5.0,
            fdc_flows_m3s=[1.0, 0.8, 0.6], fdc_exceedances=[0.1, 0.5, 0.9]
        )
        assert len(result) == 4

    def test_rated_power_positive(self):
        flows, exceedances = self._flat_fdc(1.0)
        q_r, p_kw, energy, cf = optimize_rated_power(
            "Kaplan", 1.0, 5.0, flows, exceedances
        )
        assert p_kw > 0, "Rated power must be positive"

    def test_capacity_factor_in_valid_range(self):
        flows, exceedances = self._flat_fdc(2.0)
        q_r, p_kw, energy, cf = optimize_rated_power(
            "Francis", 2.0, 20.0, flows, exceedances
        )
        assert 0.0 <= cf <= 1.0, f"CF {cf} outside [0,1]"

    def test_annual_energy_physically_plausible(self):
        """Annual energy should be ≤ rated_power × 8766 h/yr."""
        from src.phase3.turbine_selection import HOURS_PER_YEAR
        flows, exceedances = self._flat_fdc(1.0)
        q_r, p_kw, energy_mwh, cf = optimize_rated_power(
            "Kaplan", 1.0, 5.0, flows, exceedances
        )
        max_mwh = p_kw * HOURS_PER_YEAR / 1_000
        assert energy_mwh <= max_mwh * 1.001, (
            f"energy {energy_mwh:.1f} MWh exceeds theoretical max {max_mwh:.1f} MWh"
        )

    def test_rated_power_formula(self):
        """P = η × ρ × g × Q × H / 1000   at rated conditions."""
        h = 10.0
        q_design = 1.0
        t_type = "Kaplan"
        flows, exceedances = self._flat_fdc(q_design)
        q_r, p_kw, _, _ = optimize_rated_power(t_type, q_design, h, flows, exceedances)
        eta = peak_efficiency(t_type)
        expected_p = eta * RHO * G * q_r * h / 1000
        assert p_kw == pytest.approx(expected_p, rel=0.01)


# ── Full sizing pipeline ──────────────────────────────────────────────────────

class TestSelectAndSizeTurbines:
    def _make_candidates(self):
        return pl.DataFrame({
            "npdes_id":      ["MN0000001", "MN0000002", "MN0000003", "MN0000004"],
            "mean_flow_mgd": [20.0,         5.0,         100.0,        0.3],
            "head_net_m":    [8.0,          25.0,         3.0,          2.0],
            "head_valid":    [True,          True,         True,         True],
            "head_source":   ["usgs_3dep",  "phase2_literature", "usgs_3dep", "design_fallback"],
        })

    def test_output_columns_present(self):
        df = select_and_size_turbines(self._make_candidates())
        for col in (
            "turbine_type", "q_rated_m3s", "rated_power_kw",
            "annual_energy_mwh", "capacity_factor", "turbine_viable",
        ):
            assert col in df.columns, f"Column '{col}' missing"

    def test_row_count_unchanged(self):
        df_in = self._make_candidates()
        df_out = select_and_size_turbines(df_in)
        assert len(df_out) == len(df_in)

    def test_high_head_facility_gets_francis_or_pelton(self):
        """25 m head / 5 MGD → Francis expected."""
        df = select_and_size_turbines(self._make_candidates())
        row = df.filter(pl.col("npdes_id") == "MN0000002")
        assert row["turbine_type"][0] in ("Francis", "Pelton")

    def test_low_head_high_flow_gets_kaplan(self):
        """3 m head / 100 MGD → Kaplan expected."""
        df = select_and_size_turbines(self._make_candidates())
        row = df.filter(pl.col("npdes_id") == "MN0000003")
        assert row["turbine_type"][0] == "Kaplan"

    def test_invalid_head_facility_not_viable(self):
        df_in = self._make_candidates().with_columns(
            pl.when(pl.col("npdes_id") == "MN0000001")
              .then(False)
              .otherwise(pl.col("head_valid"))
              .alias("head_valid")
        )
        df_out = select_and_size_turbines(df_in)
        row = df_out.filter(pl.col("npdes_id") == "MN0000001")
        assert row["turbine_viable"][0] is False

    def test_rated_power_positive_for_viable_sites(self):
        df = select_and_size_turbines(self._make_candidates())
        viable = df.filter(pl.col("turbine_viable"))
        null_power = viable["rated_power_kw"].is_null().sum()
        neg_power  = (viable["rated_power_kw"].drop_nulls() <= 0).sum()
        assert null_power == 0, "Viable sites should have non-null rated power"
        assert neg_power == 0,  "Viable sites should have positive rated power"

    def test_energy_positive_for_viable_sites(self):
        df = select_and_size_turbines(self._make_candidates())
        viable = df.filter(pl.col("turbine_viable"))
        neg_energy = (viable["annual_energy_mwh"].drop_nulls() <= 0).sum()
        assert neg_energy == 0

    def test_missing_required_columns_raises(self):
        bad_df = pl.DataFrame({"npdes_id": ["A"], "head_net_m": [5.0]})
        with pytest.raises(ValueError, match="missing columns"):
            select_and_size_turbines(bad_df)

    def test_20point_fdc_produces_nonzero_energy(self):
        """Regression test for Bug 1: 20-point Phase-1 FDC must not silently yield 0 energy.

        Phase 1 stores FDCs using 20 exceedance probabilities.  A previous bug caused
        _compute_annual_energy to return 0.0 whenever len(fdc_flows) != len(fdc_exceedances),
        which is triggered when Phase 3's 10-point exceedance grid was paired with a
        20-point FDC from Phase 1.
        """
        assert len(_PHASE1_FDC_EXCEEDANCES) == 20, (
            "This test requires the 20-point Phase 1 exceedance grid"
        )

        # Build a 20-point decreasing FDC (typical for a wastewater plant)
        fdc_flows_20 = [float(20 - i) for i in range(20)]  # 20.0 … 1.0 MGD
        fdc_exceedances_20 = _PHASE1_FDC_EXCEEDANCES[:]

        q_r, p_kw, energy_mwh, cf = optimize_rated_power(
            "Kaplan",
            q_design_m3s=fdc_flows_20[0] * 0.043813,  # ~0.876 m³/s
            h_net_m=5.0,
            fdc_flows_m3s=[v * 0.043813 for v in fdc_flows_20],
            fdc_exceedances=fdc_exceedances_20,
        )
        assert energy_mwh > 0.0, (
            f"20-point FDC produced zero annual energy (Bug 1 regression). "
            f"Got: energy={energy_mwh}, cf={cf}, p={p_kw}"
        )

    def test_full_pipeline_with_20point_fdc_column(self):
        """select_and_size_turbines must handle a flow_duration_curve column (20 points)."""
        # Construct FDC as a list-of-floats column (as Phase 1 would store it)
        fdc_values = [float(20 - i) for i in range(20)]  # 20 MGD → 1 MGD

        df = pl.DataFrame({
            "npdes_id":           ["MN0000099"],
            "mean_flow_mgd":      [12.0],
            "head_net_m":         [5.0],
            "head_valid":         [True],
            "head_source":        ["usgs_3dep"],
            "flow_duration_curve": [fdc_values],
        })

        result = select_and_size_turbines(df)
        assert len(result) == 1
        energy = result["annual_energy_mwh"][0]
        assert energy is not None and energy > 0.0, (
            f"select_and_size_turbines with 20-point FDC produced zero energy={energy}"
        )


# ── _compute_annual_energy direct tests ──────────────────────────────────────

class TestComputeAnnualEnergy:
    def test_flat_fdc_matches_p_times_hours(self):
        """Flat FDC at constant Q should equal P_rated × hours × CF ~ 1."""
        q = 1.0   # m³/s
        h = 5.0
        t_type = "Kaplan"
        eta = peak_efficiency(t_type)
        rated_power_kw = eta * 998.2 * 9.81 * q * h / 1000

        flows = [q] * 10
        exceedances = [i / 9 for i in range(10)]  # 0.0 … 1.0

        energy = _compute_annual_energy(t_type, q, h, flows, exceedances)
        expected = rated_power_kw * 8766 / 1000  # MWh if CF=1

        assert energy == pytest.approx(expected, rel=0.05), (
            f"Flat FDC energy={energy:.1f} MWh, expected≈{expected:.1f} MWh"
        )

    def test_length_mismatch_is_tolerated(self):
        """_compute_annual_energy should not return 0 on minor length mismatches."""
        flows_20 = [1.0] * 20
        exceedances_10 = [i / 9 for i in range(10)]  # shorter

        energy = _compute_annual_energy("Kaplan", 1.0, 5.0, flows_20, exceedances_10)
        assert energy > 0.0, (
            "Length mismatch (20 vs 10) should be handled by truncation, not zero return"
        )

    def test_fewer_than_two_points_returns_zero(self):
        energy = _compute_annual_energy("Kaplan", 1.0, 5.0, [1.0], [0.5])
        assert energy == 0.0

    def test_empty_fdc_returns_zero(self):
        energy = _compute_annual_energy("Kaplan", 1.0, 5.0, [], [])
        assert energy == 0.0


# ── Optimizer fallback (no CF-satisfying candidate) ───────────────────────────

class TestOptimizerFallback:
    def test_fallback_when_no_fraction_meets_cf(self):
        """When no Q fraction yields CF ≥ MIN_CF, fall back to Q_design.

        This happens with very intermittent flows (flat FDC at tiny Q).
        """
        # FDC where Q is always much smaller than Q_rated → CF never reaches MIN_CF
        tiny_q = 0.001  # m³/s
        flows = [tiny_q] * 5
        exceedances = [0.0, 0.25, 0.50, 0.75, 1.0]
        q_design = 2.0  # m³/s — huge rated flow relative to actual FDC

        q_r, p_kw, energy_mwh, cf = optimize_rated_power(
            "Kaplan",
            q_design_m3s=q_design,
            h_net_m=5.0,
            fdc_flows_m3s=flows,
            fdc_exceedances=exceedances,
        )

        # Fallback should return valid values, not crash
        assert p_kw >= 0.0
        assert energy_mwh >= 0.0
        assert 0.0 <= cf <= 1.0

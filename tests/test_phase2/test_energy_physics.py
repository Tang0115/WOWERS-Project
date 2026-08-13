"""Tests for Phase 2 energy physics and Monte Carlo sampler."""

import math

import numpy as np
import pytest

from src.phase2.energy_physics import (
    HOURS_PER_YEAR,
    MGD_TO_M3S,
    integrate_fdc_energy,
    power_kw,
    run_monte_carlo,
)

# ── power_kw ─────────────────────────────────────────────────────────────────

class TestPowerKw:
    def test_basic_value(self):
        # P = 0.82 * 998.2 * 9.81 * 1.0 * 10.0 / 1000 ≈ 80.3 kW
        p = power_kw(0.82, 1.0, 10.0)
        assert 70 < p < 90

    def test_zero_flow_returns_zero(self):
        assert power_kw(0.82, 0.0, 10.0) == 0.0

    def test_zero_head_returns_zero(self):
        assert power_kw(0.82, 1.0, 0.0) == 0.0

    def test_proportional_to_flow(self):
        p1 = power_kw(0.82, 1.0, 5.0)
        p2 = power_kw(0.82, 2.0, 5.0)
        assert abs(p2 / p1 - 2.0) < 1e-9

    def test_proportional_to_head(self):
        p1 = power_kw(0.82, 1.0, 5.0)
        p2 = power_kw(0.82, 1.0, 10.0)
        assert abs(p2 / p1 - 2.0) < 1e-9

    def test_proportional_to_efficiency(self):
        p1 = power_kw(0.50, 1.0, 5.0)
        p2 = power_kw(1.00, 1.0, 5.0)
        assert abs(p2 / p1 - 2.0) < 1e-9


# ── integrate_fdc_energy ──────────────────────────────────────────────────────

class TestIntegrateFdcEnergy:
    _FDC_CONSTANT = np.full(20, 10.0)  # constant 10 MGD across all exceedances

    def test_zero_flow_returns_zero(self):
        assert integrate_fdc_energy(np.zeros(20), 5.0, 0.82, 0.95) == 0.0

    def test_zero_head_returns_zero(self):
        assert integrate_fdc_energy(self._FDC_CONSTANT, 0.0, 0.82, 0.95) == 0.0

    def test_single_point_returns_zero(self):
        assert integrate_fdc_energy(np.array([5.0]), 5.0, 0.82, 0.95) == 0.0

    def test_availability_scales_energy(self):
        e100 = integrate_fdc_energy(self._FDC_CONSTANT, 5.0, 0.82, 1.0)
        e50  = integrate_fdc_energy(self._FDC_CONSTANT, 5.0, 0.82, 0.5)
        assert abs(e100 / e50 - 2.0) < 1e-6

    def test_head_scales_energy(self):
        e1 = integrate_fdc_energy(self._FDC_CONSTANT, 5.0,  0.82, 0.95)
        e2 = integrate_fdc_energy(self._FDC_CONSTANT, 10.0, 0.82, 0.95)
        assert abs(e2 / e1 - 2.0) < 1e-6

    def test_energy_unit_reasonableness(self):
        # Constant 1 MGD flow, 5 m head, 82% efficiency, 95% availability
        # P = 0.82 × 998.2 × 9.81 × (1×0.043813) × 5 / 1000 ≈ 1.76 kW
        # Integration width [0.01→0.95] = 0.94
        # Energy ≈ 1.76 × 0.94 × 8766 × 0.95 ≈ 13,800 kWh/yr
        e = integrate_fdc_energy(np.ones(20), 5.0, 0.82, 0.95)
        assert 12_000 < e < 16_000, f"Energy {e:.0f} kWh/yr outside expected band"

    def test_energy_hand_calc_tight(self):
        # Exact replication: 1 MGD, 5m, η=0.85, avail=0.95
        # q = 1 × 0.043813 m³/s; P = 0.85 × 998.2 × 9.81 × 0.043813 × 5 / 1000
        # Numeric: 998.2×9.81 = 9791.342; ×0.043813 = 429.03; ×5 = 2145.1 W
        # ×0.85 / 1000 = 1.8234 kW
        # Energy = 1.8234 × 0.94 × 8766 × 0.95 ≈ 14,270 kWh/yr
        e = integrate_fdc_energy(np.ones(20), 5.0, 0.85, 0.95)
        assert 13_500 < e < 15_500, f"Energy {e:.0f} kWh/yr outside tight band"

    def test_two_point_fdc_gives_nonzero(self):
        # Minimum valid case: 2 FDC points, uses first 2 exceedances [0.01, 0.05]
        e = integrate_fdc_energy(np.array([5.0, 2.0]), 5.0, 0.82, 0.95)
        assert e > 0

    def test_returns_positive_float(self):
        e = integrate_fdc_energy(self._FDC_CONSTANT, 5.0, 0.82, 0.95)
        assert isinstance(e, float)
        assert e > 0


# ── run_monte_carlo ───────────────────────────────────────────────────────────

class TestRunMonteCarlo:
    _FDC = np.linspace(20, 2, 20)   # 20 MGD at P01 down to 2 MGD at P95
    _RNG = np.random.default_rng(42)

    def _run(self, **kw):
        return run_monte_carlo(
            self._FDC,
            head_low_m=2.0, head_mode_m=5.0, head_high_m=12.0,
            n_iterations=200,
            rng=np.random.default_rng(42),
            **kw,
        )

    def test_returns_required_keys(self):
        result = self._run()
        required = {
            "energy_p10_kwh_yr", "energy_p50_kwh_yr", "energy_p90_kwh_yr",
            "energy_mean_kwh_yr", "energy_std_kwh_yr",
            "power_p50_kw", "capacity_factor_p50",
            "head_m_p10", "head_m_p50", "head_m_p90",
        }
        assert required == set(result.keys())

    def test_head_percentile_ordering(self):
        r = self._run()
        assert r["head_m_p10"] <= r["head_m_p50"] <= r["head_m_p90"]

    def test_head_within_distribution_bounds(self):
        # Distribution: Triangular(2.0, 5.0, 12.0)
        # P10 must be ≥ low bound; P90 must be ≤ high bound
        r = self._run()
        assert r["head_m_p10"] >= 2.0
        assert r["head_m_p90"] <= 12.0
        assert 2.0 <= r["head_m_p50"] <= 12.0

    def test_percentile_ordering(self):
        r = self._run()
        assert r["energy_p10_kwh_yr"] <= r["energy_p50_kwh_yr"] <= r["energy_p90_kwh_yr"]

    def test_mean_between_p10_and_p90(self):
        r = self._run()
        assert r["energy_p10_kwh_yr"] < r["energy_mean_kwh_yr"] < r["energy_p90_kwh_yr"]

    def test_capacity_factor_bounded(self):
        r = self._run()
        assert 0.0 <= r["capacity_factor_p50"] <= 1.0

    def test_power_positive(self):
        r = self._run()
        assert r["power_p50_kw"] > 0

    def test_energy_physically_reasonable(self):
        r = self._run()
        # A plant with mean ~11 MGD at 2–12 m head: expect 50k–5M kWh/yr
        assert 50_000 < r["energy_p50_kwh_yr"] < 5_000_000

    def test_zero_flow_fdc_gives_zero_cf(self):
        r = run_monte_carlo(np.zeros(20), 2.0, 5.0, 12.0,
                            n_iterations=50, rng=np.random.default_rng(0))
        assert r["capacity_factor_p50"] == 0.0

    def test_mc_convergence(self):
        # Statistical convergence: P50 with small vs large n should agree ±10%.
        # Different seeds ensure we test true MC convergence, not truncation stability.
        kw = {"head_low_m": 2.0, "head_mode_m": 5.0, "head_high_m": 12.0}
        r_small = run_monte_carlo(self._FDC, n_iterations=100,
                                  rng=np.random.default_rng(11), **kw)
        r_large = run_monte_carlo(self._FDC, n_iterations=5_000,
                                  rng=np.random.default_rng(22), **kw)
        ratio = r_small["energy_p50_kwh_yr"] / r_large["energy_p50_kwh_yr"]
        assert 0.90 <= ratio <= 1.10, f"P50 ratio {ratio:.3f} outside ±10%"

    def test_reproducible_with_seed(self):
        r1 = run_monte_carlo(self._FDC, 2.0, 5.0, 12.0, 100, rng=np.random.default_rng(7))
        r2 = run_monte_carlo(self._FDC, 2.0, 5.0, 12.0, 100, rng=np.random.default_rng(7))
        assert r1["energy_p50_kwh_yr"] == r2["energy_p50_kwh_yr"]

    def test_different_seeds_differ(self):
        r1 = run_monte_carlo(self._FDC, 2.0, 5.0, 12.0, 500, rng=np.random.default_rng(1))
        r2 = run_monte_carlo(self._FDC, 2.0, 5.0, 12.0, 500, rng=np.random.default_rng(2))
        # With 500 iterations from different seeds, P50 should differ
        assert r1["energy_p50_kwh_yr"] != r2["energy_p50_kwh_yr"]

    def test_higher_head_gives_more_energy(self):
        r_low  = run_monte_carlo(self._FDC, 1.0, 2.0,  3.0, 200, rng=np.random.default_rng(0))
        r_high = run_monte_carlo(self._FDC, 8.0, 10.0, 15.0, 200, rng=np.random.default_rng(0))
        assert r_high["energy_p50_kwh_yr"] > r_low["energy_p50_kwh_yr"]

    def test_no_rng_provided_runs_without_error(self):
        result = run_monte_carlo(self._FDC, 2.0, 5.0, 12.0, n_iterations=50)
        assert result["energy_p50_kwh_yr"] > 0

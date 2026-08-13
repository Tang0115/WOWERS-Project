"""Tests for Phase 4 revenue and electricity rate lookup."""

import pytest

from src.phase4.revenue import annual_revenue, electricity_rate


class TestElectricityRate:
    def test_known_state_returns_float(self):
        rate = electricity_rate("CA")
        assert isinstance(rate, float)
        assert rate > 0

    def test_unknown_state_returns_national_avg(self):
        rate_unknown = electricity_rate("ZZ")
        rate_avg     = electricity_rate(None)
        assert rate_unknown == rate_avg

    def test_none_returns_national_avg(self):
        rate = electricity_rate(None)
        assert 0.04 < rate < 0.20, f"National avg rate {rate} suspicious"

    def test_case_insensitive(self):
        assert electricity_rate("mn") == electricity_rate("MN")

    def test_state_rates_in_plausible_range(self):
        # All US industrial rates in [3¢, 25¢] /kWh
        for state in ("CA", "TX", "NY", "FL", "IL", "PA", "OH", "MN", "WA"):
            r = electricity_rate(state)
            assert 0.03 <= r <= 0.25, f"{state}: rate={r:.4f} out of range"

    def test_high_vs_low_rate_states(self):
        # Hawaii typically highest, Northwest/Southeast typically lower
        hi = electricity_rate("HI")
        wa = electricity_rate("WA")
        assert hi > wa, "HI industrial rate should exceed WA"


class TestAnnualRevenue:
    def test_basic_computation(self):
        # 500,000 kWh/yr × $0.085/kWh = $42,500 + REC portion
        rev = annual_revenue(500_000.0, "MN", include_rec=False)
        rate = electricity_rate("MN")
        assert abs(rev - 500_000.0 * rate) < 1.0

    def test_rec_increases_revenue(self):
        rev_no_rec  = annual_revenue(500_000.0, "MN", include_rec=False)
        rev_with_rec = annual_revenue(500_000.0, "MN", include_rec=True)
        assert rev_with_rec > rev_no_rec

    def test_zero_energy_returns_zero(self):
        assert annual_revenue(0.0, "TX") == 0.0

    def test_none_state_uses_national_avg(self):
        rev_none  = annual_revenue(100_000.0, None,  include_rec=False)
        rev_avg   = annual_revenue(100_000.0, "XX",  include_rec=False)
        assert abs(rev_none - rev_avg) < 1e-6

    def test_revenue_scales_with_energy(self):
        r1 = annual_revenue(100_000.0, "CA")
        r2 = annual_revenue(200_000.0, "CA")
        assert abs(r2 / r1 - 2.0) < 1e-9

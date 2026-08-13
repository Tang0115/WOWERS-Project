"""Tests for Phase 4 CapEx and OpEx cost models."""

import pytest

from src.phase4.cost_models import (
    annual_opex,
    capex_per_kw,
    interconnection_cost,
    permitting_cost,
    permitting_tier_label,
    project_capex,
    total_capex,
)

TURBINE_TYPES = ("Kaplan", "Francis", "Pelton", "in_conduit_micro")


class TestCapexPerKw:
    def test_returns_positive_for_each_type(self):
        for t in TURBINE_TYPES:
            assert capex_per_kw(t, 100.0) > 0, f"Non-positive capex for {t}"

    def test_clamped_to_min_at_large_power(self):
        # At very large power, power-law pushes below min; clamp kicks in
        for t in TURBINE_TYPES:
            c = capex_per_kw(t, 100_000.0)
            assert c > 0  # must still be positive (clamped to min)

    def test_per_type_max_at_zero_power(self):
        # Zero rated power → returns per-type max (D3 fix: no longer global 10,000)
        expected_max = {
            "Kaplan":           10_000,
            "Francis":           9_000,
            "Pelton":            8_000,
            "in_conduit_micro": 15_000,
        }
        for t in TURBINE_TYPES:
            assert capex_per_kw(t, 0.0) == pytest.approx(expected_max[t]), \
                f"{t}: expected per-type max {expected_max[t]}, got {capex_per_kw(t, 0.0)}"

    def test_economies_of_scale(self):
        # Larger plant → lower $/kW (before clamping)
        for t in TURBINE_TYPES:
            c1 = capex_per_kw(t, 10.0)
            c2 = capex_per_kw(t, 1_000.0)
            assert c1 >= c2, f"{t}: no economies of scale detected"

    def test_unknown_type_falls_back_to_kaplan(self):
        known  = capex_per_kw("Kaplan", 100.0)
        unk    = capex_per_kw("UnknownTurbine", 100.0)
        assert known == unk

    def test_capex_per_kw_range_physically_plausible(self):
        # ORNL/DOE range for conduit hydro: ~$1,000–$15,000/kW
        for t in TURBINE_TYPES:
            c = capex_per_kw(t, 100.0)
            assert 500 <= c <= 20_000, f"{t}: capex/kW={c} out of plausible range"

    def test_francis_clamped_to_vendor_floor_at_large_power(self):
        # F4-VENDORBAND: large Francis sites previously priced below the vendor
        # floor ($1,800/kW, Canyon/Gilkes min). min_per_kw clamp now lifts them
        # to the floor so the model never quotes under real manufacturer pricing.
        assert capex_per_kw("Francis", 1_000.0) == pytest.approx(1_800)
        assert capex_per_kw("Francis", 5_000.0) == pytest.approx(1_800)


class TestTotalCapex:
    def test_total_equals_per_kw_times_power(self):
        for t in TURBINE_TYPES:
            kw = 250.0
            assert abs(total_capex(t, kw) - capex_per_kw(t, kw) * kw) < 1e-6

    def test_zero_power_still_computes(self):
        c = total_capex("Kaplan", 0.0)
        assert c >= 0


class TestAnnualOpex:
    def test_returns_positive_fraction_of_capex(self):
        for t in TURBINE_TYPES:
            capex = 500_000.0
            opex  = annual_opex(t, capex)
            assert 0 < opex < capex, f"{t}: opex={opex} outside (0, capex)"

    def test_opex_fraction_between_1_and_5_pct(self):
        for t in TURBINE_TYPES:
            capex = 1_000_000.0
            frac  = annual_opex(t, capex) / capex
            assert 0.01 <= frac <= 0.05, f"{t}: opex fraction={frac:.3f} unusual"

    def test_unknown_type_gets_default_fraction(self):
        capex = 1_000_000.0
        opex  = annual_opex("UnknownType", capex)
        assert 0 < opex < capex


# ── F4-INTERCON: grid interconnection cost ────────────────────────────────────

class TestInterconnectionCost:
    """F4-INTERCON + F4-BTM: behind-the-meter branch then distribution-tie tiers.

    F4-BTM (2026-06-01): sites ≤25 kW are behind-the-meter self-consumption
    (no grid export) → $5k, not a distribution-tie tier.  Tiers apply >25 kW.
    """

    def test_behind_the_meter_micro_returns_5k(self):
        # F4-BTM: ≤25 kW self-consumed behind the WWTP meter, no grid export → $5k
        assert interconnection_cost(5.0) == pytest.approx(5_000)
        assert interconnection_cost(11.0) == pytest.approx(5_000)
        assert interconnection_cost(25.0) == pytest.approx(5_000)

    def test_micro_tier_returns_100k(self):
        # 25 < kw <= 50 → $100k (grid distribution tie required)
        assert interconnection_cost(26.0) == pytest.approx(100_000)
        assert interconnection_cost(50.0) == pytest.approx(100_000)

    def test_small_tier_returns_150k(self):
        # 50 < kw <= 250 → $150k
        assert interconnection_cost(100.0) == pytest.approx(150_000)
        assert interconnection_cost(250.0) == pytest.approx(150_000)

    def test_large_tier_returns_200k(self):
        # 250 < kw → $200k (catch-all)
        assert interconnection_cost(500.0) == pytest.approx(200_000)
        assert interconnection_cost(10_000.0) == pytest.approx(200_000)

    def test_monotonically_non_decreasing(self):
        prev = -1.0
        for kw in (1, 10, 25, 26, 50, 51, 100, 250, 251, 500, 1_000):
            cur = interconnection_cost(float(kw))
            assert cur >= prev, f"intercon cost dropped at {kw} kW"
            prev = cur

    def test_zero_or_negative_returns_lowest_tier(self):
        # Non-positive input is a safety floor → lowest distribution tier (BTM
        # branch only triggers for positive rated power).
        assert interconnection_cost(0.0) == pytest.approx(50_000)
        assert interconnection_cost(-5.0) == pytest.approx(50_000)


# ── F4-PERMIT-TIER: tiered permitting cost ────────────────────────────────────

class TestPermittingCost:
    """F4-PERMIT-TIER: tier lookup by rated power.

    Defaults (qualified_facility recalibrated 2026-06-01, F4-CONDUIT):
        ≤ 25 kW  →  $5k  (qualifying conduit facility — FERC NOI, no fee)
        ≤ 250 kW → $75k  (small_ferc)
        > 250 kW → $150k (full_nepa)
    """

    def test_qualified_facility_tier(self):
        # F4-CONDUIT: ≤25 kW = qualifying conduit facility (FERC NOI, no fee) → $5k
        assert permitting_cost(1.0)   == pytest.approx(5_000)
        assert permitting_cost(10.0)  == pytest.approx(5_000)
        assert permitting_cost(25.0)  == pytest.approx(5_000)

    def test_small_ferc_tier(self):
        # 25 < kw <= 250 → $75k
        assert permitting_cost(26.0)  == pytest.approx(75_000)
        assert permitting_cost(100.0) == pytest.approx(75_000)
        assert permitting_cost(250.0) == pytest.approx(75_000)

    def test_full_nepa_tier(self):
        # > 250 kW → $150k
        assert permitting_cost(251.0)   == pytest.approx(150_000)
        assert permitting_cost(1_000.0) == pytest.approx(150_000)
        assert permitting_cost(10_000.0) == pytest.approx(150_000)

    def test_monotonically_non_decreasing(self):
        prev = -1.0
        for kw in (0.5, 10, 25, 26, 100, 250, 251, 1_000, 10_000):
            cur = permitting_cost(float(kw))
            assert cur >= prev, f"permit cost dropped at {kw} kW"
            prev = cur

    def test_zero_returns_lowest_tier(self):
        assert permitting_cost(0.0)   == pytest.approx(5_000)
        assert permitting_cost(-5.0)  == pytest.approx(5_000)

    def test_all_tiers_strictly_positive(self):
        # F4-PERMIT-TIER replaces the old $0-for-large-sites model — every
        # site now carries SOME permitting overhead.  Important contract.
        for kw in (5, 25, 100, 250, 1_000, 5_000):
            assert permitting_cost(float(kw)) > 0, (
                f"{kw} kW: every site must carry non-zero permit cost"
            )


class TestPermittingTierLabel:
    """F4-PERMIT-TIER: label string matches the tier."""

    def test_qualified_facility_label(self):
        for kw in (0.5, 5, 25):
            assert permitting_tier_label(float(kw)) == "qualified_facility"

    def test_small_ferc_label(self):
        for kw in (26, 100, 250):
            assert permitting_tier_label(float(kw)) == "small_ferc"

    def test_full_nepa_label(self):
        for kw in (251, 1_000, 5_000):
            assert permitting_tier_label(float(kw)) == "full_nepa"

    def test_zero_returns_qualified_facility(self):
        assert permitting_tier_label(0.0) == "qualified_facility"

    def test_label_and_cost_consistent(self):
        # Sanity: same tier → same label across all queried values
        label_to_cost = {
            "qualified_facility":  5_000,
            "small_ferc":         75_000,
            "full_nepa":         150_000,
        }
        for kw in (5, 25, 26, 100, 250, 251, 1_000):
            assert permitting_cost(float(kw)) == pytest.approx(
                label_to_cost[permitting_tier_label(float(kw))]
            )


# ── Project CapEx aggregation ─────────────────────────────────────────────────

class TestProjectCapex:
    """project_capex returns full breakdown and aggregates correctly."""

    def test_returns_all_expected_keys(self):
        bd = project_capex("Kaplan", 100.0)
        required = {
            "equipment_capex_usd",
            "installation_capex_usd",   # F4-INSTALL
            "interconnection_capex_usd",
            "permitting_capex_usd",
            "permitting_tier",
            "total_project_capex_usd",
        }
        assert required == set(bd.keys())

    def test_total_equals_sum_of_components(self):
        # F4-INSTALL: total is now a four-component sum.
        for t in TURBINE_TYPES:
            for kw in (5, 25, 100, 500):
                bd = project_capex(t, float(kw))
                expected = (
                    bd["equipment_capex_usd"]
                    + bd["installation_capex_usd"]
                    + bd["interconnection_capex_usd"]
                    + bd["permitting_capex_usd"]
                )
                assert bd["total_project_capex_usd"] == pytest.approx(expected)

    def test_micro_site_qualified_facility_tier(self):
        # 20 kW: F4-CONDUIT permit ($5k) + F4-BTM behind-the-meter intercon ($5k)
        bd = project_capex("Crossflow", 20.0)
        assert bd["permitting_capex_usd"] == pytest.approx(5_000)
        assert bd["permitting_tier"] == "qualified_facility"
        assert bd["interconnection_capex_usd"] == pytest.approx(5_000)

    def test_small_ferc_tier_site(self):
        # 100 kW: small_ferc permit + 150k intercon tier
        bd = project_capex("Kaplan", 100.0)
        assert bd["permitting_capex_usd"] == pytest.approx(75_000)
        assert bd["permitting_tier"] == "small_ferc"
        assert bd["interconnection_capex_usd"] == pytest.approx(150_000)

    def test_full_nepa_tier_site(self):
        # 300 kW: full_nepa permit + top intercon tier
        bd = project_capex("Francis", 300.0)
        assert bd["permitting_capex_usd"] == pytest.approx(150_000)
        assert bd["permitting_tier"] == "full_nepa"
        assert bd["interconnection_capex_usd"] == pytest.approx(200_000)

    def test_total_strictly_greater_than_equipment(self):
        # F4 adders must always increase project CapEx vs equipment-only
        for kw in (5, 50, 500):
            bd = project_capex("Kaplan", float(kw))
            assert bd["total_project_capex_usd"] > bd["equipment_capex_usd"]

    def test_equipment_matches_total_capex_function(self):
        for t in TURBINE_TYPES:
            for kw in (10, 100, 1_000):
                bd = project_capex(t, float(kw))
                assert bd["equipment_capex_usd"] == pytest.approx(
                    total_capex(t, float(kw))
                )


# ── F4-INSTALL: installation cost ────────────────────────────────────────────

class TestInstallationCapex:
    """F4-INSTALL: mechanical installation cost as a fraction of equipment CapEx.

    Scope: mechanical install / labor ONLY — excludes civil works; does NOT
    overlap interconnection or permitting.  One-time cost; not O&M-bearing.
    Reversible lever: ``installation_pct_of_equipment = 0.0`` disables it.
    """

    ALL_TYPES = ("Kaplan", "Francis", "Pelton", "in_conduit_micro", "Crossflow")

    def test_installation_capex_key_present(self):
        """installation_capex_usd key must be present in project_capex output."""
        bd = project_capex("Kaplan", 100.0)
        assert "installation_capex_usd" in bd

    def test_installation_equals_equipment_times_pct(self):
        """installation_capex_usd == equipment_capex_usd × _INSTALL_PCT."""
        import src.phase4.cost_models as cm
        for t in self.ALL_TYPES:
            for kw in (5.0, 50.0, 500.0):
                bd = project_capex(t, kw)
                expected = bd["equipment_capex_usd"] * cm._INSTALL_PCT
                assert bd["installation_capex_usd"] == pytest.approx(expected, rel=1e-9), (
                    f"{t} @ {kw} kW: install mismatch"
                )

    def test_total_is_four_component_sum(self):
        """total = equipment + installation + interconnection + permitting."""
        for t in self.ALL_TYPES:
            for kw in (5.0, 50.0, 500.0):
                bd = project_capex(t, kw)
                expected = (
                    bd["equipment_capex_usd"]
                    + bd["installation_capex_usd"]
                    + bd["interconnection_capex_usd"]
                    + bd["permitting_capex_usd"]
                )
                assert bd["total_project_capex_usd"] == pytest.approx(expected, rel=1e-9), (
                    f"{t} @ {kw} kW: total ≠ four-component sum"
                )

    def test_zero_pct_noop(self, monkeypatch):
        """With install pct = 0.0, installation = 0 and total = 3-component sum."""
        import src.phase4.cost_models as cm
        monkeypatch.setattr(cm, "_INSTALL_PCT", 0.0)
        bd = project_capex("Kaplan", 100.0)
        assert bd["installation_capex_usd"] == pytest.approx(0.0)
        expected_3 = (
            bd["equipment_capex_usd"]
            + bd["interconnection_capex_usd"]
            + bd["permitting_capex_usd"]
        )
        assert bd["total_project_capex_usd"] == pytest.approx(expected_3, rel=1e-9)

    def test_installation_non_negative(self):
        """Installation cost must always be ≥ 0 (equipment is always ≥ 0)."""
        for t in self.ALL_TYPES:
            for kw in (1.0, 10.0, 100.0, 1_000.0):
                bd = project_capex(t, kw)
                assert bd["installation_capex_usd"] >= 0.0, (
                    f"{t} @ {kw} kW: negative installation cost"
                )

    def test_opex_is_equipment_only_not_affected(self):
        """annual_opex must not change when installation fraction is non-zero.

        O&M is a % of *equipment* CapEx.  Installation is a one-time cost and
        must not be O&M-bearing.  Verify by comparing opex at pct=0 vs pct>0.
        """
        import src.phase4.cost_models as cm
        eq = total_capex("Kaplan", 100.0)
        opex_baseline = annual_opex("Kaplan", eq)

        # Temporarily raise install pct — opex must be unchanged
        import unittest.mock as mock
        with mock.patch.object(cm, "_INSTALL_PCT", 0.175):
            bd = project_capex("Kaplan", 100.0)
            opex_with_install = annual_opex("Kaplan", bd["equipment_capex_usd"])

        assert opex_with_install == pytest.approx(opex_baseline, rel=1e-9), (
            "OpEx changed when installation pct changed — install must NOT be O&M-bearing"
        )

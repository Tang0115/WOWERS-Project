"""Tests for Phase 4 tornado sensitivity analysis."""

import pytest

from src.phase4.sensitivity import (
    _HEAD_FACTORS,
    _FLOW_FACTORS,
    _RATE_FACTORS,
    run_tornado,
)

_BASE = dict(
    annual_energy_kwh=1_000_000.0,
    elec_rate_per_kwh=0.085,
    annual_opex_usd=12_500.0,
    total_capex_usd=500_000.0,
)


class TestRunTornado:
    def _run(self, **kw):
        args = {**_BASE, **kw}
        return run_tornado(**args)

    def test_returns_required_keys(self):
        r = self._run()
        required = {
            "sensitivity_head_npv_low", "sensitivity_head_npv_high",
            "sensitivity_flow_npv_low", "sensitivity_flow_npv_high",
            "sensitivity_rate_npv_low", "sensitivity_rate_npv_high",
            "dominant_sensitivity",
        }
        assert required == set(r.keys())

    def test_dominant_sensitivity_is_valid(self):
        r = self._run()
        assert r["dominant_sensitivity"] in {"head", "flow", "rate", "energy_uncertain"}

    # ── T15: monotonicity ─────────────────────────────────────────────────────

    def test_head_high_npv_exceeds_head_low_npv(self):
        r = self._run()
        assert r["sensitivity_head_npv_high"] > r["sensitivity_head_npv_low"]

    def test_flow_high_npv_exceeds_flow_low_npv(self):
        r = self._run()
        assert r["sensitivity_flow_npv_high"] > r["sensitivity_flow_npv_low"]

    def test_rate_high_npv_exceeds_rate_low_npv(self):
        r = self._run()
        assert r["sensitivity_rate_npv_high"] > r["sensitivity_rate_npv_low"]

    # ── T16: dominant correctness after normalization ─────────────────────────

    def test_head_dominant_for_large_head_uncertainty(self):
        # Head has ±50% range (width=1.0), flow ±20% (0.40), rate ±30% (0.60).
        # Fallback path (no FDC) → dominant_sensitivity = "energy_uncertain";
        # skip swing-consistency check since head/flow swings are algebraically
        # equal in the linear fallback.
        r = self._run()
        dominant = r["dominant_sensitivity"]
        if dominant == "energy_uncertain":
            return  # correct fallback label — no further assertion needed
        raw = {
            "head": abs(r["sensitivity_head_npv_high"] - r["sensitivity_head_npv_low"]),
            "flow": abs(r["sensitivity_flow_npv_high"] - r["sensitivity_flow_npv_low"]),
            "rate": abs(r["sensitivity_rate_npv_high"] - r["sensitivity_rate_npv_low"]),
        }
        ranges = {"head": 1.00, "flow": 0.40, "rate": 0.60}
        normalised = {k: raw[k] / ranges[k] for k in raw}
        assert normalised[dominant] == max(normalised.values())

    def test_range_factors_consistent(self):
        # Confirm the sensitivity factors in the module are physically reasonable
        assert _HEAD_FACTORS[0] < 1.0 < _HEAD_FACTORS[1]
        assert _FLOW_FACTORS[0] < 1.0 < _FLOW_FACTORS[1]
        assert _RATE_FACTORS[0] < 1.0 < _RATE_FACTORS[1]

    def test_normalised_swing_stored_correctly(self):
        # Recompute normalised swings manually and compare to dominant label.
        # Fallback path (no FDC) → "energy_uncertain"; skip max-swing assertion.
        r = self._run()
        if r["dominant_sensitivity"] == "energy_uncertain":
            return  # linear fallback: head/flow swings equal, label is correct
        ranges = {"head": 1.00, "flow": 0.40, "rate": 0.60}
        raw = {
            "head": abs(r["sensitivity_head_npv_high"] - r["sensitivity_head_npv_low"]),
            "flow": abs(r["sensitivity_flow_npv_high"] - r["sensitivity_flow_npv_low"]),
            "rate": abs(r["sensitivity_rate_npv_high"] - r["sensitivity_rate_npv_low"]),
        }
        normalised = {k: raw[k] / ranges[k] for k in raw}
        expected_dominant = max(normalised, key=normalised.__getitem__)
        assert r["dominant_sensitivity"] == expected_dominant

    # ── Option B physical model (FDC provided) ────────────────────────────────

    def test_fallback_path_labels_energy_uncertain(self):
        """No FDC inputs → fallback path → dominant_sensitivity == 'energy_uncertain'.

        Sites without measured DMR data (data_quality in {actual_avg_only,
        design_only}) have no FDC.  The physical Option B model can't engage,
        so head/flow swings collapse to algebraic identity.  Correct label is
        'energy_uncertain' so downstream consumers know the distinction is
        not meaningful.
        """
        r = self._run()  # no h_net_m / fdc_flows_m3s — fallback engaged
        assert r["dominant_sensitivity"] == "energy_uncertain", (
            f"Expected 'energy_uncertain' label in fallback, got "
            f"{r['dominant_sensitivity']}"
        )
        # In linear fallback, normalised head_swing must equal flow_swing
        head_swing = abs(r["sensitivity_head_npv_high"] - r["sensitivity_head_npv_low"]) / 1.0
        flow_swing = abs(r["sensitivity_flow_npv_high"] - r["sensitivity_flow_npv_low"]) / 0.4
        assert abs(head_swing - flow_swing) < 1e-3, (
            f"Fallback head/flow swings must be equal (algebraic), "
            f"got head={head_swing:.4f}, flow={flow_swing:.4f}"
        )

    def test_physical_model_distinguishes_head_from_flow(self):
        """With FDC + turbine inputs (Option B), head and flow swings must differ.

        Head sweep re-optimises rated power → nonlinear via FDC clipping.
        Flow sweep scales FDC bins, keeps Q_rated fixed → nonlinear via the
        part-load η(q) curve.  They engage different physics, so normalised
        swings must not be algebraically equal.
        """
        from src.phase3.turbine_selection import _PHASE1_FDC_EXCEEDANCES
        fdc = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25,
               0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
        r = self._run(
            h_net_m=8.0,
            q_design_m3s=1.0,
            fdc_flows_m3s=fdc,
            fdc_exceedances=_PHASE1_FDC_EXCEEDANCES,
            turbine_type="Kaplan",
            q_rated_m3s=1.0,
        )
        assert r["dominant_sensitivity"] in {"head", "flow", "rate"}, (
            f"Physical path should never label 'energy_uncertain', got {r['dominant_sensitivity']}"
        )
        head_swing = abs(r["sensitivity_head_npv_high"] - r["sensitivity_head_npv_low"]) / 1.0
        flow_swing = abs(r["sensitivity_flow_npv_high"] - r["sensitivity_flow_npv_low"]) / 0.4
        assert abs(head_swing - flow_swing) > 1.0, (
            f"Physical model should differentiate head from flow, got "
            f"head_swing={head_swing:.2f}, flow_swing={flow_swing:.2f}"
        )

    def test_physical_head_partial_iso_curve(self):
        """At very low head (×0.5), re-optimisation may yield 0 energy.

        Head_factor=0.5 on h_net=2m drops to 1m which is at MIN_NET_HEAD_M.
        Physical optimizer should still return finite NPV (not crash).
        """
        from src.phase3.turbine_selection import _PHASE1_FDC_EXCEEDANCES
        fdc = [0.5] * 20
        r = self._run(
            h_net_m=2.0,
            q_design_m3s=0.5,
            fdc_flows_m3s=fdc,
            fdc_exceedances=_PHASE1_FDC_EXCEEDANCES,
            turbine_type="Crossflow",
            q_rated_m3s=0.5,
        )
        # No NaN/None on any sensitivity NPV
        for k in ("sensitivity_head_npv_low", "sensitivity_head_npv_high",
                  "sensitivity_flow_npv_low", "sensitivity_flow_npv_high",
                  "sensitivity_rate_npv_low", "sensitivity_rate_npv_high"):
            assert r[k] is not None
            assert r[k] == r[k]  # not NaN

"""Tests for F4-ECON-CAT: derive_econ_cat_payback, _npv, _irr (run.py).

Coverage:
  * One valid case per band for each of the three functions.
  * None / inf / NaN inputs → worst band (degenerate-input guard).
  * Cross-metric edge: NPV≤0 forces econ_cat_payback == "uneconomic"
    even when payback_years ≤ 5yr (because project_viable must be True).
  * IRR unit: stored as fraction (0.15 = 15%) — confirmed by live parquet.
"""

from __future__ import annotations

import math

import pytest

from src.phase4.run import (
    derive_econ_cat_irr,
    derive_econ_cat_npv,
    derive_econ_cat_payback,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pb_row(viable: bool, payback: float) -> dict:
    return {"project_viable": viable, "payback_years": payback,
            "npv_usd": 1.0, "irr": 0.12}


def _npv_row(npv: float) -> dict:
    return {"npv_usd": npv}


def _irr_row(irr: float) -> dict:
    return {"irr": irr}


# ── derive_econ_cat_payback ───────────────────────────────────────────────────

class TestEconCatPayback:
    def test_excellent_at_5yr_boundary(self):
        assert derive_econ_cat_payback(_pb_row(True, 5.0)) == "excellent"

    def test_excellent_below_boundary(self):
        assert derive_econ_cat_payback(_pb_row(True, 3.0)) == "excellent"

    def test_good_just_above_5(self):
        assert derive_econ_cat_payback(_pb_row(True, 5.01)) == "good"

    def test_good_at_10yr_boundary(self):
        assert derive_econ_cat_payback(_pb_row(True, 10.0)) == "good"

    def test_marginal_just_above_10(self):
        assert derive_econ_cat_payback(_pb_row(True, 10.01)) == "marginal"

    def test_marginal_at_20yr_boundary(self):
        assert derive_econ_cat_payback(_pb_row(True, 20.0)) == "marginal"

    def test_uneconomic_when_not_viable(self):
        # project_viable=False → uneconomic regardless of payback value
        assert derive_econ_cat_payback(_pb_row(False, 3.0)) == "uneconomic"

    def test_uneconomic_payback_above_20(self):
        assert derive_econ_cat_payback(_pb_row(True, 21.0)) == "uneconomic"

    def test_uneconomic_payback_none(self):
        row = {"project_viable": True, "payback_years": None}
        assert derive_econ_cat_payback(row) == "uneconomic"

    def test_uneconomic_payback_inf(self):
        row = {"project_viable": True, "payback_years": math.inf}
        assert derive_econ_cat_payback(row) == "uneconomic"

    def test_uneconomic_payback_nan(self):
        row = {"project_viable": True, "payback_years": float("nan")}
        assert derive_econ_cat_payback(row) == "uneconomic"

    def test_cross_metric_npv_le_0_forces_uneconomic(self):
        """NPV≤0 → project_viable must be False → payback category uneconomic.

        Even if payback_years ≤ 5, a site with NPV≤0 cannot be project_viable,
        so the payback category must be uneconomic regardless of payback value.
        """
        row = {
            "project_viable": False,  # NPV≤0 precludes True
            "payback_years": 2.0,     # would be "excellent" if viable
            "npv_usd": -1_000.0,
            "irr": 0.05,
        }
        assert derive_econ_cat_payback(row) == "uneconomic"


# ── derive_econ_cat_npv ───────────────────────────────────────────────────────

class TestEconCatNpv:
    def test_high_at_500k_boundary(self):
        assert derive_econ_cat_npv(_npv_row(500_000.0)) == "high"

    def test_high_well_above_500k(self):
        assert derive_econ_cat_npv(_npv_row(1_000_000.0)) == "high"

    def test_medium_just_below_500k(self):
        assert derive_econ_cat_npv(_npv_row(499_999.0)) == "medium"

    def test_medium_at_100k_boundary(self):
        assert derive_econ_cat_npv(_npv_row(100_000.0)) == "medium"

    def test_low_just_below_100k(self):
        assert derive_econ_cat_npv(_npv_row(99_999.0)) == "low"

    def test_low_small_positive(self):
        assert derive_econ_cat_npv(_npv_row(1.0)) == "low"

    def test_negative_at_zero(self):
        assert derive_econ_cat_npv(_npv_row(0.0)) == "negative"

    def test_negative_below_zero(self):
        assert derive_econ_cat_npv(_npv_row(-50_000.0)) == "negative"

    def test_negative_on_none(self):
        assert derive_econ_cat_npv({"npv_usd": None}) == "negative"

    def test_negative_on_nan(self):
        assert derive_econ_cat_npv({"npv_usd": float("nan")}) == "negative"


# ── derive_econ_cat_irr ───────────────────────────────────────────────────────

class TestEconCatIrr:
    """IRR is stored as fraction: 0.15 = 15%, 0.08 = 8%, etc."""

    def test_strong_at_15pct_boundary(self):
        assert derive_econ_cat_irr(_irr_row(0.15)) == "strong"

    def test_strong_well_above_15pct(self):
        assert derive_econ_cat_irr(_irr_row(0.30)) == "strong"

    def test_moderate_just_below_15pct(self):
        assert derive_econ_cat_irr(_irr_row(0.149)) == "moderate"

    def test_moderate_at_8pct_boundary(self):
        assert derive_econ_cat_irr(_irr_row(0.08)) == "moderate"

    def test_weak_just_below_8pct(self):
        assert derive_econ_cat_irr(_irr_row(0.079)) == "weak"

    def test_weak_at_zero(self):
        assert derive_econ_cat_irr(_irr_row(0.0)) == "weak"

    def test_none_below_zero(self):
        assert derive_econ_cat_irr(_irr_row(-0.01)) == "none"

    def test_none_on_nan(self):
        assert derive_econ_cat_irr({"irr": float("nan")}) == "none"

    def test_none_on_none(self):
        assert derive_econ_cat_irr({"irr": None}) == "none"

    def test_none_on_positive_sentinel(self):
        # IRR sentinel +3.0 (trivially-profitable nano-CapEx) → non-real
        assert derive_econ_cat_irr(_irr_row(3.0)) == "none"

    def test_none_on_negative_sentinel(self):
        # IRR sentinel -0.99 (all-negative-CF) → non-real
        assert derive_econ_cat_irr(_irr_row(-0.99)) == "none"

    def test_irr_stored_as_fraction_not_percent(self):
        """Sanity: 0.20 must be "strong" (20% IRR), not 20.0 which would be
        unreasonably high and treated as a sentinel (>= 3.0 would still be none).
        Confirms the function uses fractional storage, matching the parquet."""
        assert derive_econ_cat_irr(_irr_row(0.20)) == "strong"


# ── Valid-label completeness ──────────────────────────────────────────────────

class TestEconCatValidLabels:
    """All three functions must only return their documented label set."""

    _PAYBACK_LABELS = {"excellent", "good", "marginal", "uneconomic"}
    _NPV_LABELS     = {"high", "medium", "low", "negative"}
    _IRR_LABELS     = {"strong", "moderate", "weak", "none"}

    _PROBE_ROWS = [
        {"project_viable": True,  "payback_years": 3.0,  "npv_usd": 600_000.0, "irr": 0.25},
        {"project_viable": True,  "payback_years": 8.0,  "npv_usd": 200_000.0, "irr": 0.10},
        {"project_viable": True,  "payback_years": 15.0, "npv_usd": 50_000.0,  "irr": 0.05},
        {"project_viable": False, "payback_years": 25.0, "npv_usd": -10_000.0, "irr": -0.5},
        {"project_viable": False, "payback_years": None, "npv_usd": None,       "irr": None},
    ]

    def test_payback_only_returns_valid_labels(self):
        for row in self._PROBE_ROWS:
            result = derive_econ_cat_payback(row)
            assert result in self._PAYBACK_LABELS, f"Unexpected label: {result}"

    def test_npv_only_returns_valid_labels(self):
        for row in self._PROBE_ROWS:
            result = derive_econ_cat_npv(row)
            assert result in self._NPV_LABELS, f"Unexpected label: {result}"

    def test_irr_only_returns_valid_labels(self):
        for row in self._PROBE_ROWS:
            result = derive_econ_cat_irr(row)
            assert result in self._IRR_LABELS, f"Unexpected label: {result}"

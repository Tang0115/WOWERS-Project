"""Tests for Phase 4 site_tier (A/B/C) classification."""

from __future__ import annotations

import math

import polars as pl
import pytest

from src.phase4.run import derive_site_tier


class TestDeriveSiteTier:
    def test_tier_a_when_project_viable(self):
        row = {
            "project_viable": True,
            "npv_usd": 50_000.0,
            "payback_years": 8.0,
            "irr": 0.12,
        }
        assert derive_site_tier(row) == "A"

    def test_tier_b_cash_flow_positive_below_minrev(self):
        row = {
            "project_viable": False,
            "npv_usd": 10_000.0,
            "payback_years": 12.0,
            "irr": 0.08,
        }
        assert derive_site_tier(row) == "B"

    def test_tier_c_negative_npv(self):
        row = {
            "project_viable": False,
            "npv_usd": -5_000.0,
            "payback_years": 25.0,
            "irr": -0.5,
        }
        assert derive_site_tier(row) == "C"

    def test_tier_c_irr_sentinel_high(self):
        row = {
            "project_viable": False,
            "npv_usd": 1_000.0,
            "payback_years": 5.0,
            "irr": 3.0,
        }
        assert derive_site_tier(row) == "C"

    def test_tier_c_nan_irr_blocks_tier_b(self):
        row = {
            "project_viable": False,
            "npv_usd": 1_000.0,
            "payback_years": 5.0,
            "irr": math.nan,
        }
        assert derive_site_tier(row) == "C"

    def test_tier_c_missing_irr_key(self):
        row = {
            "project_viable": False,
            "npv_usd": 1_000.0,
            "payback_years": 5.0,
        }
        assert derive_site_tier(row) == "C"

    def test_synthetic_dataframe_all_tiers_assigned(self):
        rows = [
            {"project_viable": True, "npv_usd": 1.0, "payback_years": 5.0, "irr": 0.1},
            {"project_viable": False, "npv_usd": 1.0, "payback_years": 10.0, "irr": 0.05},
            {"project_viable": False, "npv_usd": -1.0, "payback_years": 30.0, "irr": math.nan},
        ]
        tiers = [derive_site_tier(r) for r in rows]
        assert set(tiers) == {"A", "B", "C"}
        assert all(t in {"A", "B", "C"} for t in tiers)

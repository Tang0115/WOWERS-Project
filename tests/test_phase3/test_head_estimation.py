"""Tests for phase3/head_estimation.py — physics correctness + edge cases."""

from __future__ import annotations

import polars as pl
import pytest

from src.phase3.head_estimation import (
    HEAD_LOSS_FRACTION,
    MIN_NET_HEAD_M,
    _compute_head_row,
    estimate_head,
    head_summary_stats,
)


class TestComputeHeadRow:
    """Unit tests for the row-level head computation function."""

    def test_both_elevations_available(self):
        """Standard case: both facility and outfall elevation known."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=300.0,
            elev_outfall_m=290.0,
            head_p50_literature_m=10.0,
        )
        assert gross == pytest.approx(10.0)
        assert net == pytest.approx(10.0 * (1 - HEAD_LOSS_FRACTION))
        assert src == "usgs_3dep"
        assert valid is True
        assert conf == "high"

    def test_falls_back_to_literature_when_no_outfall_elevation(self):
        """When outfall elevation is None, should use literature head."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=300.0,
            elev_outfall_m=None,
            head_p50_literature_m=8.0,
        )
        assert src == "phase2_literature"
        assert net == pytest.approx(8.0 * (1 - HEAD_LOSS_FRACTION))
        assert conf == "medium"

    def test_falls_back_to_design_when_no_data(self):
        """Worst case: no elevation, no literature head."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=None,
            elev_outfall_m=None,
            head_p50_literature_m=None,
        )
        assert src == "design_fallback"
        assert net is not None
        assert conf == "low"

    def test_negative_head_elevation_rejected(self):
        """Outfall HIGHER than facility: 3DEP result is invalid, falls back to literature."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=290.0,
            elev_outfall_m=300.0,   # outfall uphill — physically impossible
            head_p50_literature_m=5.0,
        )
        # 3DEP-derived head is negative → plausibility gate rejects it
        assert src != "usgs_3dep", "Negative head should not use 3DEP"
        assert net is not None

    def test_wildly_divergent_3dep_rejected(self):
        """3DEP head that differs 3× from literature should be rejected."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=400.0,
            elev_outfall_m=300.0,   # gross = 100 m
            head_p50_literature_m=5.0,  # literature says ~5 m — 20× divergence
        )
        assert src != "usgs_3dep", "Wildly divergent 3DEP head should not be used"

    def test_f2_regression_2_5x_ratio_now_accepted(self):
        """F2 fix: divergence ratio raised from 2.0 → 4.0.

        A 3DEP reading at 2.5× literature P50 was previously rejected
        (ratio 1.5 > 2.0) but must now be accepted (1.5 < 4.0).
        Example: literature=3.27m, 3DEP head=3.27*3.5=~11.4m → accepted.
        """
        lit_p50 = 3.27  # national median literature head
        # 3DEP gross head at 2.5× literature (ratio = (2.5-1) = 1.5 — within 4.0)
        threeDEP_gross = lit_p50 * 3.5   # ratio = 2.5, well inside 4.0
        from src.phase3.head_estimation import HEAD_LOSS_FRACTION
        facility_elev = 200.0
        outfall_elev = facility_elev - threeDEP_gross

        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=facility_elev,
            elev_outfall_m=outfall_elev,
            head_p50_literature_m=lit_p50,
        )
        assert src == "usgs_3dep", (
            f"3DEP head at 3.5× literature should pass divergence gate "
            f"(ratio={2.5:.1f} < 4.0); got src={src}"
        )
        assert net == pytest.approx(threeDEP_gross * (1 - HEAD_LOSS_FRACTION), rel=1e-4)

    def test_head_below_minimum_is_invalid(self):
        """Net head below MIN_NET_HEAD_M should set head_valid = False."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=200.0,
            elev_outfall_m=199.5,   # gross = 0.5 m → net < 1 m
            head_p50_literature_m=None,
        )
        # Separate assertions: bugs that set valid=True are not masked by the or-branch
        assert not valid, (
            f"head_valid should be False for sub-minimum net head; got valid={valid}, net={net}"
        )

    def test_loss_fraction_applied_correctly(self):
        """Verify the arithmetic of the loss fraction multiplier."""
        gross, net, src, valid, conf = _compute_head_row(
            elev_facility_m=320.0,
            elev_outfall_m=300.0,   # gross = 20 m
            head_p50_literature_m=20.0,
        )
        if src == "usgs_3dep":
            assert net == pytest.approx(20.0 * (1 - HEAD_LOSS_FRACTION), rel=1e-4)


class TestEstimateHead:
    """Integration tests for estimate_head()."""

    def _make_df(self, n: int = 4) -> pl.DataFrame:
        return pl.DataFrame({
            "npdes_id":     [f"MN{i:07d}" for i in range(n)],
            "elevation_m":  [300.0, None,  310.0, 290.0],
            "elev_outfall_m": [290.0, None, None, 295.0],
            "head_m_p50":   [10.0,  8.0,  7.0,  None],
            "mean_flow_mgd":[10.0,  5.0,  3.0,  2.0],
        })

    def test_output_columns_present(self):
        df = estimate_head(self._make_df())
        for col in ("head_gross_m", "head_net_m", "head_source", "head_valid", "head_confidence"):
            assert col in df.columns, f"Column '{col}' missing from estimate_head output"

    def test_row_count_unchanged(self):
        df_in = self._make_df()
        df_out = estimate_head(df_in)
        assert len(df_out) == len(df_in)

    def test_no_facility_is_all_null_head(self):
        """Every row should have a head_net_m value (design fallback ensures this)."""
        df = estimate_head(self._make_df())
        null_count = df["head_net_m"].is_null().sum()
        assert null_count == 0, f"{null_count} rows have null head_net_m"

    def test_missing_required_column_raises(self):
        bad_df = pl.DataFrame({"npdes_id": ["A"], "some_col": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            estimate_head(bad_df)

    def test_source_column_valid_values(self):
        df = estimate_head(self._make_df())
        valid_sources = {"usgs_3dep", "phase2_literature", "design_fallback"}
        actual = set(df["head_source"].to_list())
        assert actual.issubset(valid_sources), f"Unknown source values: {actual - valid_sources}"

    def test_confidence_column_valid_values(self):
        df = estimate_head(self._make_df())
        valid_conf = {"high", "medium", "low"}
        actual = set(df["head_confidence"].to_list())
        assert actual.issubset(valid_conf)


class TestHeadSummaryStats:
    def test_returns_dict_with_expected_keys(self):
        df = pl.DataFrame({
            "npdes_id":    ["A", "B", "C"],
            "elevation_m": [300.0, None, 280.0],
            "head_net_m":  [8.5, 5.0, 3.0],
            "head_valid":  [True, True, False],
            "head_source": ["usgs_3dep", "phase2_literature", "design_fallback"],
            "head_confidence": ["high", "medium", "low"],
        })
        stats = head_summary_stats(df)
        for k in ("n_viable", "head_p50_m", "head_mean_m"):
            assert k in stats, f"Key '{k}' missing from head_summary_stats"
        assert stats["n_viable"] == 2  # only head_valid=True rows

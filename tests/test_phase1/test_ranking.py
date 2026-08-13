"""Tests for phase1/ranking.py — architecture verification plan §1.5 items 3–4."""

from __future__ import annotations

import polars as pl
import pytest

from src.phase1.ranking import compute_ranking


@pytest.fixture
def ranked_df(sample_timeseries, sample_potw_facilities):
    from src.phase1.flow_features import compute_flow_features
    features = compute_flow_features(sample_timeseries, sample_potw_facilities)
    return compute_ranking(features)


class TestRanking:

    def test_ranking_score_in_unit_interval(self, ranked_df):
        scores = ranked_df["ranking_score"].drop_nulls()
        assert (scores >= 0.0).all(), "Scores below 0"
        assert (scores <= 1.0).all(), "Scores above 1"

    def test_rank_is_unique_and_sequential(self, ranked_df):
        ranks = ranked_df["rank"].sort().to_list()
        expected = list(range(1, len(ranked_df) + 1))
        assert ranks == expected, f"Ranks not sequential: {ranks}"

    def test_higher_mean_flow_wins_if_equal_consistency(self, sample_potw_facilities):
        """Given two plants with identical CV/n_years, higher mean flow should rank higher."""
        from src.phase1.flow_features import _compute_fdc
        import numpy as np

        # Create a minimal features df with two plants only varying by mean_flow
        df = pl.DataFrame({
            "npdes_id": ["A", "B"],
            "facility_name": ["Big Flow", "Small Flow"],
            "city": ["City", "City"],
            "state_code": ["MN", "MN"],
            "zip": ["55401", "55401"],
            "latitude": [44.9, 44.9],
            "longitude": [-93.2, -93.2],
            "facility_type_indicator": ["POTW", "POTW"],
            "facility_type_code": ["POT", "POT"],
            "major_minor": ["M", "M"],
            "design_flow_mgd": [50.0, 5.0],
            "actual_avg_flow_mgd": [40.0, 4.0],
            "permit_status_code": ["EFF", "EFF"],
            "mean_flow_mgd": [40.0, 4.0],
            "median_flow_mgd": [40.0, 4.0],
            "std_flow_mgd": [2.0, 0.2],
            "cv_flow": [0.05, 0.05],
            "p10_flow_mgd": [37.0, 3.7],
            "p25_flow_mgd": [38.5, 3.85],
            "p75_flow_mgd": [41.5, 4.15],
            "p90_flow_mgd": [43.0, 4.3],
            "min_flow_mgd": [35.0, 3.5],
            "max_flow_mgd": [45.0, 4.5],
            "n_months_data": [36, 36],
            "n_years_data": [3, 3],
            "pct_missing": [0.0, 0.0],
            "utilization_ratio": [0.80, 0.80],
            "flow_trend_mgd_per_year": [0.0, 0.0],
            "seasonal_amplitude_mgd": [3.0, 0.3],
            "flow_duration_curve": [
                _compute_fdc(np.full(36, 40.0)),
                _compute_fdc(np.full(36, 4.0)),
            ],
            "data_quality": ["dmr", "dmr"],
        })

        ranked = compute_ranking(df)
        rank_a = ranked.filter(pl.col("npdes_id") == "A")["rank"][0]
        rank_b = ranked.filter(pl.col("npdes_id") == "B")["rank"][0]
        assert rank_a < rank_b, (
            f"Big flow plant (rank {rank_a}) should outrank small flow plant (rank {rank_b})"
        )

    def test_steady_flow_ranks_better_than_variable_same_mean(self):
        """Same mean flow but lower CV should yield higher ranking score."""
        import numpy as np
        from src.phase1.flow_features import _compute_fdc

        flows_steady = np.full(36, 10.0)
        flows_variable = np.concatenate([np.full(18, 1.0), np.full(18, 19.0)])

        df = pl.DataFrame({
            "npdes_id": ["Steady", "Variable"],
            "facility_name": ["Steady", "Variable"],
            "city": ["C", "C"],
            "state_code": ["MN", "MN"],
            "zip": ["55401", "55401"],
            "latitude": [44.9, 44.9],
            "longitude": [-93.2, -93.2],
            "facility_type_indicator": ["POTW", "POTW"],
            "facility_type_code": ["POT", "POT"],
            "major_minor": ["M", "M"],
            "design_flow_mgd": [12.0, 12.0],
            "actual_avg_flow_mgd": [10.0, 10.0],
            "permit_status_code": ["EFF", "EFF"],
            "mean_flow_mgd": [10.0, 10.0],
            "median_flow_mgd": [10.0, 10.0],
            "std_flow_mgd": [0.0, float(np.std(flows_variable, ddof=1))],
            "cv_flow": [0.0, float(np.std(flows_variable, ddof=1) / 10.0)],
            "p10_flow_mgd": [10.0, float(np.percentile(flows_variable, 10))],
            "p25_flow_mgd": [10.0, float(np.percentile(flows_variable, 25))],
            "p75_flow_mgd": [10.0, float(np.percentile(flows_variable, 75))],
            "p90_flow_mgd": [10.0, float(np.percentile(flows_variable, 90))],
            "min_flow_mgd": [10.0, 1.0],
            "max_flow_mgd": [10.0, 19.0],
            "n_months_data": [36, 36],
            "n_years_data": [3, 3],
            "pct_missing": [0.0, 0.0],
            "utilization_ratio": [0.83, 0.83],
            "flow_trend_mgd_per_year": [0.0, 0.0],
            "seasonal_amplitude_mgd": [0.0, 18.0],
            "flow_duration_curve": [
                _compute_fdc(flows_steady),
                _compute_fdc(flows_variable),
            ],
            "data_quality": ["dmr", "dmr"],
        })

        ranked = compute_ranking(df)
        score_steady = ranked.filter(pl.col("npdes_id") == "Steady")["ranking_score"][0]
        score_variable = ranked.filter(pl.col("npdes_id") == "Variable")["ranking_score"][0]
        assert score_steady > score_variable, (
            f"Steady score ({score_steady:.4f}) should exceed variable score ({score_variable:.4f})"
        )

    def test_no_temp_columns_in_output(self, ranked_df):
        """Intermediate _norm_ columns should be stripped from output."""
        for col in ranked_df.columns:
            assert not col.startswith("_norm_"), f"Temp column leaked: {col}"
            assert not col.endswith("_capped"), f"Temp column leaked: {col}"
        assert "_mean_flow_for_ranking" not in ranked_df.columns

    def test_design_zero_zeros_ranking_score(self):
        """Sites with design_flow_mgd == 0 must have ranking_score == 0 and rank last.

        Regression guard: KYP000044/KYP000040 Boyd County (design=0, mean=243 MGD)
        previously ranked #2/#5 due to high raw DMR flow.  The post-fix code zeros
        ranking_score for design<=0 so misclassified industrial discharges don't
        appear at the top of P1 visualisations.
        """
        import numpy as np
        from src.phase1.flow_features import _compute_fdc

        flows = np.full(36, 40.0)
        fdc = _compute_fdc(flows)
        df = pl.DataFrame({
            "npdes_id": ["Good", "Industrial", "Tiny"],
            "facility_name": ["Real POTW", "Misclassified", "Small Real"],
            "city": ["C", "C", "C"],
            "state_code": ["MN", "MN", "MN"],
            "zip": ["55401", "55401", "55401"],
            "latitude": [44.9, 44.9, 44.9],
            "longitude": [-93.2, -93.2, -93.2],
            "facility_type_indicator": ["POTW", "POTW", "POTW"],
            "facility_type_code": ["POT", "POT", "POT"],
            "major_minor": ["M", "M", "m"],
            # Industrial has high mean_flow but design=0 — should be zeroed
            "design_flow_mgd": [50.0, 0.0, 5.0],
            "actual_avg_flow_mgd": [40.0, 240.0, 4.0],
            "permit_status_code": ["EFF", "EFF", "EFF"],
            "mean_flow_mgd": [40.0, 240.0, 4.0],
            "median_flow_mgd": [40.0, 240.0, 4.0],
            "std_flow_mgd": [2.0, 5.0, 0.2],
            "cv_flow": [0.05, 0.02, 0.05],
            "p10_flow_mgd": [37.0, 235.0, 3.7],
            "p25_flow_mgd": [38.5, 237.0, 3.85],
            "p75_flow_mgd": [41.5, 243.0, 4.15],
            "p90_flow_mgd": [43.0, 245.0, 4.3],
            "min_flow_mgd": [35.0, 230.0, 3.5],
            "max_flow_mgd": [45.0, 250.0, 4.5],
            "n_months_data": [36, 36, 36],
            "n_years_data": [3, 3, 3],
            "pct_missing": [0.0, 0.0, 0.0],
            "utilization_ratio": [0.80, None, 0.80],
            "flow_trend_mgd_per_year": [0.0, 0.0, 0.0],
            "seasonal_amplitude_mgd": [3.0, 5.0, 0.3],
            "flow_duration_curve": [fdc, _compute_fdc(np.full(36, 240.0)), _compute_fdc(np.full(36, 4.0))],
            "data_quality": ["dmr", "dmr_limited", "dmr"],
        })

        ranked = compute_ranking(df)
        ind = ranked.filter(pl.col("npdes_id") == "Industrial").to_dicts()[0]
        good = ranked.filter(pl.col("npdes_id") == "Good").to_dicts()[0]
        assert ind["ranking_score"] == 0.0, (
            f"design=0 site should have score=0, got {ind['ranking_score']}"
        )
        assert ind["rank"] == 3, (
            f"design=0 site should rank LAST (3 of 3), got rank {ind['rank']}"
        )
        # The real POTW (Good) should now be #1 — not the high-flow industrial
        assert good["rank"] == 1, (
            f"Real POTW should outrank misclassified industrial, got rank {good['rank']}"
        )

    def test_design_null_ranks_normally(self):
        """design_flow_mgd = null sites are NOT zeroed — only explicit design=0 is zeroed.

        Null design_flow means missing permit data (clerical gap), not an industrial
        misclassification. ~2,106 corpus sites fall here. Zeroing them collapses
        legitimate POTWs with good DMR history. Leave in normal ranking.
        Documented in WOWERS_PROJECT_JOURNAL.md (F1 trade-off resolution).
        """
        import numpy as np
        from src.phase1.flow_features import _compute_fdc

        fdc = _compute_fdc(np.full(36, 40.0))
        df = pl.DataFrame({
            "npdes_id": ["WithDesign", "NoDesign"],
            "facility_name": ["A", "B"],
            "city": ["C", "C"],
            "state_code": ["MN", "MN"],
            "zip": ["55401", "55401"],
            "latitude": [44.9, 44.9],
            "longitude": [-93.2, -93.2],
            "facility_type_indicator": ["POTW", "POTW"],
            "facility_type_code": ["POT", "POT"],
            "major_minor": ["M", "M"],
            "design_flow_mgd": [50.0, None],
            "actual_avg_flow_mgd": [40.0, 40.0],
            "permit_status_code": ["EFF", "EFF"],
            "mean_flow_mgd": [40.0, 40.0],
            "median_flow_mgd": [40.0, 40.0],
            "std_flow_mgd": [2.0, 2.0],
            "cv_flow": [0.05, 0.05],
            "p10_flow_mgd": [37.0, 37.0],
            "p25_flow_mgd": [38.5, 38.5],
            "p75_flow_mgd": [41.5, 41.5],
            "p90_flow_mgd": [43.0, 43.0],
            "min_flow_mgd": [35.0, 35.0],
            "max_flow_mgd": [45.0, 45.0],
            "n_months_data": [36, 36],
            "n_years_data": [3, 3],
            "pct_missing": [0.0, 0.0],
            "utilization_ratio": [0.80, None],
            "flow_trend_mgd_per_year": [0.0, 0.0],
            "seasonal_amplitude_mgd": [3.0, 3.0],
            "flow_duration_curve": [fdc, fdc],
            "data_quality": ["dmr", "dmr"],
        })

        ranked = compute_ranking(df)
        no_design = ranked.filter(pl.col("npdes_id") == "NoDesign").to_dicts()[0]
        with_design = ranked.filter(pl.col("npdes_id") == "WithDesign").to_dicts()[0]
        assert no_design["ranking_score"] > 0.0, (
            "design=null sites should rank normally (not zeroed) — "
            f"got {no_design['ranking_score']}"
        )
        # Null-design site scores lower than full-design (utilization_ratio is null → 0),
        # but must still rank in normal pack (not zeroed to exactly 0.0)
        assert no_design["ranking_score"] < with_design["ranking_score"], (
            "null-design site should rank below full-data site (missing utilization)"
        )

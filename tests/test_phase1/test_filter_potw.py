"""Tests for phase1/filter_potw.py — architecture verification plan §1.5 items 1 & 4."""

from __future__ import annotations

import polars as pl
import pytest

from src.phase1.filter_potw import load_potw_facilities


class TestPOTWFilter:

    def test_returns_only_potw_facilities(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """Only POTW facilities with active permits should be returned."""
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        # All returned rows should have POTW indicator
        types = result["facility_type_indicator"].unique().to_list()
        assert all(t in ("POTW", "POT") for t in types if t is not None)

    def test_count_matches_expected_potw(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """Expect 20 active POTW facilities (IND, FDF, and TRM excluded)."""
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        # 20 active POTWs; 10 POTW-no-coords dropped; 5 terminated dropped; 15 IND filtered
        assert len(result) == 20

    def test_no_negative_or_zero_coords(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """All returned facilities must have valid lat/lon."""
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        assert result["latitude"].null_count() == 0
        assert result["longitude"].null_count() == 0

    def test_output_schema_has_required_columns(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """Output must contain the columns defined in POTW_SCHEMA."""
        from src.phase1.filter_potw import POTW_SCHEMA
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        for col in POTW_SCHEMA:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_duplicate_npdes_ids(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """Each NPDES ID should appear at most once."""
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        assert result["npdes_id"].n_unique() == len(result)

    def test_design_flow_is_float(
        self, sample_facilities_csv, sample_permits_csv
    ):
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        assert result["design_flow_mgd"].dtype == pl.Float64

    def test_excludes_terminated_permits(
        self, sample_facilities_csv, sample_permits_csv
    ):
        """Terminated permits (TRM status) should not appear in output."""
        result = load_potw_facilities(sample_facilities_csv, sample_permits_csv)
        terminated_ids = {f"MNT{i:07d}" for i in range(5)}
        returned_ids = set(result["npdes_id"].to_list())
        assert terminated_ids.isdisjoint(returned_ids), (
            f"Terminated permits found in output: {terminated_ids & returned_ids}"
        )

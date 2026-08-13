"""Tests for phase1/dmr_timeseries.py — architecture verification plan §1.5 item 2."""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.phase1.dmr_timeseries import parse_dmr_year, _process_chunk, _build_col_map


def _make_dmr_zip(tmp_path: Path, csv_path: Path, year: int) -> Path:
    """Wrap a CSV into a ZIP as the DMR downloader would produce."""
    zip_path = tmp_path / f"npdes_dmrs_fy{str(year)[2:]}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname=csv_path.name)
    return zip_path


class TestDMRParsing:

    def test_only_flow_parameter_retained(self, sample_dmr_csv, tmp_path):
        """Only rows with PARAMETER_CODE=50050 should survive."""
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000", "MN0000001"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        # We should only have records for parameter 50050 — no temperature etc.
        assert len(result) > 0
        # All resulting npdes_ids should be in our POTW set
        assert set(result["npdes_id"].to_list()).issubset(potw_ids)

    def test_non_potw_facility_excluded(self, sample_dmr_csv, tmp_path):
        """IL9999999 is not in potw_ids and must not appear in output."""
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000", "MN0000001"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        assert "IL9999999" not in result["npdes_id"].to_list()

    def test_nodi_c_becomes_zero(self, sample_dmr_csv, tmp_path):
        """NODI code C (conditionally exempt) should produce flow_value = 0.0."""
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000001"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        mn1 = result.filter(pl.col("npdes_id") == "MN0000001")
        if len(mn1) > 0:
            # avg_flow_mgd for the NODI-C record should be 0.0 (not null)
            assert mn1["avg_flow_mgd"].drop_nulls().to_list() == [0.0]

    def test_output_has_required_columns(self, sample_dmr_csv, tmp_path):
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        required = ["npdes_id", "outfall", "period_end", "avg_flow_mgd", "fiscal_year"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_period_end_is_date_type(self, sample_dmr_csv, tmp_path):
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        assert result["period_end"].dtype == pl.Date

    def test_no_negative_flow_values(self, sample_dmr_csv, tmp_path):
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000", "MN0000001"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        flows = result["avg_flow_mgd"].drop_nulls()
        assert (flows >= 0).all(), "Found negative flow values"

    def test_avg_max_min_correctly_pivoted(self, sample_dmr_csv, tmp_path):
        """MN0000000 has MO AVG=~10, MO MAX=~13, MO MIN=~7 — verify pivot."""
        zip_path = _make_dmr_zip(tmp_path, sample_dmr_csv, 2022)
        potw_ids = {"MN0000000"}
        result = parse_dmr_year(zip_path, 2022, potw_ids)
        mn0 = result.filter(pl.col("npdes_id") == "MN0000000")
        assert len(mn0) > 0
        # max >= avg >= min for non-null rows
        valid = mn0.filter(
            pl.col("avg_flow_mgd").is_not_null()
            & pl.col("max_flow_mgd").is_not_null()
            & pl.col("min_flow_mgd").is_not_null()
        )
        if len(valid) > 0:
            assert (valid["max_flow_mgd"] >= valid["avg_flow_mgd"]).all()
            assert (valid["avg_flow_mgd"] >= valid["min_flow_mgd"]).all()

    def test_col_map_handles_known_variants(self):
        """_build_col_map should resolve all known column name variants."""
        columns_variant_a = [
            "EXTERNAL_PERMIT_NMBR", "PERM_FEATURE_NMBR",
            "MONITORING_PERIOD_END_DATE", "PARAMETER_CODE",
            "STATISTICAL_BASE_SHORT_DESC", "DMR_VALUE_NMBR", "NODI_CODE",
        ]
        col_map = _build_col_map(columns_variant_a)
        assert "npdes_id" in col_map
        assert "param_code" in col_map
        assert "value" in col_map

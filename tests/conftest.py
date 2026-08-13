"""Shared fixtures for all Phase 1 tests."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_facilities_csv(tmp_path: Path) -> Path:
    """50-row synthetic ICIS_FACILITIES.CSV with mix of types."""
    rows = []
    # 20 POTWs with coordinates
    for i in range(20):
        rows.append({
            "NPDES_ID": f"MN{i:07d}",
            "FACILITY_TYPE_CODE": "POT",
            "FACILITY_NAME": f"Test POTW {i}",
            "LOCATION_ADDRESS": f"{i} Main St",
            "CITY": "Minneapolis",
            "STATE_CODE": "MN",
            "ZIP": "55401",
            "GEOCODE_LATITUDE": 44.9 + i * 0.01,
            "GEOCODE_LONGITUDE": -93.2 + i * 0.01,
        })
    # 15 industrial (non-POTW)
    for i in range(15):
        rows.append({
            "NPDES_ID": f"MNI{i:06d}",
            "FACILITY_TYPE_CODE": "IND",
            "FACILITY_NAME": f"Industrial {i}",
            "LOCATION_ADDRESS": f"{i} Industry Rd",
            "CITY": "St. Paul",
            "STATE_CODE": "MN",
            "ZIP": "55102",
            "GEOCODE_LATITUDE": 44.95 + i * 0.01,
            "GEOCODE_LONGITUDE": -93.1 + i * 0.01,
        })
    # 10 with missing coordinates
    for i in range(10):
        rows.append({
            "NPDES_ID": f"MNC{i:07d}",
            "FACILITY_TYPE_CODE": "POT",
            "FACILITY_NAME": f"POTW No Coords {i}",
            "LOCATION_ADDRESS": "",
            "CITY": "Unknown",
            "STATE_CODE": "MN",
            "ZIP": "",
            "GEOCODE_LATITUDE": None,
            "GEOCODE_LONGITUDE": None,
        })
    # 5 federal (FDF) — should be excluded
    for i in range(5):
        rows.append({
            "NPDES_ID": f"MNF{i:07d}",
            "FACILITY_TYPE_CODE": "FDF",
            "FACILITY_NAME": f"Federal Facility {i}",
            "LOCATION_ADDRESS": "1 Federal Way",
            "CITY": "Washington",
            "STATE_CODE": "DC",
            "ZIP": "20001",
            "GEOCODE_LATITUDE": 38.9,
            "GEOCODE_LONGITUDE": -77.0,
        })

    df = pl.DataFrame(rows)
    path = tmp_path / "ICIS_FACILITIES.CSV"
    df.write_csv(path)
    return path


@pytest.fixture
def sample_permits_csv(tmp_path: Path) -> Path:
    """Synthetic ICIS_PERMITS.CSV matching sample_facilities_csv."""
    rows = []
    # 20 active POTW permits
    for i in range(20):
        rows.append({
            "EXTERNAL_PERMIT_NMBR": f"MN{i:07d}",
            "FACILITY_TYPE_INDICATOR": "POTW",
            "TOTAL_DESIGN_FLOW_NMBR": float(10 + i * 5),
            "ACTUAL_AVERAGE_FLOW_NMBR": float(7 + i * 4),
            "MAJOR_MINOR_STATUS_FLAG": "M" if i < 10 else "N",
            "PERMIT_STATUS_CODE": "EFF",
        })
    # 15 industrial — non-POTW
    for i in range(15):
        rows.append({
            "EXTERNAL_PERMIT_NMBR": f"MNI{i:06d}",
            "FACILITY_TYPE_INDICATOR": "IND",
            "TOTAL_DESIGN_FLOW_NMBR": 1.0,
            "ACTUAL_AVERAGE_FLOW_NMBR": 0.8,
            "MAJOR_MINOR_STATUS_FLAG": "N",
            "PERMIT_STATUS_CODE": "EFF",
        })
    # 5 POTW but terminated — should be excluded
    for i in range(5):
        rows.append({
            "EXTERNAL_PERMIT_NMBR": f"MNT{i:07d}",
            "FACILITY_TYPE_INDICATOR": "POTW",
            "TOTAL_DESIGN_FLOW_NMBR": 5.0,
            "ACTUAL_AVERAGE_FLOW_NMBR": 3.0,
            "MAJOR_MINOR_STATUS_FLAG": "N",
            "PERMIT_STATUS_CODE": "TRM",  # terminated
        })

    df = pl.DataFrame(rows)
    path = tmp_path / "ICIS_PERMITS.CSV"
    df.write_csv(path)
    return path


@pytest.fixture
def sample_dmr_csv(tmp_path: Path) -> Path:
    """Synthetic DMR CSV with flow (50050) and non-flow parameter codes."""
    rows = []
    # Flow records for MN0000000 — 24 months, avg + max + min
    for month in range(1, 25):
        year = 2022 + (month - 1) // 12
        m = ((month - 1) % 12) + 1
        period = date(year, m, 28)
        base_flow = 10.0 + month * 0.1
        for stat, value in [("MO AVG", base_flow), ("MO MAX", base_flow * 1.3), ("MO MIN", base_flow * 0.7)]:
            rows.append({
                "EXTERNAL_PERMIT_NMBR": "MN0000000",
                "PERM_FEATURE_NMBR": "001",
                "MONITORING_PERIOD_END_DATE": period.strftime("%m/%d/%Y"),
                "PARAMETER_CODE": "50050",
                "STATISTICAL_BASE_SHORT_DESC": stat,
                "DMR_VALUE_NMBR": str(value),
                "NODI_CODE": "",
            })
    # Non-flow parameter (should be filtered out)
    rows.append({
        "EXTERNAL_PERMIT_NMBR": "MN0000000",
        "PERM_FEATURE_NMBR": "001",
        "MONITORING_PERIOD_END_DATE": "01/31/2022",
        "PARAMETER_CODE": "00010",  # Temperature — not flow
        "STATISTICAL_BASE_SHORT_DESC": "MO AVG",
        "DMR_VALUE_NMBR": "18.5",
        "NODI_CODE": "",
    })
    # NODI code C (should become 0.0)
    rows.append({
        "EXTERNAL_PERMIT_NMBR": "MN0000001",
        "PERM_FEATURE_NMBR": "001",
        "MONITORING_PERIOD_END_DATE": "03/31/2022",
        "PARAMETER_CODE": "50050",
        "STATISTICAL_BASE_SHORT_DESC": "MO AVG",
        "DMR_VALUE_NMBR": "",
        "NODI_CODE": "C",
    })
    # Record for non-POTW facility (should be filtered)
    rows.append({
        "EXTERNAL_PERMIT_NMBR": "IL9999999",
        "PERM_FEATURE_NMBR": "001",
        "MONITORING_PERIOD_END_DATE": "01/31/2022",
        "PARAMETER_CODE": "50050",
        "STATISTICAL_BASE_SHORT_DESC": "MO AVG",
        "DMR_VALUE_NMBR": "50.0",
        "NODI_CODE": "",
    })

    df = pl.DataFrame(rows)
    path = tmp_path / "NPDES_DMRS_FY2022.CSV"
    df.write_csv(path)
    return path


@pytest.fixture
def sample_timeseries() -> pl.DataFrame:
    """Pre-built monthly flow time series for 3 facilities."""
    rows = []
    # MN0000000: 36 months, steady flow ~10 MGD
    for i in range(36):
        year = 2021 + i // 12
        month = (i % 12) + 1
        rows.append({
            "npdes_id": "MN0000000",
            "outfall": "001",
            "period_end": date(year, month, 28),
            "fiscal_year": year,
            "avg_flow_mgd": 10.0 + (i % 3) * 0.5,
            "max_flow_mgd": 13.0,
            "min_flow_mgd": 7.0,
            "is_estimated": False,
        })
    # MN0000001: 12 months, highly variable
    for i in range(12):
        rows.append({
            "npdes_id": "MN0000001",
            "outfall": "001",
            "period_end": date(2022, i + 1, 28),
            "fiscal_year": 2022,
            "avg_flow_mgd": 5.0 + (i % 6) * 3.0,  # swings between 5 and 20
            "max_flow_mgd": 25.0,
            "min_flow_mgd": 2.0,
            "is_estimated": False,
        })
    # MN0000002: 6 months, limited data
    for i in range(6):
        rows.append({
            "npdes_id": "MN0000002",
            "outfall": "001",
            "period_end": date(2023, i + 1, 28),
            "fiscal_year": 2023,
            "avg_flow_mgd": 2.5,
            "max_flow_mgd": 3.0,
            "min_flow_mgd": 2.0,
            "is_estimated": False,
        })

    return pl.DataFrame(rows).with_columns(
        pl.col("period_end").cast(pl.Date)
    )


@pytest.fixture
def sample_potw_facilities() -> pl.DataFrame:
    """Minimal POTW facilities DataFrame for feature/ranking tests."""
    return pl.DataFrame({
        "npdes_id": ["MN0000000", "MN0000001", "MN0000002", "MN0000003"],
        "facility_name": ["Big Plant", "Variable Plant", "Small Plant", "No Data Plant"],
        "city": ["Minneapolis", "St Paul", "Duluth", "Rochester"],
        "state_code": ["MN", "MN", "MN", "MN"],
        "zip": ["55401", "55102", "55802", "55901"],
        "latitude": [44.98, 44.95, 46.78, 44.02],
        "longitude": [-93.27, -93.09, -92.10, -92.46],
        "facility_type_indicator": ["POTW", "POTW", "POTW", "POTW"],
        "facility_type_code": ["POT", "POT", "POT", "POT"],
        "major_minor": ["M", "M", "N", "N"],
        "design_flow_mgd": [15.0, 20.0, 3.0, 5.0],
        "actual_avg_flow_mgd": [11.0, 12.0, 2.2, 3.5],
        "permit_status_code": ["EFF", "EFF", "EFF", "EFF"],
    })

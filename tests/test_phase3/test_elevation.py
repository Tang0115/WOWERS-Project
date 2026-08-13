"""Tests for phase3/elevation.py — cache logic and DataFrame contract."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from src.phase3.elevation import (
    _cache_key,
    _cache_path,
    _read_cache,
    _write_cache,
    fetch_elevations,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_facilities():
    return pl.DataFrame({
        "npdes_id":  ["MN0000001", "MN0000002", "MN0000003", "MN0000004"],
        "latitude":  [44.98,       45.01,       None,        46.80],
        "longitude": [-93.27,      -93.10,      None,        -92.10],
        "mean_flow_mgd": [10.0, 5.0, 2.0, 8.0],
    })


# ── Cache key tests ───────────────────────────────────────────────────────────

class TestCacheKey:
    def test_rounding_precision(self):
        key1 = _cache_key(44.98765432, -93.12345678)
        key2 = _cache_key(44.98765, -93.12346)
        assert key1 == key2, "Cache key should round to 5 decimal places"

    def test_distinct_points(self):
        key1 = _cache_key(44.0, -93.0)
        key2 = _cache_key(45.0, -93.0)
        assert key1 != key2


class TestCacheReadWrite:
    def test_write_and_read_roundtrip(self, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        _write_cache(44.98, -93.27, 287.5)
        result = _read_cache(44.98, -93.27)
        assert result == pytest.approx(287.5)

    def test_read_miss_returns_none(self, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        result = _read_cache(99.0, -99.0)
        assert result is None

    def test_write_none_elevation(self, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        _write_cache(44.98, -93.27, None)
        # A cached None should not be returned as a valid hit (read returns None)
        result = _read_cache(44.98, -93.27)
        assert result is None

    def test_corrupt_cache_returns_none(self, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        p = _cache_path(44.98, -93.27)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")

        result = _read_cache(44.98, -93.27)
        assert result is None


# ── fetch_elevations contract ─────────────────────────────────────────────────

class TestFetchElevations:
    def test_output_has_required_columns(self, sample_facilities, tmp_path, monkeypatch):
        """fetch_elevations must return DataFrame with elevation_m and elev_source."""
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        # Populate cache for valid-coord rows so no API call needed
        _write_cache(44.98, -93.27, 300.0)
        _write_cache(45.01, -93.10, 290.0)
        _write_cache(46.80, -92.10, 400.0)

        result = fetch_elevations(sample_facilities)

        assert "elevation_m" in result.columns
        assert "elev_source" in result.columns
        assert len(result) == len(sample_facilities)

    def test_no_coords_gets_no_coords_source(self, sample_facilities, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        _write_cache(44.98, -93.27, 300.0)
        _write_cache(45.01, -93.10, 290.0)
        _write_cache(46.80, -92.10, 400.0)

        result = fetch_elevations(sample_facilities)
        no_coord_rows = result.filter(pl.col("npdes_id") == "MN0000003")
        assert no_coord_rows["elev_source"][0] == "no_coords"
        assert no_coord_rows["elevation_m"][0] is None

    def test_cache_hits_get_cache_source(self, sample_facilities, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        _write_cache(44.98, -93.27, 300.0)
        _write_cache(45.01, -93.10, 290.0)
        _write_cache(46.80, -92.10, 400.0)

        result = fetch_elevations(sample_facilities)
        cache_rows = result.filter(pl.col("elev_source") == "cache")
        assert len(cache_rows) == 3  # the 3 rows with coordinates that we cached

    def test_missing_columns_raises(self):
        bad_df = pl.DataFrame({"npdes_id": ["A"], "latitude": [44.0]})
        with pytest.raises(ValueError, match="missing columns"):
            fetch_elevations(bad_df)


# ── Ocean sentinel detection ──────────────────────────────────────────────────

class TestOceanSentinel:
    """EPQS returns -1000000 for ocean / off-grid points; elevation_m should be None."""

    @pytest.mark.asyncio
    async def test_ocean_sentinel_produces_none_elevation(self, tmp_path, monkeypatch):
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(elev_mod, "MAX_RETRIES", 1)

        ocean_response = {"value": "-1000000"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ocean_response

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        semaphore = __import__("asyncio").Semaphore(1)
        result = await elev_mod._query_one(mock_client, semaphore, "TX0000001", 29.5, -95.0)

        assert result["elevation_m"] is None, (
            "Ocean sentinel -1000000 should produce elevation_m=None"
        )
        assert result["elev_source"] == "usgs_3dep"

    def test_null_cached_elevation_returns_none_on_read(self, tmp_path, monkeypatch):
        """A cached null (API failure) should not be treated as a valid cache hit."""
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)

        _write_cache(44.98, -93.27, None)
        result = _read_cache(44.98, -93.27)
        assert result is None, (
            "_read_cache must return None for a null-cached elevation "
            "(not raise TypeError)"
        )


# ── API mock path ─────────────────────────────────────────────────────────────

class TestFetchElevationsAPIPath:
    """Tests that exercise the actual async API call path, not just the cache."""

    def test_api_result_populates_elevation(self, tmp_path, monkeypatch):
        """fetch_elevations should call the API for cache-miss entries."""
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(elev_mod, "MAX_RETRIES", 1)
        monkeypatch.setattr(elev_mod, "REQUEST_DELAY", 0.0)

        api_response = {"value": "312.5"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = api_response

        async def fake_get(*_args, **_kwargs):
            return mock_resp

        single_facility = pl.DataFrame({
            "npdes_id":  ["CA0000001"],
            "latitude":  [34.05],
            "longitude": [-118.25],
        })

        with patch("httpx.AsyncClient") as mock_cls:
            instance = MagicMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = instance

            result = fetch_elevations(single_facility)

        assert result["elevation_m"][0] == pytest.approx(312.5)
        assert result["elev_source"][0] == "usgs_3dep"

    def test_api_failure_produces_failed_source(self, tmp_path, monkeypatch):
        """All-retries-exhausted should yield elev_source='failed' without crashing."""
        import src.phase3.elevation as elev_mod
        monkeypatch.setattr(elev_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(elev_mod, "MAX_RETRIES", 1)
        monkeypatch.setattr(elev_mod, "REQUEST_DELAY", 0.0)
        monkeypatch.setattr(elev_mod, "RETRY_BACKOFF", 0.0)

        import httpx

        single_facility = pl.DataFrame({
            "npdes_id":  ["FL0000001"],
            "latitude":  [25.77],
            "longitude": [-80.19],
        })

        with patch("httpx.AsyncClient") as mock_cls:
            instance = MagicMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=httpx.RequestError("timeout"))
            mock_cls.return_value = instance

            result = fetch_elevations(single_facility)

        assert result["elevation_m"][0] is None
        assert result["elev_source"][0] == "failed"

        # Second run: the null should be cached and not raise TypeError
        with patch("httpx.AsyncClient") as mock_cls2:
            instance2 = MagicMock()
            instance2.__aenter__ = AsyncMock(return_value=instance2)
            instance2.__aexit__ = AsyncMock(return_value=False)
            instance2.get = AsyncMock(side_effect=httpx.RequestError("timeout"))
            mock_cls2.return_value = instance2

            result2 = fetch_elevations(single_facility)

        assert result2["elevation_m"][0] is None  # no TypeError from _read_cache

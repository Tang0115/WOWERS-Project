"""Phase 3 — USGS 3DEP Elevation Queries.

Queries the USGS Elevation Point Query Service (EPQS) v1 API to retrieve
terrain elevation for each POTW facility's lat/lon coordinate.

Workflow
--------
1. Load ranked_candidates.parquet from Phase 1/2 output.
2. For each facility, check the disk cache first (JSON file keyed by rounded
   lat/lon) to avoid redundant API calls across pipeline re-runs.
3. Batch remaining queries with asyncio + httpx, honouring rate-limit settings
   from settings.yaml (max_concurrent_requests, request_delay_s).
4. Persist results to data/elevation_cache/ and return a Polars DataFrame.

API endpoint  https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&wkid=4326&units=Meters
Returns       {"value": <meters_asl>, ...}  or  {"value": "-1000000"} for ocean/error
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase3.elevation")

# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg(key: str, default=None):
    return config.get(f"phase3.{key}", default)


EPQS_URL: str = config.get("usgs.elevation_url", "https://epqs.nationalmap.gov/v1/json")
CACHE_DIR: Path = config.project_root() / _cfg("elevation_cache_dir", "data/elevation_cache")
COORD_PREC: int = _cfg("coord_precision", 5)
MAX_CONCURRENT: int = _cfg("max_concurrent_requests", 10)
REQUEST_DELAY: float = _cfg("request_delay_s", 0.5)
API_TIMEOUT: float = _cfg("api_timeout_s", 30)
MAX_RETRIES: int = _cfg("max_retries", 3)
RETRY_BACKOFF: float = _cfg("retry_backoff_s", 2.0)

# Sentinel: EPQS returns -1000000 for ocean / off-grid points
_OCEAN_SENTINEL = -1_000_000.0


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, COORD_PREC)},{round(lon, COORD_PREC)}"


def _cache_path(lat: float, lon: float) -> Path:
    key = _cache_key(lat, lon)
    # Shard into 1-degree grid subdirs to keep directory size manageable
    lat_bucket = int(lat)
    lon_bucket = int(lon)
    return CACHE_DIR / f"{lat_bucket}_{lon_bucket}" / f"{key}.json"


def _read_cache(lat: float, lon: float) -> Optional[float]:
    p = _cache_path(lat, lon)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        raw = data.get("elevation_m")          # key exists → None if JSON null
        if raw is None:
            return None                        # cached API failure — not a valid hit
        elev = float(raw)
        return None if elev != elev else elev  # NaN check
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def _write_cache(lat: float, lon: float, elevation_m: Optional[float]) -> None:
    p = _cache_path(lat, lon)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"elevation_m": elevation_m}, f)


# ── Single-point async query ──────────────────────────────────────────────────

async def _query_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    npdes_id: str,
    lat: float,
    lon: float,
) -> dict:
    """Fetch elevation for one point with retry logic."""
    async with semaphore:
        params = {"x": lon, "y": lat, "wkid": "4326", "units": "Meters"}
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.get(EPQS_URL, params=params, timeout=API_TIMEOUT)
                resp.raise_for_status()
                body = resp.json()

                # EPQS v1 returns {"value": "<float>"} or {"value": "-1000000"}
                raw = body.get("value", "")
                elev = float(raw) if raw not in ("", None) else None

                if elev is not None and abs(elev - _OCEAN_SENTINEL) < 1.0:
                    elev = None  # ocean / error sentinel

                _write_cache(lat, lon, elev)
                return {"npdes_id": npdes_id, "latitude": lat, "longitude": lon,
                        "elevation_m": elev, "elev_source": "usgs_3dep"}

            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF * attempt)

        log.warning(f"  Elevation query failed for {npdes_id} ({lat},{lon}) after "
                    f"{MAX_RETRIES} attempts: {last_exc}")
        _write_cache(lat, lon, None)
        return {"npdes_id": npdes_id, "latitude": lat, "longitude": lon,
                "elevation_m": None, "elev_source": "failed"}


# ── Batch async runner ────────────────────────────────────────────────────────

async def _fetch_elevations_async(
    points: list[tuple[str, float, float]],
) -> list[dict]:
    """Async-fetch elevations for all points, respecting rate limits.

    Args:
        points: List of (npdes_id, lat, lon) tuples with no cache hit.

    Returns:
        List of result dicts (one per point).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[dict] = []

    # Process in batches to insert REQUEST_DELAY between waves
    batch_size = MAX_CONCURRENT
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            tasks = [
                _query_one(client, semaphore, npdes_id, lat, lon)
                for npdes_id, lat, lon in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            log.info(
                f"  Elevation batch {batch_start + len(batch)}/{len(points)} done"
            )
            if batch_start + batch_size < len(points):
                await asyncio.sleep(REQUEST_DELAY)

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_elevations(facilities: pl.DataFrame) -> pl.DataFrame:
    """Fetch USGS 3DEP elevation for each facility.

    Args:
        facilities: DataFrame with columns [npdes_id, latitude, longitude].
                    Any facility missing lat/lon is passed through with
                    elevation_m=None and elev_source='no_coords'.

    Returns:
        DataFrame with original columns plus:
            elevation_m   (float | null)  — metres above sea level
            elev_source   (str)           — 'usgs_3dep', 'cache', 'no_coords',
                                            or 'failed'
    """
    required = {"npdes_id", "latitude", "longitude"}
    missing_cols = required - set(facilities.columns)
    if missing_cols:
        raise ValueError(f"facilities DataFrame missing columns: {missing_cols}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = facilities.select(["npdes_id", "latitude", "longitude"]).to_dicts()
    cached_results: list[dict] = []
    to_fetch: list[tuple[str, float, float]] = []

    for row in rows:
        npdes_id = row["npdes_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        if lat is None or lon is None:
            cached_results.append({
                "npdes_id": npdes_id, "latitude": lat, "longitude": lon,
                "elevation_m": None, "elev_source": "no_coords",
            })
            continue

        cached = _read_cache(lat, lon)
        if cached is not None:
            cached_results.append({
                "npdes_id": npdes_id, "latitude": lat, "longitude": lon,
                "elevation_m": cached, "elev_source": "cache",
            })
        else:
            to_fetch.append((npdes_id, lat, lon))

    n_cached = len(cached_results)
    n_fetch = len(to_fetch)
    log.info(f"Elevation: {n_cached:,} from cache, {n_fetch:,} need API queries")

    api_results: list[dict] = []
    if to_fetch:
        api_results = asyncio.run(_fetch_elevations_async(to_fetch))

    all_results = cached_results + api_results

    elev_df = pl.DataFrame({
        "npdes_id":    [r["npdes_id"]    for r in all_results],
        "elevation_m": [r["elevation_m"] for r in all_results],
        "elev_source": [r["elev_source"] for r in all_results],
    })

    result = facilities.join(elev_df, on="npdes_id", how="left")
    elapsed = time.time() - t0
    n_ok = elev_df["elevation_m"].drop_nulls().len()
    n_total = len(elev_df)
    log.info(
        f"Elevation complete: {n_ok:,}/{n_total:,} valid elevations in {elapsed:.1f}s"
    )
    return result


def load_elevation_results(path: Path) -> pl.DataFrame:
    """Load previously saved elevation_data.parquet."""
    return pl.read_parquet(path)

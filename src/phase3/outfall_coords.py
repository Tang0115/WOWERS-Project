"""Phase 3 — NPDES Outfall Coordinate Loader.

Loads outfall discharge-point coordinates and returns one representative outfall
per NPDES permit.  Two sources are consulted in priority order:

  1. ``npdes_outfalls_layer.csv`` (LATLONG_TYPE = "Permitted Feature",
     SUB_TYPE_DESC = "External Outfall") — actual discharge-pipe coordinates.
     When a facility has multiple external outfalls the one with the lowest
     PERM_FEATURE_NMBR is selected.

  2. ``NPDES_PERM_FEATURE_COORDS.csv`` — generic permit-feature coordinates used
     as a fallback for NPDES IDs not present in the outfalls layer.  This file
     does not label feature type, so coordinates may be facility centroids rather
     than true discharge points.  It is retained only for coverage.

Selection priority within each source:
  a. PERM_FEATURE_NMBR == '001'   (primary discharge outfall)
  b. Lowest numeric PERM_FEATURE_NMBR  (e.g. '002', '003')
  c. First row available

Rows with null or geo-implausible coordinates are dropped.

Path overrides via settings.yaml:
  phase3.outfall_coords_path          → NPDES_PERM_FEATURE_COORDS.csv
  phase3.outfalls_layer_path          → npdes_outfalls_layer.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import polars as pl

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase3.outfall_coords")

# ── Config ────────────────────────────────────────────────────────────────────

def _cfg(key: str, default=None):
    return config.get(f"phase3.{key}", default)


OUTFALL_COORDS_PATH: Path = (
    config.project_root()
    / _cfg(
        "outfall_coords_path",
        "data/raw/npdes_downloads/NPDES_PERM_FEATURE_COORDS.csv",
    )
)

OUTFALLS_LAYER_PATH: Path = (
    config.project_root()
    / _cfg(
        "outfalls_layer_path",
        "data/raw/npdes_downloads/npdes_outfalls_layer.csv",
    )
)

# Loose bounding box: continental US + AK + HI + territories
_LAT_MIN, _LAT_MAX = 10.0, 72.0
_LON_MIN, _LON_MAX = -180.0, -60.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_valid_coord(lat: float, lon: float) -> bool:
    return _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX


def _feature_priority(feat: str) -> int:
    """Lower = more preferred.  '001' → 0, numeric → int value, text → 9999."""
    stripped = feat.strip()
    if stripped == "001":
        return 0
    try:
        return int(stripped)
    except ValueError:
        return 9_999


def _pick_one_per_permit(records: list[dict]) -> pl.DataFrame:
    """Given a list of coord records, return one per npdes_id (lowest priority wins)."""
    if not records:
        return pl.DataFrame(
            {"npdes_id": pl.Series([], dtype=pl.Utf8),
             "lat_outfall": pl.Series([], dtype=pl.Float64),
             "lon_outfall": pl.Series([], dtype=pl.Float64)}
        )
    df = pl.DataFrame(records)
    df = (
        df.sort(["npdes_id", "_prio"])
          .unique(subset=["npdes_id"], keep="first")
          .drop(["_prio", "perm_feature"])
    )
    return df


# ── Source loaders ─────────────────────────────────────────────────────────────

def _load_outfalls_layer(
    npdes_ids: Optional[set[str]],
    path: Path,
) -> pl.DataFrame:
    """Load External Outfall discharge-point coordinates from npdes_outfalls_layer.csv.

    Filters to rows where SUB_TYPE_DESC == "External Outfall" and
    LATLONG_TYPE == "Permitted Feature".  Returns one coordinate per permit.
    """
    empty = pl.DataFrame(
        {"npdes_id": pl.Series([], dtype=pl.Utf8),
         "lat_outfall": pl.Series([], dtype=pl.Float64),
         "lon_outfall": pl.Series([], dtype=pl.Float64)}
    )
    if not path.exists():
        log.warning(
            f"npdes_outfalls_layer.csv not found at {path}. "
            "Will rely solely on NPDES_PERM_FEATURE_COORDS.csv for outfall coords."
        )
        return empty

    log.info(f"Loading External Outfall coords from {path.name} …")
    records: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Only accept actual discharge-pipe points
            if (row.get("LATLONG_TYPE") or "").strip() != "Permitted Feature":
                continue
            if (row.get("SUB_TYPE_DESC") or "").strip() != "External Outfall":
                continue

            npdes_id = (row.get("EXTERNAL_PERMIT_NMBR") or "").strip().upper()
            if not npdes_id:
                continue
            if npdes_ids is not None and npdes_id not in npdes_ids:
                continue

            feat    = (row.get("PERM_FEATURE_NMBR") or "").strip()
            lat_raw = (row.get("LATITUDE83")  or "").strip()
            lon_raw = (row.get("LONGITUDE83") or "").strip()

            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except ValueError:
                continue

            if not _is_valid_coord(lat, lon):
                continue

            records.append({
                "npdes_id":    npdes_id,
                "perm_feature": feat,
                "lat_outfall": lat,
                "lon_outfall": lon,
                "_prio":       _feature_priority(feat),
            })

    result = _pick_one_per_permit(records)
    log.info(
        f"  {path.name}: {len(records):,} valid External Outfall rows → "
        f"{len(result):,} permits"
    )
    return result


def _load_perm_feature_coords(
    npdes_ids: Optional[set[str]],
    path: Path,
    exclude_ids: Optional[set[str]] = None,
) -> pl.DataFrame:
    """Load coordinates from NPDES_PERM_FEATURE_COORDS.csv (fallback source).

    Args:
        npdes_ids:   Optional allowlist. Pass None to load all.
        path:        Path to NPDES_PERM_FEATURE_COORDS.csv.
        exclude_ids: Optional set of NPDES IDs already covered by the
                     outfalls layer — skipped here to avoid redundant I/O
                     on the 626k-row file.
    """
    empty = pl.DataFrame(
        {"npdes_id": pl.Series([], dtype=pl.Utf8),
         "lat_outfall": pl.Series([], dtype=pl.Float64),
         "lon_outfall": pl.Series([], dtype=pl.Float64)}
    )
    if not path.exists():
        log.warning(
            f"NPDES_PERM_FEATURE_COORDS.csv not found at {path}. "
            "Phase 3 will use literature-based head for all sites. "
            "Place the file (or a symlink) at the path above to enable "
            "USGS 3DEP real-elevation head calculation."
        )
        return empty

    log.info(f"Loading fallback outfall coords from {path.name} …")
    records: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            npdes_id = (row.get("EXTERNAL_PERMIT_NMBR") or "").strip().upper()
            if not npdes_id:
                continue
            if npdes_ids is not None and npdes_id not in npdes_ids:
                continue
            if exclude_ids is not None and npdes_id in exclude_ids:
                continue  # already covered by outfalls layer — skip

            feat    = (row.get("PERM_FEATURE_NMBR") or "").strip()
            lat_raw = (row.get("LATITUDE_MEASURE")  or "").strip()
            lon_raw = (row.get("LONGITUDE_MEASURE") or "").strip()

            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except ValueError:
                continue

            if not _is_valid_coord(lat, lon):
                continue

            records.append({
                "npdes_id":    npdes_id,
                "perm_feature": feat,
                "lat_outfall": lat,
                "lon_outfall": lon,
                "_prio":       _feature_priority(feat),
            })

    result = _pick_one_per_permit(records)
    log.info(
        f"  {path.name}: {len(records):,} valid rows → {len(result):,} permits"
    )
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def load_primary_outfall_coords(
    npdes_ids: Optional[list[str]] = None,
    path: Optional[Path] = None,
    outfalls_layer_path: Optional[Path] = None,
) -> pl.DataFrame:
    """Load one representative outfall coordinate per NPDES permit.

    Consults two sources in priority order:
      1. npdes_outfalls_layer.csv  (External Outfall discharge-pipe coords)
      2. NPDES_PERM_FEATURE_COORDS.csv  (generic permit-feature coords, fallback)

    Args:
        npdes_ids:           Optional allowlist of NPDES permit numbers.
                             Pass ``None`` to load all.
        path:                Override path for NPDES_PERM_FEATURE_COORDS.csv.
        outfalls_layer_path: Override path for npdes_outfalls_layer.csv.

    Returns:
        DataFrame with columns:
            npdes_id    (str)    NPDES permit number (upper-cased)
            lat_outfall (float)  outfall latitude  (NAD83)
            lon_outfall (float)  outfall longitude (NAD83)
    """
    id_set: Optional[set[str]] = set(npdes_ids) if npdes_ids is not None else None

    layer_path  = outfalls_layer_path or OUTFALLS_LAYER_PATH
    coords_path = path or OUTFALL_COORDS_PATH

    # Source 1: External Outfall discharge-pipe coordinates (preferred)
    primary = _load_outfalls_layer(id_set, layer_path)

    # Source 2: Generic permit-feature coords for IDs not covered by source 1
    already_covered: set[str] = (
        set(primary["npdes_id"].to_list()) if len(primary) > 0 else set()
    )

    if id_set is not None:
        remaining_ids: Optional[set[str]] = id_set - already_covered
        # If all requested IDs are covered, skip source 2 entirely
        if not remaining_ids:
            fallback = pl.DataFrame(
                {"npdes_id": pl.Series([], dtype=pl.Utf8),
                 "lat_outfall": pl.Series([], dtype=pl.Float64),
                 "lon_outfall": pl.Series([], dtype=pl.Float64)}
            )
        else:
            fallback = _load_perm_feature_coords(remaining_ids, coords_path)
    else:
        # Load all — but pass exclude_ids so rows already in the outfalls layer
        # are skipped inline, avoiding redundant I/O on the 626k-row PFC file.
        fallback = _load_perm_feature_coords(
            None, coords_path, exclude_ids=already_covered or None
        )

    # Deduplicate: outfalls-layer row always wins over PFC fallback row for same ID.
    combined = pl.concat([primary, fallback]).unique(subset=["npdes_id"], keep="first")
    n_total = len(combined)
    n_from_layer  = len(primary)
    n_from_coords = len(fallback)
    log.info(
        f"Outfall coords: {n_total:,} permits total "
        f"({n_from_layer:,} from outfalls layer, "
        f"{n_from_coords:,} from NPDES_PERM_FEATURE_COORDS fallback)"
    )
    return combined

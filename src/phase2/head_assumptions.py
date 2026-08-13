"""Phase 2 — Head assumptions by facility archetype.

Phase 2 cannot measure actual hydraulic head (that is Phase 3's job via USGS
3DEP).  Instead, each facility is classified into one of three archetypes based
on design_flow_mgd, and head is sampled from a triangular distribution derived
from published literature on US gravity-outfall wastewater plants.

Archetype boundaries and triangular parameters are stored in settings.yaml
under ``head_assumptions`` so they can be tuned without code changes.

Reference distributions (Triangular(low, mode, high) in metres):
  large_potw  (>10 MGD)  : Tri(3, 8, 15)
  medium_potw (1–10 MGD) : Tri(2, 5, 10)
  small_potw  (<1 MGD)   : Tri(1, 3, 6)
"""

from __future__ import annotations

from typing import NamedTuple

from src.common import config

# ── Archetype definitions ─────────────────────────────────────────────────────

class HeadDistribution(NamedTuple):
    low_m:    float
    mode_m:   float
    high_m:   float


def _load(key: str) -> HeadDistribution:
    section = config.get(f"head_assumptions.{key}", {})
    return HeadDistribution(
        low_m=float(section.get("low_m",     2.0)),
        mode_m=float(section.get("default_m", 5.0)),
        high_m=float(section.get("high_m",   10.0)),
    )


LARGE_POTW_HEAD:  HeadDistribution = _load("large_potw")
MEDIUM_POTW_HEAD: HeadDistribution = _load("medium_potw")
SMALL_POTW_HEAD:  HeadDistribution = _load("small_potw")

# Design-flow breakpoints (MGD)
_LARGE_THRESHOLD_MGD:  float = 10.0
_SMALL_THRESHOLD_MGD:  float = 1.0


# ── Public API ────────────────────────────────────────────────────────────────

def classify_archetype(design_flow_mgd: float | None) -> str:
    """Return the facility archetype string for a given design flow.

    Args:
        design_flow_mgd: Permitted design flow (MGD).  None or ≤ 0 treated
                         as small_potw (conservative default).

    Returns:
        One of ``"large_potw"``, ``"medium_potw"``, or ``"small_potw"``.
    """
    if design_flow_mgd is None or design_flow_mgd <= 0:
        return "small_potw"
    if design_flow_mgd > _LARGE_THRESHOLD_MGD:
        return "large_potw"
    if design_flow_mgd >= _SMALL_THRESHOLD_MGD:
        return "medium_potw"
    return "small_potw"


def get_head_distribution(archetype: str) -> HeadDistribution:
    """Return the triangular head distribution for an archetype.

    Args:
        archetype: One of the archetype strings returned by
                   :func:`classify_archetype`.

    Returns:
        :class:`HeadDistribution` with ``low_m``, ``mode_m``, ``high_m``.

    Raises:
        KeyError: if ``archetype`` is unrecognised.
    """
    mapping = {
        "large_potw":  LARGE_POTW_HEAD,
        "medium_potw": MEDIUM_POTW_HEAD,
        "small_potw":  SMALL_POTW_HEAD,
    }
    if archetype not in mapping:
        raise KeyError(f"Unknown archetype: {archetype!r}. "
                       f"Valid: {list(mapping)}")
    return mapping[archetype]


def head_params_for_flow(design_flow_mgd: float | None) -> HeadDistribution:
    """Convenience wrapper: classify archetype then return head distribution."""
    return get_head_distribution(classify_archetype(design_flow_mgd))

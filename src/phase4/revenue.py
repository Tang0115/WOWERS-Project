"""Phase 4 — Revenue computation and state electricity rate lookup.

Electricity rates are loaded from ``config/electricity_rates/state_rates.yaml``.
REC (Renewable Energy Certificate) value is stored in ``config/settings.yaml``
under ``financials.rec_value_per_kwh``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.common import config, logging_setup

log = logging_setup.get("wowers.phase4.revenue")

_RATES_FILE: Path = config.project_root() / "config" / "electricity_rates" / "state_rates.yaml"
_REC_PER_KWH: float = float(config.get("financials.rec_value_per_kwh", 0.01))


@lru_cache(maxsize=1)
def _load_rates() -> dict[str, float]:
    """Load the state electricity rate YAML (cached after first call)."""
    if not _RATES_FILE.exists():
        log.warning(
            f"State rates file not found: {_RATES_FILE}. "
            "Using national average $0.081/kWh for all states."
        )
        return {"national_avg": 0.081}
    with _RATES_FILE.open() as f:
        data = yaml.safe_load(f)
    states: dict[str, float] = {k: float(v) for k, v in data.get("states", {}).items()}
    states["national_avg"] = float(data.get("national_avg", 0.081))
    return states


def electricity_rate(state_code: str | None) -> float:
    """Return the 2023 industrial electricity rate for a state ($/kWh).

    Args:
        state_code: Two-letter US state abbreviation (e.g. ``"MN"``).
                    None or unrecognised codes fall back to national average.

    Returns:
        Electricity rate in $/kWh.
    """
    rates = _load_rates()
    if state_code and state_code.upper() in rates:
        return rates[state_code.upper()]
    return rates.get("national_avg", 0.081)


def annual_revenue(
    annual_energy_kwh: float,
    state_code:        str | None,
    include_rec:       bool = True,
) -> float:
    """Year-1 gross revenue from electricity sales plus optional RECs.

    Args:
        annual_energy_kwh: Annual generation (kWh/yr).
        state_code:        Two-letter state code for rate lookup.
        include_rec:       Whether to add the REC value to the electricity rate.

    Returns:
        Annual revenue in USD/yr.
    """
    rate = electricity_rate(state_code)
    if include_rec:
        rate += _REC_PER_KWH
    return annual_energy_kwh * rate

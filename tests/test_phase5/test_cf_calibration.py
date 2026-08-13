"""P5-CF-CALIB — unit tests for capacity-factor calibration pure functions.

All tests use small synthetic polars frames built in-memory.
No drive, no Excel files, no parquets required.

Coverage:
  - parse_cf_pct: '%' string → fraction; nulls; edge values
  - recompute_cf: Net_Generation_MWh / (Capacity_MW × Hours) formula
  - filter_clean_hy: HY filter, CF bounds, Capacity_MW floor
  - bucket_stats: correct p-tiles, plant/plant-year counts, year filter
  - calibration_band: multiplier math, monotonicity, 5-row output
  - conduit_cf_stats: metered conduit CF, municipal subset, scale-trend reporting
  - phase2_viable_cf_stats: join + headline GWh computation
  - recompute_validation: |diff| statistics

One real-drive integration test skips when SANDISK/fastexcel unavailable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl
import pytest

# Import the pure functions from the script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cf_calibration import (
    parse_cf_pct,
    recompute_cf,
    filter_clean_hy,
    bucket_stats,
    calibration_band,
    conduit_cf_stats,
    phase2_viable_cf_stats,
    recompute_validation,
    _DEFAULT_EHA_DIR,
    _EHA_CF_FILE,
    CF_DROP_ABOVE,
    CF_DROP_BELOW_EQ,
    WWTP_CENTRAL_CF,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cf_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal EHA CF frame from a list of row dicts."""
    return pl.DataFrame(rows)


def _base_row(
    eha_id: str = "hc001",
    plant: str = "Alpha Dam",
    state: str = "CA",
    type_: str = "HY",
    year: int = 2020,
    cap_mw: float = 1.0,
    net_gen_mwh: float = 4_380.0,   # → CF 0.5 at 8760 h
    hours: int = 8_760,
    cf_str: str = "50%",
) -> dict:
    return {
        "EHA_PtID": eha_id,
        "PlantName": plant,
        "State": state,
        "Type": type_,
        "Year": year,
        "Capacity_MW": cap_mw,
        "Net_Generation_MWh": net_gen_mwh,
        "Hours": hours,
        "Capacity_Factor": cf_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# parse_cf_pct
# ══════════════════════════════════════════════════════════════════════════════

class TestParseCfPct:

    def test_integer_pct(self):
        s = pl.Series(["22%", "50%", "100%"])
        out = parse_cf_pct(s)
        assert out[0] == pytest.approx(0.22)
        assert out[1] == pytest.approx(0.50)
        assert out[2] == pytest.approx(1.00)

    def test_decimal_pct(self):
        s = pl.Series(["55.74%", "27.03%"])
        out = parse_cf_pct(s)
        assert out[0] == pytest.approx(0.5574, rel=1e-3)
        assert out[1] == pytest.approx(0.2703, rel=1e-3)

    def test_null_for_unparseable(self):
        s = pl.Series(["N/A", "", "abc"])
        out = parse_cf_pct(s)
        assert out.is_null().all()

    def test_zero_pct(self):
        s = pl.Series(["0%"])
        out = parse_cf_pct(s)
        assert out[0] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# recompute_cf
# ══════════════════════════════════════════════════════════════════════════════

class TestRecomputeCf:

    def test_basic_formula(self):
        df = _make_cf_frame([_base_row(cap_mw=2.0, net_gen_mwh=8_760.0, hours=8_760)])
        out = recompute_cf(df)
        # CF = 8760 / (2 * 8760) = 0.5
        assert out["cf_calc"][0] == pytest.approx(0.5)

    def test_mwh_not_kwh(self):
        # 200 kW = 0.2 MW; 1100 MWh/yr → CF=0.628
        df = _make_cf_frame([_base_row(cap_mw=0.2, net_gen_mwh=1_100.0, hours=8_760)])
        out = recompute_cf(df)
        expected = 1_100.0 / (0.2 * 8_760)
        assert out["cf_calc"][0] == pytest.approx(expected, rel=1e-5)

    def test_cf_given_parsed(self):
        df = _make_cf_frame([_base_row(cf_str="50%")])
        out = recompute_cf(df)
        assert out["cf_given"][0] == pytest.approx(0.50)

    def test_columns_added(self):
        df = _make_cf_frame([_base_row()])
        out = recompute_cf(df)
        assert "cf_calc" in out.columns
        assert "cf_given" in out.columns

    def test_recompute_matches_given_within_rounding(self):
        # 22%: Net=1927.12, Cap=1.0, Hours=8760 → CF≈0.22004
        df = _make_cf_frame([_base_row(cap_mw=1.0, net_gen_mwh=1_927.12, hours=8_760, cf_str="22%")])
        out = recompute_cf(df)
        diff = abs(out["cf_calc"][0] - out["cf_given"][0])
        assert diff < 0.01   # rounding only


# ══════════════════════════════════════════════════════════════════════════════
# filter_clean_hy
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterCleanHy:

    def _prepped(self, rows: list[dict]) -> pl.DataFrame:
        return recompute_cf(_make_cf_frame(rows))

    def test_keeps_valid_hy_rows(self):
        df = self._prepped([_base_row(type_="HY")])
        out = filter_clean_hy(df)
        assert out.height == 1

    def test_drops_non_hy(self):
        df = self._prepped([_base_row(type_="PS")])
        out = filter_clean_hy(df)
        assert out.height == 0

    def test_drops_cf_zero(self):
        df = self._prepped([_base_row(net_gen_mwh=0.0)])
        out = filter_clean_hy(df)
        assert out.height == 0

    def test_drops_cf_above_threshold(self):
        # CF > 1.2: 15000 MWh / (1 MW * 8760 h) = 1.712
        df = self._prepped([_base_row(net_gen_mwh=15_000.0)])
        out = filter_clean_hy(df)
        assert out.height == 0

    def test_keeps_cf_exactly_at_upper_limit(self):
        # CF = 1.2 exactly (boundary inclusive)
        # 1.2 * 1.0 * 8760 = 10512 MWh
        df = self._prepped([_base_row(net_gen_mwh=10_512.0)])
        out = filter_clean_hy(df)
        assert out.height == 1

    def test_drops_capacity_below_floor(self):
        df = self._prepped([_base_row(cap_mw=0.05)])  # < 0.1 MW
        out = filter_clean_hy(df)
        assert out.height == 0

    def test_keeps_capacity_at_floor(self):
        # cap_mw=0.1, net_gen=438 MWh → CF = 438/(0.1*8760) = 0.5; valid
        df = self._prepped([_base_row(cap_mw=0.1, net_gen_mwh=438.0)])
        out = filter_clean_hy(df)
        assert out.height == 1

    def test_drops_negative_cf(self):
        df = self._prepped([_base_row(net_gen_mwh=-100.0)])
        out = filter_clean_hy(df)
        assert out.height == 0


# ══════════════════════════════════════════════════════════════════════════════
# bucket_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestBucketStats:

    def _clean_frame(self) -> pl.DataFrame:
        """3 plants: 0.5 MW, 1.0 MW, 5.0 MW; 2 years each."""
        rows = [
            _base_row("p1", cap_mw=0.5,  net_gen_mwh=2190.0, year=2020),  # CF=0.5
            _base_row("p1", cap_mw=0.5,  net_gen_mwh=2190.0, year=2021),
            _base_row("p2", cap_mw=1.0,  net_gen_mwh=3504.0, year=2020),  # CF=0.4
            _base_row("p2", cap_mw=1.0,  net_gen_mwh=3504.0, year=2021),
            _base_row("p3", cap_mw=5.0,  net_gen_mwh=8760.0, year=2020),  # CF=0.2
            _base_row("p3", cap_mw=5.0,  net_gen_mwh=8760.0, year=2021),
        ]
        df = recompute_cf(_make_cf_frame(rows))
        return filter_clean_hy(df)

    def test_plant_count(self):
        stats = bucket_stats(self._clean_frame(), max_mw=5.0)
        assert stats["n_plants"] == 3

    def test_plant_years(self):
        stats = bucket_stats(self._clean_frame(), max_mw=5.0)
        assert stats["n_plant_years"] == 6

    def test_max_mw_filter(self):
        stats = bucket_stats(self._clean_frame(), max_mw=1.0)
        assert stats["n_plants"] == 2   # 0.5 and 1.0 only

    def test_year_filter(self):
        stats = bucket_stats(self._clean_frame(), max_mw=5.0, year_min=2021)
        assert stats["n_plant_years"] == 3  # one year each for 3 plants

    def test_percentile_order(self):
        stats = bucket_stats(self._clean_frame(), max_mw=5.0)
        assert stats["p10"] <= stats["p25"] <= stats["p50"]
        assert stats["p50"] <= stats["p75"] <= stats["p90"]


# ══════════════════════════════════════════════════════════════════════════════
# calibration_band
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationBand:

    def _empirical(self) -> dict:
        return {"p25": 0.254, "p50": 0.390, "p75": 0.541}

    def test_returns_5_rows(self):
        band = calibration_band(409.1, 0.872, self._empirical(), WWTP_CENTRAL_CF)
        assert len(band) == 5

    def test_multiplier_formula(self):
        band = calibration_band(100.0, 0.80, {"p25": 0.40, "p50": 0.50, "p75": 0.60}, 0.60)
        # p25 row: mult = 0.40/0.80 = 0.5; gwh = 50.0
        p25_row = [r for r in band if "p25" in r["tier"]][0]
        assert p25_row["multiplier"] == pytest.approx(0.5, rel=1e-3)
        assert p25_row["gwh"] == pytest.approx(50.0, rel=1e-3)

    def test_monotonicity_floor_le_optimistic_le_ceiling(self):
        band = calibration_band(409.1, 0.872, self._empirical(), WWTP_CENTRAL_CF)
        gwh_by_tier = {r["tier"]: r["gwh"] for r in band}
        floor_min  = gwh_by_tier["Floor (river-hydro p25)"]
        floor_med  = gwh_by_tier["Floor (river-hydro p50)"]
        floor_max  = gwh_by_tier["Floor (river-hydro p75)"]
        # Tier relabelled 2026-07-25: "Central (WWTP-appropriate)" → "Optimistic
        # (Conduit 3 proj.)". The CF value is unchanged at 0.60; only the label moved,
        # because the tier rests on a design projection rather than a measurement.
        optimistic = gwh_by_tier["Optimistic (Conduit 3 proj.)"]
        ceiling    = gwh_by_tier["Physics ceiling (Phase 2)"]
        assert floor_min <= floor_med <= floor_max <= optimistic <= ceiling

    def test_physics_ceiling_equals_headline(self):
        band = calibration_band(400.0, 0.80, self._empirical(), 0.60)
        ceiling = [r for r in band if "ceiling" in r["tier"].lower()][0]
        assert ceiling["gwh"] == pytest.approx(400.0, rel=1e-4)

    def test_optimistic_above_floor_p50(self):
        band = calibration_band(409.1, 0.872, self._empirical(), WWTP_CENTRAL_CF)
        p50_gwh    = [r["gwh"] for r in band if "p50" in r["tier"]][0]
        optimistic = [r["gwh"] for r in band if "Optimistic" in r["tier"]][0]
        assert optimistic > p50_gwh


# ══════════════════════════════════════════════════════════════════════════════
# conduit_cf_stats  (added 2026-07-25 with the measured-conduit calibration tier)
# ══════════════════════════════════════════════════════════════════════════════

class TestConduitCfStats:

    def _frame(self) -> pl.DataFrame:
        """Four plants: two municipal, two canal. CF = energy / (kw × 8760).

        Capacities differ so the scale-trend correlation is defined; the frame
        deliberately omits the descriptive columns (state, source_year,
        water_source) to prove the function does not require them.
        """
        return pl.DataFrame([
            # 1000 kW × 8760 = 8,760,000 → CF 0.10
            {"site_name": "Muni A", "capacity_kw": 1000.0,
             "annual_energy_kwh": 876_000.0, "is_wwtp_type": True},
            # 2000 kW → CF 0.20
            {"site_name": "Muni B", "capacity_kw": 2000.0,
             "annual_energy_kwh": 3_504_000.0, "is_wwtp_type": True},
            # 4000 kW → CF 0.30
            {"site_name": "Canal A", "capacity_kw": 4000.0,
             "annual_energy_kwh": 10_512_000.0, "is_wwtp_type": False},
            # 8000 kW → CF 0.40
            {"site_name": "Canal B", "capacity_kw": 8000.0,
             "annual_energy_kwh": 28_032_000.0, "is_wwtp_type": False},
        ])

    def test_cf_formula_and_counts(self):
        s = conduit_cf_stats(self._frame())
        assert s["all"]["n"] == 4
        assert s["municipal"]["n"] == 2
        # median of 0.10/0.20/0.30/0.40 lands between the two middle values
        assert 0.20 <= s["all"]["p50"] <= 0.30

    def test_municipal_subset_is_the_wwtp_rows_only(self):
        s = conduit_cf_stats(self._frame())
        # municipal rows are CF 0.10 and 0.20, so the subset mean is 0.15
        assert s["municipal"]["mean"] == pytest.approx(0.15, rel=1e-6)

    def test_drops_out_of_range_cf(self):
        """Rows outside (0, CF_DROP_ABOVE] are dropped, matching filter_clean_hy."""
        bad = pl.DataFrame([
            {"site_name": "Zero", "capacity_kw": 1000.0,
             "annual_energy_kwh": 0.0, "is_wwtp_type": False},
            {"site_name": "Impossible", "capacity_kw": 1000.0,
             "annual_energy_kwh": 99_000_000.0, "is_wwtp_type": False},
            {"site_name": "Good", "capacity_kw": 1000.0,
             "annual_energy_kwh": 1_752_000.0, "is_wwtp_type": False},
        ])
        s = conduit_cf_stats(bad)
        assert s["all"]["n"] == 1
        # none of those rows are municipal, so that subset is empty; polars returns
        # None from quantile() on an empty series and must not reach float()
        assert s["municipal"]["n"] == 0
        assert s["municipal"]["p50"] is None

    def test_reports_scale_trend_and_quartiles(self):
        """The scale test must be reported, since it is what rules out extrapolating
        the 0.60 tier down to WOWERS scale."""
        s = conduit_cf_stats(self._frame())
        assert -1.0 <= s["corr_logkw_cf"] <= 1.0
        assert len(s["quartiles"]) == 4
        assert sum(q["n"] for q in s["quartiles"]) >= s["all"]["n"]

    def test_correlation_is_none_when_undefined(self):
        """Constant capacity leaves the correlation undefined. It must come back as
        None rather than NaN, because the caller prints a caveat off the back of it
        and a NaN comparison would silently suppress that caveat."""
        flat = pl.DataFrame([
            {"site_name": "A", "capacity_kw": 1000.0,
             "annual_energy_kwh": 876_000.0, "is_wwtp_type": False},
            {"site_name": "B", "capacity_kw": 1000.0,
             "annual_energy_kwh": 1_752_000.0, "is_wwtp_type": False},
            {"site_name": "C", "capacity_kw": 1000.0,
             "annual_energy_kwh": 2_628_000.0, "is_wwtp_type": False},
        ])
        assert conduit_cf_stats(flat)["corr_logkw_cf"] is None

    def test_muni_rows_sorted_ascending_by_cf(self):
        s = conduit_cf_stats(self._frame())
        cfs = s["muni_rows"]["cf"].to_list()
        assert cfs == sorted(cfs)


# ══════════════════════════════════════════════════════════════════════════════
# phase2_viable_cf_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2ViableCfStats:

    def _make_p2(self) -> pl.DataFrame:
        return pl.DataFrame({
            "npdes_id":            ["A", "B", "C"],
            "capacity_factor_p50": [0.85, 0.90, 0.80],
            "energy_p50_kwh_yr":   [1e6, 2e6, 1.5e6],
        })

    def _make_p4_viable(self) -> pl.DataFrame:
        # 3 viable sites so quantile(0.5) on odd-length series is unambiguous
        return pl.DataFrame({
            "npdes_id":          ["A", "B", "C"],
            "project_viable":    [True, True, True],
            "annual_energy_kwh": [1e6, 2e6, 1.5e6],
        })

    def test_viable_count(self):
        stats = phase2_viable_cf_stats(self._make_p2(), self._make_p4_viable())
        assert stats["n_viable_sites"] == 3

    def test_headline_gwh(self):
        stats = phase2_viable_cf_stats(self._make_p2(), self._make_p4_viable())
        # A + B + C: 1e6 + 2e6 + 1.5e6 = 4.5e6 kWh = 4.5 GWh
        assert stats["headline_gwh"] == pytest.approx(4.5)

    def test_cf_median(self):
        stats = phase2_viable_cf_stats(self._make_p2(), self._make_p4_viable())
        # sorted: [0.80, 0.85, 0.90] → median = 0.85 (middle element)
        assert stats["cf_p50"] == pytest.approx(0.85)

    def test_non_viable_excluded(self):
        p4 = pl.DataFrame({
            "npdes_id":          ["A", "B", "C"],
            "project_viable":    [True, False, False],
            "annual_energy_kwh": [1e6, 2e6, 1.5e6],
        })
        stats = phase2_viable_cf_stats(self._make_p2(), p4)
        assert stats["n_viable_sites"] == 1
        assert stats["headline_gwh"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# recompute_validation
# ══════════════════════════════════════════════════════════════════════════════

class TestRecomputeValidation:

    def test_zero_diff_when_identical(self):
        rows = [
            _base_row(net_gen_mwh=4_380.0, cap_mw=1.0, hours=8_760, cf_str="50%"),
        ]
        df = filter_clean_hy(recompute_cf(_make_cf_frame(rows)))
        val = recompute_validation(df)
        # cf_calc = 4380/8760 = 0.5, cf_given = 0.50 → diff = 0
        assert val["mean"] < 0.001

    def test_nonzero_diff_with_rounding(self):
        # cf_str "22%" → 0.22; actual = 22.1% → diff ≈ 0.001
        net = 0.221 * 1.0 * 8_760   # 1936.76 MWh
        rows = [_base_row(net_gen_mwh=net, cap_mw=1.0, hours=8_760, cf_str="22%")]
        df = filter_clean_hy(recompute_cf(_make_cf_frame(rows)))
        val = recompute_validation(df)
        assert val["n"] >= 1
        assert val["mean"] >= 0.0

    def test_returns_expected_keys(self):
        rows = [_base_row()]
        df = filter_clean_hy(recompute_cf(_make_cf_frame(rows)))
        val = recompute_validation(df)
        assert set(val.keys()) == {"n", "mean", "p25", "p75"}

    def test_all_finite(self):
        rows = [_base_row(), _base_row("p2", cap_mw=2.0, net_gen_mwh=5256.0)]
        df = filter_clean_hy(recompute_cf(_make_cf_frame(rows)))
        val = recompute_validation(df)
        for v in val.values():
            assert math.isfinite(v)


# ══════════════════════════════════════════════════════════════════════════════
# Real-drive integration (skips when drive or fastexcel unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _eha_cf_available() -> bool:
    try:
        import fastexcel  # noqa: F401
    except ImportError:
        return False
    return (_DEFAULT_EHA_DIR / _EHA_CF_FILE).exists()


@pytest.mark.skipif(
    not _eha_cf_available(),
    reason="EHA CF drive or fastexcel engine not available",
)
def test_integration_real_eha_cf():
    """Spot-check real EHA data against expected known values."""
    from cf_calibration import _load_eha_cf

    df = _load_eha_cf(_DEFAULT_EHA_DIR)

    # Basic sanity
    assert df.height > 20_000, "Expected >20k clean HY rows"
    assert (df["cf_calc"] > 0).all()
    assert (df["cf_calc"] <= CF_DROP_ABOVE).all()

    # 0.1–5 MW bucket
    stats = bucket_stats(df, max_mw=5.0)
    assert stats["n_plants"] >= 600
    assert stats["n_plant_years"] >= 9_000
    # p50 should be in the 0.35–0.45 range
    assert 0.35 <= stats["p50"] <= 0.45, f"0.1-5 MW p50 CF = {stats['p50']}"

    # Recompute vs given diff should be small (rounding only)
    val = recompute_validation(df)
    assert val["mean"] < 0.005, f"Recompute diff too large: {val['mean']}"

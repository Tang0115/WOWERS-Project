"""Phase 5 ground-truth ingest — unit tests on synthetic frames.

No external drive or Excel engine required: every test feeds standardized
in-memory polars frames to the pure transforms. One optional integration test
reads the real EIA file and skips when the drive/engine is absent.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from src.phase5 import ground_truth as gt


# ── Fixtures: standardized frames as produced by the IO layer ─────────────────
def _generators() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "plant_code": [1, 1, 2, 3, 4],
            "prime_mover": ["HY", "HY", "HY", "PS", "ST"],  # plant1 has 2 hydro gens
            "nameplate_mw": [10.0, 5.0, 2.0, 50.0, 100.0],
        }
    )


def _generation() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "plant_code": [1, 2, 3, 4, 1],
            "prime_mover": ["HY", "HY", "PS", "ST", "HY"],  # plant1 hydro on 2 rows
            "net_gen_mwh": [40_000.0, 8_000.0, -500.0, 250_000.0, 5_000.0],
        }
    )


def _plants() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "plant_code": [1, 2, 3, 4],
            "plant_name": ["Alpha Dam", "Beta Conduit", "Gamma PS", "Delta Coal"],
            "state_code": ["CA", "CO", "WA", "TX"],
            "latitude": [38.1, 39.2, 47.3, 31.4],
            "longitude": [-120.1, -105.2, -121.3, -97.4],
        }
    )


# ── prime-mover filter + capacity aggregation ─────────────────────────────────
class TestAggregateCapacity:
    def test_keeps_only_hydro_and_sums_to_plant_kw(self):
        out = gt.aggregate_capacity_kw(_generators()).sort("plant_code")
        # PS (plant3) and ST (plant4) dropped; plant1 = (10+5)*1000.
        assert out["plant_code"].to_list() == [1, 2]
        cap = dict(zip(out["plant_code"], out["actual_installed_kw"]))
        assert cap[1] == pytest.approx(15_000.0)
        assert cap[2] == pytest.approx(2_000.0)

    def test_pumped_storage_excluded(self):
        df = pl.DataFrame({"plant_code": [9], "prime_mover": ["PS"], "nameplate_mw": [30.0]})
        assert gt.aggregate_capacity_kw(df).height == 0


# ── generation aggregation ────────────────────────────────────────────────────
class TestAggregateGeneration:
    def test_keeps_only_hydro_and_sums_to_plant_kwh(self):
        out = gt.aggregate_generation_kwh(_generation()).sort("plant_code")
        assert out["plant_code"].to_list() == [1, 2]
        en = dict(zip(out["plant_code"], out["actual_annual_energy_kwh"]))
        assert en[1] == pytest.approx((40_000.0 + 5_000.0) * 1_000.0)  # two hydro rows
        assert en[2] == pytest.approx(8_000.0 * 1_000.0)

    def test_mwh_to_kwh_conversion(self):
        df = pl.DataFrame({"plant_code": [1], "prime_mover": ["HY"], "net_gen_mwh": [1.0]})
        out = gt.aggregate_generation_kwh(df)
        assert out["actual_annual_energy_kwh"][0] == pytest.approx(1_000.0)


# ── full assemble ─────────────────────────────────────────────────────────────
class TestAssemble:
    def test_schema_columns_and_dtypes(self):
        out = gt.assemble_ground_truth(_generators(), _generation(), _plants(), year=2019)
        assert out.columns == list(gt.CANONICAL_SCHEMA.keys())
        assert dict(out.schema) == dict(gt.CANONICAL_SCHEMA)

    def test_only_hydro_plants_with_both_labels_survive(self):
        out = gt.assemble_ground_truth(_generators(), _generation(), _plants(), year=2019).sort(
            "source_plant_code"
        )
        # Only plants 1 & 2 have both hydro capacity and hydro generation.
        assert out["source_plant_code"].to_list() == [1, 2]

    def test_label_values_and_metadata(self):
        out = gt.assemble_ground_truth(_generators(), _generation(), _plants(), year=2019).sort(
            "source_plant_code"
        )
        r0 = out.row(0, named=True)
        assert r0["actual_installed_kw"] == pytest.approx(15_000.0)
        assert r0["actual_annual_energy_kwh"] == pytest.approx(45_000_000.0)
        assert r0["ground_truth_source"] == "EIA"
        assert r0["facility_name"] == "Alpha Dam"
        assert r0["state_code"] == "CA"
        assert r0["latitude"] == pytest.approx(38.1)
        assert r0["source_year"] == 2019
        # EIA never reports head/flow.
        assert r0["actual_head_m"] is None
        assert r0["actual_flow_m3s"] is None

    def test_capacity_without_generation_dropped(self):
        # plant 5 has hydro capacity but no generation row → inner join drops it.
        gens = pl.DataFrame(
            {"plant_code": [5], "prime_mover": ["HY"], "nameplate_mw": [3.0]}
        )
        gens_no = pl.DataFrame(
            {"plant_code": [99], "prime_mover": ["HY"], "net_gen_mwh": [1.0]}
        )
        plants = pl.DataFrame(
            {
                "plant_code": [5],
                "plant_name": ["Orphan"],
                "state_code": ["NV"],
                "latitude": [1.0],
                "longitude": [2.0],
            }
        )
        out = gt.assemble_ground_truth(gens, gens_no, plants, year=2019)
        assert out.height == 0

    def test_zero_and_negative_labels_dropped(self):
        gens = pl.DataFrame(
            {"plant_code": [1, 2], "prime_mover": ["HY", "HY"], "nameplate_mw": [0.0, 5.0]}
        )
        gen = pl.DataFrame(
            {"plant_code": [1, 2], "prime_mover": ["HY", "HY"], "net_gen_mwh": [100.0, -10.0]}
        )
        plants = pl.DataFrame(
            {
                "plant_code": [1, 2],
                "plant_name": ["Z", "N"],
                "state_code": ["CA", "CA"],
                "latitude": [1.0, 2.0],
                "longitude": [1.0, 2.0],
            }
        )
        # plant1 zero capacity, plant2 negative generation → both dropped.
        out = gt.assemble_ground_truth(gens, gen, plants, year=2019)
        assert out.height == 0


# ── column-name matcher ───────────────────────────────────────────────────────
class TestFindCol:
    def test_matches_all_substrings_case_insensitive(self):
        cols = ["Plant Code", "Nameplate Capacity (MW)", "Reported\nPrime Mover"]
        assert gt._find_col(cols, "nameplate", "capacity") == "Nameplate Capacity (MW)"
        assert gt._find_col(cols, "prime", "mover") == "Reported\nPrime Mover"

    def test_raises_when_absent(self):
        with pytest.raises(KeyError):
            gt._find_col(["A", "B"], "nonexistent")


# ── optional real-file integration (skips without drive/engine) ───────────────
def _eia_available() -> bool:
    try:
        import fastexcel  # noqa: F401
    except ImportError:
        return False
    try:
        return gt._resolve_eia_dir().is_dir()
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _eia_available(), reason="EIA drive or fastexcel engine not available")
def test_integration_real_eia_ingest():
    df = gt.ingest_eia_year()
    assert df.columns == list(gt.CANONICAL_SCHEMA.keys())
    assert df.height > 0
    assert (df["actual_installed_kw"] > 0).all()
    assert (df["actual_annual_energy_kwh"] > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# P5-EHA — unit tests for EHA (DOE HydroSource) ingest
# ══════════════════════════════════════════════════════════════════════════════

# ── Synthetic standardised frames (as produced by _load_eha_plants / _load_eha_cf)

def _eha_plants() -> pl.DataFrame:
    """Minimal plant frame with conventional + pumped-storage cases."""
    return pl.DataFrame(
        {
            "eha_pt_id":   ["hc001_p01", "hc002_p01", "hc003_p01", "hc004_p01"],
            "eia_pt_id":   [1001,         1002,         1003,         None],    # one null EIA id
            "pt_name":     ["Alpha Hydro", "Beta Canal", "Gamma PS-Only", "Delta No-Cap"],
            "state_code":  ["CA",          "CO",          "WA",            "OR"],
            "latitude":    [38.1,          39.2,          47.3,            44.5],
            "longitude":   [-120.1,        -105.2,        -121.3,          -122.1],
            "ch_mw":       [10.0,          2.5,           None,            0.0],  # ps-only=None, dead=0
        }
    )


def _eha_cf() -> pl.DataFrame:
    """Multi-year CF frame: two years, conventional-hydro type only."""
    return pl.DataFrame(
        {
            "eha_pt_id":   ["hc001_p01", "hc001_p01", "hc002_p01", "hc002_p01", "hc005_p01"],
            "type":        ["HY",         "HY",         "HY",         "HY",         "HY"],
            "year":        [2021,          2022,          2021,          2022,          2022],
            "net_gen_mwh": [40_000.0,      45_000.0,      8_000.0,       9_000.0,       5_000.0],
            "capacity_mw": [10.0,          10.0,          2.5,           2.5,           1.0],
        }
    )


# ── eha_filter_capacity ───────────────────────────────────────────────────────

class TestEhaFilterCapacity:
    def test_drops_null_ch_mw(self):
        out = gt.eha_filter_capacity(_eha_plants())
        assert "hc003_p01" not in out["eha_pt_id"].to_list()

    def test_drops_zero_ch_mw(self):
        out = gt.eha_filter_capacity(_eha_plants())
        assert "hc004_p01" not in out["eha_pt_id"].to_list()

    def test_keeps_positive_ch_mw(self):
        out = gt.eha_filter_capacity(_eha_plants())
        assert set(out["eha_pt_id"].to_list()) == {"hc001_p01", "hc002_p01"}

    def test_pumped_storage_only_excluded(self):
        # plant with ch_mw=None is pumped-storage-only → excluded
        ps_only = pl.DataFrame(
            {"eha_pt_id": ["ps1"], "eia_pt_id": [99], "pt_name": ["PS Plant"],
             "state_code": ["WA"], "latitude": [47.0], "longitude": [-120.0], "ch_mw": [None]}
        )
        assert gt.eha_filter_capacity(ps_only).height == 0


# ── eha_capacity_kw ───────────────────────────────────────────────────────────

class TestEhaCapacityKw:
    def test_mw_to_kw_conversion(self):
        out = gt.eha_capacity_kw(_eha_plants())
        cap = {r["eha_pt_id"]: r["actual_installed_kw"] for r in out.iter_rows(named=True)}
        assert cap["hc001_p01"] == pytest.approx(10_000.0)   # 10 MW × 1000
        assert cap["hc002_p01"] == pytest.approx(2_500.0)    # 2.5 MW × 1000

    def test_ch_mw_column_removed(self):
        out = gt.eha_capacity_kw(_eha_plants())
        assert "ch_mw" not in out.columns

    def test_metadata_columns_preserved(self):
        out = gt.eha_capacity_kw(_eha_plants())
        for col in ("eha_pt_id", "eia_pt_id", "pt_name", "state_code", "latitude", "longitude"):
            assert col in out.columns


# ── eha_energy_kwh ────────────────────────────────────────────────────────────

class TestEhaEnergyKwh:
    def test_year_filter(self):
        # Only 2022 rows should appear
        out = gt.eha_energy_kwh(_eha_cf(), year=2022)
        # hc001_p01 2022 only: 45_000 MWh → 45_000_000 kWh
        assert "hc001_p01" in out["eha_pt_id"].to_list()
        row = out.filter(pl.col("eha_pt_id") == "hc001_p01").row(0, named=True)
        assert row["actual_annual_energy_kwh"] == pytest.approx(45_000_000.0)

    def test_year_2021(self):
        out = gt.eha_energy_kwh(_eha_cf(), year=2021)
        row = out.filter(pl.col("eha_pt_id") == "hc001_p01").row(0, named=True)
        assert row["actual_annual_energy_kwh"] == pytest.approx(40_000_000.0)

    def test_mwh_to_kwh_conversion(self):
        cf = pl.DataFrame(
            {"eha_pt_id": ["x"], "type": ["HY"], "year": [2022],
             "net_gen_mwh": [1.0], "capacity_mw": [0.1]}
        )
        out = gt.eha_energy_kwh(cf, year=2022)
        assert out["actual_annual_energy_kwh"][0] == pytest.approx(1_000.0)

    def test_type_filter_excludes_non_hy(self):
        cf = pl.DataFrame(
            {"eha_pt_id": ["p", "q"], "type": ["PS", "ST"], "year": [2022, 2022],
             "net_gen_mwh": [999.0, 888.0], "capacity_mw": [10.0, 5.0]}
        )
        out = gt.eha_energy_kwh(cf, year=2022)
        assert out.height == 0

    def test_plants_not_in_year_absent(self):
        # hc005_p01 only has 2022 data
        out2021 = gt.eha_energy_kwh(_eha_cf(), year=2021)
        assert "hc005_p01" not in out2021["eha_pt_id"].to_list()


# ── assemble_eha_ground_truth ─────────────────────────────────────────────────

class TestAssembleEha:
    def test_schema_columns_and_dtypes(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022)
        assert out.columns == list(gt.CANONICAL_SCHEMA.keys())
        assert dict(out.schema) == dict(gt.CANONICAL_SCHEMA)

    def test_ground_truth_source_is_eha(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022)
        assert (out["ground_truth_source"] == "EHA").all()

    def test_head_and_flow_are_null(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022)
        assert out["actual_head_m"].is_null().all()
        assert out["actual_flow_m3s"].is_null().all()

    def test_only_plants_with_both_labels_survive(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022).sort("source_plant_code")
        # hc001 + hc002 both in plant file (CH_MW>0) and CF 2022 → survive
        # hc003 (null CH_MW), hc004 (0 CH_MW), hc005 (no plant record) → dropped
        assert out.height == 2
        names = set(out["facility_name"].to_list())
        assert names == {"Alpha Hydro", "Beta Canal"}

    def test_label_values_correct(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022).sort("source_plant_code")
        r0 = out.row(0, named=True)  # Alpha Hydro (EIA_PtID=1001)
        assert r0["actual_installed_kw"] == pytest.approx(10_000.0)
        assert r0["actual_annual_energy_kwh"] == pytest.approx(45_000_000.0)
        assert r0["state_code"] == "CA"
        assert r0["latitude"] == pytest.approx(38.1)
        assert r0["source_year"] == 2022

    def test_source_plant_code_is_eia_pt_id(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022).sort("source_plant_code")
        codes = out["source_plant_code"].to_list()
        assert codes[0] == 1001  # Alpha Hydro EIA_PtID
        assert codes[1] == 1002  # Beta Canal EIA_PtID

    def test_zero_energy_rows_dropped(self):
        # Net_gen = 0 → energy = 0 → dropped by > 0 filter
        cf_zero = pl.DataFrame(
            {"eha_pt_id": ["hc001_p01"], "type": ["HY"], "year": [2022],
             "net_gen_mwh": [0.0], "capacity_mw": [10.0]}
        )
        out = gt.assemble_eha_ground_truth(_eha_plants(), cf_zero, year=2022)
        assert "hc001_p01" not in (out["facility_name"].to_list()
                                   if out.height > 0 else [])

    def test_negative_energy_rows_dropped(self):
        cf_neg = pl.DataFrame(
            {"eha_pt_id": ["hc001_p01"], "type": ["HY"], "year": [2022],
             "net_gen_mwh": [-100.0], "capacity_mw": [10.0]}
        )
        out = gt.assemble_eha_ground_truth(_eha_plants(), cf_neg, year=2022)
        assert out.height == 0

    def test_eha_pt_id_not_in_canonical_schema(self):
        out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022)
        assert "eha_pt_id" not in out.columns

    def test_plant_absent_from_cf_dropped(self):
        # hc001_p01 has no 2021 row in our fixture → dropped for year=2021
        # (hc001 only has 2021 and 2022; hc002 only has 2021 and 2022)
        # hc005_p01 has 2022 CF but is NOT in the plant file → inner join drops it
        cf_2022_only_hc005 = pl.DataFrame(
            {"eha_pt_id": ["hc005_p01"], "type": ["HY"], "year": [2022],
             "net_gen_mwh": [5000.0], "capacity_mw": [1.0]}
        )
        out = gt.assemble_eha_ground_truth(_eha_plants(), cf_2022_only_hc005, year=2022)
        assert out.height == 0  # hc005 not in plant file → inner join → 0 rows

    def test_schema_identical_to_eia_schema(self):
        # EHA and EIA must produce the EXACT same schema (enables combine_ground_truth)
        eha_out = gt.assemble_eha_ground_truth(_eha_plants(), _eha_cf(), year=2022)
        eia_out = gt.assemble_ground_truth(
            pl.DataFrame({"plant_code": [1], "prime_mover": ["HY"], "nameplate_mw": [10.0]}),
            pl.DataFrame({"plant_code": [1], "prime_mover": ["HY"], "net_gen_mwh": [45_000.0]}),
            pl.DataFrame({"plant_code": [1], "plant_name": ["EIA Plant"], "state_code": ["CA"],
                          "latitude": [38.0], "longitude": [-120.0]}),
            year=2022,
        )
        assert eha_out.columns == eia_out.columns
        assert dict(eha_out.schema) == dict(eia_out.schema)


# ── optional real-file integration (skips without drive/engine) ───────────────

def _eha_available() -> bool:
    try:
        import fastexcel  # noqa: F401
    except ImportError:
        return False
    try:
        return gt._resolve_eha_dir().is_dir()
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _eha_available(), reason="EHA drive or fastexcel engine not available")
def test_integration_real_eha_ingest():
    df = gt.ingest_eha()
    assert df.columns == list(gt.CANONICAL_SCHEMA.keys())
    assert dict(df.schema) == dict(gt.CANONICAL_SCHEMA)
    assert df.height > 0
    assert (df["actual_installed_kw"] > 0).all()
    assert (df["actual_annual_energy_kwh"] > 0).all()
    assert (df["ground_truth_source"] == "EHA").all()
    assert df["actual_head_m"].is_null().all()
    assert df["actual_flow_m3s"].is_null().all()
    # Fleet GWh sanity: EHA 2022 should be well above 100,000 GWh (US fleet)
    fleet_gwh = df["actual_annual_energy_kwh"].sum() / 1e6
    assert fleet_gwh > 100_000, f"EHA fleet {fleet_gwh:.0f} GWh looks too low"

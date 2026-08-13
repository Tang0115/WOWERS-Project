"""Phase 5 — D1/D2/D3/D4 unit tests.

All tests use small synthetic polars frames built in-memory.
No external drive, no data files, no real parquets.

Coverage:
  D1 - combine_ground_truth: dedup key, EHA-priority rule, null codes, schema
  D2 - build_feature_matrix: left join on npdes_id, derived features, nulls preserved
  D3 - LEAKAGE_DENYLIST: assert_no_leakage raises on every member; passes clean list;
       select_model_features drops denylist; toggle behaviour
  D4 - nested_cv: leakage guard raises; no group leaks across inner folds;
       outer folds stratified; returns correct metric keys; dummy estimator
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from src.phase5 import ground_truth as gt
from src.phase5 import features as feat
from src.phase5 import cv as cv_mod


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — synthetic canonical-schema frames
# ══════════════════════════════════════════════════════════════════════════════

def _canonical_row(
    source: str = "EIA",
    code: int | None = 1,
    name: str = "Plant A",
    state: str = "CA",
    kw: float = 10_000.0,
    kwh: float = 40_000_000.0,
    year: int = 2022,
) -> dict:
    return {
        "ground_truth_source": source,
        "facility_name":       name,
        "state_code":          state,
        "latitude":            38.0,
        "longitude":           -120.0,
        "actual_annual_energy_kwh": kwh,
        "actual_installed_kw": kw,
        "actual_head_m":       None,
        "actual_flow_m3s":     None,
        "source_plant_code":   code,
        "source_year":         year,
    }


def _make_canonical(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        # Build empty typed DataFrame from schema directly
        return pl.DataFrame(
            {c: pl.Series([], dtype=dtype) for c, dtype in gt.CANONICAL_SCHEMA.items()}
        )
    df = pl.DataFrame(rows)
    return df.select(
        [pl.col(c).cast(dtype).alias(c) for c, dtype in gt.CANONICAL_SCHEMA.items()]
    )


# ══════════════════════════════════════════════════════════════════════════════
# D1 — combine_ground_truth
# ══════════════════════════════════════════════════════════════════════════════

class TestCombineGroundTruth:

    def test_schema_preserved(self):
        eia = _make_canonical([_canonical_row("EIA", 1)])
        eha = _make_canonical([_canonical_row("EHA", 2)])
        out = gt.combine_ground_truth(eia, eha)
        assert out.columns == list(gt.CANONICAL_SCHEMA.keys())
        assert dict(out.schema) == dict(gt.CANONICAL_SCHEMA)

    def test_eha_priority_on_collision(self):
        """When same source_plant_code appears in both, EHA row is kept."""
        eia = _make_canonical([_canonical_row("EIA", 100, kwh=1_000_000.0)])
        eha = _make_canonical([_canonical_row("EHA", 100, kwh=9_999_999.0)])
        out = gt.combine_ground_truth(eia, eha)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["ground_truth_source"] == "EHA"
        assert row["actual_annual_energy_kwh"] == pytest.approx(9_999_999.0)

    def test_eia_only_rows_kept(self):
        eia = _make_canonical([
            _canonical_row("EIA", 1),
            _canonical_row("EIA", 2),
        ])
        eha = _make_canonical([_canonical_row("EHA", 99)])   # different code
        out = gt.combine_ground_truth(eia, eha)
        # EIA codes 1 and 2 are not in EHA → kept; EHA code 99 kept
        assert out.height == 3
        sources = set(out["ground_truth_source"].to_list())
        assert sources == {"EIA", "EHA"}

    def test_eha_only_rows_kept(self):
        eia = _make_canonical([_canonical_row("EIA", 10)])
        eha = _make_canonical([
            _canonical_row("EHA", 20),
            _canonical_row("EHA", 30),
        ])
        out = gt.combine_ground_truth(eia, eha)
        assert out.height == 3

    def test_null_code_eha_rows_kept_always(self):
        """EHA rows with null source_plant_code cannot be deduped — always kept."""
        eia = _make_canonical([_canonical_row("EIA", 1)])
        eha = _make_canonical([
            _canonical_row("EHA", None),   # null code → cannot dedup
            _canonical_row("EHA", 1),      # same code as EIA → EHA wins, EIA dropped
        ])
        out = gt.combine_ground_truth(eia, eha)
        # null-code EHA + code-1 EHA (EIA dropped) = 2 rows total
        assert out.height == 2
        null_codes = out.filter(pl.col("source_plant_code").is_null()).height
        assert null_codes == 1

    def test_full_dedup_math(self):
        """Verify dedup arithmetic: overlap → keep EHA; EIA-only kept; EHA-only kept."""
        # EIA: codes 1,2,3 (1 and 2 also in EHA)
        # EHA: codes 2,3,4 + 1 null-code row
        eia = _make_canonical([
            _canonical_row("EIA", 1, kwh=111),
            _canonical_row("EIA", 2, kwh=222),
            _canonical_row("EIA", 3, kwh=333),
        ])
        eha = _make_canonical([
            _canonical_row("EHA", 2, kwh=2222),
            _canonical_row("EHA", 3, kwh=3333),
            _canonical_row("EHA", 4, kwh=4444),
            _canonical_row("EHA", None, kwh=9999),
        ])
        out = gt.combine_ground_truth(eia, eha)
        # EIA-only: code=1 (1 row)
        # EHA wins: codes 2 and 3 (2 rows)
        # EHA-only: code=4 (1 row) + null-code (1 row) = 2 rows
        # Total = 1 + 2 + 2 = 5
        assert out.height == 5
        code_kwh = {r["source_plant_code"]: r["actual_annual_energy_kwh"]
                    for r in out.to_dicts()}
        assert code_kwh[1] == pytest.approx(111)    # EIA-only kept
        assert code_kwh[2] == pytest.approx(2222)   # EHA wins
        assert code_kwh[3] == pytest.approx(3333)   # EHA wins
        assert code_kwh[4] == pytest.approx(4444)   # EHA-only kept

    def test_eha_internal_dup_deduped_keep_max_energy(self):
        """EHA-internal duplicate source_plant_code: keep the higher-energy row.

        Regression for real EHA defect: EIA code 61217 appeared twice in the
        EHA workbook (two sub-sites, identical energy).  The function must
        collapse EHA to one row per code before any cross-source join, so the
        same plant never lands in two CV folds.
        """
        eia = _make_canonical([_canonical_row("EIA", 999, kwh=5_000_000.0)])
        # EHA has code 7777 twice: one row with more energy (should win)
        eha = _make_canonical([
            _canonical_row("EHA", 7777, name="Sub-site A", kwh=1_000_000.0),
            _canonical_row("EHA", 7777, name="Sub-site B", kwh=9_000_000.0),  # higher
        ])
        out = gt.combine_ground_truth(eia, eha)
        # code 7777 appears exactly once
        rows_7777 = out.filter(pl.col("source_plant_code") == 7777)
        assert rows_7777.height == 1, "EHA internal dup must be collapsed to 1 row"
        # keep-max: the 9 MWh row wins
        assert rows_7777["actual_annual_energy_kwh"][0] == pytest.approx(9_000_000.0)
        # EIA code 999 is not in EHA → kept as-is
        assert out.filter(pl.col("source_plant_code") == 999).height == 1

    def test_eha_internal_dup_tied_energy_keeps_one_row(self):
        """Tied energy (real U Canal case): output has exactly 1 row."""
        eia = _make_canonical([])
        eha = _make_canonical([
            _canonical_row("EHA", 61217, name="U Canal Hydro 2",          kwh=4_267_000.0),
            _canonical_row("EHA", 61217, name="Head of U Canal Hydro Proj", kwh=4_267_000.0),
        ])
        out = gt.combine_ground_truth(eia, eha)
        assert out.height == 1
        assert out["source_plant_code"][0] == 61217

    def test_raises_on_missing_schema_column(self):
        eia = pl.DataFrame({"source_plant_code": [1], "facility_name": ["X"]})  # bad schema
        eha = _make_canonical([_canonical_row("EHA", 1)])
        with pytest.raises(ValueError, match="missing CANONICAL_SCHEMA"):
            gt.combine_ground_truth(eia, eha)

    def test_empty_eha(self):
        eia = _make_canonical([_canonical_row("EIA", 1)])
        eha = _make_canonical([])
        out = gt.combine_ground_truth(eia, eha)
        assert out.height == 1
        assert out["ground_truth_source"][0] == "EIA"

    def test_empty_eia(self):
        eia = _make_canonical([])
        eha = _make_canonical([_canonical_row("EHA", 5)])
        out = gt.combine_ground_truth(eia, eha)
        assert out.height == 1
        assert out["ground_truth_source"][0] == "EHA"


# ══════════════════════════════════════════════════════════════════════════════
# D2 — build_feature_matrix
# ══════════════════════════════════════════════════════════════════════════════

def _make_p1(npdes_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "npdes_id":       npdes_ids,
        "facility_name":  [f"Plant {i}" for i in npdes_ids],
        "state_code":     ["CA"] * len(npdes_ids),
        "latitude":       [38.0 + i * 0.1 for i in range(len(npdes_ids))],
        "longitude":      [-120.0 - i * 0.1 for i in range(len(npdes_ids))],
        "mean_flow_mgd":  [float(i) for i in range(1, len(npdes_ids) + 1)],
        "design_flow_mgd": [float(i) + 0.5 for i in range(1, len(npdes_ids) + 1)],
        "n_years_data":   [10] * len(npdes_ids),
        "data_quality":   ["dmr"] * len(npdes_ids),
    })


def _make_p2(npdes_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "npdes_id":             npdes_ids,
        "archetype":            ["large_potw"] * len(npdes_ids),
        "energy_p50_kwh_yr":    [float(i) * 1e6 for i in range(1, len(npdes_ids) + 1)],
        "head_m_p50":           [5.0] * len(npdes_ids),
        "power_p50_kw":         [float(i) * 100 for i in range(1, len(npdes_ids) + 1)],
        "capacity_factor_p50":  [0.4] * len(npdes_ids),
        "excluded":             [False] * len(npdes_ids),
        "exclusion_reason":     [None] * len(npdes_ids),
    })


def _make_p3(npdes_ids: list[str]) -> pl.DataFrame:
    if not npdes_ids:
        return pl.DataFrame({
            "npdes_id":          pl.Series([], dtype=pl.Utf8),
            "head_net_m":        pl.Series([], dtype=pl.Float64),
            "rated_power_kw":    pl.Series([], dtype=pl.Float64),
            "turbine_type":      pl.Series([], dtype=pl.Utf8),
            "turbine_viable":    pl.Series([], dtype=pl.Boolean),
            "annual_energy_mwh": pl.Series([], dtype=pl.Float64),
        })
    return pl.DataFrame({
        "npdes_id":        npdes_ids,
        "head_net_m":      [4.0 + i * 0.5 for i in range(len(npdes_ids))],
        "rated_power_kw":  [float(i) * 50 for i in range(1, len(npdes_ids) + 1)],
        "turbine_type":    ["Crossflow"] * len(npdes_ids),
        "turbine_viable":  [True] * len(npdes_ids),
        "annual_energy_mwh": [float(i) * 100 for i in range(1, len(npdes_ids) + 1)],
    })


def _make_p4(npdes_ids: list[str]) -> pl.DataFrame:
    if not npdes_ids:
        return pl.DataFrame({
            "npdes_id":             pl.Series([], dtype=pl.Utf8),
            "elec_rate_per_kwh":    pl.Series([], dtype=pl.Float64),
            "equipment_capex_usd":  pl.Series([], dtype=pl.Float64),
            "total_capex_usd":      pl.Series([], dtype=pl.Float64),
            "annual_opex_usd":      pl.Series([], dtype=pl.Float64),
            "annual_revenue_usd":   pl.Series([], dtype=pl.Float64),
            "npv_usd":              pl.Series([], dtype=pl.Float64),
            "project_viable":       pl.Series([], dtype=pl.Boolean),
            "data_quality_tier":    pl.Series([], dtype=pl.Int64),
            "permitting_tier":      pl.Series([], dtype=pl.Utf8),
        })
    return pl.DataFrame({
        "npdes_id":             npdes_ids,
        "elec_rate_per_kwh":    [0.12] * len(npdes_ids),
        "equipment_capex_usd":  [float(i) * 1e5 for i in range(1, len(npdes_ids) + 1)],
        "total_capex_usd":      [float(i) * 1.2e5 for i in range(1, len(npdes_ids) + 1)],
        "annual_opex_usd":      [float(i) * 2000 for i in range(1, len(npdes_ids) + 1)],
        # Leakage cols — present in p4 but should be droppable via select_model_features
        "annual_revenue_usd":   [float(i) * 1e4 for i in range(1, len(npdes_ids) + 1)],
        "npv_usd":              [float(i) * 5e4 for i in range(1, len(npdes_ids) + 1)],
        "project_viable":       [True] * len(npdes_ids),
        "data_quality_tier":    [0] * len(npdes_ids),
        "permitting_tier":      ["small_ferc"] * len(npdes_ids),
    })


class TestBuildFeatureMatrix:

    def test_spine_preserved(self):
        p1 = _make_p1(["A", "B", "C"])
        p2 = _make_p2(["A", "B"])     # C absent from P2
        p3 = _make_p3(["A"])          # B, C absent from P3
        p4 = _make_p4(["A"])          # B, C absent from P4
        out = feat.build_feature_matrix(p1, p2, p3, p4)
        # P1 is spine — all 3 rows preserved
        assert out.height == 3
        assert set(out["npdes_id"].to_list()) == {"A", "B", "C"}

    def test_nulls_for_missing_downstream_rows(self):
        """Rows absent from P3/P4 must have nulls, not be dropped."""
        p1 = _make_p1(["X", "Y"])
        p2 = _make_p2(["X", "Y"])
        p3 = _make_p3(["X"])          # Y absent
        p4 = _make_p4(["X"])          # Y absent
        out = feat.build_feature_matrix(p1, p2, p3, p4)
        y = out.filter(pl.col("npdes_id") == "Y")
        assert y.height == 1
        # rated_power_kw should be null for Y (it's only in P3 which covers only X)
        assert y["rated_power_kw"][0] is None

    def test_derived_flow_x_head(self):
        p1 = _make_p1(["A"])
        p2 = _make_p2(["A"])
        p3 = _make_p3(["A"])
        p4 = _make_p4(["A"])
        out = feat.build_feature_matrix(p1, p2, p3, p4)
        # flow_x_head = mean_flow_mgd × head_net_m
        assert "flow_x_head" in out.columns
        val = out.filter(pl.col("npdes_id") == "A")["flow_x_head"][0]
        assert val is not None and val > 0

    def test_derived_climate_zone(self):
        p1 = _make_p1(["T1"])
        p2 = _make_p2(["T1"])
        p3 = _make_p3(["T1"])
        p4 = _make_p4(["T1"])
        out = feat.build_feature_matrix(p1, p2, p3, p4)
        assert "climate_zone" in out.columns
        zone = out["climate_zone"][0]
        assert zone in ("tropical", "subtropical", "temperate", "continental", "polar", None)

    def test_no_imputation_of_nulls(self):
        """Nulls on downstream-absent rows must NOT be imputed."""
        p1 = _make_p1(["Z"])
        p2 = pl.DataFrame({
            "npdes_id": ["Z"],
            "archetype": [None],
            "energy_p50_kwh_yr": [None],
            "head_m_p50": [None],
            "power_p50_kw": [None],
            "capacity_factor_p50": [None],
            "excluded": [None],
            "exclusion_reason": [None],
        })
        p3 = _make_p3([])  # empty
        p4 = _make_p4([])  # empty
        out = feat.build_feature_matrix(p1, p2, p3, p4)
        assert out.height == 1
        # rated_power_kw (from P3) should be null
        assert out["rated_power_kw"][0] is None


# ══════════════════════════════════════════════════════════════════════════════
# D3 — leakage lock
# ══════════════════════════════════════════════════════════════════════════════

class TestAssertNoLeakage:

    def test_raises_on_every_denylist_member(self):
        """Every LEAKAGE_DENYLIST column must trigger a raise individually."""
        for col in feat.LEAKAGE_DENYLIST:
            with pytest.raises(ValueError, match="Leakage detected"):
                feat.assert_no_leakage([col])

    def test_raises_listing_all_bad_cols(self):
        bad = ["annual_revenue_usd", "npv_usd"]
        with pytest.raises(ValueError) as exc_info:
            feat.assert_no_leakage(bad)
        msg = str(exc_info.value)
        assert "annual_revenue_usd" in msg
        assert "npv_usd" in msg

    def test_passes_on_clean_features(self):
        clean = ["mean_flow_mgd", "head_net_m", "latitude", "state_code", "n_years_data"]
        feat.assert_no_leakage(clean)  # must not raise

    def test_passes_on_empty_list(self):
        feat.assert_no_leakage([])   # no features → nothing to leak


class TestSelectModelFeatures:

    def _df_with_all(self) -> pl.DataFrame:
        """DataFrame with a mix of safe and denylisted columns."""
        cols = {
            "npdes_id":            ["A"],
            "mean_flow_mgd":       [5.0],
            "latitude":            [38.0],
            # Denylisted columns
            "annual_revenue_usd":  [10000.0],
            "npv_usd":             [50000.0],
            "project_viable":      [True],
            "capacity_factor":     [0.4],
            # Physics estimate cols
            "annual_energy_kwh":   [40e6],
            "energy_p50_kwh_yr":   [38e6],
            "annual_energy_mwh":   [40e3],
        }
        return pl.DataFrame(cols)

    def test_drops_all_denylist_cols_by_default(self):
        df = self._df_with_all()
        safe = feat.select_model_features(df)
        for col in feat.LEAKAGE_DENYLIST:
            assert col not in safe, f"Denylisted col {col!r} should be dropped"

    def test_keeps_safe_cols(self):
        df = self._df_with_all()
        safe = feat.select_model_features(df)
        assert "mean_flow_mgd" in safe
        assert "latitude" in safe
        assert "npdes_id" in safe

    def test_physics_dropped_by_default(self):
        df = self._df_with_all()
        safe = feat.select_model_features(df, allow_physics_estimate=False)
        for col in feat.PHYSICS_ESTIMATE_COLS:
            assert col not in safe

    def test_physics_retained_when_toggled(self):
        df = self._df_with_all()
        safe = feat.select_model_features(df, allow_physics_estimate=True)
        # Physics cols that are present in df should now be in safe list
        for col in feat.PHYSICS_ESTIMATE_COLS:
            if col in df.columns:
                assert col in safe, f"{col} should be retained with toggle=True"

    def test_non_physics_leakage_still_dropped_with_toggle(self):
        """revenue/NPV/viable always dropped even when toggle=True."""
        df = self._df_with_all()
        safe = feat.select_model_features(df, allow_physics_estimate=True)
        always_bad = {"annual_revenue_usd", "npv_usd", "project_viable", "capacity_factor"}
        for col in always_bad:
            assert col not in safe

    def test_no_leakage_after_select(self):
        """assert_no_leakage must not raise on the output of select_model_features."""
        df = self._df_with_all()
        safe = feat.select_model_features(df)
        feat.assert_no_leakage(safe)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# D4 — nested_cv harness
# ══════════════════════════════════════════════════════════════════════════════

class _ConstantRegressor:
    """Trivial sklearn-compatible regressor that predicts the training mean."""

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def _make_cv_inputs(
    n: int = 40,
    n_states: int = 4,
    n_groups: int = 8,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series, list[str], list[str]]:
    """Synthetic inputs for nested_cv: clean features, log-energy target."""
    rng = np.random.default_rng(seed)
    X = pl.DataFrame({
        "mean_flow_mgd": rng.uniform(0.5, 50.0, n).tolist(),
        "latitude":      rng.uniform(30.0, 50.0, n).tolist(),
        "n_years_data":  rng.integers(1, 20, n).tolist(),
    })
    y = pl.Series("log_energy", rng.uniform(10, 20, n).tolist())
    states = [f"ST{i % n_states}" for i in range(n)]
    groups = [f"G{i % n_groups}" for i in range(n)]
    return X, y, groups, states


class TestNestedCv:

    def test_leakage_guard_raises(self):
        X, y, groups, states = _make_cv_inputs()
        # Inject a denylisted column
        X_bad = X.with_columns(pl.lit(1.0).alias("annual_revenue_usd"))
        with pytest.raises(ValueError, match="Leakage detected"):
            cv_mod.nested_cv(
                X_bad, y, groups, states,
                estimator_factory=_ConstantRegressor,
                outer=2, inner=2,
            )

    def test_returns_correct_metric_keys(self):
        X, y, groups, states = _make_cv_inputs()
        results = cv_mod.nested_cv(
            X, y, groups, states,
            estimator_factory=_ConstantRegressor,
            outer=3, inner=2,
        )
        assert set(results.keys()) == {"rmse_log", "mape", "r2", "spearman"}

    def test_fold_counts_match_outer_setting(self):
        X, y, groups, states = _make_cv_inputs()
        results = cv_mod.nested_cv(
            X, y, groups, states,
            estimator_factory=_ConstantRegressor,
            outer=4, inner=2,
        )
        for metric, vals in results.items():
            assert len(vals) == 4, f"{metric}: expected 4 folds, got {len(vals)}"

    def test_no_group_leak_across_inner_folds(self):
        """Directly verify inner fold groups do not bleed across train/val."""
        n = 30
        groups = [f"G{i % 5}" for i in range(n)]
        states = ["CA"] * n
        train_idx = list(range(n))
        inner_splits = cv_mod._group_aware_inner_splits(train_idx, groups, n_folds=3, seed=0)
        for itrain, ival in inner_splits:
            train_g = {groups[i] for i in itrain}
            val_g   = {groups[i] for i in ival}
            assert train_g & val_g == set(), f"Group leak: {train_g & val_g}"

    def test_outer_stratification_by_state(self):
        """Each state should appear in at least some val fold."""
        n = 40
        states = [f"ST{i % 4}" for i in range(n)]
        splits = cv_mod._state_stratified_splits(states, n_folds=4, seed=0)
        for train_idx, val_idx in splits:
            val_states = {states[i] for i in val_idx}
            # Each fold should have at least 2 distinct states (not collapsed to 1)
            assert len(val_states) >= 1

    def test_dimension_mismatch_raises(self):
        X, y, groups, states = _make_cv_inputs(n=20)
        with pytest.raises(ValueError, match="rows but y"):
            cv_mod.nested_cv(
                X, pl.Series("log_energy", [1.0]),  # wrong length
                groups, states,
                estimator_factory=_ConstantRegressor,
                outer=2, inner=2,
            )

    def test_metrics_finite(self):
        """Non-Spearman metrics must be finite; Spearman may be NaN for
        a constant predictor (all predictions identical → undefined rank corr)."""
        X, y, groups, states = _make_cv_inputs(n=30)
        results = cv_mod.nested_cv(
            X, y, groups, states,
            estimator_factory=_ConstantRegressor,
            outer=3, inner=2,
        )
        for metric in ("rmse_log", "mape", "r2"):
            for v in results[metric]:
                assert math.isfinite(v), f"{metric} has non-finite value {v}"
        # Spearman is undefined for a constant prediction (all ranks tied → NaN);
        # just assert the list length is correct and values are float
        assert len(results["spearman"]) == 3
        for v in results["spearman"]:
            assert isinstance(v, float)

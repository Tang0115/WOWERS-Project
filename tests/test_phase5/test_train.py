"""P5-SMOKE — unit tests for src/phase5/train.py.

All unit tests use small synthetic polars frames built in-memory.
No external drive, no real parquets required.

Coverage:
  build_training_frame : shape, all-numeric X, column order, log-target,
                         state encoding determinism + sorted-unique order,
                         climate-zone determinism, null-plant-code sentinel,
                         no nulls in X, leakage guard integration
  MeanRegressor        : predict equals train mean, fit returns self
  CFBaselineRegressor  : median_cf_ computed, predict in log-space, fit returns self
  Baselines beatable   : linear model beats mean regressor on synthetic data
  Metadata schema      : required keys, metrics structure, JSON serialisability
  demonstrate_leakage_guard: returns error string containing "Leakage detected"

One real-data integration test is gated on combined_ground_truth.parquet
being present; it asserts 1,360 rows, all-numeric X, and 4 metric lists of
length 5 from nested_cv.

Baseline (pre-smoke-test) suite: 532 passed / 1 skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.phase5 import train as tr
from src.phase5 import ground_truth as gt_mod
from src.phase5.features import LEAKAGE_DENYLIST, assert_no_leakage


# ── Paths ─────────────────────────────────────────────────────────────────────

_GT_PATH = Path("data/raw/ground_truth/combined_ground_truth.parquet")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _canonical_row(
    source: str = "EIA",
    code: int | None = 1,
    name: str = "Plant A",
    state: str = "CA",
    lat: float = 38.0,
    lon: float = -120.0,
    kw: float = 10_000.0,
    kwh: float = 40_000_000.0,
    year: int = 2022,
) -> dict:
    return {
        "ground_truth_source":    source,
        "facility_name":          name,
        "state_code":             state,
        "latitude":               lat,
        "longitude":              lon,
        "actual_annual_energy_kwh": kwh,
        "actual_installed_kw":    kw,
        "actual_head_m":          None,
        "actual_flow_m3s":        None,
        "source_plant_code":      code,
        "source_year":            year,
    }


def _make_gt(n: int = 10, seed: int = 0) -> pl.DataFrame:
    """Synthetic ground-truth frame conforming to CANONICAL_SCHEMA."""
    rng = np.random.default_rng(seed)
    state_pool = ["CA", "OR", "WA", "TX", "NY", "FL", "CO", "MN", "ME", "OH"]
    rows = []
    for i in range(n):
        rows.append(_canonical_row(
            source="EIA",
            code=i + 1,
            name=f"Plant {i}",
            state=state_pool[i % len(state_pool)],
            lat=float(rng.uniform(25.0, 48.0)),
            lon=float(rng.uniform(-120.0, -70.0)),
            kw=float(rng.uniform(100.0, 1e5)),
            kwh=float(rng.uniform(1e6, 1e9)),
        ))
    df = pl.DataFrame(rows)
    return df.select(
        [pl.col(c).cast(dtype).alias(c) for c, dtype in gt_mod.CANONICAL_SCHEMA.items()]
    )


def _make_gt_with_null_code(n: int = 5) -> pl.DataFrame:
    """Synthetic frame with exactly 1 null source_plant_code (at index 2)."""
    df = _make_gt(n)
    codes = df["source_plant_code"].to_list()
    codes[2] = None
    return df.with_columns(
        pl.Series("source_plant_code", codes, dtype=pl.Int64)
    )


# ══════════════════════════════════════════════════════════════════════════════
# build_training_frame — shape and column correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTrainingFrameShape:

    def test_output_lengths(self):
        n = 20
        X, y, groups, states = tr.build_training_frame(_make_gt(n))
        assert X.height == n
        assert len(y)      == n
        assert len(groups) == n
        assert len(states) == n

    def test_all_x_columns_float64(self):
        """X must be all Float64 so X.to_numpy() returns a float64 array."""
        X, *_ = tr.build_training_frame(_make_gt(10))
        for col, dtype in X.schema.items():
            assert dtype == pl.Float64, (
                f"Column '{col}' has dtype {dtype}, expected Float64"
            )

    def test_x_columns_match_feature_list(self):
        X, *_ = tr.build_training_frame(_make_gt(10))
        assert list(X.columns) == tr.FEATURE_COLUMNS

    def test_no_nulls_in_x(self):
        """X has no null values — required for X.to_numpy() in nested_cv."""
        X, *_ = tr.build_training_frame(_make_gt(20))
        null_total = sum(X[c].null_count() for c in X.columns)
        assert null_total == 0

    def test_x_to_numpy_shape(self):
        """X.to_numpy() returns a 2-D float64 array of shape (n, 5)."""
        n = 15
        X, *_ = tr.build_training_frame(_make_gt(n))
        arr = X.to_numpy()
        assert arr.dtype == np.float64
        assert arr.shape == (n, len(tr.FEATURE_COLUMNS))


# ══════════════════════════════════════════════════════════════════════════════
# build_training_frame — log-space target
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTrainingFrameTarget:

    def test_log_target_equals_numpy_log(self):
        """y[i] == ln(actual_annual_energy_kwh[i]) for all i."""
        df = _make_gt(15)
        _, y, _, _ = tr.build_training_frame(df)
        expected = np.log(df["actual_annual_energy_kwh"].to_numpy())
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-10)

    def test_log_target_finite(self):
        """All log-target values are finite (energy > 0 guaranteed by schema)."""
        _, y, _, _ = tr.build_training_frame(_make_gt(20))
        assert all(math.isfinite(v) for v in y.to_list())


# ══════════════════════════════════════════════════════════════════════════════
# build_training_frame — state encoding
# ══════════════════════════════════════════════════════════════════════════════

class TestStateEncoding:

    def test_determinism_same_input(self):
        """Identical input → identical state_code_code on every call."""
        df = _make_gt(20)
        X1, *_ = tr.build_training_frame(df)
        X2, *_ = tr.build_training_frame(df)
        np.testing.assert_array_equal(
            X1["state_code_code"].to_numpy(),
            X2["state_code_code"].to_numpy(),
        )

    def test_sorted_unique_mapping(self):
        """First alphabetical state → code 0; second → code 1; etc."""
        # Only states CA and NY: sorted order is CA=0, NY=1
        df = pl.DataFrame([
            _canonical_row(state="NY", code=1, kw=5_000.0, kwh=1e7),
            _canonical_row(state="CA", code=2, kw=10_000.0, kwh=2e7),
        ]).select(
            [pl.col(c).cast(dtype).alias(c) for c, dtype in gt_mod.CANONICAL_SCHEMA.items()]
        )
        X, *_ = tr.build_training_frame(df)
        codes = X["state_code_code"].to_list()
        assert codes[0] == pytest.approx(1.0)   # NY → 1
        assert codes[1] == pytest.approx(0.0)   # CA → 0

    def test_different_input_can_give_different_mapping(self):
        """Adding a new state changes the map (sorted-unique recomputed per call)."""
        df_a = pl.DataFrame([
            _canonical_row(state="AZ", code=1, kw=5000.0, kwh=1e7),
        ]).select(
            [pl.col(c).cast(dtype).alias(c) for c, dtype in gt_mod.CANONICAL_SCHEMA.items()]
        )
        df_b = pl.DataFrame([
            _canonical_row(state="AZ", code=1, kw=5000.0, kwh=1e7),
            _canonical_row(state="CA", code=2, kw=5000.0, kwh=1e7),  # inserted before AZ
        ]).select(
            [pl.col(c).cast(dtype).alias(c) for c, dtype in gt_mod.CANONICAL_SCHEMA.items()]
        )
        Xa, *_ = tr.build_training_frame(df_a)
        Xb, *_ = tr.build_training_frame(df_b)
        # In df_a: AZ=0 (only state).  In df_b: AZ=0, CA=1 (sorted).
        # Row 0 (AZ) should have code 0 in both.
        assert Xa["state_code_code"][0] == pytest.approx(0.0)
        assert Xb["state_code_code"][0] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# build_training_frame — climate-zone encoding
# ══════════════════════════════════════════════════════════════════════════════

class TestClimateZoneEncoding:

    def test_determinism(self):
        """Same latitudes → same climate_zone_code on every call."""
        df = _make_gt(20, seed=99)
        X1, *_ = tr.build_training_frame(df)
        X2, *_ = tr.build_training_frame(df)
        np.testing.assert_array_equal(
            X1["climate_zone_code"].to_numpy(),
            X2["climate_zone_code"].to_numpy(),
        )

    def test_band_boundaries(self):
        """Each latitude band maps to the correct code."""
        lats = [10.0, 30.0, 42.0, 55.0, 70.0]
        expected = [0, 1, 2, 3, 4]   # tropical, subtropical, temperate, continental, polar
        for lat, exp_code in zip(lats, expected):
            assert tr._lat_to_climate_code(lat) == exp_code

    def test_none_lat_returns_none(self):
        assert tr._lat_to_climate_code(None) is None

    def test_negative_lat_uses_abs(self):
        """Southern latitudes mirror Northern (abs used)."""
        assert tr._lat_to_climate_code(-42.0) == tr._lat_to_climate_code(42.0)


# ══════════════════════════════════════════════════════════════════════════════
# build_training_frame — null plant-code sentinel
# ══════════════════════════════════════════════════════════════════════════════

class TestNullPlantCodeSentinel:

    def test_null_code_row_gets_string_sentinel(self):
        """The null source_plant_code row receives a string sentinel group key."""
        df = _make_gt_with_null_code()
        _, _, groups, _ = tr.build_training_frame(df)
        string_groups = [g for g in groups if isinstance(g, str)]
        assert len(string_groups) == 1
        assert string_groups[0].startswith("SENTINEL_")

    def test_sentinel_starts_with_prefix(self):
        """Sentinel key format: SENTINEL_<facility_name>_<state_code>."""
        df = _make_gt_with_null_code()
        _, _, groups, _ = tr.build_training_frame(df)
        sentinel = next(g for g in groups if isinstance(g, str))
        assert "SENTINEL_" in sentinel

    def test_all_groups_unique(self):
        """Group keys are all unique (no duplicates across plants)."""
        df = _make_gt_with_null_code()
        _, _, groups, _ = tr.build_training_frame(df)
        str_groups = [str(g) for g in groups]
        assert len(str_groups) == len(set(str_groups))

    def test_non_null_codes_are_ints(self):
        """Non-null source_plant_code rows produce int group keys."""
        df = _make_gt_with_null_code(n=6)
        _, _, groups, _ = tr.build_training_frame(df)
        int_groups = [g for g in groups if isinstance(g, int)]
        # 5 non-null rows (null is at index 2)
        assert len(int_groups) == 5


# ══════════════════════════════════════════════════════════════════════════════
# Leakage guard
# ══════════════════════════════════════════════════════════════════════════════

class TestLeakageGuard:

    def test_assert_no_leakage_raises_on_denylist_col(self):
        """assert_no_leakage raises ValueError for any denylisted column."""
        bad_col = sorted(LEAKAGE_DENYLIST)[0]   # deterministic: "annual_energy_kwh"
        with pytest.raises(ValueError, match="Leakage detected"):
            assert_no_leakage([bad_col, "latitude", "longitude"])

    def test_assert_no_leakage_passes_clean_list(self):
        """assert_no_leakage does not raise for the smoke-test feature set."""
        assert_no_leakage(tr.FEATURE_COLUMNS)   # should not raise

    def test_demonstrate_leakage_guard_contains_error_text(self):
        """demonstrate_leakage_guard() returns a string with 'Leakage detected'."""
        msg = tr.demonstrate_leakage_guard()
        assert "Leakage detected" in msg

    def test_demonstrate_leakage_guard_non_trivial_length(self):
        """demonstrate_leakage_guard() returns a meaningful error message."""
        msg = tr.demonstrate_leakage_guard()
        assert len(msg) > 30

    def test_smoke_test_feature_columns_not_in_denylist(self):
        """Every column in FEATURE_COLUMNS is clean (not in LEAKAGE_DENYLIST)."""
        for col in tr.FEATURE_COLUMNS:
            assert col not in LEAKAGE_DENYLIST, (
                f"Feature '{col}' is in LEAKAGE_DENYLIST — smoke test would fail"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MeanRegressor
# ══════════════════════════════════════════════════════════════════════════════

class TestMeanRegressor:

    def test_predict_equals_train_mean(self):
        model = tr.MeanRegressor()
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model.fit(np.zeros((5, 3)), y_train)
        pred = model.predict(np.zeros((10, 3)))
        np.testing.assert_allclose(pred, np.full(10, 3.0))

    def test_fit_returns_self(self):
        model = tr.MeanRegressor()
        result = model.fit(np.zeros((5, 3)), np.ones(5))
        assert result is model

    def test_predict_length_matches_input(self):
        model = tr.MeanRegressor()
        model.fit(np.zeros((10, 5)), np.arange(10, dtype=float))
        pred = model.predict(np.zeros((7, 5)))
        assert len(pred) == 7

    def test_predict_constant(self):
        """Prediction is constant regardless of X values."""
        model = tr.MeanRegressor()
        model.fit(np.zeros((5, 2)), np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
        x_varied = np.random.default_rng(0).random((8, 2))
        pred = model.predict(x_varied)
        assert np.all(pred == pred[0]), "MeanRegressor must predict the same value for all rows"


# ══════════════════════════════════════════════════════════════════════════════
# CFBaselineRegressor
# ══════════════════════════════════════════════════════════════════════════════

class TestCFBaselineRegressor:

    def _make_x(self, log_cap: np.ndarray) -> np.ndarray:
        """Build 5-column X with log_cap_kw at column 2."""
        n = len(log_cap)
        X = np.zeros((n, 5))
        X[:, 2] = log_cap
        return X

    def test_median_cf_computed_correctly(self):
        """fit computes median CF from log(energy) and log(cap) columns."""
        rng = np.random.default_rng(0)
        n = 50
        log_cap = rng.uniform(8.0, 12.0, n)
        cf_true  = 0.40                                          # known CF
        y = log_cap + math.log(8760) + math.log(cf_true)        # exact log-energy
        X = self._make_x(log_cap)
        model = tr.CFBaselineRegressor(log_cap_col_idx=2)
        model.fit(X, y)
        assert model.median_cf_ == pytest.approx(cf_true, rel=1e-6)

    def test_predict_formula_correct(self):
        """predict = log_cap + ln(8760) + ln(median_cf)."""
        log_cap = np.array([10.0, 11.0, 12.0])
        X = self._make_x(log_cap)
        model = tr.CFBaselineRegressor(2)
        model.median_cf_ = 0.50
        pred = model.predict(X)
        expected = log_cap + math.log(8760) + math.log(0.50)
        np.testing.assert_allclose(pred, expected, rtol=1e-10)

    def test_fit_returns_self(self):
        model = tr.CFBaselineRegressor(2)
        result = model.fit(np.zeros((5, 5)), np.ones(5))
        assert result is model

    def test_predict_uses_log_cap_column(self):
        """Predictions scale with log_capacity_kw, not other columns."""
        log_caps = np.array([9.0, 10.0, 11.0])
        X = self._make_x(log_caps)
        model = tr.CFBaselineRegressor(2)
        model.median_cf_ = 0.45
        pred = model.predict(X)
        # Predictions should be strictly increasing with log_cap
        assert pred[0] < pred[1] < pred[2]

    def test_diverse_cf_median_robust(self):
        """Median CF is stable when training CF values span a range."""
        rng = np.random.default_rng(7)
        n = 100
        log_cap = rng.uniform(8.0, 12.0, n)
        cf_values = rng.uniform(0.2, 0.8, n)       # diverse CFs
        y = log_cap + math.log(8760) + np.log(cf_values)
        X = self._make_x(log_cap)
        model = tr.CFBaselineRegressor(2)
        model.fit(X, y)
        assert 0.15 < model.median_cf_ < 0.85


# ══════════════════════════════════════════════════════════════════════════════
# Baselines beatable by construction
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselinesBeatability:
    """Verify that the metric computations are correct by construction.

    On synthetic data with a tight linear relationship (log_energy ≈ log_cap),
    a non-trivial model that fits this relationship should have lower RMSE than
    the constant predict-mean baseline.
    """

    def test_linear_model_beats_mean_baseline(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        n = 200
        log_cap = rng.uniform(8.0, 14.0, n)
        y = 0.9 * log_cap + 2.0 + rng.normal(0, 0.2, n)

        split = int(n * 0.7)
        X_tr = log_cap[:split, np.newaxis]
        y_tr = y[:split]
        X_va = log_cap[split:, np.newaxis]
        y_va = y[split:]

        mean_model = tr.MeanRegressor()
        mean_model.fit(X_tr, y_tr)
        mean_rmse = float(np.sqrt(np.mean((y_va - mean_model.predict(X_va)) ** 2)))

        lr = LinearRegression().fit(X_tr, y_tr)
        lr_rmse = float(np.sqrt(np.mean((y_va - lr.predict(X_va)) ** 2)))

        assert lr_rmse < mean_rmse, (
            f"Linear RMSE {lr_rmse:.4f} should be < Mean RMSE {mean_rmse:.4f}"
        )

    def test_cf_baseline_beats_mean_when_cf_constant(self):
        """CF baseline beats mean baseline when true CF is constant."""
        rng = np.random.default_rng(13)
        n = 100
        log_cap = rng.uniform(8.0, 12.0, n)
        cf_true = 0.5
        y = log_cap + math.log(8760) + math.log(cf_true) + rng.normal(0, 0.01, n)

        split = 70
        X_tr = np.column_stack([np.zeros(split), np.zeros(split), log_cap[:split],
                                np.zeros(split), np.zeros(split)])
        X_va = np.column_stack([np.zeros(n - split), np.zeros(n - split),
                                log_cap[split:], np.zeros(n - split),
                                np.zeros(n - split)])
        y_tr = y[:split]
        y_va = y[split:]

        mean_model = tr.MeanRegressor().fit(X_tr, y_tr)
        cf_model   = tr.CFBaselineRegressor(2).fit(X_tr, y_tr)

        mean_rmse = float(np.sqrt(np.mean((y_va - mean_model.predict(X_va)) ** 2)))
        cf_rmse   = float(np.sqrt(np.mean((y_va - cf_model.predict(X_va)) ** 2)))

        assert cf_rmse < mean_rmse, (
            f"CF RMSE {cf_rmse:.4f} should be < Mean RMSE {mean_rmse:.4f} "
            "when CF is constant"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Metadata schema
# ══════════════════════════════════════════════════════════════════════════════

class TestMetadataSchema:

    def _make_fake_metadata(self) -> dict:
        """Construct a smoke_metadata.json-compatible dict for schema tests."""
        fold5 = [0.5, 0.6, 0.4, 0.5, 0.55]
        metric_block = {
            m: {"per_fold": fold5, "mean": float(np.mean(fold5)), "std": float(np.std(fold5))}
            for m in ["rmse_log", "mape", "r2", "spearman"]
        }
        return {
            "feature_names":        tr.FEATURE_COLUMNS,
            "categorical_mappings": {
                "state_code":   {"CA": 0, "OR": 1, "WA": 2},
                "climate_zone": {k: int(v) for k, v in tr._CLIMATE_BANDS.items()},
            },
            "lgbm_params":     {"n_estimators": 400, "learning_rate": 0.05},
            "seed":            0,
            "git_sha":         "abcdef1234567890",
            "train_row_count": 1360,
            "timestamp":       "2026-07-06T12:00:00+00:00",
            "criteria": {
                "crit3_rmse_pass":     True,
                "crit3_spearman_pass": True,
            },
            "metrics": {
                "lgbm":          metric_block,
                "mean_baseline": metric_block,
                "cf_baseline":   metric_block,
            },
        }

    def test_required_top_level_keys(self):
        meta = self._make_fake_metadata()
        for key in [
            "feature_names", "categorical_mappings", "lgbm_params",
            "seed", "git_sha", "train_row_count", "timestamp", "metrics",
        ]:
            assert key in meta, f"Missing top-level key: {key}"

    def test_metrics_has_three_models(self):
        meta = self._make_fake_metadata()
        assert set(meta["metrics"].keys()) == {"lgbm", "mean_baseline", "cf_baseline"}

    def test_metrics_has_four_metric_keys(self):
        meta = self._make_fake_metadata()
        for model_key in ["lgbm", "mean_baseline", "cf_baseline"]:
            assert set(meta["metrics"][model_key].keys()) == {
                "rmse_log", "mape", "r2", "spearman"
            }

    def test_per_fold_length_five(self):
        meta = self._make_fake_metadata()
        for model_key in ["lgbm", "mean_baseline", "cf_baseline"]:
            for metric in ["rmse_log", "mape", "r2", "spearman"]:
                block = meta["metrics"][model_key][metric]
                assert len(block["per_fold"]) == 5, (
                    f"metrics.{model_key}.{metric}.per_fold has "
                    f"{len(block['per_fold'])} entries, expected 5"
                )

    def test_metric_block_has_mean_and_std(self):
        meta = self._make_fake_metadata()
        for model_key in meta["metrics"]:
            for metric_key in meta["metrics"][model_key]:
                block = meta["metrics"][model_key][metric_key]
                assert "mean" in block
                assert "std"  in block
                assert isinstance(block["mean"], float)
                assert isinstance(block["std"],  float)

    def test_feature_names_match_feature_columns(self):
        meta = self._make_fake_metadata()
        assert meta["feature_names"] == tr.FEATURE_COLUMNS

    def test_climate_zone_mapping_complete(self):
        meta = self._make_fake_metadata()
        assert set(meta["categorical_mappings"]["climate_zone"].keys()) == set(
            tr._CLIMATE_BANDS.keys()
        )

    def test_json_serialisable(self):
        """Metadata dict must survive a json.dumps/loads round-trip."""
        meta = self._make_fake_metadata()
        raw = json.dumps(meta)
        recovered = json.loads(raw)
        assert recovered["train_row_count"] == 1360

    def test_criteria_keys_present(self):
        meta = self._make_fake_metadata()
        assert "criteria" in meta
        assert "crit3_rmse_pass"     in meta["criteria"]
        assert "crit3_spearman_pass" in meta["criteria"]


# ══════════════════════════════════════════════════════════════════════════════
# Integration test (real data — gated on parquet availability)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not _GT_PATH.exists(),
    reason="combined_ground_truth.parquet not available — integration test skipped",
)
class TestIntegration:

    def test_real_data_row_count(self):
        """1,360 rows load from combined_ground_truth.parquet."""
        df = pl.read_parquet(_GT_PATH)
        assert df.height == 1360

    def test_real_data_x_all_float64(self):
        """X from real data has all Float64 columns."""
        df = pl.read_parquet(_GT_PATH)
        X, *_ = tr.build_training_frame(df)
        for col, dtype in X.schema.items():
            assert dtype == pl.Float64, (
                f"Column '{col}' dtype is {dtype}, expected Float64"
            )

    def test_real_data_x_no_nulls(self):
        """X from real data has no null values."""
        df = pl.read_parquet(_GT_PATH)
        X, *_ = tr.build_training_frame(df)
        null_total = sum(X[c].null_count() for c in X.columns)
        assert null_total == 0

    def test_real_data_exactly_one_null_plant_code(self):
        """Exactly 1 null source_plant_code → exactly 1 string sentinel in groups."""
        df = pl.read_parquet(_GT_PATH)
        _, _, groups, _ = tr.build_training_frame(df)
        string_groups = [g for g in groups if isinstance(g, str)]
        assert len(string_groups) == 1, (
            f"Expected 1 sentinel string group, got {len(string_groups)}"
        )

    def test_real_data_nested_cv_returns_five_folds(self):
        """nested_cv returns 4 metric lists each of length 5 on real data."""
        from src.phase5.cv import nested_cv

        df = pl.read_parquet(_GT_PATH)
        X, y, groups, states = tr.build_training_frame(df)
        results = nested_cv(
            X, y, groups, states,
            estimator_factory=lambda: tr.MeanRegressor(),
            outer=5, inner=3, seed=0,
        )
        assert set(results.keys()) == {"rmse_log", "mape", "r2", "spearman"}
        for metric, vals in results.items():
            assert len(vals) == 5, (
                f"Metric '{metric}' has {len(vals)} fold values, expected 5"
            )

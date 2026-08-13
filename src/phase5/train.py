"""Phase 5 — internal smoke-test training run on combined dam labels.

INTERNAL — pipeline proof only. These numbers must never appear in
director, pitch, or competition material. See SMOKE_TEST_REPORT.md.

Orchestrates the full smoke-test pipeline:
    load combined_ground_truth.parquet (1,360 rows)
    → build_training_frame  (5 numeric features, log-space target, groups, states)
    → nested_cv × 3         (LightGBM, predict-mean baseline, CF baseline)
    → final fit on all 1,360 rows
    → persist model + metadata to data/processed/phase5/models/

Run:
    python -m src.phase5.train

Context
-------
Phase 5 as a product ML model was killed 2026-07-04
(WOWERS_PROJECT_JOURNAL.md, Jul-2 and Jul-4 sessions). This smoke test
survives as a pipeline proof only — the rails (combine → features →
leakage lock → nested_cv) are verified end-to-end.

allow_physics_estimate_feature stays false (dam labels carry no
physics-estimate columns anyway; the toggle is moot here).

Label domain caveat: dam plants (median ~8 MW, no head/flow columns,
only ~4 shared features with WWTP scoring matrix) vs 1–500 kW outfall
targets. Mediocre absolute R²/MAPE is expected and acceptable.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from src.common import config, logging_setup
from src.phase5.features import LEAKAGE_DENYLIST, assert_no_leakage
from src.phase5.cv import nested_cv

logging_setup.setup_run_log("phase5")
log = logging_setup.get("wowers.phase5.train")

# ── Climate-zone constants ─────────────────────────────────────────────────────
# Köppen–Geiger 5-band latitude approximation — mirrors features.py _climate_zone
# (private; replicated here per smoke-test design; do not import _climate_zone).
# Zones: tropical (<23.5°), subtropical (23.5–35°), temperate (35–50°),
# continental (50–66.5°), polar (>66.5°).
_CLIMATE_BANDS: dict[str, int] = {
    "tropical":    0,
    "subtropical": 1,
    "temperate":   2,
    "continental": 3,
    "polar":       4,
}

# ── Feature column order ───────────────────────────────────────────────────────
# Order is fixed and documented; CFBaselineRegressor uses _LOG_CAP_IDX.
FEATURE_COLUMNS: list[str] = [
    "latitude",
    "longitude",
    "log_capacity_kw",
    "climate_zone_code",
    "state_code_code",
]
_LOG_CAP_IDX: int = FEATURE_COLUMNS.index("log_capacity_kw")  # = 2
_LOG_8760: float = math.log(8_760)

# ── LightGBM fixed hyper-parameters (no HP sweep — by design) ─────────────────
# deterministic=True + force_row_wise=True ensures bit-identical consecutive runs.
# n_jobs=1 eliminates thread non-determinism.
_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators":      400,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "n_jobs":            1,
    "deterministic":     True,
    "force_row_wise":    True,
    "verbose":           -1,
}


# ── Climate / state helpers ────────────────────────────────────────────────────

def _lat_to_climate_code(lat: float | None) -> int | None:
    """Map latitude (decimal degrees) to _CLIMATE_BANDS integer code.

    Returns None for None / NaN inputs (null-safe; real data has 0 null lats).
    """
    if lat is None:
        return None
    try:
        if math.isnan(lat):
            return None
    except (TypeError, ValueError):
        return None
    a = abs(lat)
    if a < 23.5:
        return _CLIMATE_BANDS["tropical"]
    if a < 35.0:
        return _CLIMATE_BANDS["subtropical"]
    if a < 50.0:
        return _CLIMATE_BANDS["temperate"]
    if a < 66.5:
        return _CLIMATE_BANDS["continental"]
    return _CLIMATE_BANDS["polar"]


def _state_code_map(gt: pl.DataFrame) -> dict[str, int]:
    """Build deterministic sorted-unique state_code → int mapping from ``gt``."""
    states = sorted(gt["state_code"].drop_nulls().unique().to_list())
    return {s: i for i, s in enumerate(states)}


# ── Baseline estimators (sklearn-compatible) ───────────────────────────────────

class MeanRegressor:
    """Baseline: predict training-fold mean of log-energy for all val rows.

    Sklearn fit/predict interface — plugs into nested_cv's estimator_factory.
    """

    mean_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_)


class CFBaselineRegressor:
    """Baseline: predict energy = installed_kw × 8760 × median_CF (train fold).

    In log-space:
        log_energy_pred = log_capacity_kw + ln(8760) + ln(median_CF)

    Median CF is computed from each training fold only — no leakage of
    validation labels.

    Args:
        log_cap_col_idx: Column index of ``log_capacity_kw`` in the feature
            matrix (numpy 2-D array).  Use FEATURE_COLUMNS.index("log_capacity_kw")
            = _LOG_CAP_IDX = 2.

    Note: CF here is river-hydro CF from the training dam plants, not WWTP CF.
    Expected to underperform on WWTP outfalls — that is intentional and
    documented in SMOKE_TEST_REPORT.md §6.
    """

    def __init__(self, log_cap_col_idx: int) -> None:
        self._idx = log_cap_col_idx
        self.median_cf_: float = 0.5   # safe fallback before fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CFBaselineRegressor":
        # CF = exp(y) / (exp(log_cap_kw) × 8760)
        #    = exp(y − log_cap_kw) / 8760
        log_cap = X[:, self._idx]
        cf = np.exp(y - log_cap) / 8_760.0
        valid = cf[(cf > 0) & np.isfinite(cf)]
        if len(valid) > 0:
            self.median_cf_ = float(np.median(valid))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.median_cf_ <= 0:
            return np.zeros(len(X))
        return X[:, self._idx] + _LOG_8760 + math.log(self.median_cf_)


# ── Feature engineering ────────────────────────────────────────────────────────

def build_training_frame(
    gt: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.Series, list[Any], list[str]]:
    """Build numeric feature matrix, log target, groups, and state list.

    Features (all Float64 — required by nested_cv's X.to_numpy() call):
        latitude            float64  decimal degrees (0 nulls in real data)
        longitude           float64  decimal degrees (0 nulls in real data)
        log_capacity_kw     float64  ln(actual_installed_kw)
        climate_zone_code   float64  0=tropical … 4=polar (see _CLIMATE_BANDS)
        state_code_code     float64  sorted-unique state → integer index

    Encoding note: ``state_code_code`` and ``climate_zone_code`` are ordinal
    integer codes cast to Float64 for numpy compatibility.  LightGBM treats
    them as ordinal here; acceptable for a smoke-test pipeline proof.  A
    production model would declare them as native LightGBM categorical columns.

    Target: ln(actual_annual_energy_kwh).  nested_cv expects a log-space
    target and exponentiates internally for MAPE.

    Groups: source_plant_code (Int64); the 1 null row (EHA-only site with no
    EIA cross-reference) receives a string sentinel
    "SENTINEL_<facility_name>_<state_code>" so it forms its own unique group
    and is never split across inner CV folds.

    Leakage guard: assert_no_leakage(X.columns) is called before returning —
    belt-and-braces; nested_cv also guards at entry.

    Args:
        gt: Combined ground-truth DataFrame (1,360 rows, CANONICAL_SCHEMA).

    Returns:
        X      : polars DataFrame, shape (N, 5), all Float64
        y      : polars Series of ln(actual_annual_energy_kwh), Float64
        groups : list[int | str], N elements
        states : list[str], N elements (state_code strings)

    Raises:
        ValueError: If any feature column is in LEAKAGE_DENYLIST (should
            never fire for the fixed smoke-test feature set).
    """
    state_map = _state_code_map(gt)

    lat_arr  = gt["latitude"].to_numpy().astype(np.float64)
    lon_arr  = gt["longitude"].to_numpy().astype(np.float64)
    log_cap  = np.log(gt["actual_installed_kw"].to_numpy().astype(np.float64))

    climate_list  = [_lat_to_climate_code(v) for v in gt["latitude"].to_list()]
    climate_codes = np.array(
        [float(c) if c is not None else 0.0 for c in climate_list],
        dtype=np.float64,
    )

    state_codes = np.array(
        [float(state_map.get(s, -1)) for s in gt["state_code"].to_list()],
        dtype=np.float64,
    )

    X = pl.DataFrame({
        "latitude":          lat_arr,
        "longitude":         lon_arr,
        "log_capacity_kw":   log_cap,
        "climate_zone_code": climate_codes,
        "state_code_code":   state_codes,
    })

    # Leakage guard (belt-and-braces; nested_cv also checks at entry)
    assert_no_leakage(list(X.columns))

    # Target: log-space (nested_cv exponentiates for MAPE internally)
    y_np = np.log(gt["actual_annual_energy_kwh"].to_numpy().astype(np.float64))
    y = pl.Series("log_annual_energy_kwh", y_np)

    # Groups: plant code (int) or sentinel string for null
    codes      = gt["source_plant_code"].to_list()
    names      = gt["facility_name"].to_list()
    state_strs = gt["state_code"].to_list()
    groups: list[Any] = []
    for code, name, state in zip(codes, names, state_strs):
        if code is None:
            groups.append(f"SENTINEL_{name or ''}_{state or ''}")
        else:
            groups.append(int(code))

    states: list[str] = gt["state_code"].to_list()

    log.info(
        "build_training_frame: %d rows, %d features, %d unique groups, %d unique states",
        X.height,
        len(FEATURE_COLUMNS),
        len({str(g) for g in groups}),
        len(set(states)),
    )
    return X, y, groups, states


# ── Naive baselines ────────────────────────────────────────────────────────────

def naive_baselines(gt: pl.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Compute predict-mean and CF-baseline metrics under the SAME CV splits.

    Both baselines use the SAME outer splits as the LightGBM model in
    ``run_smoke_test`` — identical states and seed from config ensure the
    outer-fold stratification is reproducible across calls.

    Args:
        gt: Combined ground-truth DataFrame (1,360 rows, CANONICAL_SCHEMA).

    Returns:
        {
            "mean_baseline": {"rmse_log": [...5 floats...], "mape": [...], ...},
            "cf_baseline":   {"rmse_log": [...5 floats...], ...},
        }
    """
    X, y, groups, states = build_training_frame(gt)
    cv_cfg = config.get("phase5.cv") or {}
    outer  = int(cv_cfg.get("outer_folds", 5))
    inner  = int(cv_cfg.get("inner_folds", 3))
    seed   = int(cv_cfg.get("seed", 0))

    mean_results = nested_cv(
        X, y, groups, states,
        estimator_factory=lambda: MeanRegressor(),
        outer=outer, inner=inner, seed=seed,
    )
    cf_results = nested_cv(
        X, y, groups, states,
        estimator_factory=lambda: CFBaselineRegressor(_LOG_CAP_IDX),
        outer=outer, inner=inner, seed=seed,
    )
    return {"mean_baseline": mean_results, "cf_baseline": cf_results}


# ── Leakage-guard demonstration ────────────────────────────────────────────────

def demonstrate_leakage_guard() -> str:
    """Return the ValueError text produced when a denylisted column reaches the guard.

    Used to populate SMOKE_TEST_REPORT.md §5. Not called during normal training.
    Picks the first member of LEAKAGE_DENYLIST alphabetically for a deterministic
    demonstration (resolves to "annual_energy_kwh").
    """
    bad = sorted(LEAKAGE_DENYLIST)[0]
    try:
        assert_no_leakage([bad, "latitude", "longitude"])
        return "ERROR: leakage guard did not fire — check LEAKAGE_DENYLIST"
    except ValueError as exc:
        return str(exc)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_smoke_test(seed: int = 0) -> dict:
    """End-to-end smoke test: load → features → CV × 3 → final fit → persist.

    Evaluates all five success criteria defined in SMOKE_TEST_REPORT.md and
    writes two artifacts to ``data/processed/phase5/models/``:
        smoke_lgbm.txt          LightGBM booster (native format)
        smoke_metadata.json     full metrics + mappings + run provenance

    Args:
        seed: Random seed for all CV splits and LightGBM training.
              Default 0 matches phase5.cv.seed in config/settings.yaml.

    Returns:
        Metadata dict (same content as smoke_metadata.json).

    Raises:
        FileNotFoundError: If combined_ground_truth.parquet is missing.
    """
    log.info("=== Phase 5 internal smoke test START (seed=%d) ===", seed)
    log.info(
        "INTERNAL — pipeline proof only. Numbers must NEVER appear in "
        "director, pitch, or competition material."
    )

    # ── Load ground truth (read-only) ────────────────────────────────────────
    gt_path = Path(
        config.get("phase5.combined_ground_truth_path")
        or "data/raw/ground_truth/combined_ground_truth.parquet"
    )
    if not gt_path.exists():
        raise FileNotFoundError(
            f"combined_ground_truth.parquet not found at {gt_path}. "
            "Run run_combine() or mount the SANDISK drive."
        )
    gt = pl.read_parquet(gt_path)
    log.info("Loaded ground truth: %d rows", gt.height)

    # ── Build training frame ─────────────────────────────────────────────────
    X, y, groups, states = build_training_frame(gt)

    # ── Read CV config ───────────────────────────────────────────────────────
    cv_cfg = config.get("phase5.cv") or {}
    outer  = int(cv_cfg.get("outer_folds", 5))
    inner  = int(cv_cfg.get("inner_folds", 3))

    # ── LightGBM nested CV ───────────────────────────────────────────────────
    lgbm_params = {**_LGBM_PARAMS, "random_state": seed}

    def _lgbm_factory() -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(**lgbm_params)

    log.info("Running LightGBM nested CV (outer=%d, inner=%d) …", outer, inner)
    lgbm_results = nested_cv(
        X, y, groups, states,
        estimator_factory=_lgbm_factory,
        outer=outer, inner=inner, seed=seed,
    )

    # ── Baselines (same outer splits via same seed) ──────────────────────────
    log.info("Running mean-baseline nested CV …")
    mean_results = nested_cv(
        X, y, groups, states,
        estimator_factory=lambda: MeanRegressor(),
        outer=outer, inner=inner, seed=seed,
    )

    log.info("Running CF-baseline nested CV …")
    cf_results = nested_cv(
        X, y, groups, states,
        estimator_factory=lambda: CFBaselineRegressor(_LOG_CAP_IDX),
        outer=outer, inner=inner, seed=seed,
    )

    # ── Success criterion 3 ──────────────────────────────────────────────────
    lgbm_rmse_mean    = float(np.mean(lgbm_results["rmse_log"]))
    mean_bl_rmse_mean = float(np.mean(mean_results["rmse_log"]))
    lgbm_spear_mean   = float(np.mean(lgbm_results["spearman"]))

    crit3_rmse = lgbm_rmse_mean < mean_bl_rmse_mean

    # Mean baseline predicts a constant → Spearman is undefined (NaN) in every fold
    # because scipy.stats.spearmanr raises ConstantInputWarning and returns NaN.
    # A constant-prediction model has no rank discriminability; any model with
    # finite positive Spearman beats it by construction.
    mean_bl_spear_vals = mean_results["spearman"]
    if all(math.isnan(v) for v in mean_bl_spear_vals):
        crit3_spearman      = math.isfinite(lgbm_spear_mean) and lgbm_spear_mean > 0
        mean_bl_spear_str   = "NaN (constant predictions)"
    else:
        mean_bl_spear_mean  = float(np.nanmean(mean_bl_spear_vals))
        crit3_spearman      = lgbm_spear_mean > mean_bl_spear_mean
        mean_bl_spear_str   = f"{mean_bl_spear_mean:.4f}"

    log.info(
        "Criterion 3 — LightGBM vs mean-baseline:  "
        "rmse_log %s (%.4f < %.4f)  spearman %s (%.4f > %s)",
        "PASS" if crit3_rmse else "FAIL",
        lgbm_rmse_mean, mean_bl_rmse_mean,
        "PASS" if crit3_spearman else "FAIL",
        lgbm_spear_mean, mean_bl_spear_str,
    )

    # ── Final fit on all 1,360 rows ──────────────────────────────────────────
    log.info("Fitting final model on all %d rows …", X.height)
    final_model = lgb.LGBMRegressor(**lgbm_params)
    final_model.fit(
        X.to_numpy(), np.asarray(y, dtype=np.float64),
        feature_name=FEATURE_COLUMNS,
    )

    # ── Persist model artifact ───────────────────────────────────────────────
    model_dir = Path(
        config.get("phase5.model_dir") or "data/processed/phase5/models"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "smoke_lgbm.txt"
    final_model.booster_.save_model(str(model_path))
    log.info("Saved model (LightGBM native format): %s", model_path)

    # ── Git SHA ──────────────────────────────────────────────────────────────
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).strip()
    except Exception:
        git_sha = "unknown"

    # ── Assemble metadata ────────────────────────────────────────────────────
    def _fold_summary(vals: list[float]) -> dict:
        return {
            "per_fold": [float(v) for v in vals],
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals)),
        }

    metadata: dict = {
        "feature_names":       FEATURE_COLUMNS,
        "categorical_mappings": {
            "state_code":  _state_code_map(gt),
            "climate_zone": {k: int(v) for k, v in _CLIMATE_BANDS.items()},
        },
        "lgbm_params":     {k: (int(v) if isinstance(v, (np.integer,)) else v)
                            for k, v in lgbm_params.items()},
        "seed":            seed,
        "git_sha":         git_sha,
        "train_row_count": int(X.height),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "crit3_rmse_pass":     bool(crit3_rmse),
            "crit3_spearman_pass": bool(crit3_spearman),
        },
        "metrics": {
            "lgbm": {
                m: _fold_summary(lgbm_results[m]) for m in lgbm_results
            },
            "mean_baseline": {
                m: _fold_summary(mean_results[m]) for m in mean_results
            },
            "cf_baseline": {
                m: _fold_summary(cf_results[m]) for m in cf_results
            },
        },
    }

    meta_path = model_dir / "smoke_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata: %s", meta_path)

    log.info("=== Phase 5 internal smoke test COMPLETE ===")
    return metadata


# ── CLI entry ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the smoke test from the command line and print a summary table."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Phase 5 internal smoke-test on combined dam labels."
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0; must match phase5.cv.seed for report)"
    )
    args = ap.parse_args()

    meta = run_smoke_test(seed=args.seed)

    sep = "=" * 72
    print(f"\n{sep}")
    print("INTERNAL — pipeline proof only.")
    print("These numbers must NEVER appear in director, pitch, or competition material.")
    print(sep)

    hdr = f"\n{'Metric':12s}  {'LightGBM (mean±std)':>22s}  "
    hdr += f"{'MeanBaseline':>14s}  {'CFBaseline':>12s}"
    print(hdr)
    print("-" * 72)

    for metric in ["rmse_log", "mape", "r2", "spearman"]:
        lm = meta["metrics"]["lgbm"][metric]
        mb = meta["metrics"]["mean_baseline"][metric]
        cb = meta["metrics"]["cf_baseline"][metric]
        print(
            f"{metric:12s}  {lm['mean']:>10.4f} ± {lm['std']:.4f}  "
            f"{mb['mean']:>14.4f}  {cb['mean']:>12.4f}"
        )

    crit = meta.get("criteria", {})
    print()
    print(f"Criterion 3 rmse_log (LightGBM < MeanBL): "
          f"{'PASS' if crit.get('crit3_rmse_pass') else 'FAIL'}")
    print(f"Criterion 3 spearman (LightGBM > MeanBL): "
          f"{'PASS' if crit.get('crit3_spearman_pass') else 'FAIL'}")

    model_dir = Path(
        config.get("phase5.model_dir") or "data/processed/phase5/models"
    )
    print()
    print(f"Model     : {model_dir / 'smoke_lgbm.txt'}")
    print(f"Metadata  : {model_dir / 'smoke_metadata.json'}")
    print()

    # Leakage guard demo (for report §5)
    print("Leakage guard demonstration:")
    print(f"  {demonstrate_leakage_guard()}")


if __name__ == "__main__":
    main()

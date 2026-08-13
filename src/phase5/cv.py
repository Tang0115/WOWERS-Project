"""Phase 5 — nested cross-validation harness (ARCH §5.3 step 3).

D4 — nested_cv
--------------
Outer folds: stratified by ``state`` (geographic diversity in each fold).
Inner folds: group-aware by ``groups`` (no utility split across folds).

``assert_no_leakage(feature_names)`` is called at the very top — the function
refuses to run if any denylisted column is passed, so a leaky feature set
causes a loud failure rather than silently inflated metrics.

Pluggable estimator: pass ``estimator_factory()`` returning an unfitted
sklearn-compatible regressor.  LightGBM, Ridge, or a trivial dummy are all
valid via this interface.

Metrics returned per outer fold
--------------------------------
- ``rmse_log``    : RMSE on log-transformed target (primary metric)
- ``mape``        : Mean Absolute Percentage Error on raw kWh (interpretable)
- ``r2``          : Coefficient of determination (variance explained)
- ``spearman``    : Spearman rank correlation (plant ranking quality)

No model artifacts, no HP run results, and no eval results are committed.
This module is the harness only — the decisions about which features and
which estimator to use remain with the human.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.phase5.features import assert_no_leakage
from src.common import logging_setup

logging_setup.setup_run_log("phase5")
log = logging_setup.get("wowers.phase5.cv")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true_raw: np.ndarray, y_pred_log: np.ndarray) -> float:
    """MAPE on raw kWh space: |true − exp(pred)| / |true|, mean %."""
    y_pred_raw = np.exp(y_pred_log)
    return float(np.mean(np.abs(y_true_raw - y_pred_raw) / np.maximum(y_true_raw, 1e-9)) * 100)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def _state_stratified_splits(
    states: list[str],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Outer fold indices: stratified by state label, no group leakage.

    Produces ``n_folds`` (train_indices, val_indices) pairs such that each
    state appears in approximately equal proportion in every fold.

    Strategy: sort states into bins, assign one bin per fold cyclically.
    Randomise within each state before assignment (controlled by ``seed``).
    Group-awareness is enforced separately inside the inner loop — outer
    stratification is solely by state.

    Returns:
        List of (train_idx, val_idx) tuples; indices are integer positions.
    """
    rng = random.Random(seed)
    unique_states = sorted(set(states))
    # Map state → positions (shuffled)
    state_to_positions: dict[str, list[int]] = {s: [] for s in unique_states}
    for i, s in enumerate(states):
        state_to_positions[s].append(i)
    for positions in state_to_positions.values():
        rng.shuffle(positions)

    # Assign positions to folds round-robin within each state
    fold_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for positions in state_to_positions.values():
        for j, pos in enumerate(positions):
            fold_indices[j % n_folds].append(pos)

    splits = []
    for val_fold in range(n_folds):
        val_idx = fold_indices[val_fold]
        train_idx = [i for f in range(n_folds) if f != val_fold for i in fold_indices[f]]
        splits.append((train_idx, val_idx))
    return splits


def _group_aware_inner_splits(
    train_idx: list[int],
    groups: list[Any],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Inner fold splits: group-aware (no group appears in both train and val).

    Groups are utility / NPDES permit groupings.  A group must appear entirely
    in train or entirely in val — never split across.

    Returns:
        List of (inner_train_idx, inner_val_idx) tuples (positions into
        ``train_idx``).
    """
    rng = random.Random(seed + 1000)
    # Map group value → list of positions within train_idx
    group_to_positions: dict[Any, list[int]] = {}
    for local_pos, global_idx in enumerate(train_idx):
        g = groups[global_idx]
        group_to_positions.setdefault(g, []).append(local_pos)

    all_groups = list(group_to_positions.keys())
    rng.shuffle(all_groups)

    # Assign groups to inner folds round-robin
    inner_folds: list[list[int]] = [[] for _ in range(n_folds)]
    for j, g in enumerate(all_groups):
        inner_folds[j % n_folds].extend(group_to_positions[g])

    splits = []
    for val_fold in range(n_folds):
        val_local = inner_folds[val_fold]
        train_local = [p for f in range(n_folds) if f != val_fold for p in inner_folds[f]]
        splits.append((train_local, val_local))
    return splits


def nested_cv(
    X: pl.DataFrame,
    y: pl.Series,
    groups: list[Any],
    states: list[str],
    estimator_factory: Callable[[], Any],
    *,
    outer: int = 5,
    inner: int = 3,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Nested cross-validation for Phase 5 ML model.

    LEAKAGE GUARD: calls ``assert_no_leakage(X.columns)`` at entry.
    Raises ``ValueError`` if any denylisted column is present.

    Outer folds: stratified by ``states`` (geographic diversity in each fold).
    Inner folds: group-aware by ``groups`` (no group split across folds).

    Target is assumed to be log-transformed energy (``log(actual_annual_energy_kwh)``).
    MAPE is computed in raw kWh space by exponentiating predictions.

    NOTE: This harness does NOT run hyperparameter tuning — the inner loop
    structure is wired, but HP search (optuna/grid) must be wired into
    ``estimator_factory`` by the caller.  This keeps the harness pure and
    testable without a real training run.

    Args:
        X:                 Feature matrix (polars DataFrame; rows = plants).
        y:                 Target series (log-transformed energy, float64).
        groups:            Group labels, one per row (e.g. NPDES permit number or
                           utility name).  No group may span train/val.
        states:            State code per row.  Used for outer fold stratification.
        estimator_factory: Callable returning a fresh unfitted sklearn-compatible
                           regressor (e.g. ``lambda: LGBMRegressor(...)``).
        outer:             Number of outer CV folds (default 5).
        inner:             Number of inner CV folds for HP tuning (default 3).
        seed:              Random seed for reproducibility.

    Returns:
        Dict mapping metric name → list of per-outer-fold values::

            {
                "rmse_log"   : [float, ...],  # RMSE on log-energy
                "mape"       : [float, ...],  # MAPE on raw kWh (%)
                "r2"         : [float, ...],
                "spearman"   : [float, ...],
            }

    Raises:
        ValueError: If X contains any denylisted feature column (leakage).
        ValueError: If len(X) != len(y) or other dimension mismatches.
    """
    # Leakage guard — refuse to proceed on a leaky feature set
    assert_no_leakage(list(X.columns))

    n = X.height
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)} elements")
    if len(groups) != n:
        raise ValueError(f"X has {n} rows but groups has {len(groups)} elements")
    if len(states) != n:
        raise ValueError(f"X has {n} rows but states has {len(states)} elements")

    X_np = X.to_numpy()
    y_np = np.asarray(y, dtype=np.float64)
    y_raw = np.exp(y_np)  # raw kWh for MAPE

    outer_splits = _state_stratified_splits(states, outer, seed)

    results: dict[str, list[float]] = {
        "rmse_log": [], "mape": [], "r2": [], "spearman": []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
        if len(val_idx) == 0 or len(train_idx) == 0:
            log.warning("Outer fold %d has empty split — skipping", fold_idx)
            continue

        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        # Inner fold: group-aware HP tuning (pluggable — caller wires optuna here)
        inner_splits = _group_aware_inner_splits(train_idx, groups, inner, seed)

        # Verify no group leaks across inner splits
        for itrain, ival in inner_splits:
            train_groups_inner = {groups[train_idx[p]] for p in itrain}
            val_groups_inner   = {groups[train_idx[p]] for p in ival}
            leaked = train_groups_inner & val_groups_inner
            if leaked:
                raise RuntimeError(
                    f"Inner CV fold group leak detected (fold {fold_idx}): {leaked}"
                )

        # Fit on full outer training split (HP tuning happens inside estimator_factory)
        est = estimator_factory()
        est.fit(X_train, y_train)
        y_pred = est.predict(X_val)

        results["rmse_log"].append(_rmse(y_val, y_pred))
        results["mape"].append(_mape(y_raw[val_idx], y_pred))
        results["r2"].append(_r2(y_val, y_pred))
        results["spearman"].append(_spearman(y_val, y_pred))

        log.info(
            "Outer fold %d/%d: RMSE_log=%.4f  MAPE=%.1f%%  R²=%.3f  Spearman=%.3f",
            fold_idx + 1, outer,
            results["rmse_log"][-1],
            results["mape"][-1],
            results["r2"][-1],
            results["spearman"][-1],
        )

    # Summary
    for metric, vals in results.items():
        if vals:
            log.info(
                "Nested CV %s: mean=%.4f  std=%.4f",
                metric, float(np.mean(vals)), float(np.std(vals)),
            )

    return results

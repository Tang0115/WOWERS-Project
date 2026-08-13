"""Tests for Phase 2 Monte Carlo exclusion filter — W13 small-POTW gate.

Also covers the P2-SEED site-keyed seeding invariants (2026-07-06 incident):
prior positional seeding (seed + row_index) meant that removing WI0025194 from
17,158 → 17,148 facilities re-drew ~1,090 scorecard rows and flipped 3 viability
calls.  The site-keyed scheme (sha256(npdes_id) XOR base_seed) makes per-facility
draws independent of row position.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.phase2.monte_carlo import (
    _exclude, _site_seed_sequence, estimate_all_facilities,
    _MIN_VIABLE_FLOW_MGD, _OUTPUT_KEYS,
)


# ── _exclude: zero / null flow ─────────────────────────────────────────────────

class TestExcludeNoUsableFlow:
    def test_none_flow_excluded(self):
        assert _exclude({"mean_flow_mgd": None}) == "no_usable_flow"

    def test_zero_flow_excluded(self):
        assert _exclude({"mean_flow_mgd": 0.0}) == "no_usable_flow"

    def test_negative_flow_excluded(self):
        assert _exclude({"mean_flow_mgd": -1.0}) == "no_usable_flow"

    def test_nan_flow_excluded(self):
        """NaN slips past None check and <= 0 check — must be caught explicitly."""
        assert _exclude({"mean_flow_mgd": float("nan")}) == "no_usable_flow"

    def test_missing_key_excluded(self):
        """Row dict with no mean_flow_mgd key → .get() returns None → excluded."""
        assert _exclude({}) == "no_usable_flow"

    def test_string_flow_excluded(self):
        """Non-numeric value in mean_flow_mgd field → excluded (isinstance guard)."""
        assert _exclude({"mean_flow_mgd": "1.5"}) == "no_usable_flow"


# ── _exclude: W13 small-POTW gate ─────────────────────────────────────────────

class TestExcludeSmallPotw:
    def test_below_threshold_excluded(self):
        row = {"mean_flow_mgd": _MIN_VIABLE_FLOW_MGD - 0.01}
        assert _exclude(row) == "small_potw"

    def test_well_below_threshold_excluded(self):
        row = {"mean_flow_mgd": 0.1}
        assert _exclude(row) == "small_potw"

    def test_at_threshold_not_excluded(self):
        """Boundary: mean_flow == threshold passes (filter is strictly < threshold)."""
        row = {"mean_flow_mgd": _MIN_VIABLE_FLOW_MGD}
        assert _exclude(row) is None

    def test_above_threshold_not_excluded(self):
        row = {"mean_flow_mgd": 1.0}
        assert _exclude(row) is None

    def test_large_flow_not_excluded(self):
        row = {"mean_flow_mgd": 500.0}
        assert _exclude(row) is None

    def test_threshold_value_matches_config(self):
        """_MIN_VIABLE_FLOW_MGD should match settings.yaml phase2.min_viable_mean_flow_mgd."""
        from src.common import config
        expected = float(config.get("phase2.min_viable_mean_flow_mgd", 0.5))
        assert _MIN_VIABLE_FLOW_MGD == expected


# ── _exclude: sparse DMR gate (regression — W13 must not break existing gate) ──

class TestExcludeSparseDmr:
    def test_dmr_limited_sparse_excluded(self):
        row = {"mean_flow_mgd": 2.0, "data_quality": "dmr_limited", "n_months_data": 2}
        assert _exclude(row) == "sparse_dmr_artifact"

    def test_dmr_limited_sufficient_not_excluded(self):
        row = {"mean_flow_mgd": 2.0, "data_quality": "dmr_limited", "n_months_data": 12}
        assert _exclude(row) is None

    def test_normal_dmr_not_excluded(self):
        row = {"mean_flow_mgd": 2.0, "data_quality": "dmr", "n_months_data": 120}
        assert _exclude(row) is None

    def test_dmr_limited_none_months_excluded(self):
        """n_months_data == None with dmr_limited → sparse_dmr_artifact."""
        row = {"mean_flow_mgd": 2.0, "data_quality": "dmr_limited", "n_months_data": None}
        assert _exclude(row) == "sparse_dmr_artifact"

    def test_missing_data_quality_key_not_excluded(self):
        """data_quality key absent → defaults to 'design_only' → not dmr_limited → passes."""
        row = {"mean_flow_mgd": 2.0}
        assert _exclude(row) is None


# ── Priority order: no_usable_flow before small_potw ──────────────────────────

class TestExcludePriority:
    def test_zero_flow_takes_priority_over_small_potw(self):
        """Flow == 0 → no_usable_flow, not small_potw (zero check runs first)."""
        row = {"mean_flow_mgd": 0.0}
        assert _exclude(row) == "no_usable_flow"

    def test_small_potw_takes_priority_over_sparse_dmr(self):
        """Flow < threshold with sparse DMR → small_potw reported first."""
        row = {"mean_flow_mgd": 0.3, "data_quality": "dmr_limited", "n_months_data": 1}
        assert _exclude(row) == "small_potw"


# ── estimate_all_facilities: integration smoke test ───────────────────────────

class TestEstimateAllFacilities:
    # Minimal schema — real Phase 1 parquet has more columns (facility_name, state, etc.)
    # _process_one only reads npdes_id, mean_flow_mgd, design_flow_mgd,
    # flow_duration_curve, data_quality, n_months_data — so minimal schema is intentional.

    def test_small_potw_excluded_in_batch(self):
        """estimate_all_facilities marks small sites excluded=True, exclusion_reason=small_potw."""
        from src.phase2.monte_carlo import estimate_all_facilities

        # A002 uses flat FDC fallback (None); A003 uses realistic 20-point FDC
        fdc_20pt = [5.0 - i * 0.2 for i in range(20)]  # 5.0 → 1.2 MGD descending

        candidates = pl.DataFrame({
            "npdes_id":          ["A001", "A002", "A003"],
            "mean_flow_mgd":     [0.2,    1.0,    5.0],
            "design_flow_mgd":   [0.3,    1.5,    6.0],
            "flow_duration_curve": [None,  None,  fdc_20pt],
            "data_quality":      ["dmr",  "dmr",  "dmr"],
            "n_months_data":     [60,     60,     60],
        })

        results = estimate_all_facilities(candidates, n_iterations=100, seed=0)

        a001 = results.filter(pl.col("npdes_id") == "A001").to_dicts()[0]
        a002 = results.filter(pl.col("npdes_id") == "A002").to_dicts()[0]
        a003 = results.filter(pl.col("npdes_id") == "A003").to_dicts()[0]

        # Excluded site: correct flags, all energy fields None
        assert a001["excluded"] is True
        assert a001["exclusion_reason"] == "small_potw"
        assert a001["energy_p50_kwh_yr"] is None

        # Success path: not excluded, energy and archetype populated
        assert a002["excluded"] is False
        assert a002["energy_p50_kwh_yr"] is not None
        assert a002["archetype"] is not None

        # Success path with real 20-pt FDC: same contracts hold
        assert a003["excluded"] is False
        assert a003["energy_p50_kwh_yr"] is not None

    def test_exclusion_count_reflects_small_potw_filter(self):
        """All facilities below threshold → all excluded."""
        from src.phase2.monte_carlo import estimate_all_facilities

        candidates = pl.DataFrame({
            "npdes_id":          ["X001", "X002"],
            "mean_flow_mgd":     [0.1,    0.4],
            "design_flow_mgd":   [0.2,    0.5],
            "flow_duration_curve": [None, None],
            "data_quality":      ["dmr",  "dmr"],
            "n_months_data":     [24,     24],
        })

        results = estimate_all_facilities(candidates, n_iterations=100, seed=0)
        excluded = results.filter(pl.col("excluded") == True)
        assert len(excluded) == 2
        reasons = set(excluded["exclusion_reason"].to_list())
        assert reasons == {"small_potw"}

    def test_excluded_row_schema_matches_success_row(self):
        """Excluded and non-excluded rows must have identical column sets (no drift)."""
        candidates = pl.DataFrame({
            "npdes_id":          ["S001", "S002"],
            "mean_flow_mgd":     [0.2,    2.0],
            "design_flow_mgd":   [0.3,    3.0],
            "flow_duration_curve": [None, None],
            "data_quality":      ["dmr",  "dmr"],
            "n_months_data":     [60,     60],
        })

        results = estimate_all_facilities(candidates, n_iterations=50, seed=0)
        assert len(results) == 2
        # Column set must exactly match _OUTPUT_KEYS — catches silent drift if
        # success-path dict gains a new key that exclusion rows don't emit.
        assert set(results.columns) == set(_OUTPUT_KEYS)


# ── P2-SEED: site-keyed seeding invariants ────────────────────────────────────


def _viable_candidates(ids: list[str], flows: list[float]) -> pl.DataFrame:
    """Build a minimal Phase-1-like DataFrame for estimate_all_facilities.

    All facilities are above the W13 threshold so none are excluded —
    this keeps tests focused on energy values, not exclusion logic.
    """
    return pl.DataFrame({
        "npdes_id":            ids,
        "mean_flow_mgd":       [float(f) for f in flows],
        "design_flow_mgd":     [float(f) * 1.5 for f in flows],
        "flow_duration_curve": [None] * len(ids),
        "data_quality":        ["dmr"] * len(ids),
        "n_months_data":       [60] * len(ids),
    })


class TestSiteKeyedSeed:
    """Invariance tests for site-keyed Monte Carlo seeding (P2-SEED fix).

    The 2026-07-06 P1-COORD-GUARD incident removed 10 bad-coordinate rows from
    Phase 1.  Because seeds were positional (seed + row_index), removing rows
    shifted seeds for all subsequent facilities, re-drawing ~1,090 scorecard rows
    and flipping 3 viability calls.  Site-keyed seeding (sha256(npdes_id) mixed
    with base_seed) makes each facility's draw independent of row position.
    """

    N_ITER = 2_000   # enough for stable p50 estimates; fast enough for CI

    def test_determinism_same_run_twice(self):
        """Identical inputs + seed → bit-identical energy outputs on two runs."""
        cands = _viable_candidates(["A001", "B001", "C001"], [2.0, 3.0, 5.0])
        r1 = estimate_all_facilities(cands, n_iterations=self.N_ITER, seed=42)
        r2 = estimate_all_facilities(cands, n_iterations=self.N_ITER, seed=42)
        for col in ["energy_p50_kwh_yr", "energy_p10_kwh_yr", "energy_p90_kwh_yr"]:
            assert r1[col].to_list() == r2[col].to_list(), (
                f"Non-deterministic output in {col}"
            )

    def test_removal_invariance(self):
        """Removing B from [A,B,C] does NOT change A's or C's energy estimates.

        This test WOULD FAIL on the old positional scheme (seed + row_index):
          - In [A,B,C]: A→seed+0, B→seed+1, C→seed+2
          - In [A,C]:   A→seed+0,           C→seed+1  ← different seed → different draws
        With site-keyed seeding:
          - A's seed = f(base_seed, "A001") regardless of B's presence.
          - C's seed = f(base_seed, "C001") regardless of B's presence.

        Root cause of 2026-07-06 incident: cited here so the test documents why
        this invariant matters for production correctness.
        """
        cands_abc = _viable_candidates(["A001", "B001", "C001"], [2.0, 3.0, 5.0])
        cands_ac  = _viable_candidates(["A001", "C001"],          [2.0,       5.0])

        r_abc = estimate_all_facilities(cands_abc, n_iterations=self.N_ITER, seed=42)
        r_ac  = estimate_all_facilities(cands_ac,  n_iterations=self.N_ITER, seed=42)

        a_in_abc = r_abc.filter(pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        a_in_ac  = r_ac.filter( pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        c_in_abc = r_abc.filter(pl.col("npdes_id") == "C001")["energy_p50_kwh_yr"][0]
        c_in_ac  = r_ac.filter( pl.col("npdes_id") == "C001")["energy_p50_kwh_yr"][0]

        assert a_in_abc == a_in_ac, (
            f"A001 energy_p50 changed when B001 removed: {a_in_abc} vs {a_in_ac}. "
            "Positional seeding would cause this; site-keyed seeding must not."
        )
        assert c_in_abc == c_in_ac, (
            f"C001 energy_p50 changed when B001 removed: {c_in_abc} vs {c_in_ac}."
        )

    def test_insertion_invariance(self):
        """Inserting B into [A,C] does NOT change A's or C's energy estimates."""
        cands_ac  = _viable_candidates(["A001", "C001"],          [2.0,       5.0])
        cands_abc = _viable_candidates(["A001", "B001", "C001"], [2.0, 3.0, 5.0])

        r_ac  = estimate_all_facilities(cands_ac,  n_iterations=self.N_ITER, seed=42)
        r_abc = estimate_all_facilities(cands_abc, n_iterations=self.N_ITER, seed=42)

        a_in_ac  = r_ac.filter( pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        a_in_abc = r_abc.filter(pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        c_in_ac  = r_ac.filter( pl.col("npdes_id") == "C001")["energy_p50_kwh_yr"][0]
        c_in_abc = r_abc.filter(pl.col("npdes_id") == "C001")["energy_p50_kwh_yr"][0]

        assert a_in_ac == a_in_abc
        assert c_in_ac == c_in_abc

    def test_different_base_seed_gives_different_draws(self):
        """Changing base_seed changes energy estimates (deliberate re-baseline)."""
        cands = _viable_candidates(["A001"], [5.0])
        r42 = estimate_all_facilities(cands, n_iterations=self.N_ITER, seed=42)
        r99 = estimate_all_facilities(cands, n_iterations=self.N_ITER, seed=99)
        e42 = r42.filter(pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        e99 = r99.filter(pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        assert e42 != e99, (
            "Different base seeds must produce different draws; "
            f"got identical energy {e42} for seeds 42 and 99."
        )

    def test_different_npdes_id_gives_different_draws(self):
        """Same flow, different npdes_id → different seeds → different energy."""
        cands = _viable_candidates(["A001", "A002"], [5.0, 5.0])
        r = estimate_all_facilities(cands, n_iterations=self.N_ITER, seed=42)
        e_a = r.filter(pl.col("npdes_id") == "A001")["energy_p50_kwh_yr"][0]
        e_b = r.filter(pl.col("npdes_id") == "A002")["energy_p50_kwh_yr"][0]
        assert e_a != e_b, (
            "Different NPDES IDs must produce different random draws "
            f"(both same flow=5.0 MGD, but got identical energy {e_a})."
        )

    def test_sequential_equals_parallel(self):
        """n_workers=1 (sequential) and n_workers=2 (parallel) → identical outputs."""
        cands = _viable_candidates(
            ["A001", "B001", "C001", "D001"], [2.0, 3.0, 5.0, 8.0]
        )
        r_seq = estimate_all_facilities(cands, n_iterations=self.N_ITER, n_workers=1, seed=42)
        r_par = estimate_all_facilities(cands, n_iterations=self.N_ITER, n_workers=2, seed=42)

        for col in ["energy_p50_kwh_yr", "energy_p10_kwh_yr", "energy_p90_kwh_yr"]:
            seq_vals = r_seq.sort("npdes_id")[col].to_list()
            par_vals = r_par.sort("npdes_id")[col].to_list()
            assert seq_vals == par_vals, (
                f"Sequential and parallel paths differ on {col}. "
                "Old parallel scheme used seed + batch_offset; new scheme uses same base_seed."
            )


class TestSiteKeySeedSequence:
    """Unit tests for _site_seed_sequence — the core seeding primitive."""

    def test_same_inputs_deterministic(self):
        """Same (base_seed, npdes_id) always produces same SeedSequence."""
        import numpy as np
        s1 = _site_seed_sequence(42, "TX0001234")
        s2 = _site_seed_sequence(42, "TX0001234")
        rng1 = np.random.default_rng(s1)
        rng2 = np.random.default_rng(s2)
        assert rng1.random() == rng2.random()

    def test_different_base_seed_different_draw(self):
        """Changing base_seed changes the RNG draw."""
        import numpy as np
        s1 = np.random.default_rng(_site_seed_sequence(42, "TX0001234")).random()
        s2 = np.random.default_rng(_site_seed_sequence(99, "TX0001234")).random()
        assert s1 != s2

    def test_different_npdes_id_different_draw(self):
        """Different npdes_id → different draw even with same base_seed."""
        import numpy as np
        s1 = np.random.default_rng(_site_seed_sequence(42, "TX0001234")).random()
        s2 = np.random.default_rng(_site_seed_sequence(42, "TX0001235")).random()
        assert s1 != s2

    def test_returns_seed_sequence(self):
        """_site_seed_sequence returns a np.random.SeedSequence."""
        import numpy as np
        result = _site_seed_sequence(42, "TEST001")
        assert isinstance(result, np.random.SeedSequence)

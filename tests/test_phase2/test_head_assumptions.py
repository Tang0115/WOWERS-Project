"""Tests for Phase 2 head assumption classification and distribution lookup."""

import pytest

from src.phase2.head_assumptions import (
    HeadDistribution,
    classify_archetype,
    get_head_distribution,
    head_params_for_flow,
)


class TestClassifyArchetype:
    def test_large_potw_above_10_mgd(self):
        assert classify_archetype(15.0) == "large_potw"

    def test_large_potw_just_above_threshold(self):
        assert classify_archetype(10.01) == "large_potw"

    def test_medium_potw_at_upper_boundary(self):
        assert classify_archetype(10.0) == "medium_potw"

    def test_medium_potw_mid_range(self):
        assert classify_archetype(5.0) == "medium_potw"

    def test_medium_potw_at_lower_boundary(self):
        assert classify_archetype(1.0) == "medium_potw"

    def test_small_potw_below_1_mgd(self):
        assert classify_archetype(0.5) == "small_potw"

    def test_small_potw_near_zero(self):
        assert classify_archetype(0.001) == "small_potw"

    def test_none_flow_defaults_to_small(self):
        assert classify_archetype(None) == "small_potw"

    def test_zero_flow_defaults_to_small(self):
        assert classify_archetype(0.0) == "small_potw"

    def test_negative_flow_defaults_to_small(self):
        assert classify_archetype(-5.0) == "small_potw"


class TestGetHeadDistribution:
    def test_large_returns_distribution(self):
        d = get_head_distribution("large_potw")
        assert isinstance(d, HeadDistribution)
        # low < mode < high (triangular constraint)
        assert d.low_m < d.mode_m < d.high_m

    def test_medium_returns_distribution(self):
        d = get_head_distribution("medium_potw")
        assert d.low_m < d.mode_m < d.high_m

    def test_small_returns_distribution(self):
        d = get_head_distribution("small_potw")
        assert d.low_m < d.mode_m < d.high_m

    def test_large_head_greater_than_small(self):
        large = get_head_distribution("large_potw")
        small = get_head_distribution("small_potw")
        assert large.mode_m > small.mode_m

    def test_unknown_archetype_raises_key_error(self):
        with pytest.raises(KeyError):
            get_head_distribution("nonexistent_archetype")

    def test_head_values_physically_plausible(self):
        for arch in ("large_potw", "medium_potw", "small_potw"):
            d = get_head_distribution(arch)
            assert d.low_m >= 0.5, f"{arch}: low_m too small"
            assert d.high_m <= 50.0, f"{arch}: high_m unrealistically large for POTW"


class TestHeadParamsForFlow:
    def test_convenience_wrapper_matches_full_pipeline(self):
        for flow_mgd in (0.1, 3.0, 50.0):
            archetype = classify_archetype(flow_mgd)
            expected  = get_head_distribution(archetype)
            actual    = head_params_for_flow(flow_mgd)
            assert actual == expected

    def test_none_flow_works(self):
        d = head_params_for_flow(None)
        assert isinstance(d, HeadDistribution)

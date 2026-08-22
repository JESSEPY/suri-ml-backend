import pytest
from ml.confidence import (
    apply_sample_size_floor, confidence_from_metrics, downgrade_tier,
    LOW_SAMPLE_THRESHOLD, TIER_PCT_CEILING,
)


class TestConfidenceFromMetrics:
    @pytest.mark.parametrize("mae,mean_actual,expected_tier", [
        (1.0, 100.0, "high"),      # 1% error
        (14.9, 100.0, "high"),     # just under 15%
        (15.0, 100.0, "high"),     # exactly at the boundary -> high (spec: <=15% is high, inclusive)
        (15.1, 100.0, "fair"),
        (40.0, 100.0, "fair"),     # exactly at the fair/limited boundary -> fair (spec: <=40% is fair)
        (40.1, 100.0, "limited"),
        (90.0, 100.0, "limited"),
    ])
    def test_tier_boundaries(self, mae, mean_actual, expected_tier):
        tier, _ = confidence_from_metrics(mae, mean_actual)
        assert tier == expected_tier

    def test_pct_is_100_minus_normalized_error(self):
        _, pct = confidence_from_metrics(mae=10.0, mean_actual=100.0)
        assert pct == 90

    def test_pct_never_exceeds_100(self):
        _, pct = confidence_from_metrics(mae=0.0, mean_actual=100.0)
        assert pct == 100

    def test_pct_floors_at_5_not_0_or_negative(self):
        """A model doing much worse than the mean (mae > mean_actual) would
        otherwise produce a negative or zero percentage, which reads as
        nonsensical rather than 'very low confidence'."""
        _, pct = confidence_from_metrics(mae=500.0, mean_actual=100.0)
        assert pct == 5

    def test_zero_mean_actual_is_defensive_limited(self):
        """Shouldn't happen for a real fertilizer target (always > 0 on
        average), but must not divide by zero if it ever does."""
        tier, pct = confidence_from_metrics(mae=1.0, mean_actual=0.0)
        assert tier == "limited"
        assert pct == 5


class TestDowngradeTier:
    def test_high_downgrades_to_fair(self):
        assert downgrade_tier("high") == "fair"

    def test_fair_downgrades_to_limited(self):
        assert downgrade_tier("fair") == "limited"

    def test_limited_stays_limited(self):
        assert downgrade_tier("limited") == "limited"


class TestApplySampleSizeFloor:
    """A tier computed from very few held-out rows isn't trustworthy no
    matter how good its error looks -- a single test row can produce any
    normalized error at all, good or bad, by chance. See
    docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md's
    successor note on this (added alongside this floor)."""

    def test_below_threshold_forces_limited(self):
        tier, pct = apply_sample_size_floor("high", 91, n=3)
        assert tier == "limited"

    def test_below_threshold_caps_pct_at_the_limited_ceiling(self):
        tier, pct = apply_sample_size_floor("high", 91, n=1)
        assert pct == TIER_PCT_CEILING["limited"]

    def test_at_threshold_is_not_downgraded(self):
        tier, pct = apply_sample_size_floor("high", 91, n=LOW_SAMPLE_THRESHOLD)
        assert (tier, pct) == ("high", 91)

    def test_above_threshold_is_unchanged(self):
        assert apply_sample_size_floor("fair", 70, n=170) == ("fair", 70)

    def test_already_limited_below_threshold_is_unchanged(self):
        assert apply_sample_size_floor("limited", 20, n=1) == ("limited", 20)

    def test_none_n_is_unchanged(self):
        """n is None for artifacts trained before sample-count tracking
        existed -- treated as "unknown", not "too few", since forcing every
        crop in an old artifact to Limited would be a false downgrade, not
        an honest one."""
        assert apply_sample_size_floor("high", 91, n=None) == ("high", 91)


class TestTierPctCeiling:
    def test_every_tier_has_a_ceiling(self):
        assert set(TIER_PCT_CEILING.keys()) == {"high", "fair", "limited"}

    def test_ceilings_are_descending(self):
        assert TIER_PCT_CEILING["limited"] < TIER_PCT_CEILING["fair"] < TIER_PCT_CEILING["high"]

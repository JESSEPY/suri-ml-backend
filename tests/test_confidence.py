import pytest
from ml.confidence import confidence_from_metrics, downgrade_tier, TIER_PCT_CEILING


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


class TestTierPctCeiling:
    def test_every_tier_has_a_ceiling(self):
        assert set(TIER_PCT_CEILING.keys()) == {"high", "fair", "limited"}

    def test_ceilings_are_descending(self):
        assert TIER_PCT_CEILING["limited"] < TIER_PCT_CEILING["fair"] < TIER_PCT_CEILING["high"]

import pytest
from data.crop_baselines import CROP_BASELINES, CROPS
from ml.agronomy import (
    total_nitrogen_from_om, ph_p_penalty, nutrient_requirements,
)


class TestTotalNitrogen:
    def test_matches_real_scc417_report(self):
        """The real report shows OM 3.21% -> Total N 0.1605%, exactly OM x 0.05."""
        assert total_nitrogen_from_om(3.21) == pytest.approx(0.1605)

    def test_is_exactly_five_percent_of_om(self):
        assert total_nitrogen_from_om(2.0) == pytest.approx(0.10)


class TestPhPhosphorusPenalty:
    def test_no_penalty_at_optimal_ph(self):
        """Phosphorus availability peaks at pH 6.5 — no extra P needed."""
        assert ph_p_penalty(6.5) == pytest.approx(1.0)

    def test_penalty_is_symmetric_around_optimum(self):
        assert ph_p_penalty(5.5) == pytest.approx(ph_p_penalty(7.5))

    def test_penalty_increases_away_from_optimum(self):
        assert ph_p_penalty(8.5) > ph_p_penalty(7.5) > ph_p_penalty(6.5)

    def test_penalty_is_capped_at_thirty_percent(self):
        assert ph_p_penalty(14.0) == pytest.approx(1.30)
        assert ph_p_penalty(0.0) == pytest.approx(1.30)


class TestNutrientRequirements:
    def test_golden_record_scc417_within_tolerance(self):
        """Real report: Sorghum, pH 8.17, OM 3.21, P 18.96, K 138 -> 50/40/60.
        The generator only approximates DA's real rule, so tolerance is +/-15
        kg/ha — but a miss wider than that means the formulas disagree with real
        agronomic practice badly enough to investigate."""
        n, p, k = nutrient_requirements("Sorghum", 8.17, 3.21, 18.96, 138.0)
        assert abs(n - 50) <= 15
        assert abs(p - 40) <= 15
        assert abs(k - 60) <= 15

    def test_returns_whole_numbers(self):
        n, p, k = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 120.0)
        assert isinstance(n, int) and isinstance(p, int) and isinstance(k, int)

    def test_high_om_reduces_nitrogen(self):
        low = nutrient_requirements("Corn", 6.5, 1.0, 15.0, 120.0)[0]
        high = nutrient_requirements("Corn", 6.5, 5.0, 15.0, 120.0)[0]
        assert high < low

    def test_high_phosphorus_reduces_p2o5(self):
        low = nutrient_requirements("Corn", 6.5, 2.5, 5.0, 120.0)[1]
        high = nutrient_requirements("Corn", 6.5, 2.5, 35.0, 120.0)[1]
        assert high < low

    def test_high_potassium_reduces_k2o(self):
        low = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 40.0)[2]
        high = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 240.0)[2]
        assert high < low

    def test_extreme_ph_increases_p2o5(self):
        """The non-linear pH penalty is the specific reason this pipeline needs
        a tree model rather than a linear one."""
        optimal = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 120.0)[1]
        alkaline = nutrient_requirements("Corn", 8.5, 2.5, 15.0, 120.0)[1]
        assert alkaline > optimal

    @pytest.mark.parametrize("crop", CROPS)
    def test_all_crops_produce_clamped_values(self, crop):
        for ph, om, p, k in [(4.5, 0.5, 3.0, 30.0), (8.5, 6.0, 40.0, 250.0)]:
            n, p2o5, k2o = nutrient_requirements(crop, ph, om, p, k)
            assert 20 <= n <= 200
            assert 20 <= p2o5 <= 150
            assert 20 <= k2o <= 200

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", 6.5, 2.5, 15.0, 120.0)


class TestCropBaselines:
    def test_sorghum_anchored_to_real_report(self):
        assert CROP_BASELINES["Sorghum"] == (50, 40, 60)

    def test_crops_list_matches_baseline_keys(self):
        assert set(CROPS) == set(CROP_BASELINES.keys())

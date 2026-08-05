import pytest

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from data.nutrient_recommendation_guide import CROPS


class TestTotalNitrogenFromOm:
    def test_derives_total_n_at_the_lab_s_five_percent_ratio(self):
        assert total_nitrogen_from_om(2.0) == pytest.approx(0.10)

    def test_zero_om_is_zero_n(self):
        assert total_nitrogen_from_om(0.0) == 0.0


class TestNutrientRequirements:
    def test_returns_a_numeric_triple(self):
        n, p, k = nutrient_requirements("Corn", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert isinstance(n, float) and isinstance(p, float) and isinstance(k, float)

    def test_tree_crop_rates_are_not_rounded_to_zero(self):
        """See data/nutrient_recommendation_guide.py's nutrient_requirements
        docstring — this must stay a float pass-through, not re-introduce
        an int cast at this delegation layer."""
        n, p, k = nutrient_requirements("Banana", om=1.0, p_ppm=3.0, k_cmol_kg=0.1)
        assert n > 0 and p > 0 and k > 0

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)

    def test_variety_with_no_matching_row_falls_back_gracefully(self):
        """Cassava has no variety-specific guide rows — passing one must not
        raise, since resolve_variety/(guide row keying) for Cassava ignores
        variety entirely."""
        without = nutrient_requirements("Cassava", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        with_variety = nutrient_requirements("Cassava", om=2.5, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert without == with_variety

    def test_known_variety_changes_the_result(self):
        opv = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="OPV")
        hybrid = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert opv != hybrid

    @pytest.mark.parametrize("crop", CROPS)
    def test_every_supported_crop_resolves_without_error(self, crop):
        n, p, k = nutrient_requirements(crop, om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert n >= 0 and p >= 0 and k >= 0

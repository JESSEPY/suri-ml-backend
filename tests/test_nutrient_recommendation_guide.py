import pytest

from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, target_unit_for_crop, resolve_variety,
    resolve_cultivation_method, resolve_age_band, nutrient_requirements,
)


class TestKConversion:
    def test_standard_conversion_factor(self):
        """1 cmol(+)/kg K = 391 mg/kg (ppm): equivalent weight 39.1 g/mol, charge 1."""
        assert k_cmol_to_ppm(1.0) == pytest.approx(391.0)

    def test_zero_stays_zero(self):
        assert k_cmol_to_ppm(0.0) == 0.0


class TestTargetUnit:
    @pytest.mark.parametrize("crop", ["Rice", "Corn", "Cassava", "Peanut", "Tomato"])
    def test_field_crops_are_kg_ha(self, crop):
        assert target_unit_for_crop(crop) == "kg_ha"

    @pytest.mark.parametrize("crop", ["Mango", "Cacao", "Coconut", "Banana"])
    def test_tree_crops_are_kg_tree(self, crop):
        assert target_unit_for_crop(crop) == "kg_tree"

    def test_every_supported_crop_has_a_unit(self):
        for crop in CROPS:
            assert target_unit_for_crop(crop) in ("kg_ha", "kg_tree")


class TestResolvers:
    def test_variety_none_becomes_the_string_none(self):
        assert resolve_variety("Cassava", None) == "None"

    def test_variety_passthrough(self):
        assert resolve_variety("Rice", "Hybrid") == "Hybrid"

    def test_cultivation_method_only_applies_to_rice(self):
        assert resolve_cultivation_method("Corn", "Upland") == "None"

    def test_cultivation_method_defaults_to_lowland_wet(self):
        assert resolve_cultivation_method("Rice", None) == "Lowland Wet"

    def test_cultivation_method_passthrough_for_known_value(self):
        assert resolve_cultivation_method("Rice", "Upland") == "Upland"

    def test_age_band_only_applies_to_coconut_cacao_mango(self):
        assert resolve_age_band("Tomato", "bearing") == "None"

    def test_age_band_bearing_alias_resolves_per_crop(self):
        assert resolve_age_band("Coconut", "bearing") == "9yrs & up"
        assert resolve_age_band("Cacao", "bearing") == "Over 7yrs"
        assert resolve_age_band("Mango", "bearing") == "10yrs & up"

    def test_age_band_omitted_is_the_same_as_bearing(self):
        assert resolve_age_band("Coconut", None) == resolve_age_band("Coconut", "bearing")

    def test_age_band_passthrough_for_known_value(self):
        assert resolve_age_band("Coconut", "1-5yrs") == "1-5yrs"

    def test_variety_only_applies_to_rice_and_corn(self):
        assert resolve_variety("Cassava", "Hybrid") == "None"
        assert resolve_variety("Tomato", "Hybrid") == "None"
        assert resolve_variety("Mango", "Hybrid") == "None"

    def test_variety_still_passes_through_for_rice_and_corn(self):
        assert resolve_variety("Rice", "Hybrid") == "Hybrid"
        assert resolve_variety("Corn", "Hybrid") == "Hybrid"

    def test_variety_not_tracked_by_this_crop_falls_back_to_none(self):
        """Corn doesn't track Inbred (a Rice-only variety) -- passing it for
        Corn must not silently accept an axis value this crop's guide rows
        don't distinguish."""
        assert resolve_variety("Corn", "Inbred") == "None"


class TestNutrientRequirements:
    def test_returns_a_numeric_triple(self):
        n, p, k = nutrient_requirements("Corn", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert isinstance(n, float) and isinstance(p, float) and isinstance(k, float)

    def test_tree_crop_rates_are_not_rounded_to_zero(self):
        """The guide's Banana|Fruit rates are all sub-1 (e.g. 0.16, 0.05,
        0.33) — an int() cast anywhere in this path would silently collapse
        every real Banana recommendation to (0, 0, 0)."""
        n, p, k = nutrient_requirements("Banana", om=1.0, p_ppm=3.0, k_cmol_kg=0.1)
        assert n > 0 and p > 0 and k > 0

    def test_boundary_value_lands_in_the_band_that_prints_it(self):
        """The guide prints bands as '0-2.0' then '2.1-3.5' — a 0.1 gap, not
        a continuous boundary. A value of exactly 2.0 belongs in the first
        band ('0-2.0'), not the second."""
        at_boundary = nutrient_requirements("Rice", om=2.0, p_ppm=3.0, k_cmol_kg=0.1, cultivation_method="Upland")[0]
        just_over = nutrient_requirements("Rice", om=2.1, p_ppm=3.0, k_cmol_kg=0.1, cultivation_method="Upland")[0]
        assert at_boundary == 45   # Rice|Upland's "0-2.0" band
        assert just_over == 40     # Rice|Upland's "2.1-3.5" band

    def test_lower_om_increases_nitrogen(self):
        """Guide bands are descending as soil fertility improves — this is a
        property of the real table, not a formula we chose."""
        low_om = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3)[0]
        high_om = nutrient_requirements("Corn", om=5.0, p_ppm=15.0, k_cmol_kg=0.3)[0]
        assert low_om > high_om

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)

    def test_variety_selects_a_different_guide_row(self):
        native = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="OPV")
        hybrid = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert native != hybrid

    def test_cultivation_method_selects_a_different_guide_row(self):
        wet = nutrient_requirements("Rice", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, cultivation_method="Lowland Wet")
        upland = nutrient_requirements("Rice", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, cultivation_method="Upland")
        assert wet != upland

    def test_age_band_selects_a_different_guide_row(self):
        young = nutrient_requirements("Coconut", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="Below 1yr")
        mature = nutrient_requirements("Coconut", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="9yrs & up")
        assert young != mature

    def test_no_age_band_defaults_to_bearing(self):
        default = nutrient_requirements("Cacao", om=1.0, p_ppm=10.0, k_cmol_kg=0.3)
        explicit_bearing = nutrient_requirements("Cacao", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="bearing")
        assert default == explicit_bearing


class TestGoldenRecordRiceInbred:
    """Fertilizer Computation Guide.pptx worked example: Rice (Inbred), Wet
    Season -> N/P/K 90/60/40 kg/ha at soil-test bands described only as
    'Low'/'Low'/'Sufficient' (no raw pH/OM/P/K numbers given in the pptx).
    Wet-season N and P match this file's Lowland Wet band-0 (lowest OM/P)
    values exactly. K does not currently match any band at band-0's OM/P —
    a known, documented discrepancy (see Task 1's Context note and the
    plan's Global Constraints). This test asserts the two nutrients that do
    match; the K mismatch is intentionally not asserted here so this test
    suite doesn't silently start passing by weakening a real check — chasing
    the K discrepancy requires re-reading the source PDF, tracked as
    follow-up work, not papered over in this test."""

    def test_wet_season_low_om_low_p_matches_n_and_p(self):
        n, p, _k = nutrient_requirements(
            "Rice", om=1.0, p_ppm=3.0, k_cmol_kg=0.5, cultivation_method="Lowland Wet",
        )
        assert n == 90
        assert p == 60

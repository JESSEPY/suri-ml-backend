import pytest

from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, target_unit_for_crop, resolve_variety,
    resolve_cultivation_method, resolve_age_band, nutrient_requirements,
    max_total_need,
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
            nutrient_requirements("Nonesuch", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)

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


class TestMaxTotalNeed:
    """max_total_need is the worst-case (poorest-soil) total N+P2O5+K2O for
    a crop config -- a fixed property of that crop's guide row, independent
    of any soil reading. It normalizes nutrient_requirements()'s soil-driven
    total into a 0-1 fraction of the crop's own range, so crops with very
    different absolute scales (e.g. Peanut vs. Tomato) can't be compared by
    raw kg/ha magnitude."""

    def test_known_value_peanut(self):
        # Peanut|None bands: om max 10, p max 40, k max 45 -> 95.
        assert max_total_need("Peanut") == 95

    def test_known_value_rice_lowland_wet(self):
        # Rice|Lowland Wet bands: om max 90, p max 60, k max 60 -> 210.
        assert max_total_need("Rice", cultivation_method="Lowland Wet") == 210

    def test_known_value_corn_native(self):
        # Corn|Native bands: om max 60, p max 60, k max 60 -> 180.
        assert max_total_need("Corn") == 180

    def test_independent_of_soil_reading(self):
        """A fixed property of the crop config, not computed from a soil
        test -- unlike nutrient_requirements(), it takes no om/p/k args."""
        assert max_total_need("Rice", cultivation_method="Upland") == max_total_need(
            "Rice", cultivation_method="Upland"
        )

    def test_cultivation_method_selects_a_different_guide_row(self):
        wet = max_total_need("Rice", cultivation_method="Lowland Wet")
        upland = max_total_need("Rice", cultivation_method="Upland")
        assert wet != upland

    def test_variety_selects_a_different_guide_row(self):
        native = max_total_need("Corn", variety="OPV")
        hybrid = max_total_need("Corn", variety="Hybrid")
        assert native != hybrid

    def test_age_band_selects_a_different_guide_row(self):
        young = max_total_need("Coconut", age_band="Below 1yr")
        mature = max_total_need("Coconut", age_band="9yrs & up")
        assert young != mature

    def test_no_age_band_defaults_to_bearing(self):
        assert max_total_need("Cacao") == max_total_need("Cacao", age_band="bearing")

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            max_total_need("Nonesuch")

    @pytest.mark.parametrize("crop", ["Rice", "Corn", "Cassava", "Peanut", "Tomato",
                                       "Mango", "Cacao", "Coconut", "Banana"])
    def test_every_crop_is_at_least_the_worst_case_nutrient_requirement(self, crop):
        """The worst-case total (poorest possible soil: om=0, p=0, k=0) must
        never exceed max_total_need -- if it did, a real soil reading could
        produce a need fraction > 1."""
        n, p, k = nutrient_requirements(crop, om=0.0, p_ppm=0.0, k_cmol_kg=0.0)
        assert n + p + k <= max_total_need(crop)


class TestDerivedRegistry:
    """The registry is derived from GUIDE_ROWS rather than restated by hand.
    These pin the derivation to exactly today's values, so the refactor is
    provably behaviour-preserving."""

    def test_canonical_crops_match_the_guide_rows(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, GUIDE_ROWS

        expected = sorted({key.split("|", 1)[0] for key in GUIDE_ROWS})
        assert CANONICAL_CROPS == expected

    def test_the_nine_original_crops_are_all_present(self):
        from data.nutrient_recommendation_guide import CROPS

        for crop in ["Corn", "Rice", "Cassava", "Peanut", "Mango",
                     "Cacao", "Banana", "Tomato", "Coconut"]:
            assert crop in CROPS

    def test_target_unit_is_derived_and_still_correct(self):
        from data.nutrient_recommendation_guide import TARGET_UNIT

        for crop in ["Rice", "Corn", "Cassava", "Peanut", "Tomato"]:
            assert TARGET_UNIT[crop] == "kg_ha"
        for crop in ["Mango", "Cacao", "Coconut", "Banana"]:
            assert TARGET_UNIT[crop] == "kg_tree"

    def test_every_crop_has_a_target_unit(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, TARGET_UNIT

        assert set(TARGET_UNIT) == set(CANONICAL_CROPS)

    def test_a_crop_whose_rows_disagree_on_unit_is_rejected(self):
        """A mixed-unit crop would make target_unit_for_crop() arbitrary."""
        from data.nutrient_recommendation_guide import GuideRow, _derive_target_units

        rows = {
            "Bogus|A": GuideRow("kg_ha", ((2.0, 1),), ((6.0, 1),), ((35.0, 1),)),
            "Bogus|B": GuideRow("kg_tree", ((2.0, 1),), ((6.0, 1),), ((35.0, 1),)),
        }
        with pytest.raises(ValueError, match="Bogus"):
            _derive_target_units(rows)


class TestGuideRowKeyGeneralization:
    """Corn's Native/Hybrid split is a variety axis, same shape as the
    sub-axes new crops need. Pinning it here guards the generalization."""

    def test_corn_defaults_to_the_native_row(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Corn", None, None, None) == "Corn|Native"
        assert _guide_row_key("Corn", "OPV", None, None) == "Corn|Native"
        assert _guide_row_key("Corn", "Hybrid", None, None) == "Corn|Hybrid"

    def test_rice_still_keys_on_cultivation_method(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Rice", None, None, None) == "Rice|Lowland Wet"
        assert _guide_row_key("Rice", None, "Upland", None) == "Rice|Upland"

    def test_tree_crops_still_key_on_age_band(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Coconut", None, None, None) == "Coconut|9yrs & up"
        assert _guide_row_key("Coconut", None, None, "1-5yrs") == "Coconut|1-5yrs"

    def test_unsplit_crops_key_on_the_bare_name(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Cassava", None, None, None) == "Cassava"


class TestCropAliases:
    def test_an_alias_resolves_to_its_canonical_crop(self):
        from data.nutrient_recommendation_guide import CROP_ALIASES, resolve_crop

        for alias, canonical in CROP_ALIASES.items():
            assert resolve_crop(alias) == canonical

    def test_a_canonical_crop_resolves_to_itself(self):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop("Cassava") == "Cassava"

    def test_an_unknown_crop_raises(self):
        from data.nutrient_recommendation_guide import resolve_crop

        with pytest.raises(KeyError, match="Nonesuch"):
            resolve_crop("Nonesuch")

    def test_every_alias_points_at_a_real_canonical_crop(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROP_ALIASES

        for alias, canonical in CROP_ALIASES.items():
            assert canonical in CANONICAL_CROPS, alias

    def test_no_alias_shadows_a_canonical_crop(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROP_ALIASES

        assert not (set(CROP_ALIASES) & set(CANONICAL_CROPS))

    def test_crops_exposes_canonical_names_and_aliases(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROPS, CROP_ALIASES

        assert set(CROPS) == set(CANONICAL_CROPS) | set(CROP_ALIASES)

    def test_nutrient_requirements_accepts_an_alias(self):
        from data.nutrient_recommendation_guide import CROP_ALIASES, nutrient_requirements

        if not CROP_ALIASES:
            pytest.skip("no aliases registered yet")
        alias, canonical = next(iter(CROP_ALIASES.items()))
        assert nutrient_requirements(alias, 2.5, 15.0, 0.3) == \
               nutrient_requirements(canonical, 2.5, 15.0, 0.3)


class TestVegetableRows:
    """One assertion per band edge per crop. Values are the guide's own;
    a failure here means either the transcription or the band arithmetic is
    wrong, and both are worth failing loudly on."""

    @pytest.mark.parametrize("crop,om,p,k,expected", [
        # (crop, %OM, Olsen P ppm, K ppm, (N, P2O5, K2O))
        ("Alugbati", 1.0, 3.0, 20.0, (90, 60, 60)),
        ("Alugbati", 6.0, 35.0, 400.0, (0, 0, 0)),
        ("Ampalaya", 1.0, 3.0, 20.0, (40, 80, 45)),
        ("Asparagus", 3.0, 8.0, 45.0, (60, 30, 45)),
        ("Cabbage", 1.0, 3.0, 20.0, (150, 80, 75)),
        ("Cabbage", 5.0, 12.0, 100.0, (100, 40, 30)),
        ("Cauliflower", 1.0, 3.0, 20.0, (90, 80, 90)),
        ("Celery", 1.0, 3.0, 20.0, (60, 90, 60)),
        ("Chayote", 1.0, 3.0, 20.0, (60, 80, 65)),
        ("Cucumber", 1.0, 3.0, 20.0, (60, 100, 45)),
        ("Eggplant", 1.0, 3.0, 20.0, (90, 80, 90)),
        ("Lettuce", 1.0, 3.0, 20.0, (60, 60, 75)),
        ("Radish", 1.0, 3.0, 20.0, (90, 90, 120)),
        ("Radish", 3.0, 8.0, 280.0, (60, 60, 30)),
        ("Squash", 1.0, 3.0, 20.0, (60, 80, 60)),
        ("Strawberry", 1.0, 3.0, 20.0, (45, 30, 60)),
    ])
    def test_band_lookups(self, crop, om, p, k, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(crop, om, p, k / K_CMOL_TO_PPM) == expected

    def test_all_vegetables_are_per_hectare(self):
        from data.nutrient_recommendation_guide import target_unit_for_crop

        for crop in ["Alugbati", "Ampalaya", "Asparagus", "Cabbage", "Cauliflower",
                     "Celery", "Chayote", "Cucumber", "Eggplant", "Lettuce",
                     "Radish", "Squash", "Strawberry"]:
            assert target_unit_for_crop(crop) == "kg_ha"

    @pytest.mark.parametrize("alias,canonical", [
        ("Kangkong", "Alugbati"), ("Pechay", "Cabbage"), ("Mustard", "Cabbage"),
        ("Spinach", "Cabbage"), ("Broccoli", "Cabbage"), ("Muskmelon", "Cucumber"),
        ("Watermelon", "Cucumber"), ("Okra", "Eggplant"), ("Malunggay", "Lettuce"),
        ("Carrots", "Radish"), ("Upo", "Squash"), ("Patola", "Squash"),
    ])
    def test_co_listed_species_alias_to_their_row(self, alias, canonical):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop(alias) == canonical


class TestCondimentRootcropAndLegumeRows:
    @pytest.mark.parametrize("crop,om,p,k,expected", [
        ("Garlic", 1.0, 3.0, 20.0, (120, 80, 150)),
        ("Garlic", 6.0, 35.0, 400.0, (0, 0, 0)),
        ("Onion", 1.0, 3.0, 20.0, (120, 80, 150)),
        ("Gabi", 1.0, 3.0, 20.0, (40, 60, 90)),
        ("Potato", 1.0, 3.0, 20.0, (120, 100, 120)),
        ("Beans", 1.0, 3.0, 20.0, (10, 60, 60)),
        ("Peas", 1.0, 3.0, 20.0, (20, 40, 45)),
        ("Sugarcane", 1.0, 3.0, 20.0, (180, 120, 160)),
        ("Sugarcane", 5.0, 18.0, 280.0, (75, 20, 60)),
    ])
    def test_band_lookups(self, crop, om, p, k, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(crop, om, p, k / K_CMOL_TO_PPM) == expected

    def test_ginger_defaults_to_the_improved_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Ginger", 1.0, 3.0, k) == (60, 60, 120)
        assert nutrient_requirements("Ginger", 1.0, 3.0, k, variety="Local") == (40, 50, 90)

    def test_onion_spring_is_a_separate_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Onion", 1.0, 3.0, k, variety="Spring") == (130, 60, 45)

    @pytest.mark.parametrize("alias,canonical", [
        ("Chili Pepper", "Garlic"),
        ("Camote", "Cassava"),
        ("Ube", "Gabi"),
        ("Yam", "Gabi"), ("Taro", "Gabi"), ("Singkamas", "Potato"),
        ("Sitao", "Beans"), ("Lima", "Beans"), ("Cowpea", "Peas"), ("Tapilan", "Peas"),
    ])
    def test_co_listed_species_alias_to_their_row(self, alias, canonical):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop(alias) == canonical


class TestFreeSpeciesAliases:
    @pytest.mark.parametrize("alias,canonical", [
        ("Sorghum", "Corn"), ("Camote", "Cassava"), ("Yacon", "Cassava"),
        ("Soybean", "Peanut"), ("Mongo", "Peanut"), ("Santol", "Mango"),
        ("Caimito", "Mango"), ("Durian", "Mango"), ("Marang", "Mango"),
    ])
    def test_alias_resolves_and_predicts_identically(self, alias, canonical):
        from data.nutrient_recommendation_guide import nutrient_requirements, resolve_crop

        assert resolve_crop(alias) == canonical
        assert nutrient_requirements(alias, 2.5, 15.0, 0.3) == \
               nutrient_requirements(canonical, 2.5, 15.0, 0.3)

    def test_tree_crop_aliases_keep_the_tree_unit(self):
        from data.nutrient_recommendation_guide import target_unit_for_crop

        for alias in ["Santol", "Caimito", "Durian", "Marang"]:
            assert target_unit_for_crop(alias) == "kg_tree"

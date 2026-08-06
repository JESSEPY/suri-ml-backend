import pytest

from data.parse_real_soil_database import (
    parse_numeric_cell, normalize_lab_code, normalize_crop,
    normalize_cultivation_method, normalize_age_band, find_column,
)


class TestParseNumericCell:
    def test_real_number_passes_through_as_float(self):
        assert parse_numeric_cell(8.17) == pytest.approx(8.17)

    def test_integer_becomes_float(self):
        assert parse_numeric_cell(5) == pytest.approx(5.0)

    def test_none_is_missing(self):
        assert parse_numeric_cell(None) is None

    def test_stk_sentinel_is_missing(self):
        """'STK' marks a field tested via the rapid kit instead of the lab —
        not a lab measurement, must not be coerced into a number."""
        assert parse_numeric_cell("STK") is None

    def test_nil_and_ngl_sentinels_are_missing(self):
        assert parse_numeric_cell("NIL") is None
        assert parse_numeric_cell("NGL") is None

    def test_malformed_number_typo_is_missing(self):
        """A real data-entry typo found in the source file ('4..67' for pH
        4.67) — any non-numeric-typed cell is treated as absent, not parsed
        with a lenient regex that might silently mis-parse it."""
        assert parse_numeric_cell("4..67") is None


class TestNormalizeLabCode:
    def test_strips_whitespace_and_prefix(self):
        assert normalize_lab_code("SCC 417", "2022") == "2022-SCC417"

    def test_zero_padding_collapses_to_the_same_code(self):
        assert normalize_lab_code("SCC 01", "2024") == normalize_lab_code("SCC 1", "2024") == "2024-SCC1"

    def test_different_years_produce_different_codes(self):
        """The same 'SCC 417' number is reused every year for an unrelated
        physical sample — namespacing by year keeps them distinct."""
        assert normalize_lab_code("SCC 417", "2022") != normalize_lab_code("SCC 417", "2023")

    def test_case_insensitive_prefix(self):
        assert normalize_lab_code("scc417", "2023") == "2023-SCC417"


class TestNormalizeCrop:
    def test_known_sole_crop_with_no_variety(self):
        assert normalize_crop("Cassava") == ("Cassava", None)

    def test_case_and_whitespace_insensitive(self):
        assert normalize_crop("  cassava  ") == ("Cassava", None)

    def test_hybrid_rice_maps_to_rice_with_variety(self):
        assert normalize_crop("Hybrid Rice") == ("Rice", "Hybrid")

    def test_inbred_rice_maps_to_rice_with_variety(self):
        assert normalize_crop("Inbred Rice") == ("Rice", "Inbred")

    def test_opv_corn_maps_to_corn_with_variety(self):
        assert normalize_crop("OPV Corn") == ("Corn", "OPV")
        assert normalize_crop("Corn OPV") == ("Corn", "OPV")

    def test_hybrid_corn_maps_to_corn_with_variety(self):
        assert normalize_crop("Hybrid Corn") == ("Corn", "Hybrid")

    def test_multi_crop_text_is_rejected(self):
        assert normalize_crop("Ampalaya, Eggplant, Squash") is None

    def test_empty_or_none_is_rejected(self):
        assert normalize_crop("") is None
        assert normalize_crop(None) is None
        assert normalize_crop("   ") is None

    def test_unknown_crop_is_rejected(self):
        """Crops below the real-sample floor (Cabbage/Sorghum/Sugarcane) or
        not in the 9-crop list at all must not silently pass through."""
        assert normalize_crop("Sorghum") is None
        assert normalize_crop("Ampalaya") is None


class TestNormalizeCultivationMethod:
    def test_detects_upland_for_rice(self):
        assert normalize_cultivation_method("Rice", "Rice Upland") == "Upland"

    def test_detects_lowland_wet_for_rice(self):
        assert normalize_cultivation_method("Rice", "Rice Lowland Wet") == "Lowland Wet"

    def test_not_stated_returns_none_not_a_guess(self):
        """Absence must be represented as None (resolved to the default by
        data.nutrient_recommendation_guide.resolve_cultivation_method later),
        not guessed here."""
        assert normalize_cultivation_method("Rice", "Rice") is None

    def test_non_rice_crop_is_always_none(self):
        assert normalize_cultivation_method("Corn", "Corn Upland") is None


class TestNormalizeAgeBand:
    def test_detects_a_stated_age_band(self):
        assert normalize_age_band("Coconut", "Coconut 6-8yrs") == "6-8yrs"

    def test_not_stated_returns_none_not_a_guess(self):
        assert normalize_age_band("Coconut", "Coconut") is None

    def test_crop_with_no_age_dimension_is_always_none(self):
        """Banana isn't age-banded — it always uses the Fruit row (see
        data.nutrient_recommendation_guide) regardless of any age text."""
        assert normalize_age_band("Banana", "Banana 6-8yrs") is None


class TestFindColumn:
    def test_finds_exact_match(self):
        header = ("No.", "Laboratory Code", "pH (H2O, 1:1)", "Crops")
        assert find_column(header, "pH (H2O, 1:1)") == 2

    def test_raises_when_not_found(self):
        """A missing expected column means the sheet's layout has drifted
        further than accounted for — this must fail loudly, not silently
        skip the field or guess a nearby column."""
        header = ("No.", "Laboratory Code")
        with pytest.raises(ValueError):
            find_column(header, "pH (H2O, 1:1)")

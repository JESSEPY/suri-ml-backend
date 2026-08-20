import pytest

from data.nutrient_recommendation_guide import CANONICAL_CROPS
from data.parse_real_soil_database import (
    parse_numeric_cell, normalize_lab_code, normalize_crop,
    normalize_cultivation_method, normalize_age_band, find_column,
    CROP_VARIETY_MAP,
)

import openpyxl
from data.parse_real_soil_database import (
    build_training_row, parse_sheet, SHEET_CONFIGS, LAB_DETAIL_HEADERS,
    SOIL_DATA_2025_HEADERS,
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
        """A crop CROP_VARIETY_MAP has no entry for must not silently pass
        through. "Nonesuch" can never become real; Sorghum and Ampalaya
        used to belong here too, back when this system only supported 9
        crops -- they're now real, recognized crops (see
        TestCornRiceVariants / TestVegetableVariants), so asserting them
        rejected here would be testing stale behavior on purpose."""
        assert normalize_crop("Nonesuch") is None

    def test_crop_with_trailing_cultivation_method_text_still_resolves(self):
        """The Crops field is free text that often trails the crop name with
        cultivation-method/age-band words (see normalize_cultivation_method /
        normalize_age_band, which re-scan this same raw text) -- the crop
        itself must still resolve from a leading, whole-word match."""
        assert normalize_crop("Rice Upland") == ("Rice", None)
        assert normalize_crop("Rice Lowland Wet") == ("Rice", None)
        assert normalize_crop("Coconut 6-8yrs") == ("Coconut", None)

    def test_variety_prefix_still_wins_over_the_bare_crop_name(self):
        """Longest-key-first matching must not let the bare "rice" key shadow
        a variety-specific key like "hybrid rice"."""
        assert normalize_crop("Hybrid Rice Lowland Wet") == ("Rice", "Hybrid")


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


class TestBuildTrainingRow:
    def test_complete_row_computes_remarks_and_targets(self):
        raw = {
            "laboratory_code": "2022-SCC417", "crop": "Corn", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.4, "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
            "potassium_cmol_kg": 0.35, "total_nitrogen_pct": None,
        }
        row = build_training_row(raw)
        assert row is not None
        assert row["data_source"] == "lab"
        assert row["crop"] == "Corn"
        assert row["variety"] == "None"
        assert row["cultivation_method"] == "None"
        assert row["age_band"] == "None"
        assert row["om_remark"] == "LOW"
        assert row["target_unit"] == "kg_ha"
        assert isinstance(row["target_n"], float)
        assert row["target_n"] > 0

    def test_k_remark_uses_the_converted_ppm_not_the_raw_cmol_kg(self):
        """0.35 cmol/kg -> 136.85 ppm -> SUFFICIENT. Feeding the raw 0.35
        into derive_k_remark directly would misclassify it as DEFICIENT."""
        raw = {
            "laboratory_code": "2022-SCC417", "crop": "Corn", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.4, "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
            "potassium_cmol_kg": 0.35, "total_nitrogen_pct": None,
        }
        assert build_training_row(raw)["k_remark"] == "SUFFICIENT"

    def test_tree_crop_row_gets_kg_tree_unit(self):
        raw = {
            "laboratory_code": "2022-SCC5", "crop": "Coconut", "variety": None,
            "cultivation_method": None, "age_band": "9yrs & up",
            "ph": 6.5, "organic_matter_pct": 2.0, "phosphorus_ppm": 10.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        row = build_training_row(raw)
        assert row["target_unit"] == "kg_tree"

    def test_total_nitrogen_derived_when_absent(self):
        raw = {
            "laboratory_code": "2022-SCC1", "crop": "Rice", "variety": "Hybrid",
            "cultivation_method": None, "age_band": None,
            "ph": 6.5, "organic_matter_pct": 2.0, "phosphorus_ppm": 10.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        row = build_training_row(raw)
        assert row["total_nitrogen_pct"] == pytest.approx(0.10)

    def test_missing_om_or_phosphorus_or_potassium_returns_none(self):
        """A real target needs numeric OM/P/K — nutrient_requirements() can't
        run without them, and there's no real substitute (see the design
        spec's "Data source scope" section for why this must not be
        patched with an invented number)."""
        base = {
            "laboratory_code": "2022-SCC1", "crop": "Corn", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.5, "organic_matter_pct": 2.5, "phosphorus_ppm": 15.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        for missing_field in ("organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg"):
            incomplete = {**base, missing_field: None}
            assert build_training_row(incomplete) is None

    def test_variety_none_becomes_the_string_none(self):
        raw = {
            "laboratory_code": "2022-SCC1", "crop": "Cassava", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.5, "organic_matter_pct": 2.5, "phosphorus_ppm": 15.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        assert build_training_row(raw)["variety"] == "None"


class TestParseSheetIntegration:
    """Exercises parse_sheet against a small in-memory workbook shaped like
    the real Laboratory Details sheets, to catch column-mapping regressions
    without depending on the actual multi-thousand-row source file."""

    @pytest.fixture
    def workbook(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Laboratory Details2022"
        ws.cell(row=3, column=2, value="Laboratory Code")
        ws.cell(row=3, column=15, value="Crops")
        ws.cell(row=5, column=17, value="pH (H2O, 1:1)")
        ws.cell(row=5, column=18, value="Available Phosphorus (Avail. P), mg/kg")
        ws.cell(row=5, column=19, value="Exchangeable Potassium (K), cmol/kg")
        ws.cell(row=5, column=20, value="Organic Matter (OM), %")
        ws.cell(row=5, column=21, value="Total Nitrogen (N), %")
        ws.cell(row=6, column=2, value="SCC 1")
        ws.cell(row=6, column=15, value="Corn")
        ws.cell(row=6, column=17, value=6.4)
        ws.cell(row=6, column=18, value=18.0)
        ws.cell(row=6, column=19, value=0.35)
        ws.cell(row=6, column=20, value=2.8)
        ws.cell(row=6, column=21, value=0.14)
        # A row that was field-tested instead of lab-tested for phosphorus
        ws.cell(row=7, column=2, value="SCC 2")
        ws.cell(row=7, column=15, value="Rice Upland")
        ws.cell(row=7, column=17, value=6.8)
        ws.cell(row=7, column=18, value="STK")
        ws.cell(row=7, column=19, value=0.4)
        ws.cell(row=7, column=20, value=2.0)
        # A multi-crop row that must be excluded
        ws.cell(row=8, column=2, value="SCC 3")
        ws.cell(row=8, column=15, value="Corn, Rice")
        ws.cell(row=8, column=17, value=6.5)
        ws.cell(row=8, column=18, value=15.0)
        ws.cell(row=8, column=19, value=0.3)
        ws.cell(row=8, column=20, value=2.5)
        path = tmp_path / "test.xlsx"
        wb.save(path)
        loaded = openpyxl.load_workbook(path, read_only=True, data_only=True)
        return loaded["Laboratory Details2022"]

    def test_reads_the_complete_row(self, workbook):
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        codes = {r["laboratory_code"] for r in rows}
        assert "2022-SCC1" in codes

    def test_excludes_multi_crop_rows(self, workbook):
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        assert "2022-SCC3" not in {r["laboratory_code"] for r in rows}

    def test_stk_flagged_phosphorus_is_none_not_a_string(self, workbook):
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        scc2 = next(r for r in rows if r["laboratory_code"] == "2022-SCC2")
        assert scc2["phosphorus_ppm"] is None

    def test_stated_cultivation_method_is_captured(self, workbook):
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        scc2 = next(r for r in rows if r["laboratory_code"] == "2022-SCC2")
        assert scc2["cultivation_method"] == "Upland"


class TestBuildTrainingRowDefaultResolution:
    """The stored cultivation_method/age_band feature must match what
    nutrient_requirements() actually resolved and used to compute the
    target — not the raw unstated value. Storing the literal string "None"
    for a Rice row whose target was silently computed against the
    "Lowland Wet" default would desynchronize the feature from the label
    the model is trained to predict."""

    def test_rice_with_no_stated_cultivation_method_stores_the_resolved_default(self):
        raw = {
            "laboratory_code": "2022-SCC9", "crop": "Rice", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.5, "organic_matter_pct": 2.0, "phosphorus_ppm": 10.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        assert build_training_row(raw)["cultivation_method"] == "Lowland Wet"

    def test_coconut_with_no_stated_age_band_stores_the_bearing_default(self):
        raw = {
            "laboratory_code": "2022-SCC10", "crop": "Coconut", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.5, "organic_matter_pct": 2.0, "phosphorus_ppm": 10.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        assert build_training_row(raw)["age_band"] == "9yrs & up"


class TestCropVarietyMapConsistency:
    def test_every_mapped_crop_is_canonical(self):
        """A CROP_VARIETY_MAP entry targeting an alias (or a typo) would
        silently desynchronize by_crop training metrics from the canonical
        name Task 7's untrained-crop guard and ml/train.py's evaluation
        both key on."""
        for text, (crop, variety) in CROP_VARIETY_MAP.items():
            assert crop in CANONICAL_CROPS, f"{text!r} maps to non-canonical {crop!r}"

    def test_the_original_nine_crops_still_resolve(self):
        """Behavior-preservation pin for the pre-existing map — this task
        changes normalize_crop's validation, not its entries."""
        assert normalize_crop("Corn") == ("Corn", None)
        assert normalize_crop("OPV Corn") == ("Corn", "OPV")
        assert normalize_crop("Hybrid Rice") == ("Rice", "Hybrid")
        assert normalize_crop("Cassava") == ("Cassava", None)


class TestCornRiceVariants:
    @pytest.mark.parametrize("text,expected", [
        ("sweetcorn", ("Corn", None)),
        ("Sweet Corn", ("Corn", None)),
        ("Corn Hybrid", ("Corn", "Hybrid")),
        ("CORN HYBRID YELLOW", ("Corn", "Hybrid")),
        ("Corn Tinigib", ("Corn", None)),
        ("corn tiniguib", ("Corn", None)),
        ("Sorghum", ("Corn", None)),
        ("Adlai", ("Corn", "Hybrid")),
        ("Upland Rice", ("Rice", None)),
        ("Rice Inbred", ("Rice", "Inbred")),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

    def test_upland_rice_cultivation_method_is_still_detected_downstream(self):
        """normalize_crop only resolves the crop; normalize_cultivation_method
        runs separately against the same raw text and must still find
        'upland' regardless of which CROP_VARIETY_MAP key matched."""
        assert normalize_cultivation_method("Rice", "Upland Rice") == "Upland"


class TestVegetableVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Lettuce", ("Lettuce", None)),
        ("Pechay", ("Cabbage", None)),
        ("Ampalaya", ("Ampalaya", None)),
        ("Eggplant", ("Eggplant", None)),
        ("Talong", ("Eggplant", None)),
        ("Okra", ("Eggplant", None)),
        ("Sayote", ("Chayote", None)),
        ("Cabbage", ("Cabbage", None)),
        ("Squash", ("Squash", None)),
        ("Carrots", ("Radish", None)),
        ("Cauliflower", ("Cauliflower", None)),
        ("Radish", ("Radish", None)),
        ("Raddish", ("Radish", None)),
        ("Cucumber", ("Cucumber", None)),
        ("Pipino", ("Cucumber", None)),
        ("Watermelon", ("Cucumber", None)),
        ("Red Cabbage", ("Cabbage", None)),
        ("Repolyo", ("Cabbage", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected


class TestCondimentRootcropLegumeVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Sugarcane", ("Sugarcane", None)),
        ("Onions", ("Onion", None)),
        ("Onion", ("Onion", None)),
        ("Sibuyas", ("Onion", None)),
        ("Spring Onion", ("Onion", "Spring")),
        ("Spring Onions", ("Onion", "Spring")),
        ("Green Onion", ("Onion", "Spring")),
        ("Ginger", ("Ginger", None)),
        ("Gabi", ("Gabi", None)),
        ("Ube", ("Gabi", None)),
        ("Ubi", ("Gabi", None)),
        ("Ube Kinampay", ("Gabi", None)),
        ("Kamote", ("Cassava", None)),
        ("Sweet Potato", ("Cassava", None)),
        ("Beans", ("Beans", None)),
        ("String Beans", ("Beans", None)),
        ("Baguio Beans", ("Beans", None)),
        ("Mung Beans", ("Peanut", None)),
        ("Mungbean", ("Peanut", None)),
        ("Sili Haba", ("Garlic", None)),
        ("Sili Labuyo", ("Garlic", None)),
        ("Sili Espada", ("Garlic", None)),
        ("Sili", ("Garlic", None)),
        ("Sili Atsal", ("Garlic", None)),
        ("Finger Pepper", ("Garlic", None)),
        ("Hot Pepper", ("Garlic", None)),
        ("Green Chili", ("Garlic", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

    def test_bell_pepper_is_not_mapped(self):
        """Botanically distinct from the guide's Chili Pepper/Garlic
        grouping -- deliberately excluded, not force-fit. See the plan's
        'Deliberately Excluded' table."""
        assert normalize_crop("Bell Pepper") is None
        assert normalize_crop("Sweet Pepper") is None


class TestTreeAndPlantationVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Calamansi", ("Citrus", None)),
        ("Lemon", ("Citrus", None)),
        ("Avocado", ("Avocado", None)),
        ("Jackfruit", ("Avocado", None)),
        ("Banana Lakatan", ("Banana", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

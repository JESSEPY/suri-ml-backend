import pytest

from data.nutrient_recommendation_guide import CANONICAL_CROPS
from data.parse_real_soil_database import (
    parse_numeric_cell, normalize_lab_code, normalize_crop, normalize_crops,
    normalize_cultivation_method, normalize_age_band, find_column,
    CROP_VARIETY_MAP,
)

import openpyxl
from data.parse_real_soil_database import (
    build_training_row, parse_sheet, SHEET_CONFIGS, LAB_DETAIL_HEADERS,
    SOIL_DATA_2025_HEADERS, unmatched_crop_segments, collect_unrecognized_crop_texts,
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

    def test_no_crop_arg_is_unchanged(self):
        """Backward-compat pin: omitting the new crop parameter must produce
        the exact same code as before intercrop expansion existed, since
        already-ingested Supabase rows are keyed on this unsuffixed code."""
        assert normalize_lab_code("SCC 417", "2022", None) == "2022-SCC417"

    def test_crop_suffix_when_provided(self):
        """A source row that expands into multiple training rows (one per
        intercropped species) needs a unique code per row -- the upsert
        primary key is (laboratory_code, data_source), so two rows sharing
        one unsuffixed code would silently overwrite each other on ingest."""
        assert normalize_lab_code("SCC 417", "2024", "Rice") == "2024-SCC417-RICE"

    def test_crop_suffix_is_uppercased(self):
        assert normalize_lab_code("SCC 1", "2024", "corn") == "2024-SCC1-CORN"

    def test_variety_is_also_suffixed_when_crop_matches_but_variety_differs(self):
        """Two intercrop segments can resolve to the same crop with
        different varieties (e.g. 'Corn Tundan' -> (Corn, None) and 'Corn
        OPV' -> (Corn, OPV), real source: '2022-SCC877') -- normalize_crops
        correctly keeps both as distinct rows (different (crop, variety)),
        but a crop-only suffix would still collide. Real source example:
        '2022-SCC877' listed both and both landed on '...-CORN' before this
        fix."""
        assert normalize_lab_code("SCC 877", "2022", "Corn", "OPV") == "2022-SCC877-CORN-OPV"
        assert normalize_lab_code("SCC 877", "2022", "Corn", None) == "2022-SCC877-CORN"

    def test_variety_omitted_keeps_the_crop_only_suffix(self):
        """Backward-compat within this same feature: calling with just crop
        (no variety arg) must be unchanged from before variety-suffixing
        was added, since most crops have no variety axis at all."""
        assert normalize_lab_code("SCC 1", "2024", "Rice") == "2024-SCC1-RICE"


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


class TestNormalizeCrops:
    """normalize_crops() is the intercrop-aware counterpart to normalize_crop:
    where normalize_crop rejects any comma/slash-separated text outright,
    normalize_crops splits it and returns one (crop, variety, raw_segment)
    tuple per recognized species, silently dropping unrecognized segments --
    recovering training rows from intercropped soil readings instead of
    discarding them."""

    def test_single_crop_text_returns_one_entry(self):
        assert normalize_crops("Cassava") == [("Cassava", None, "Cassava")]

    def test_two_recognized_crops_returns_two_entries(self):
        assert normalize_crops("Corn, Rice") == [
            ("Corn", None, "Corn"), ("Rice", None, "Rice"),
        ]

    def test_slash_separator_also_splits(self):
        assert normalize_crops("Corn/Rice") == [
            ("Corn", None, "Corn"), ("Rice", None, "Rice"),
        ]

    def test_partial_recognition_drops_unrecognized_segment(self):
        """One real crop + one gibberish/unmapped species keeps the real
        one instead of rejecting the whole row -- this is the recovery this
        function exists for."""
        assert normalize_crops("Rice, Nonesuch") == [("Rice", None, "Rice")]

    def test_all_unrecognized_returns_empty_list(self):
        assert normalize_crops("Nonesuch, AlsoNonesuch") == []

    def test_blank_or_none_returns_empty_list(self):
        assert normalize_crops("") == []
        assert normalize_crops(None) == []
        assert normalize_crops("   ") == []

    def test_variety_still_resolves_per_segment(self):
        assert normalize_crops("Hybrid Rice, OPV Corn") == [
            ("Rice", "Hybrid", "Hybrid Rice"), ("Corn", "OPV", "OPV Corn"),
        ]

    def test_segment_preserves_qualifier_text_for_downstream_scoping(self):
        """The returned raw_segment is that crop's own slice of the original
        text (e.g. still containing 'Upland'), not the whole cell -- callers
        use it to scope cultivation-method/age-band inference per crop
        instead of letting one crop's qualifier bleed into another's."""
        assert normalize_crops("Rice Upland, Corn") == [
            ("Rice", None, "Rice Upland"), ("Corn", None, "Corn"),
        ]

    def test_synonym_segments_resolving_to_the_same_crop_are_deduplicated(self):
        """Kamote is just the Filipino word for Cassava -- CROP_VARIETY_MAP
        maps both to the same canonical (crop, variety). Emitting two rows
        for one physical sample under the same resolved crop is pure
        duplication (identical features AND identical targets), and gave
        both rows the same crop-suffixed laboratory_code -- a genuine
        collision the ingest upsert key correctly rejects. Real source
        example: '2024-SCC705', crops='Corn, Cassava, Kamote'."""
        assert normalize_crops("Cassava, Kamote") == [("Cassava", None, "Cassava")]

    def test_three_way_synonym_collision_dedups_to_one(self):
        """Real source example: '2022-SCC898' lists both 'Okra' and
        'Eggplant' among 18 species -- both resolve to canonical Eggplant."""
        assert normalize_crops("Okra, Eggplant, Talong") == [("Eggplant", None, "Okra")]

    def test_distinct_crops_are_not_deduplicated(self):
        """Non-regression: two genuinely different crops must both survive
        deduplication -- only an identical (crop, variety) resolution
        collapses."""
        assert normalize_crops("Corn, Rice") == [
            ("Corn", None, "Corn"), ("Rice", None, "Rice"),
        ]

    def test_same_crop_different_variety_is_not_deduplicated(self):
        """Dedup key is (crop, variety), not crop alone -- Hybrid Rice and
        Inbred Rice are different resolved identities (different target
        rows in the guide) and must both survive."""
        assert normalize_crops("Hybrid Rice, Inbred Rice") == [
            ("Rice", "Hybrid", "Hybrid Rice"), ("Rice", "Inbred", "Inbred Rice"),
        ]


class TestUnmatchedCropSegments:
    """unmatched_crop_segments() is normalize_crops()'s counterpart: where
    normalize_crops() returns what it recognized (silently dropping the
    rest), this returns exactly what it silently dropped -- the audit trail
    for growing CROP_VARIETY_MAP, see collect_unrecognized_crop_texts."""

    def test_single_unrecognized_crop(self):
        assert unmatched_crop_segments("Nonesuch") == ["Nonesuch"]

    def test_recognized_crop_has_no_unmatched_segments(self):
        assert unmatched_crop_segments("Corn") == []

    def test_partial_recognition_returns_only_the_unmatched_segment(self):
        assert unmatched_crop_segments("Rice, Nonesuch") == ["Nonesuch"]

    def test_all_recognized_returns_empty(self):
        assert unmatched_crop_segments("Corn, Rice") == []

    def test_blank_or_none_returns_empty(self):
        assert unmatched_crop_segments("") == []
        assert unmatched_crop_segments(None) == []

    def test_preserves_original_casing_and_qualifier_text(self):
        """Not lowercased, not stripped of trailing qualifier words -- the
        raw text is what a human reviewer needs to see to decide whether
        it's a real missing synonym or just noise."""
        assert unmatched_crop_segments("Nonesuch Upland") == ["Nonesuch Upland"]


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


class TestBuildTrainingRowRangeValidation:
    """build_training_row() must reject a numerically-parsed but physically
    implausible value the same way it already rejects a missing one -- a
    corrupted reading must not become a training label any more than an
    absent one should. Bounds are data.training_upload.NUMERIC_BOUNDS, the
    exact same table main.py's live /predict enforces (kept in sync by
    tests/test_training_upload.py::TestBoundsMatchTheApi) -- reused here
    rather than re-declared, so the bulk parser and the live API can never
    silently drift apart on what counts as plausible."""

    BASE = {
        "laboratory_code": "2022-SCC1", "crop": "Corn", "variety": None,
        "cultivation_method": None, "age_band": None,
        "ph": 6.5, "organic_matter_pct": 2.5, "phosphorus_ppm": 15.0,
        "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
    }

    def test_organic_matter_above_bound_returns_none(self):
        """Real source example: 2024-SCC1791 has organic_matter_pct=356.0,
        almost certainly a decimal-point typo (3.56 or 35.6) -- previously
        silently corrupted its derived Total Nitrogen and fertilizer-target
        labels (see docs/superpowers/specs/2026-08-20-da7-extraction-data-
        quality-findings.md's successor)."""
        raw = {**self.BASE, "organic_matter_pct": 356.0}
        assert build_training_row(raw) is None

    def test_potassium_negative_returns_none(self):
        """Real source example: 2024-SCC1050, potassium_cmol_kg=-0.0051 --
        a concentration cannot be negative."""
        raw = {**self.BASE, "potassium_cmol_kg": -0.0051}
        assert build_training_row(raw) is None

    def test_potassium_above_bound_returns_none(self):
        raw = {**self.BASE, "potassium_cmol_kg": 8.54}
        assert build_training_row(raw) is None

    def test_ph_above_bound_returns_none(self):
        raw = {**self.BASE, "ph": 15.0}
        assert build_training_row(raw) is None

    def test_ph_negative_returns_none(self):
        raw = {**self.BASE, "ph": -1.0}
        assert build_training_row(raw) is None

    def test_phosphorus_above_bound_returns_none(self):
        raw = {**self.BASE, "phosphorus_ppm": 501.0}
        assert build_training_row(raw) is None

    def test_values_exactly_at_the_bound_are_still_valid(self):
        """Bounds are inclusive (ge/le, not gt/lt) -- a value exactly at the
        edge is real, not a violation."""
        raw = {
            **self.BASE,
            "ph": 14.0, "organic_matter_pct": 100.0,
            "phosphorus_ppm": 500.0, "potassium_cmol_kg": 6.0,
        }
        assert build_training_row(raw) is not None


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
        # An intercropped row: both species recognized, must expand to 2 rows
        ws.cell(row=8, column=2, value="SCC 3")
        ws.cell(row=8, column=15, value="Corn, Rice")
        ws.cell(row=8, column=17, value=6.5)
        ws.cell(row=8, column=18, value=15.0)
        ws.cell(row=8, column=19, value=0.3)
        ws.cell(row=8, column=20, value=2.5)
        # An intercropped row exercising per-segment age-band scoping: "NT"
        # is a valid age-band label for BOTH Cacao and Mango (see
        # _AGE_BAND_LABELS), so scanning the whole cell instead of each
        # crop's own segment would wrongly give Mango "NT" too.
        ws.cell(row=9, column=2, value="SCC 4")
        ws.cell(row=9, column=15, value="Cacao NT, Mango")
        ws.cell(row=9, column=17, value=6.6)
        ws.cell(row=9, column=18, value=12.0)
        ws.cell(row=9, column=19, value=0.25)
        ws.cell(row=9, column=20, value=2.2)
        # A partially-recognized intercrop row: "Nonesuch" has no
        # CROP_VARIETY_MAP entry, but Rice does -- the row still produces a
        # valid Rice training row (partial recognition), while "Nonesuch"
        # itself should surface in the unrecognized-crop audit trail.
        ws.cell(row=10, column=2, value="SCC 5")
        ws.cell(row=10, column=15, value="Rice, Nonesuch")
        ws.cell(row=10, column=17, value=6.7)
        ws.cell(row=10, column=18, value=13.0)
        ws.cell(row=10, column=19, value=0.28)
        ws.cell(row=10, column=20, value=2.3)
        # A wholly-unrecognized single-crop row -- no training row at all,
        # but still contributes to the audit trail.
        ws.cell(row=11, column=2, value="SCC 6")
        ws.cell(row=11, column=15, value="Nonesuch")
        ws.cell(row=11, column=17, value=6.7)
        ws.cell(row=11, column=18, value=13.0)
        ws.cell(row=11, column=19, value=0.28)
        ws.cell(row=11, column=20, value=2.3)
        # Same unrecognized text on a second row -- the audit trail counts
        # repeats, since that's exactly the signal worth acting on first.
        ws.cell(row=12, column=2, value="SCC 7")
        ws.cell(row=12, column=15, value="Rice, Nonesuch")
        ws.cell(row=12, column=17, value=6.7)
        ws.cell(row=12, column=18, value=13.0)
        ws.cell(row=12, column=19, value=0.28)
        ws.cell(row=12, column=20, value=2.3)
        path = tmp_path / "test.xlsx"
        wb.save(path)
        loaded = openpyxl.load_workbook(path, read_only=True, data_only=True)
        return loaded["Laboratory Details2022"]

    def test_reads_the_complete_row(self, workbook):
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        codes = {r["laboratory_code"] for r in rows}
        assert "2022-SCC1" in codes

    def test_multi_crop_row_expands_to_one_row_per_recognized_crop(self, workbook):
        """Behavior change from the pre-intercrop-expansion parser: a row
        naming multiple recognized species now contributes one training row
        per species (sharing the same soil-chemistry reading) instead of
        being dropped entirely."""
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        codes = {r["laboratory_code"] for r in rows}
        assert "2022-SCC3-CORN" in codes
        assert "2022-SCC3-RICE" in codes
        # Bare unsuffixed code must never appear for an expanded row --
        # it would collide with a genuinely single-crop row's code.
        assert "2022-SCC3" not in codes

        corn_row = next(r for r in rows if r["laboratory_code"] == "2022-SCC3-CORN")
        rice_row = next(r for r in rows if r["laboratory_code"] == "2022-SCC3-RICE")
        assert corn_row["crop"] == "Corn"
        assert rice_row["crop"] == "Rice"
        # Both rows share the one physical sample's chemistry reading.
        for r in (corn_row, rice_row):
            assert r["ph"] == pytest.approx(6.5)
            assert r["organic_matter_pct"] == pytest.approx(2.5)
            assert r["phosphorus_ppm"] == pytest.approx(15.0)
            assert r["potassium_cmol_kg"] == pytest.approx(0.3)

    def test_intercrop_age_band_scoped_to_each_crops_own_segment(self, workbook):
        """"NT" is a valid age-band label for both Cacao and Mango. Cacao's
        segment ("Cacao NT") states it; Mango's segment ("Mango") doesn't.
        If age-band inference scanned the whole cell instead of each crop's
        own split segment, Mango would wrongly inherit Cacao's "NT"."""
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        cacao_row = next(r for r in rows if r["laboratory_code"] == "2022-SCC4-CACAO")
        mango_row = next(r for r in rows if r["laboratory_code"] == "2022-SCC4-MANGO")
        assert cacao_row["age_band"] == "NT"
        assert mango_row["age_band"] is None

    def test_partially_recognized_row_still_produces_its_recognized_crop_row(self, workbook):
        """SCC5 ('Rice, Nonesuch') must still yield a Rice training row --
        the audit trail below is additional information, not a reason to
        drop the row's recoverable signal. Exactly one crop was recognized
        (Nonesuch doesn't count), so is_intercropped is False and the code
        stays unsuffixed -- same as any other single-recognized-crop row."""
        config = {**SHEET_CONFIGS[0]}
        rows = parse_sheet(workbook, config)
        assert "2022-SCC5" in {r["laboratory_code"] for r in rows}

    def test_collect_unrecognized_crop_texts_records_unmatched_segments(self, workbook):
        """SCC5 and SCC7 both name 'Nonesuch' alongside a recognized crop;
        SCC6 names only 'Nonesuch'. All three should count toward the same
        audit-trail entry -- repetition is exactly the signal worth acting
        on first when deciding which real-world synonym to add next."""
        config = {**SHEET_CONFIGS[0]}
        counts = collect_unrecognized_crop_texts(workbook, config)
        assert counts == {"Nonesuch": 3}

    def test_collect_unrecognized_crop_texts_ignores_fully_recognized_rows(self, workbook):
        """SCC1/SCC2/SCC3/SCC4 are all fully recognized (or STK-flagged, not
        crop-related) -- none of their text should appear in the audit
        trail."""
        config = {**SHEET_CONFIGS[0]}
        counts = collect_unrecognized_crop_texts(workbook, config)
        assert "Corn" not in counts
        assert "Rice" not in counts
        assert "Rice Upland" not in counts

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


class TestCropCoverageExpansionPlanSynonyms:
    """These crops/groupings were explicitly named in the crop-coverage-
    expansion plan (docs/superpowers/plans/2026-08-18-crop-coverage-
    expansion.md's Phase 1/2 row groupings, and the "9 free species" list in
    docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md
    section 2.1) -- the guide-table canonical crop already exists
    (confirmed against CANONICAL_CROPS), but CROP_VARIETY_MAP itself never
    actually got the corresponding entry. Found via the unrecognized-crop
    audit trail (collect_unrecognized_crop_texts) surfacing real source text
    that should have already resolved."""

    @pytest.mark.parametrize("text,expected", [
        # "9 free species": Camote/Yacon share Cassava's row, Soybean/Mongo
        # share Peanut's row, Santol/Caimito/Durian/Marang share Mango's row
        # -- a DIFFERENT tree-fruit grouping than the Phase 2 Avocado row
        # below (Chico/Atis/Guyabano), even though both are "tropical
        # fruit trees" -- verified against the design doc's exact wording,
        # not assumed, since getting this wrong would silently corrupt
        # these crops' N/P2O5/K2O targets.
        ("Camote", ("Cassava", None)),
        ("Soybean", ("Peanut", None)),
        ("Mongo", ("Peanut", None)),
        ("Monggo", ("Peanut", None)),
        ("Mungbeans", ("Peanut", None)),
        # Phase 1 vegetables: "Alugbati/Kangkong", "Lettuce/Malunggay",
        # "Squash/Upo/Patola", "Gabi/Ube/Yam/Taro",
        # "Cabbage/Pechay/Mustard/Spinach/Broccoli".
        ("Alugbati", ("Alugbati", None)),
        ("Kangkong", ("Alugbati", None)),
        ("Malunggay", ("Lettuce", None)),
        ("Patola", ("Squash", None)),
        ("Upo", ("Squash", None)),
        ("Taro", ("Gabi", None)),
        ("Broccoli", ("Cabbage", None)),
        ("Spinach", ("Cabbage", None)),
        # Phase 1 legumes: "Peas/Tapilan/Cowpea".
        ("Cowpea", ("Peas", None)),
        # Phase 2: standalone Pineapple row.
        ("Pineapple", ("Pineapple", None)),
        # Phase 2 tree/plantation row:
        # "Avocado/Chico/Atis/Guyabano/Apple/Cashew/Jackfruit/Macopa/
        # Tambis/Balimbing".
        ("Guyabano", ("Avocado", None)),
        ("Chico", ("Avocado", None)),
        ("Atis", ("Avocado", None)),
        # "9 free species" Mango row (see the comment above this
        # parametrize list) -- NOT the Avocado row above, despite also
        # being tropical fruit trees.
        ("Caimito", ("Mango", None)),
        ("Durian", ("Mango", None)),
        # Phase 2 tree/plantation row: "Lanzones/Rambutan/Mangosteen".
        ("Rambutan", ("Lanzones", None)),
        ("Mangosteen", ("Lanzones", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected


class TestAdditionalRealWorldVariants:
    """Not from the crop-coverage-expansion plan's own prose (which focused
    on the guide-table transcription work), but unambiguous real-world
    synonyms found in the same unrecognized-crop audit trail -- either a
    Cebuano/Visayan dialect word (Region 7 is the source region), an
    established plural, or a standard English synonym."""

    @pytest.mark.parametrize("text,expected", [
        ("Kamatis", ("Tomato", None)),  # Filipino word for tomato
        ("Peanuts", ("Peanut", None)),  # plural of an existing key
        ("Kalabasa", ("Squash", None)),  # Filipino word for squash
        ("Lady Finger", ("Eggplant", None)),  # international name for okra
        ("Lubi", ("Coconut", None)),  # Cebuano for coconut
        ("Mais", ("Corn", None)),  # Filipino/Cebuano for corn
        ("Saging", ("Banana", None)),  # Cebuano/Tagalog for banana
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

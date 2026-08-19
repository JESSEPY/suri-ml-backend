"""Upload-template parsing and validation.

Every derived value (total N, the four remarks, the three targets, both
formulations) is asserted to equal what the existing production functions
return for the same inputs -- the point of this module is that it never
re-implements agronomy, only routes to it.
"""

import io

import openpyxl
import pytest

from data.nutrient_recommendation_guide import k_cmol_to_ppm, target_unit_for_crop
from data.training_upload import (
    TEMPLATE_COLUMNS, build_training_row, parse_upload, validate_rows,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

VALID_ROW = {
    "laboratory_code": "2026-UP1",
    "crop": "Corn",
    "ph": "6.4",
    "organic_matter_pct": "2.8",
    "phosphorus_ppm": "18.0",
    "potassium_cmol_kg": "0.35",
}


def csv_bytes(rows, columns=TEMPLATE_COLUMNS):
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    return ("\n".join(lines) + "\n").encode()


class TestParseUpload:
    def test_reads_a_csv_into_dicts_keyed_by_header(self):
        rows = parse_upload(csv_bytes([VALID_ROW]), "upload.csv")
        assert len(rows) == 1
        assert rows[0]["laboratory_code"] == "2026-UP1"
        assert rows[0]["crop"] == "Corn"

    def test_reads_an_xlsx_into_the_same_shape(self):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(TEMPLATE_COLUMNS)
        sheet.append([VALID_ROW.get(column, "") for column in TEMPLATE_COLUMNS])
        buffer = io.BytesIO()
        workbook.save(buffer)

        rows = parse_upload(buffer.getvalue(), "upload.xlsx")
        assert len(rows) == 1
        assert rows[0]["laboratory_code"] == "2026-UP1"

    def test_rejects_an_unsupported_extension(self):
        with pytest.raises(ValueError, match="\\.csv, \\.xlsx"):
            parse_upload(b"whatever", "upload.pdf")

    def test_rejects_a_file_missing_a_required_column(self):
        columns = [c for c in TEMPLATE_COLUMNS if c != "ph"]
        with pytest.raises(ValueError, match="ph"):
            parse_upload(csv_bytes([VALID_ROW], columns=columns), "upload.csv")

    def test_skips_fully_blank_trailing_rows(self):
        content = csv_bytes([VALID_ROW]) + b",,,,,,,,,,\n"
        assert len(parse_upload(content, "upload.csv")) == 1


class TestValidateRowsAccepts:
    def test_accepts_a_well_formed_row(self):
        result = validate_rows([VALID_ROW], existing_codes=set())
        assert result.rejected == []
        assert len(result.accepted) == 1
        assert result.accepted[0]["laboratory_code"] == "2026-UP1"
        assert result.accepted[0]["row_number"] == 2

    def test_derived_values_match_the_production_functions_exactly(self):
        result = validate_rows([VALID_ROW], existing_codes=set())
        derived = result.accepted[0]["derived"]

        ph, om, p, k = 6.4, 2.8, 18.0, 0.35
        n_t, p_t, k_t = nutrient_requirements("Corn", om, p, k)
        app1, app2 = plan_applications(n_t, p_t, k_t, target_unit_for_crop("Corn"))

        assert derived["total_nitrogen_pct"] == pytest.approx(total_nitrogen_from_om(om))
        assert derived["ph_remark"] == derive_ph_remark(ph)
        assert derived["om_remark"] == derive_om_remark(om)
        assert derived["p_remark"] == derive_p_remark(p)
        assert derived["k_remark"] == derive_k_remark(k_cmol_to_ppm(k))
        assert derived["target_n"] == n_t
        assert derived["target_p2o5"] == p_t
        assert derived["target_k2o"] == k_t
        assert derived["app1_formulation"] == app1
        assert derived["app2_formulation"] == app2
        assert derived["target_unit"] == target_unit_for_crop("Corn")

    def test_optional_metadata_is_carried_through(self):
        row = {**VALID_ROW, "farm_area_sqm": "5000", "sampling_site": "Argao",
               "date_of_sampling": "2026-03-04"}
        accepted = validate_rows([row], existing_codes=set()).accepted[0]
        built = build_training_row(accepted["raw"])
        assert built["farm_area_sqm"] == 5000.0
        assert built["sampling_site"] == "Argao"
        assert built["date_of_sampling"] == "2026-03-04"

    def test_data_source_is_always_lab(self):
        accepted = validate_rows([VALID_ROW], existing_codes=set()).accepted[0]
        assert build_training_row(accepted["raw"])["data_source"] == "lab"


class TestValidateRowsRejects:
    def _reason(self, row, existing_codes=frozenset()):
        result = validate_rows([row], existing_codes=set(existing_codes))
        assert result.accepted == []
        assert len(result.rejected) == 1
        return result.rejected[0]["reason"]

    def test_unknown_crop_lists_the_valid_ones(self):
        reason = self._reason({**VALID_ROW, "crop": "Papaya"})
        assert "Papaya" in reason
        assert "Corn" in reason

    def test_missing_laboratory_code(self):
        assert "laboratory_code" in self._reason({**VALID_ROW, "laboratory_code": ""})

    def test_duplicate_of_an_existing_row(self):
        reason = self._reason(VALID_ROW, existing_codes={"2026-UP1"})
        assert "already" in reason.lower()

    def test_duplicate_within_the_same_file(self):
        result = validate_rows([VALID_ROW, VALID_ROW], existing_codes=set())
        assert len(result.accepted) == 1
        assert len(result.rejected) == 1
        assert "twice" in result.rejected[0]["reason"].lower()

    def test_ph_out_of_range(self):
        assert "ph" in self._reason({**VALID_ROW, "ph": "15"}).lower()

    def test_non_numeric_ph(self):
        assert "ph" in self._reason({**VALID_ROW, "ph": "acidic"}).lower()

    @pytest.mark.parametrize("column", ["organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg"])
    def test_missing_a_field_the_target_lookup_needs(self, column):
        reason = self._reason({**VALID_ROW, column: ""})
        assert column in reason

    def test_organic_matter_out_of_range(self):
        assert "organic_matter_pct" in self._reason({**VALID_ROW, "organic_matter_pct": "140"})

    def test_negative_phosphorus(self):
        assert "phosphorus_ppm" in self._reason({**VALID_ROW, "phosphorus_ppm": "-1"})

    def test_potassium_above_the_supported_ceiling(self):
        assert "potassium_cmol_kg" in self._reason({**VALID_ROW, "potassium_cmol_kg": "9"})

    def test_unresolvable_age_band(self):
        reason = self._reason({
            "laboratory_code": "2026-UP2", "crop": "Coconut", "ph": "6.4",
            "organic_matter_pct": "2.8", "phosphorus_ppm": "18.0",
            "potassium_cmol_kg": "0.35", "age_band": "700yrs",
        })
        assert "age_band" in reason

    def test_a_bad_row_does_not_block_the_good_rows_around_it(self):
        good_a = {**VALID_ROW, "laboratory_code": "2026-A"}
        bad = {**VALID_ROW, "laboratory_code": "2026-B", "crop": "Nope"}
        good_b = {**VALID_ROW, "laboratory_code": "2026-C"}

        result = validate_rows([good_a, bad, good_b], existing_codes=set())
        assert [entry["laboratory_code"] for entry in result.accepted] == ["2026-A", "2026-C"]
        assert result.rejected[0]["row_number"] == 3

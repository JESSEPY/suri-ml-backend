"""Parse and validate the admin upload template.

The template carries only RAW measurements. Every derived value -- total
nitrogen, the four remarks, the three targets, both formulations -- is
computed here by calling the same production functions /predict uses, so an
admin can neither supply nor mistype a derived value, and an uploaded row can
never disagree with what the service would have derived itself.

Validation is per-row and non-fatal: one malformed row is rejected with a
specific reason while every other row in the file still goes through.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import openpyxl

from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, resolve_age_band, resolve_cultivation_method,
    resolve_variety, target_unit_for_crop,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)

# Raw measurements only -- nothing derived, nothing an admin should be
# transcribing by hand. data_source is deliberately absent: every
# admin-uploaded row is real lab data and is stamped "lab" by
# build_training_row(), matching data/parse_real_soil_database.py.
REQUIRED_COLUMNS = [
    "laboratory_code", "crop", "ph",
    "organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg",
]

OPTIONAL_COLUMNS = [
    "variety", "cultivation_method", "age_band",
    "farm_area_sqm", "sampling_site", "date_of_sampling",
]

TEMPLATE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Mirrors main.py's SoilTestRequest Pydantic bounds exactly, so a row that
# would be refused at /predict is also refused as training data. The keys are
# annotated_types' constraint names on purpose -- Pydantic distinguishes ge
# from gt, and flattening both to "a low bound" is how farm_area_sqm's gt=0
# would quietly become ge=0 here. tests/test_training_upload.py's
# TestBoundsMatchTheApi reads SoilTestRequest's real metadata and fails if
# this table ever drifts from it again.
NUMERIC_BOUNDS: dict[str, dict[str, float]] = {
    "ph": {"ge": 0.0, "le": 14.0},
    "organic_matter_pct": {"ge": 0.0, "le": 100.0},
    "phosphorus_ppm": {"ge": 0.0, "le": 500.0},
    "potassium_cmol_kg": {"ge": 0.0, "le": 6.0},
    "farm_area_sqm": {"gt": 0.0, "le": 2_000_000.0},
}

_BOUND_CHECKS = {
    "ge": (lambda value, limit: value >= limit, "at least"),
    "gt": (lambda value, limit: value > limit, "greater than"),
    "le": (lambda value, limit: value <= limit, "at most"),
    "lt": (lambda value, limit: value < limit, "less than"),
}

# The ml_soil_training_data column order (data/parse_real_soil_database.py's
# OUTPUT_COLUMNS).
TRAINING_ROW_COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "cultivation_method", "age_band", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


@dataclass(frozen=True)
class ValidationResult:
    accepted: list[dict]
    rejected: list[dict]


class UploadFormatError(ValueError):
    """The file itself is unusable (wrong type, missing a required column) --
    as opposed to a single bad row, which is reported per-row instead."""


def _require_columns(header: list[str]) -> None:
    present = {(column or "").strip() for column in header}
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise UploadFormatError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected header: {', '.join(TEMPLATE_COLUMNS)}"
        )


def _parse_csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise UploadFormatError("The file is empty.")
    _require_columns(header)

    keys = [(column or "").strip() for column in header]
    rows = []
    for values in reader:
        if not any((value or "").strip() for value in values):
            continue
        rows.append({
            key: (values[index] if index < len(values) else "")
            for index, key in enumerate(keys)
        })
    return rows


def _parse_xlsx(content: bytes) -> list[dict]:
    workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    iterator = sheet.iter_rows(values_only=True)
    try:
        header = [str(cell) if cell is not None else "" for cell in next(iterator)]
    except StopIteration:
        raise UploadFormatError("The file is empty.")
    _require_columns(header)

    keys = [column.strip() for column in header]
    rows = []
    for values in iterator:
        if not any(value is not None and str(value).strip() for value in values):
            continue
        rows.append({
            key: ("" if index >= len(values) or values[index] is None else str(values[index]))
            for index, key in enumerate(keys)
        })
    return rows


def parse_upload(content: bytes, filename: str) -> list[dict]:
    lowered = filename.lower()
    if lowered.endswith(".csv"):
        return _parse_csv(content)
    if lowered.endswith((".xlsx", ".xlsm")):
        return _parse_xlsx(content)
    raise UploadFormatError(
        f"Unsupported file type {filename!r}. Upload .csv, .xlsx, or .xlsm."
    )


def _text(raw: dict, column: str) -> str:
    return str(raw.get(column, "") or "").strip()


def _optional_text(raw: dict, column: str) -> str | None:
    value = _text(raw, column)
    return value or None


def _number(raw: dict, column: str, required: bool) -> float | None:
    value = _text(raw, column)
    if not value:
        if required:
            raise ValueError(f"{column} is required and was empty.")
        return None
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{column} must be a number, got {value!r}.")

    bounds = NUMERIC_BOUNDS.get(column)
    if bounds is not None:
        for constraint, limit in bounds.items():
            holds, phrase = _BOUND_CHECKS[constraint]
            if not holds(parsed, limit):
                raise ValueError(f"{column} must be {phrase} {limit}, got {parsed}.")
    elif parsed <= 0:
        raise ValueError(f"{column} must be greater than 0, got {parsed}.")
    return parsed


def build_training_row(raw: dict) -> dict:
    """Turn one validated raw template row into a full ml_soil_training_data
    row. Raises ValueError/KeyError for anything unusable -- validate_rows()
    is what turns those into per-row rejection reasons."""
    crop = _text(raw, "crop")
    if crop not in CROPS:
        raise ValueError(f"Unknown crop {crop!r}. Valid crops: {', '.join(CROPS)}")

    ph = _number(raw, "ph", required=True)
    om = _number(raw, "organic_matter_pct", required=True)
    phosphorus = _number(raw, "phosphorus_ppm", required=True)
    potassium = _number(raw, "potassium_cmol_kg", required=True)
    farm_area = _number(raw, "farm_area_sqm", required=False)

    variety = _optional_text(raw, "variety")
    cultivation_method = _optional_text(raw, "cultivation_method")
    age_band = _optional_text(raw, "age_band")

    # Raises KeyError for an age_band this crop's guide rows don't have --
    # the same failure /predict turns into a 422.
    target_n, target_p2o5, target_k2o = nutrient_requirements(
        crop, om, phosphorus, potassium,
        variety=variety, cultivation_method=cultivation_method, age_band=age_band,
    )
    target_unit = target_unit_for_crop(crop)
    app1, app2 = plan_applications(target_n, target_p2o5, target_k2o, target_unit)

    return {
        "laboratory_code": _text(raw, "laboratory_code"),
        "data_source": "lab",
        "sampling_site": _optional_text(raw, "sampling_site"),
        "date_of_sampling": _optional_text(raw, "date_of_sampling"),
        "crop": crop,
        "variety": resolve_variety(crop, variety),
        "cultivation_method": resolve_cultivation_method(crop, cultivation_method),
        "age_band": resolve_age_band(crop, age_band),
        "farm_area_sqm": farm_area,
        "ph": ph,
        "organic_matter_pct": om,
        "total_nitrogen_pct": round(total_nitrogen_from_om(om), 4),
        "phosphorus_ppm": phosphorus,
        "potassium_cmol_kg": potassium,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": derive_ph_remark(ph),
        "om_remark": derive_om_remark(om),
        "p_remark": derive_p_remark(phosphorus),
        "k_remark": derive_k_remark(k_cmol_to_ppm(potassium)),
        "target_n": target_n,
        "target_p2o5": target_p2o5,
        "target_k2o": target_k2o,
        "target_unit": target_unit,
        "app1_formulation": app1,
        "app2_formulation": app2,
    }


# Fields the admin reviews before committing -- everything the service
# derived rather than everything it stored.
_REVIEW_FIELDS = [
    "total_nitrogen_pct", "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


def validate_rows(raw_rows: list[dict], existing_codes: set[str]) -> ValidationResult:
    """One rejected row never blocks the rest of the file. row_number is
    1-based including the header, so it matches what the admin sees in Excel."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_in_file: set[str] = set()

    for offset, raw in enumerate(raw_rows):
        row_number = offset + 2
        code = _text(raw, "laboratory_code")

        if not code:
            rejected.append({
                "row_number": row_number, "laboratory_code": None,
                "reason": "laboratory_code is required and was empty.",
            })
            continue
        if code in seen_in_file:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"laboratory_code {code!r} appears twice in this file.",
            })
            continue
        if code in existing_codes:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"laboratory_code {code!r} is already in the training data.",
            })
            continue

        try:
            built = build_training_row(raw)
        except KeyError as error:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"Unrecognized age_band for this crop: {error}",
            })
            continue
        except ValueError as error:
            rejected.append({
                "row_number": row_number, "laboratory_code": code, "reason": str(error),
            })
            continue

        seen_in_file.add(code)
        accepted.append({
            "row_number": row_number,
            "laboratory_code": code,
            "raw": dict(raw),
            "derived": {field: built[field] for field in _REVIEW_FIELDS},
        })

    return ValidationResult(accepted=accepted, rejected=rejected)

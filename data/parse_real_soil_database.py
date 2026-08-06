"""Parses the real Region 7 National Soil Database export into training rows
matching the same shape ml/train.py already consumes (see
data.ingest_to_supabase.TABLE's schema, previously populated by the now-
retired data/generate_synthetic_data.py).

Column positions are NOT consistent across the source workbook's sheets —
confirmed by direct inspection: even within the three near-identical
"Laboratory Details" sheets, 2023's chemistry columns shift by one relative
to 2022/2024 starting at Organic Matter. Every column is located by matching
its header text on that specific sheet (find_column), never a hardcoded
index shared across sheets.
"""

from __future__ import annotations

import csv
import re
import sys

import openpyxl

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark
from data.nutrient_recommendation_guide import (
    CROPS, BEARING_AGE_BAND, k_cmol_to_ppm, target_unit_for_crop,
    resolve_variety, resolve_cultivation_method, resolve_age_band,
)

CROP_VARIETY_MAP: dict[str, tuple[str, str | None]] = {
    "corn": ("Corn", None),
    "opv corn": ("Corn", "OPV"),
    "corn opv": ("Corn", "OPV"),
    "hybrid corn": ("Corn", "Hybrid"),
    "rice": ("Rice", None),
    "hybrid rice": ("Rice", "Hybrid"),
    "inbred rice": ("Rice", "Inbred"),
    "cassava": ("Cassava", None),
    "peanut": ("Peanut", None),
    "mango": ("Mango", None),
    "cacao": ("Cacao", None),
    "banana": ("Banana", None),
    "tomato": ("Tomato", None),
    "coconut": ("Coconut", None),
}

_LAB_CODE_DIGITS = re.compile(r"\d+")

_CULTIVATION_METHOD_KEYWORDS = (
    ("lowland wet", "Lowland Wet"),
    ("lowland dry", "Lowland Dry"),
    ("upland", "Upland"),
)

# crop -> its guide age-band labels, matched as case-insensitive substrings
# of the free-text Crops field. Banana is deliberately absent — it isn't
# age-banded (see data.nutrient_recommendation_guide).
_AGE_BAND_LABELS: dict[str, tuple[str, ...]] = {
    "Coconut": ("Below 1yr", "1-5yrs", "6-8yrs", "9yrs & up"),
    "Cacao": ("Over 7yrs", "3yrs", "2yrs", "1yr", "NT"),
    "Mango": ("7-10yrs & up", "10yrs & up", "4-6yrs", "1-3yrs", "NT"),
}


def parse_numeric_cell(value: object) -> float | None:
    """A source cell is a real lab measurement only if openpyxl parsed it as
    a number. Anything else — None, or any string ("STK"/"NIL"/"NGL"
    sentinels marking "tested by kit instead of lab", or a data-entry typo
    like "4..67") — means that field is absent for this row, not that the
    row is unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_lab_code(raw_code: str, year_label: str) -> str:
    """'SCC 417' / 'SCC417' / 'SCC 0417' (any year) -> '{year}-SCC417',
    collapsing whitespace/zero-padding/case differences within a year, while
    keeping different years distinct (the same number is reused every year
    for an unrelated physical sample)."""
    digits = _LAB_CODE_DIGITS.findall(str(raw_code))
    number = int("".join(digits)) if digits else 0
    return f"{year_label}-SCC{number}"


def normalize_crop(raw_text: str | None) -> tuple[str, str | None] | None:
    """Returns (crop, variety) for a real, single, supported crop; None for
    multi-crop text, empty text, or any crop not in the 9-crop list
    (including the three dropped for insufficient real coverage).

    The source Crops field is free text that often trails the crop (+
    variety) name with cultivation-method or age-band words the caller
    separately extracts via normalize_cultivation_method/normalize_age_band
    on this same raw text (e.g. "Rice Upland", "Coconut 6-8yrs") -- so a
    match is accepted either on the whole cell text or on a leading,
    whole-word prefix of it. Longest keys are tried first so a
    variety-specific key (e.g. "hybrid rice") wins over its bare-crop
    substring ("rice")."""
    if not raw_text or not str(raw_text).strip():
        return None
    parts = [p.strip().lower() for p in re.split(r"[,/]", str(raw_text)) if p.strip()]
    if len(parts) != 1:
        return None
    text = parts[0]
    mapped = None
    for key in sorted(CROP_VARIETY_MAP, key=len, reverse=True):
        if text == key or text.startswith(key + " "):
            mapped = CROP_VARIETY_MAP[key]
            break
    if mapped is None or mapped[0] not in CROPS:
        return None
    return mapped


def normalize_cultivation_method(crop: str, raw_text: str) -> str | None:
    """Best-effort keyword match against the free-text Crops field. Returns
    None (not a guess) when the field doesn't state one — the caller
    resolves that to the crop's default via
    data.nutrient_recommendation_guide.resolve_cultivation_method."""
    if crop != "Rice":
        return None
    lowered = str(raw_text).lower()
    for keyword, method in _CULTIVATION_METHOD_KEYWORDS:
        if keyword in lowered:
            return method
    return None


def normalize_age_band(crop: str, raw_text: str) -> str | None:
    """Best-effort keyword match. Returns None (not a guess) when the field
    doesn't state one, or the crop isn't age-banded at all (Banana)."""
    labels = _AGE_BAND_LABELS.get(crop)
    if not labels:
        return None
    lowered = str(raw_text).lower()
    for label in labels:
        if label.lower() in lowered:
            return label
    return None


def find_column(header_row: tuple, target_text: str) -> int:
    """Locate target_text's column index in a specific sheet's header row.
    Raises ValueError rather than returning a guess — a missing expected
    column means the sheet's layout has drifted further than this parser
    accounts for, and must be re-verified against the live file, not
    silently worked around."""
    for index, value in enumerate(header_row):
        if value == target_text:
            return index
    raise ValueError(f"Column header {target_text!r} not found in sheet")


OUTPUT_CSV = "suri_real_training_data.csv"

DEFAULT_XLSX_PATH = "../suri/public/Region 7-National Soil Database_2022-2025.xlsx"

LAB_DETAIL_HEADERS = {
    "ph": "pH (H2O, 1:1)",
    "phosphorus_ppm": "Available Phosphorus (Avail. P), mg/kg",
    "potassium_cmol_kg": "Exchangeable Potassium (K), cmol/kg",
    "organic_matter_pct": "Organic Matter (OM), %",
    "total_nitrogen_pct": "Total Nitrogen (N), %",
}

SOIL_DATA_2025_HEADERS = {
    "ph": "pH",
    "organic_matter_pct": "OM",
    "phosphorus_ppm": "P",
    "potassium_cmol_kg": "K",
    "total_nitrogen_pct": "TOTAL N",
}

# One entry per usable source sheet. header_row/chem_header_row/data_start_row
# are 1-based, matching openpyxl's own row numbering. Column positions are
# NOT listed here — find_column resolves them per-sheet from chem_headers at
# chem_header_row, since they drift between sheets.
SHEET_CONFIGS = [
    {"name": "Laboratory Details2022", "year": "2022", "chem_header_row": 5,
     "data_start_row": 6, "lab_code_col": 1, "crop_col": 14, "chem_headers": LAB_DETAIL_HEADERS},
    {"name": "Laboratory Details2023", "year": "2023", "chem_header_row": 5,
     "data_start_row": 6, "lab_code_col": 1, "crop_col": 14, "chem_headers": LAB_DETAIL_HEADERS},
    {"name": "Laboratory Details2024", "year": "2024", "chem_header_row": 5,
     "data_start_row": 6, "lab_code_col": 1, "crop_col": 14, "chem_headers": LAB_DETAIL_HEADERS},
    {"name": "Soil Data 2025", "year": "2025", "chem_header_row": 2,
     "data_start_row": 4, "lab_code_col": 0, "crop_col": 9, "chem_headers": SOIL_DATA_2025_HEADERS},
]

OUTPUT_COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "cultivation_method", "age_band", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


def build_column_map(worksheet, config: dict) -> dict[str, int]:
    header_row = next(worksheet.iter_rows(
        min_row=config["chem_header_row"], max_row=config["chem_header_row"], values_only=True
    ))
    return {field: find_column(header_row, text) for field, text in config["chem_headers"].items()}


def parse_sheet(worksheet, config: dict) -> list[dict]:
    """Reads one source sheet into raw (pre-target) records: real chemistry
    values where present, None where a field was field-tested instead of
    lab-tested (or simply blank). Rows with no crop, a multi-crop entry, an
    unsupported crop, or no real pH are excluded outright — everything else
    is passed through for build_training_row to turn into a full row (or
    reject, if OM/P/K are missing)."""
    column_map = build_column_map(worksheet, config)
    rows: list[dict] = []

    for row in worksheet.iter_rows(min_row=config["data_start_row"], values_only=True):
        lab_code_raw = row[config["lab_code_col"]]
        if not lab_code_raw or not str(lab_code_raw).strip():
            continue

        crop_text = row[config["crop_col"]]
        crop_variety = normalize_crop(crop_text)
        if crop_variety is None:
            continue
        crop, variety = crop_variety

        ph = parse_numeric_cell(row[column_map["ph"]])
        if ph is None:
            continue

        rows.append({
            "laboratory_code": normalize_lab_code(lab_code_raw, config["year"]),
            "crop": crop,
            "variety": variety,
            "cultivation_method": normalize_cultivation_method(crop, str(crop_text)),
            "age_band": normalize_age_band(crop, str(crop_text)),
            "ph": ph,
            "organic_matter_pct": parse_numeric_cell(row[column_map["organic_matter_pct"]]),
            "phosphorus_ppm": parse_numeric_cell(row[column_map["phosphorus_ppm"]]),
            "potassium_cmol_kg": parse_numeric_cell(row[column_map["potassium_cmol_kg"]]),
            "total_nitrogen_pct": parse_numeric_cell(row[column_map["total_nitrogen_pct"]]),
        })

    return rows


def build_training_row(raw: dict) -> dict | None:
    """Turns one raw parsed record into a full training row, or None if it
    can't get a real target (nutrient_requirements needs numeric OM/P/K —
    see the design spec for why this is a hard requirement, not something to
    patch with an invented number)."""
    om = raw["organic_matter_pct"]
    p = raw["phosphorus_ppm"]
    k = raw["potassium_cmol_kg"]
    if om is None or p is None or k is None:
        return None

    ph = raw["ph"]
    crop = raw["crop"]
    variety = raw["variety"]
    cultivation_method = raw["cultivation_method"]
    age_band = raw["age_band"]

    total_nitrogen = raw["total_nitrogen_pct"]
    if total_nitrogen is None:
        total_nitrogen = total_nitrogen_from_om(om)

    n, p2o5, k2o = nutrient_requirements(
        crop, om, p, k,
        variety=variety, cultivation_method=cultivation_method, age_band=age_band,
    )
    app1, app2 = plan_applications(n, p2o5, k2o, target_unit_for_crop(crop))
    k_ppm = k_cmol_to_ppm(k)

    return {
        "laboratory_code": raw["laboratory_code"],
        "data_source": "lab",
        "sampling_site": None,
        "date_of_sampling": None,
        "crop": crop,
        # Resolved, not raw — must match what nutrient_requirements() (just
        # above) actually used to compute n/p2o5/k2o, including its
        # crop-specific defaults (e.g. Rice with no stated method resolves
        # to "Lowland Wet", not the literal string "None"). Storing the raw
        # unresolved value here would desynchronize this feature from the
        # label the model is trained to predict.
        "variety": resolve_variety(crop, variety),
        "cultivation_method": resolve_cultivation_method(crop, cultivation_method),
        "age_band": resolve_age_band(crop, age_band),
        "farm_area_sqm": None,
        "ph": ph,
        "organic_matter_pct": om,
        "total_nitrogen_pct": total_nitrogen,
        "phosphorus_ppm": p,
        "potassium_cmol_kg": k,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": derive_ph_remark(ph),
        "om_remark": derive_om_remark(om),
        "p_remark": derive_p_remark(p),
        "k_remark": derive_k_remark(k_ppm),
        "target_n": n, "target_p2o5": p2o5, "target_k2o": k2o,
        "target_unit": target_unit_for_crop(crop),
        "app1_formulation": app1, "app2_formulation": app2,
    }


def parse_workbook(xlsx_path: str) -> tuple[list[dict], dict]:
    """Returns (training_rows, data_quality_report). The report records why
    rows were dropped at each stage, per sheet, so a re-run after the source
    file changes (e.g. the 2025 sheets fill in further) is easy to sanity-check."""
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    training_rows: list[dict] = []
    report: dict = {"sheets": {}}

    for config in SHEET_CONFIGS:
        worksheet = workbook[config["name"]]
        raw_rows = parse_sheet(worksheet, config)
        built = [build_training_row(raw) for raw in raw_rows]
        usable = [row for row in built if row is not None]
        training_rows.extend(usable)
        report["sheets"][config["name"]] = {
            "rows_with_real_crop_and_ph": len(raw_rows),
            "rows_with_complete_om_p_k": len(usable),
            "dropped_for_incomplete_chemistry": len(raw_rows) - len(usable),
        }

    report["total_training_rows"] = len(training_rows)
    report["rows_by_crop"] = {}
    report["rows_by_variety"] = {}
    report["rows_by_cultivation_method"] = {}
    report["rows_by_age_band"] = {}
    for row in training_rows:
        report["rows_by_crop"][row["crop"]] = report["rows_by_crop"].get(row["crop"], 0) + 1
        report["rows_by_variety"][row["variety"]] = report["rows_by_variety"].get(row["variety"], 0) + 1
        report["rows_by_cultivation_method"][row["cultivation_method"]] = \
            report["rows_by_cultivation_method"].get(row["cultivation_method"], 0) + 1
        report["rows_by_age_band"][row["age_band"]] = report["rows_by_age_band"].get(row["age_band"], 0) + 1

    return training_rows, report


def write_csv(rows: list[dict], output_path: str = OUTPUT_CSV) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XLSX_PATH
    rows, report = parse_workbook(xlsx_path)
    write_csv(rows)
    print(f"Wrote {len(rows)} training rows to {OUTPUT_CSV}")
    print(f"Per-sheet: {report['sheets']}")
    print(f"Per-crop: {report['rows_by_crop']}")
    print(f"Per-variety: {report['rows_by_variety']}")
    print(f"Per-cultivation-method: {report['rows_by_cultivation_method']}")
    print(f"Per-age-band: {report['rows_by_age_band']}")

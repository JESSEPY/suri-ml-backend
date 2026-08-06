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

import re

from data.nutrient_recommendation_guide import CROPS, BEARING_AGE_BAND

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
    (including the three dropped for insufficient real coverage)."""
    if not raw_text or not str(raw_text).strip():
        return None
    parts = [p.strip().lower() for p in re.split(r"[,/]", str(raw_text)) if p.strip()]
    if len(parts) != 1:
        return None
    mapped = CROP_VARIETY_MAP.get(parts[0])
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

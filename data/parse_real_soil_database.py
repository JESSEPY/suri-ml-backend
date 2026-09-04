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

An intercropped Crops cell (more than one species named, e.g. "Corn, Rice")
expands into one training row per *recognized* species sharing that one
sample's chemistry reading, instead of being dropped outright — see
normalize_crops and parse_sheet. A genuinely single-crop cell is unaffected:
same normalize_crop path, same unsuffixed laboratory_code, as before this
was added.
"""

from __future__ import annotations

import csv
import re
import sys

import openpyxl

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark
from data import nutrient_recommendation_guide as guide
from data.training_upload import NUMERIC_BOUNDS
from data.nutrient_recommendation_guide import (
    k_cmol_to_ppm, target_unit_for_crop,
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
    # --- Corn / Rice / Sorghum real-world label variants -----------------
    # "corn tinigib"/"corn tiniguib" need no entry here: they already
    # resolve via the bare "corn" key's existing prefix match ("corn " is a
    # leading substring of both) -- confirmed by test before adding this.
    "sweetcorn": ("Corn", None),
    "sweet corn": ("Corn", None),
    "corn hybrid": ("Corn", "Hybrid"),
    "corn hybrid yellow": ("Corn", "Hybrid"),
    "sorghum": ("Corn", None),
    # Guide lists Adlai only on Corn's Hybrid row (Corn /Sorghum/Adlai) --
    # the crop-coverage-expansion plan deliberately did not register it as
    # a CROP_ALIASES entry for exactly this reason (a whole-crop alias has
    # no way to carry "Hybrid" and would wrongly resolve to Native). This
    # map entry carries the variety explicitly, so it doesn't have that
    # problem.
    "adlai": ("Corn", "Hybrid"),
    "upland rice": ("Rice", None),
    "rice inbred": ("Rice", "Inbred"),
    # --- Vegetable real-world label variants ------------------------------
    "lettuce": ("Lettuce", None),
    "malunggay": ("Lettuce", None),
    "pechay": ("Cabbage", None),
    "broccoli": ("Cabbage", None),
    "spinach": ("Cabbage", None),
    "ampalaya": ("Ampalaya", None),
    "eggplant": ("Eggplant", None),
    "talong": ("Eggplant", None),
    "okra": ("Eggplant", None),
    "lady finger": ("Eggplant", None),
    "sayote": ("Chayote", None),
    "cabbage": ("Cabbage", None),
    "red cabbage": ("Cabbage", None),
    "repolyo": ("Cabbage", None),
    "squash": ("Squash", None),
    "kalabasa": ("Squash", None),
    "patola": ("Squash", None),
    "upo": ("Squash", None),
    "cauliflower": ("Cauliflower", None),
    "carrots": ("Radish", None),
    "radish": ("Radish", None),
    "raddish": ("Radish", None),
    "cucumber": ("Cucumber", None),
    "pipino": ("Cucumber", None),
    "watermelon": ("Cucumber", None),
    # "Alugbati/Kangkong" is its own guide row (CANONICAL_CROPS has
    # "Alugbati" directly) -- not a synonym for another already-coded crop.
    "alugbati": ("Alugbati", None),
    "kangkong": ("Alugbati", None),
    # --- Condiment / rootcrop / legume / sugarcane real-world variants ---
    "sugarcane": ("Sugarcane", None),
    "onions": ("Onion", None),
    "onion": ("Onion", None),
    "sibuyas": ("Onion", None),
    "spring onion": ("Onion", "Spring"),
    "spring onions": ("Onion", "Spring"),
    "green onion": ("Onion", "Spring"),
    "ginger": ("Ginger", None),
    "gabi": ("Gabi", None),
    "ube": ("Gabi", None),
    "ubi": ("Gabi", None),
    "ube kinampay": ("Gabi", None),
    # "Gabi/Ube/Yam/Taro" is one guide row -- Taro is the missing synonym.
    "taro": ("Gabi", None),
    "kamatis": ("Tomato", None),
    "peanuts": ("Peanut", None),
    "kamote": ("Cassava", None),
    "camote": ("Cassava", None),
    "sweet potato": ("Cassava", None),
    "beans": ("Beans", None),
    "string beans": ("Beans", None),
    "baguio beans": ("Beans", None),
    "mung beans": ("Peanut", None),
    "mungbean": ("Peanut", None),
    "mungbeans": ("Peanut", None),
    "mongo": ("Peanut", None),
    "monggo": ("Peanut", None),
    "soybean": ("Peanut", None),
    "cowpea": ("Peas", None),
    # Guide co-lists "Garlic/ Chili Pepper" on one row (CROP_ALIASES already
    # has "Chili Pepper" -> Garlic); real labels overwhelmingly use "Sili"
    # (Filipino for chili) rather than the English name. Bell/sweet pepper
    # is botanically distinct and deliberately NOT mapped here -- no guide
    # entry to route it to.
    "sili haba": ("Garlic", None),
    "sili labuyo": ("Garlic", None),
    "sili espada": ("Garlic", None),
    "sili": ("Garlic", None),
    "sili atsal": ("Garlic", None),
    "finger pepper": ("Garlic", None),
    "hot pepper": ("Garlic", None),
    "green chili": ("Garlic", None),
    # --- Tree / plantation crop real-world label variants -----------------
    # "banana lakatan" needs no entry: already resolves via the existing
    # bare "banana" key's prefix match (Lakatan is a Banana variety name;
    # the guide has no Banana variety axis, so the detail is discarded --
    # same as how the pre-existing map already handles varietyless crops).
    "calamansi": ("Citrus", None),
    "lemon": ("Citrus", None),
    "avocado": ("Avocado", None),
    "jackfruit": ("Avocado", None),
    "pineapple": ("Pineapple", None),
    # Phase 2 tree/plantation row:
    # "Avocado/Chico/Atis/Guyabano/Apple/Cashew/Jackfruit/Macopa/Tambis/Balimbing".
    "guyabano": ("Avocado", None),
    "chico": ("Avocado", None),
    "atis": ("Avocado", None),
    # "9 free species" Mango row -- Santol/Caimito/Durian/Marang, a
    # DIFFERENT tropical-fruit-tree grouping than the Avocado row directly
    # above, despite the surface-level similarity. Verified against the
    # design doc's exact wording, not assumed -- getting this wrong would
    # silently corrupt these crops' N/P2O5/K2O targets.
    "caimito": ("Mango", None),
    "durian": ("Mango", None),
    # Phase 2 tree/plantation row: "Lanzones/Rambutan/Mangosteen".
    "rambutan": ("Lanzones", None),
    "mangosteen": ("Lanzones", None),
    # --- Cebuano/Visayan dialect real-world label variants (Region 7 is
    # the source region for this workbook) -------------------------------
    "lubi": ("Coconut", None),
    "mais": ("Corn", None),
    "saging": ("Banana", None),
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


def normalize_lab_code(
    raw_code: str, year_label: str, crop: str | None = None, variety: str | None = None,
) -> str:
    """'SCC 417' / 'SCC417' / 'SCC 0417' (any year) -> '{year}-SCC417',
    collapsing whitespace/zero-padding/case differences within a year, while
    keeping different years distinct (the same number is reused every year
    for an unrelated physical sample).

    `crop`, when given, appends a '-{CROP}' suffix, and `variety` (when also
    given) a further '-{VARIETY}' -- used only when one source row (one
    physical sample) expands into more than one training row via intercrop
    splitting, so each still gets a unique code. The upsert primary key in
    data/ingest_to_supabase.py is (laboratory_code, data_source); two rows
    sharing one unsuffixed code would silently overwrite each other on
    ingest. Crop alone is not always enough: normalize_crops dedupes by
    (crop, variety), so two intercrop segments can legitimately resolve to
    the same crop with different varieties (e.g. "Corn Tundan" -> (Corn,
    None) and "Corn OPV" -> (Corn, OPV), real source: '2022-SCC877') and
    both survive as distinct rows -- a crop-only suffix would collide again.
    Omitting crop (the default) reproduces the exact
    pre-intercrop-expansion code, since already-ingested Supabase rows are
    keyed on it."""
    digits = _LAB_CODE_DIGITS.findall(str(raw_code))
    number = int("".join(digits)) if digits else 0
    code = f"{year_label}-SCC{number}"
    if crop:
        code = f"{code}-{crop.upper()}"
        if variety:
            code = f"{code}-{variety.upper()}"
    return code


def _match_crop_key(lowered_segment: str) -> tuple[str, str | None] | None:
    """Matches one already-isolated, already-lowercased crop-name segment
    (no comma/slash, no surrounding whitespace) against CROP_VARIETY_MAP.
    A match is accepted either on the whole segment or on a leading,
    whole-word prefix of it (so "coconut 6-8yrs" matches the bare
    "coconut" key). Longest keys are tried first so a variety-specific key
    (e.g. "hybrid rice") wins over its bare-crop substring ("rice")."""
    for key in sorted(CROP_VARIETY_MAP, key=len, reverse=True):
        if lowered_segment == key or lowered_segment.startswith(key + " "):
            mapped = CROP_VARIETY_MAP[key]
            if mapped[0] in guide.CANONICAL_CROPS:
                return mapped
    return None


def _split_crop_text(raw_text: str | None) -> list[str]:
    """Splits the free-text Crops cell on commas/slashes into individual
    species segments, preserving each segment's original casing/qualifier
    words (e.g. "Rice Upland") -- only surrounding whitespace is stripped.
    Returns [] for blank/None text."""
    if not raw_text or not str(raw_text).strip():
        return []
    return [p.strip() for p in re.split(r"[,/]", str(raw_text)) if p.strip()]


def normalize_crop(raw_text: str | None) -> tuple[str, str | None] | None:
    """Returns (crop, variety) for a real, single, supported crop; None for
    multi-crop text, empty text, or any crop CROP_VARIETY_MAP doesn't have
    an entry for.

    The source Crops field is free text that often trails the crop (+
    variety) name with cultivation-method or age-band words the caller
    separately extracts via normalize_cultivation_method/normalize_age_band
    on this same raw text (e.g. "Rice Upland", "Coconut 6-8yrs") -- so a
    match is accepted either on the whole cell text or on a leading,
    whole-word prefix of it. Longest keys are tried first so a
    variety-specific key (e.g. "hybrid rice") wins over its bare-crop
    substring ("rice")."""
    parts = _split_crop_text(raw_text)
    if len(parts) != 1:
        return None
    return _match_crop_key(parts[0].lower())


def normalize_crops(raw_text: str | None) -> list[tuple[str, str | None, str]]:
    """The intercrop-aware counterpart to normalize_crop: for free text
    naming more than one species (e.g. "Rice Upland, Corn"), returns one
    (crop, variety, raw_segment) tuple per *recognized, distinct* species
    instead of rejecting the whole row. raw_segment is that species' own
    slice of the original text (original casing, not lowercased) -- callers
    use it to scope cultivation-method/age-band inference to just that crop
    rather than the whole cell. An unrecognized segment (dialect name not in
    CROP_VARIETY_MAP, typo, non-crop text) is silently dropped, same as how
    a wholly-unrecognized single crop is already dropped by normalize_crop
    -- so a row with one real crop and one unmatched one still contributes
    the real one instead of being discarded outright. Single-crop text
    works here too (returns a one-element list).

    Two segments that resolve to the SAME (crop, variety) are collapsed to
    one entry (first occurrence kept) -- real source text regularly names
    the same crop twice under different synonyms within one cell (e.g.
    "Cassava, Kamote" -- Kamote is just the Filipino word for Cassava;
    "Okra, Eggplant" -- Okra has no separate guide row and maps to
    Eggplant's). Emitting both would produce two training rows with
    identical chemistry AND identical targets for one physical sample, and
    (since normalize_lab_code suffixes by crop) the same laboratory_code
    twice, which the ingest upsert's uniqueness key correctly rejects."""
    results = []
    seen: set[tuple[str, str | None]] = set()
    for raw_segment in _split_crop_text(raw_text):
        mapped = _match_crop_key(raw_segment.lower())
        if mapped is not None and mapped not in seen:
            seen.add(mapped)
            results.append((mapped[0], mapped[1], raw_segment))
    return results


def unmatched_crop_segments(raw_text: str | None) -> list[str]:
    """normalize_crops()'s counterpart: returns exactly the segments of
    raw_text it silently dropped (no CROP_VARIETY_MAP entry matches),
    original casing and qualifier words intact -- a human reviewer deciding
    whether to add a new synonym needs to see the real text, not a
    normalized version of it. Used only to build an audit trail (see
    collect_unrecognized_crop_texts); normalize_crops() itself stays
    unaware this exists, so its own behavior/tests are untouched."""
    return [
        segment for segment in _split_crop_text(raw_text)
        if _match_crop_key(segment.lower()) is None
    ]


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
    lab-tested (or simply blank). Rows with no recognized crop at all, or no
    real pH, are excluded outright — everything else is passed through for
    build_training_row to turn into a full row (or reject, if OM/P/K are
    missing).

    An intercropped cell (more than one species named) expands into one
    record per *recognized* species (see normalize_crops), all sharing this
    one physical sample's chemistry reading — the pH/OM/P/K measurement is
    a property of the field, not of any single crop growing in it. Each
    species' laboratory_code gets a crop suffix in that case (see
    normalize_lab_code) so the two rows don't collide on ingest; a
    genuinely single-crop row keeps its plain, unsuffixed code exactly as
    before, so already-ingested rows are unaffected by this change."""
    column_map = build_column_map(worksheet, config)
    rows: list[dict] = []

    for row in worksheet.iter_rows(min_row=config["data_start_row"], values_only=True):
        lab_code_raw = row[config["lab_code_col"]]
        if not lab_code_raw or not str(lab_code_raw).strip():
            continue

        crop_text = row[config["crop_col"]]
        crop_matches = normalize_crops(crop_text)
        if not crop_matches:
            continue

        ph = parse_numeric_cell(row[column_map["ph"]])
        if ph is None:
            continue

        organic_matter_pct = parse_numeric_cell(row[column_map["organic_matter_pct"]])
        phosphorus_ppm = parse_numeric_cell(row[column_map["phosphorus_ppm"]])
        potassium_cmol_kg = parse_numeric_cell(row[column_map["potassium_cmol_kg"]])
        total_nitrogen_pct = parse_numeric_cell(row[column_map["total_nitrogen_pct"]])

        is_intercropped = len(crop_matches) > 1
        for crop, variety, raw_segment in crop_matches:
            rows.append({
                "laboratory_code": normalize_lab_code(
                    lab_code_raw, config["year"],
                    crop if is_intercropped else None,
                    variety if is_intercropped else None,
                ),
                "crop": crop,
                "variety": variety,
                "cultivation_method": normalize_cultivation_method(crop, raw_segment),
                "age_band": normalize_age_band(crop, raw_segment),
                "ph": ph,
                "organic_matter_pct": organic_matter_pct,
                "phosphorus_ppm": phosphorus_ppm,
                "potassium_cmol_kg": potassium_cmol_kg,
                "total_nitrogen_pct": total_nitrogen_pct,
            })

    return rows


def collect_unrecognized_crop_texts(worksheet, config: dict) -> dict[str, int]:
    """Tallies the free-text crop segments this sheet's lab-coded rows name
    that unmatched_crop_segments() couldn't resolve -- the audit trail
    normalize_crop()/normalize_crops() deliberately don't keep themselves
    (they only report the row-level accept/reject outcome). Counts
    duplicates: the same unrecognized text recurring across many rows is
    exactly the signal worth acting on first when deciding what to add to
    CROP_VARIETY_MAP next, instead of re-discarding the same near-misses on
    every re-run. Mirrors parse_sheet()'s own row-inclusion gate (lab-coded
    rows only) so the two stay comparable; a second pass over the same
    worksheet rather than piggybacking on parse_sheet's own loop, so
    parse_sheet's return contract (and the tests pinning it) stays
    untouched."""
    counts: dict[str, int] = {}
    for row in worksheet.iter_rows(min_row=config["data_start_row"], values_only=True):
        lab_code_raw = row[config["lab_code_col"]]
        if not lab_code_raw or not str(lab_code_raw).strip():
            continue
        crop_text = row[config["crop_col"]]
        for segment in unmatched_crop_segments(crop_text):
            counts[segment] = counts.get(segment, 0) + 1
    return counts


def _within_bounds(field: str, value: float) -> bool:
    """True if value satisfies every constraint NUMERIC_BOUNDS[field]
    declares (imported from data.training_upload, the exact same table
    tests/test_training_upload.py::TestBoundsMatchTheApi keeps in sync with
    main.py's live SoilTestRequest bounds). A field with no entry there
    (e.g. total_nitrogen_pct -- not an upload-template column) has no check
    here either, matching the upload validator's own scope."""
    bounds = NUMERIC_BOUNDS.get(field)
    if not bounds:
        return True
    if "ge" in bounds and value < bounds["ge"]:
        return False
    if "gt" in bounds and value <= bounds["gt"]:
        return False
    if "le" in bounds and value > bounds["le"]:
        return False
    if "lt" in bounds and value >= bounds["lt"]:
        return False
    return True


def build_training_row(raw: dict) -> dict | None:
    """Turns one raw parsed record into a full training row, or None if it
    can't get a real target (nutrient_requirements needs numeric OM/P/K —
    see the design spec for why this is a hard requirement, not something to
    patch with an invented number) or any of pH/OM/P/K parses to a real
    number but outside a physically plausible range (e.g. a stray decimal
    point turning 3.56% organic matter into 356% -- a corrupted value must
    not become a training label any more than a missing one should)."""
    om = raw["organic_matter_pct"]
    p = raw["phosphorus_ppm"]
    k = raw["potassium_cmol_kg"]
    if om is None or p is None or k is None:
        return None

    ph = raw["ph"]
    if not (
        _within_bounds("ph", ph)
        and _within_bounds("organic_matter_pct", om)
        and _within_bounds("phosphorus_ppm", p)
        and _within_bounds("potassium_cmol_kg", k)
    ):
        return None

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
    unrecognized_crop_texts: dict[str, int] = {}

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
        for text, count in collect_unrecognized_crop_texts(worksheet, config).items():
            unrecognized_crop_texts[text] = unrecognized_crop_texts.get(text, 0) + count

    # Sorted by frequency, most-repeated first -- that's the order worth
    # reviewing in when deciding what to add to CROP_VARIETY_MAP next (see
    # collect_unrecognized_crop_texts).
    report["unrecognized_crop_texts"] = dict(
        sorted(unrecognized_crop_texts.items(), key=lambda item: -item[1])
    )

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
    guide.reload_guide()
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XLSX_PATH
    rows, report = parse_workbook(xlsx_path)
    write_csv(rows)
    print(f"Wrote {len(rows)} training rows to {OUTPUT_CSV}")
    print(f"Per-sheet: {report['sheets']}")
    print(f"Per-crop: {report['rows_by_crop']}")
    print(f"Per-variety: {report['rows_by_variety']}")
    print(f"Per-cultivation-method: {report['rows_by_cultivation_method']}")
    print(f"Per-age-band: {report['rows_by_age_band']}")
    top_unrecognized = list(report["unrecognized_crop_texts"].items())[:20]
    print(f"Top unrecognized crop texts ({len(report['unrecognized_crop_texts'])} distinct, "
          f"top 20 shown): {top_unrecognized}")

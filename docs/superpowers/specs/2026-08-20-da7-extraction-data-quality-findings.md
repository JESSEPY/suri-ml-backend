# DA7 Soil Database Extraction — Data Quality & Cleaning Findings

**Source workbook:** `Region 7-National Soil Database_2022-2025.xlsx` (DA Region 7 National Soil
Database export, 4 sheets: `Laboratory Details2022`, `Laboratory Details2023`,
`Laboratory Details2024`, `Soil Data 2025`). Checksum-matched against the copy in
`suri/public/Region 7-National Soil Database_2022-2025.xlsx` (MD5 `c1d311423425dd588b72cea94c2bda77`).

**Parser:** `data/parse_real_soil_database.py`, run via `python -m data.parse_real_soil_database`.

**Output:** `suri_real_training_data.csv`, ingested by `data/ingest_to_supabase.py` into the
`ml_soil_training_data` table that `ml/train.py` trains against.

## Headline result

Of **9,654** lab-coded rows across all 4 sheets, **3,381 (35%)** became final training rows. This
is not a parsing failure or a scan error — it's the outcome of four sequential, deliberate
completeness checks applied to real, free-text lab data. Every row is accounted for exactly once:

| Stage | Rows removed | Rows remaining | Reason |
|---|---:|---:|---|
| Total lab-coded rows | — | 9,654 | Rows with a non-blank `Laboratory Code` cell |
| Blank crop text | 3,267 | 6,387 | `Crops` column empty — no crop recorded at all |
| Intercropped | 2,012 | 4,375 | `Crops` text names more than one species |
| Unrecognized single crop | 210 | 4,165 | Single-crop text `normalize_crop()` doesn't map |
| No real pH | 4 | 4,161 | `pH` cell blank or non-numeric |
| Missing OM/P/K | 780 | 3,381 | Organic matter, phosphorus, or potassium blank |
| **Final training rows** | — | **3,381** | Complete: crop + variety/method/age + pH + OM + P + K |

Reconciliation was verified by summing all six removed/retained buckets back to 9,654 exactly
(script run interactively against the live workbook and the live parser functions, not against
cached figures).

## Tools used

- **`openpyxl`** (`read_only=True, data_only=True`) — reads the `.xlsx` workbook sheet-by-sheet,
  row-by-row, without loading formulas (only their last-computed values).
- **`re`** (Python standard library) — powers `normalize_crop()`'s comma/slash split (intercrop
  detection) and the numeric-cell cleanup in `parse_numeric_cell()`.
- **`pypdf`** and **`pymupdf`** — used separately, to verify the *Nutrient Recommendation Guide*
  and *Mode of Application* reference PDFs this pipeline's downstream nutrient math depends on
  (unrelated to reading the `.xlsx` itself, noted here for completeness since both are part of the
  same data-provenance chain).
- **`pytest`** — 631 tests in `tests/test_parse_real_soil_database.py` pin every normalization
  rule described below (crop matching, numeric parsing, cultivation-method/age-band inference,
  end-to-end row building).

## How each stage cleans/normalizes the data

**1. Column location, not column index.** Column positions are not consistent across sheets —
even the three near-identical "Laboratory Details" sheets shift their chemistry columns by one
starting at Organic Matter between 2023 and 2022/2024. Every column is located by matching its
header text on that specific sheet (`build_column_map`/`find_column`), never a hardcoded index
shared across sheets. This is a correctness fix, not just cleanup: a hardcoded index would have
silently mis-attributed values on at least one of the four sheets.

**2. Row inclusion gate.** A row counts as "lab-coded" only if its `Laboratory Code` cell is
non-blank. This is the base population (9,654 rows) — everything else below removes from it.

**3. Crop text normalization (`normalize_crop`).** The `Crops` column is free text, e.g.
`"Rice Upland"`, `"Hybrid Corn"`, `"Cassava, Corn"`. Normalization:
   - Blank/whitespace-only text → rejected (3,267 rows: no crop recorded at all — see "Findings"
     below on why this isn't recoverable).
   - Text containing a comma or slash is split into parts; more than one part means intercropped
     text and the row is rejected (2,012 rows) — one soil reading can't be cleanly attributed to
     one crop without a design decision this pipeline deliberately doesn't make.
   - The remaining single part is matched, case-insensitively, against `CROP_VARIETY_MAP` — a
     71-entry dictionary (26 distinct crops) mapping real-world label variants (dialect names,
     English/Filipino synonyms, variety-qualified phrases like `"hybrid rice"`) to a canonical
     `(crop, variety)` pair from `data/nutrient_recommendation_guide.py`'s `CANONICAL_CROPS`. A
     match is accepted on the whole cell text or a leading whole-word prefix of it (so
     `"Coconut 6-8yrs"` matches the bare `"coconut"` key), with longest keys tried first so a
     variety-specific key like `"hybrid rice"` wins over the bare `"rice"` substring. Text that
     matches nothing in the map is rejected (210 rows — see "Findings" below on why this number is
     smaller than an earlier estimate).

**4. Cultivation-method and age-band inference.** For crops where it matters (cultivation method:
Rice only; age band: Coconut/Cacao/Mango), the same free `Crops` text is keyword-matched a second
time (`normalize_cultivation_method`, `normalize_age_band`) to recover method/age qualifiers the
crop-matching pass ignored. Unstated values resolve to the crop's documented default
(`resolve_cultivation_method`/`resolve_age_band`) rather than being guessed.

**5. Numeric cell parsing (`parse_numeric_cell`).** Lab result cells contain non-numeric sentinels
(`"NIL"`, `"STK"`, blanks, stray whitespace) alongside real numbers. Any cell that doesn't parse to
a real number is treated as missing, not zero — a blank/sentinel OM, P, or K value is a real
"unmeasured," never silently coerced to 0, which would corrupt the downstream nutrient-requirement
math.

**6. Chemistry-completeness gate.** `build_training_row()` requires numeric organic matter,
phosphorus, and potassium — `nutrient_requirements()` cannot compute a fertilizer recommendation
without all three, and this pipeline has no imputation step for training data (imputation exists
only at inference time, for incomplete IoT/STK requests, and is documented as deliberately absent
here — see `build_training_row`'s docstring). Rows missing any of the three are rejected (780
rows). pH is a separate, earlier gate (4 rows) since it's needed even before the OM/P/K check runs.

**7. Provenance tagging.** Every surviving row is tagged `data_source: "lab"` and carries no
`farm_area_sqm`/`sampling_site`/`date_of_sampling` (fields the DA7 export doesn't carry per-row) —
these stay `None` rather than being backfilled with a guess.

## Findings worth flagging

- **The largest loss (34%, 3,267 rows) is rows with no crop recorded at all.** This is a property
  of the source data, not something any parser change can recover — there's no secondary
  crop-name column in the workbook (`Crops` is the only crop field; the adjacent `Remarks` column
  was spot-checked on several blank-crop rows and was also empty). These are otherwise valid soil
  tests, just not usable for a crop-specific model.
- **The second-largest loss (21%, 2,012 rows) is intercropped text.** Recoverable only by making a
  design decision (e.g., attribute the reading to the first-listed crop, or duplicate it across
  all listed crops) — deliberately out of scope for this extraction pass.
- **Unrecognized single-crop text (210 rows) is the one bucket this project actively narrowed.**
  Before the `2026-08-20-training-data-extraction-expansion` plan, `CROP_VARIETY_MAP` had 9
  entries; it now has 71, covering 26 distinct crops (up from the original 9 crops recognized by
  the field-level system). The plan's own scan (taken while writing it) estimated 621 rows would
  remain unmatched — the true post-implementation count is 210. The gap between those two numbers
  reflects the plan-writing-time scan using a slightly looser intercrop/blank filter than
  `normalize_crop()`'s actual comma/slash-split logic, not any error in the shipped mapping work.
- **3,381 is the "strictest possible cut," not a data-quality verdict on the workbook.** Most of
  the 6,273 rows that don't become training rows are simply incomplete for this specific purpose
  (no crop stated, multiple crops stated, or missing chemistry) rather than wrong or corrupted.

## Reproducing this reconciliation

```bash
python -c "
import openpyxl
from data.parse_real_soil_database import SHEET_CONFIGS, normalize_crop, parse_numeric_cell, build_column_map

wb = openpyxl.load_workbook('Region 7-National Soil Database_2022-2025.xlsx', read_only=True, data_only=True)
total = blank = intercropped = unrecognized = no_ph = incomplete_chem = complete = 0

for config in SHEET_CONFIGS:
    ws = wb[config['name']]
    column_map = build_column_map(ws, config)
    for row in ws.iter_rows(min_row=config['data_start_row'], values_only=True):
        if not row[config['lab_code_col']] or not str(row[config['lab_code_col']]).strip():
            continue
        total += 1
        crop_text = row[config['crop_col']]
        if not crop_text or not str(crop_text).strip():
            blank += 1; continue
        parts = [p.strip() for p in str(crop_text).replace('/', ',').split(',') if p.strip()]
        if len(parts) != 1:
            intercropped += 1; continue
        if normalize_crop(crop_text) is None:
            unrecognized += 1; continue
        ph = parse_numeric_cell(row[column_map['ph']])
        if ph is None:
            no_ph += 1; continue
        om = parse_numeric_cell(row[column_map['organic_matter_pct']])
        p = parse_numeric_cell(row[column_map['phosphorus_ppm']])
        k = parse_numeric_cell(row[column_map['potassium_cmol_kg']])
        if om is None or p is None or k is None:
            incomplete_chem += 1
        else:
            complete += 1

print(total, blank, intercropped, unrecognized, no_ph, incomplete_chem, complete)
"
```

(The actual pipeline is simpler to just run directly: `python -m data.parse_real_soil_database`
prints `Wrote 3381 training rows` on completion — this script exists only to break that single
number down by rejection reason.)

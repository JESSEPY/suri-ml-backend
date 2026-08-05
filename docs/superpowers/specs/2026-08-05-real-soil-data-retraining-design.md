# Retraining on the Region 7 National Soil Database — Design

## Context

The fertilizer prediction model (`ml/train.py`, `ml/model.py`) has been trained entirely on
synthetic data (`data/generate_synthetic_data.py`): uniform-random chemistry values per crop,
coarsened into three fabricated `data_source` "views" (lab/stk/iot) of one fabricated ground
truth per record. A real dataset has since become available —
`suri/public/Region 7-National Soil Database_2022-2025.xlsx`, a BSWM (Bureau of Soils and Water
Management) government soil-testing export covering four sheets of real lab submissions from
2022 through part of 2025, roughly 9,650 real samples in total (`Laboratory Details2022`:
2,765; `2023`: 2,216; `2024`: 2,769; `Soil Data 2025`: 1,904; `Soil Data 2025 with coordinates`
and `Sheet2` are empty/reference-only and excluded).

**Goal:** retire the synthetic generator and retrain the model on this real data instead —
real chemistry inputs, real (formula-computed) targets, no fabricated numbers anywhere in the
corpus.

**Explicitly out of scope for this spec:** the `suri` web app's integration with this backend.
That work (three data-entry flows, the STK Color Vector CMS, report persistence) was designed
separately and is on hold until this backend's retrained contract (crop list, units, whether
`data_source` needs per-source training) is settled — this spec does not touch the `suri` repo.

## Non-goals

- Not building a real-outcome-learned model. `target_n_kg_ha`/`target_p2o5_kg_ha`/
  `target_k2o_kg_ha` are, and remain, computed by `ml/agronomy.py`'s `nutrient_requirements()` —
  a deterministic formula from crop baseline + soil-test adjustment. The real spreadsheet has
  soil chemistry, not observed fertilizer-outcome data, so the model is still learning to
  interpolate a formula — just now from real chemistry inputs instead of random ones.
- Not adding real training signal for `stk` or `iot` `data_source` categories (see "Data source
  scope" below) — both remain valid, requestable values with zero dedicated training rows.
- Not resolving Gap 6 or any other item from the (paused) web-integration gap analysis.

## Architecture

One new script, `data/parse_real_soil_database.py`, replaces
`data/generate_synthetic_data.py` as the corpus source. It reads the xlsx, cleans it, and
outputs rows in the exact shape `train.py` already expects (the same `COLUMNS` list the old
generator used, minus `is_synthetic` — see "Schema" below). Nothing downstream changes:
`data/ingest_to_supabase.py` (upserts any CSV matching that shape into
`ml_soil_training_data`) and `ml/train.py`/`ml/model.py`/`ml/preprocessing.py` are all
already source-agnostic and require no edits beyond the two listed in "Model changes."

```
Region 7-National Soil Database_2022-2025.xlsx
        │
        ▼
data/parse_real_soil_database.py   (new — cleans, normalizes, computes remarks/targets)
        │
        ▼
suri_real_training_data.csv        (same shape as the old synthetic CSV, is_synthetic column dropped)
        │
        ▼
data/ingest_to_supabase.py         (unchanged) → ml_soil_training_data (Supabase)
        │
        ▼
ml/train.py                        (unchanged) → new model artifacts
```

## Crop and variety taxonomy

Applying a floor of 20 real sole-crop samples (below which a 70/15/15 grouped split can't
reliably hold a group in every split):

| Crop | Real sole-crop samples | Status |
|---|---|---|
| Rice | 1,588 (+ 201 Hybrid, 156 Inbred, 10 Upland) | keep, split by variety |
| Corn | 897 (+ 315 OPV, 26 Hybrid, 5 Sweet, 1 Yellow) | keep, split by variety |
| Cassava | 104 | **add** |
| Peanut | 73 | **add** |
| Mango | 59 | **add** |
| Cacao | 57 | **add** |
| Banana | 50 | **add** |
| Tomato | 27 | keep (already supported) |
| Coconut | 24 | **add** |
| Cabbage | 8 | **drop** — below floor |
| Sorghum | 8 | **drop** — below floor (see "Golden test" below) |
| Sugarcane | 11 | **drop** — below floor |

Final crop list (9): **Corn, Rice, Cassava, Peanut, Mango, Cacao, Banana, Tomato, Coconut.**

**Variety** is modeled as its own field, not folded into the crop name (`"Rice-Hybrid"` etc.) —
this matches the `suri` web app's existing data model, which already collects crop and
variety as two separate selections (`ProfilingWizard.tsx`'s conditional variety `<Select>`).
Only Rice (Hybrid / Inbred / unspecified) and Corn (OPV / Hybrid / unspecified) have enough
real samples per variety to clear the floor; every other crop's `variety` is always
`"None"`.

Variety-specific yield-potential differences are real (PhilRice/DA publish different N
recommendations for hybrid vs. inbred rice, and for hybrid vs. OPV corn) — without
variety-specific baselines, adding `variety` as a feature would be a no-op (see "Baselines"),
so both are being added together in this pass.

## Units

- `potassium_ppm` → **`potassium_cmol_kg`**, matching the real lab reports' native unit
  (`Exchangeable Potassium (K), cmol/kg`) — not converted. The system's prior `ppm` choice was
  never authoritative; it was picked before any real data existed, purely to mirror
  `phosphorus_ppm`'s naming. Converting away from the government-standard unit adds a
  conversion-bug surface (see "STK" sentinel below) for no benefit, and diverges from what's
  literally printed on the lab reports web-side users will eventually transcribe.
- `phosphorus_ppm` unchanged — the file already reports P natively as `mg/kg` (=ppm).
- pH, `organic_matter_pct`, `total_nitrogen_pct` unchanged (unitless/percent scales, no
  mismatch found).

This is a backend-only, `suri-ml-backend`-internal contract change (`SoilTestRequest`,
`FEATURE_COLUMNS`, `crop_baselines.py`). The `suri` web side's corresponding change is out of
scope here (see "Explicitly out of scope").

## Data source scope: `lab` only

Every real sample was either lab-tested or field-tested (Rapid Soil Test Kit) per nutrient —
essentially never both for the same field. Confirmed empirically: of the ~1,425 rows with any
real RSTK reading (2022 + 2023 combined; 2024 has none), only **0 (2022) and 25 (2023)** also
have a *complete* real numeric lab panel, and even relaxing to "just needs a real lab pH"
doesn't help — the blocker is OM/P/K, which are almost always `"STK"`-flagged (not numeric)
in the lab columns exactly when RSTK was used instead. There is no way to compute a real
target (`nutrient_requirements()` needs numeric OM/P/K) for the vast majority of RSTK-tested
rows, and inventing one defeats the point of this whole effort.

**Resolution:** `stk` gets zero dedicated training rows, same treatment as `iot` (which has
zero real analog in this dataset at all — no probe telemetry in a lab spreadsheet). Both
remain valid `data_source` values in the API and in `ml/preprocessing.py`'s
`DATA_SOURCES`/`OneHotEncoder` categories — a request can still specify `"stk"` or `"iot"` —
they just rely on the model's general chemistry-based scoring (learned entirely from `lab`
rows) plus its existing missingness-indicator imputation, rather than a source-specific
learned pattern.

Consequence: the RSTK columns (N/P/K category readings, inconsistent per-sheet column
offsets, inconsistent category vocabularies) are **not read by the parser at all** — this
considerably simplifies `data/parse_real_soil_database.py` relative to earlier drafts of this
design that tried to use them.

## Data cleaning rules

**Sentinel values, not just nulls.** Lab chemistry cells (pH, OM, P, K, Total N) can contain
non-numeric strings meaning "not lab-tested, tested by kit instead": `"STK"` (by far the most
common — 480-917 occurrences per field per sheet), `"NIL"`, `"NGL"`. Also two literal
data-entry typos observed (`"4..67"`, `"6..40"`, double-dot). Each field must be parsed
defensively per-cell: a non-numeric value means that field is absent for that row, not that
the row is unusable — a row can have a real lab pH and a `"STK"`-flagged P simultaneously.

**Laboratory code collisions across years.** Lab codes are simple per-year sequential numbers
(`"SCC 417"`, `"SCC 1"`, etc.), reused every year — confirmed 2,767 of 2,805 distinct code
strings (98.6%) appear in more than one sheet. Two consequences if left as-is: (1)
`ingest_to_supabase.py` upserts on `(laboratory_code, data_source)` — ingesting a later year
would silently overwrite an earlier year's row sharing the same code; (2) `train.py` groups
splits by `laboratory_code` specifically to keep one physical sample's rows together — raw
collision would force unrelated real samples from different years into the same group.
**Fix:** every parsed row's `laboratory_code` is namespaced by source sheet, using the literal
labels `"2022"`, `"2023"`, `"2024"`, `"2025"` (the four sheets in date order) —
`f"{sheet_year}-{normalized_code}"`. The code itself is normalized first: strip whitespace,
strip any non-digit prefix (`"SCC"`, case-insensitive), parse the remaining digits as an
integer, and rebuild as `f"SCC{digits}"` — so `"SCC 1"`, `"SCC 01"`, and `"SCC01"` all collapse
to `"SCC1"` before the year prefix is added, giving e.g. `"2022-SCC1"`, `"2022-SCC417"`,
`"2023-SCC417"` as three distinct, stable identifiers.

**Crop normalization.** Split the free-text `Crops` field on `,`/`/`; keep only rows where
exactly one part remains (multi-crop plots excluded — the model and the web UI both assume
one crop per request); map variety labels to family (`"Hybrid Rice"` → crop `Rice`, variety
`Hybrid`; `"OPV Corn"`/`"Corn OPV"` → crop `Corn`, variety `OPV`; etc.); keep only rows whose
crop is in the 9-crop list.

**Row exclusion.** A row is dropped if: no crop listed, multi-crop, crop not in the 9-crop
list, or no real numeric pH (pH is the one required field). Missing OM/P/K/Total N are
tolerated (become nulls, handled by the model's existing missingness-indicator imputation) —
only pH is load-bearing enough to exclude on.

**Per-sheet layout differences.** The four source sheets have different column layouts for
the fields this design actually reads (Laboratory No./Province/Municipality/Barangay/
pH/OM/P/K/Total N/Crops) — confirmed by direct inspection, not assumed. The parser reads each
sheet's own header row to build its column map; no column index is shared across sheets.

## Baselines — needs research, flagged for review

`data/crop_baselines.py` needs:
- 6 new crop-level baselines: Cassava, Peanut, Mango, Cacao, Banana, Coconut.
- 4 variety-level overrides: (Rice, Hybrid), (Rice, Inbred), (Corn, OPV), (Corn, Hybrid).
- Removal of the 3 dropped crops' entries (Cabbage, Sorghum, Sugarcane).

Sourced from published Philippine DA/BSWM/PhilRice fertilizer recommendation guidelines,
following the same convention the file's own docstring already establishes for its existing
placeholder values: flagged for agronomist review before production use, isolated to this one
file so correcting them later is a single edit plus a retrain.

## Golden test — retired

`ml/train.py`'s `GOLDEN_SCC417` fixture and the `_golden_test()` function validate against a
specific Sorghum record (`laboratory_code: "SCC-26-417-417"`, pH 8.17/OM 3.21%/P 18.96 ppm/
K 138 ppm → 50/40/60 kg/ha). Traced to its origin
(`docs/superpowers/specs/2026-08-03-suri-fertilizer-ml-pipeline-design.md:44`): a specific
pair of DA documents (Basmayor, Jampang, Argao, Cebu) the original spec author had direct
access to — not part of this bulk export. Every `"SCC 417"` actually present in this file
(checked all four sheets) has a different location and different chemistry values; the code
is reused per-year, not a unique identifier. Since Sorghum is also being dropped for
insufficient real coverage, this fixture is removed rather than chased down — `crop_baselines.py`'s
docstring reference to it should be updated accordingly.

## Schema (unchanged output shape)

Same `COLUMNS` as `data/generate_synthetic_data.py` produced, with `is_synthetic` dropped
(always `False` now — dead weight) and `potassium_ppm` renamed `potassium_cmol_kg`:

```
laboratory_code, data_source, sampling_site, date_of_sampling, crop, variety, farm_area_sqm,
ph, organic_matter_pct, total_nitrogen_pct, phosphorus_ppm, potassium_cmol_kg,
temp, moist, ec, sal,
ph_remark, om_remark, p_remark, k_remark,
target_n_kg_ha, target_p2o5_kg_ha, target_k2o_kg_ha,
app1_formulation, app2_formulation
```

(`temp`/`moist`/`ec`/`sal` are always null — no `iot`-sourced rows in this corpus, consistent
with "Data source scope" above.)

## Model changes

Minimal, by design — the point of keeping `train.py`/`ml/model.py`/`ml/preprocessing.py`
source-agnostic is that almost nothing here changes:

- `ml/preprocessing.py`: `CONTINUOUS` — rename `potassium_ppm` → `potassium_cmol_kg`.
  `NOMINAL` gains `"variety"` (categories: `None`, `Hybrid`, `Inbred`, `OPV`).
- `ml/agronomy.py`: `nutrient_requirements()` gains a `variety` parameter; checks a new
  `VARIETY_BASELINES` dict first, falls back to the crop-level baseline.
- `data/crop_baselines.py`: 9-crop list (see above) plus `VARIETY_BASELINES`.
- `main.py`: `SoilTestRequest` gains optional `variety: str | None`.
- `ml/train.py`: `GOLDEN_SCC417`/`_golden_test()` removed.
- Grouped-by-`laboratory_code` splitting, the XGBoost+KRR ensemble, the classifier heads —
  all unchanged.

## Testing

- Unit tests for `data/parse_real_soil_database.py` (new file, sibling to the existing
  `tests/test_generate_synthetic_data.py` convention): sentinel-string exclusion (per-field,
  not per-row), crop/variety normalization, lab-code namespacing and collision resolution,
  row-exclusion rules, per-sheet header-based column mapping.
- A data-quality report as part of running the script: rows read per sheet, rows dropped and
  why (no crop / multi-crop / crop not in list / no pH), final per-crop and per-variety row
  counts, lab-code collision count before/after namespacing. This is the real final
  verification pass — broader and more reliable than further manual spreadsheet sampling.
- Retrain, compare new `metadata.json` (MAE/R² by source) against the last synthetic-trained
  model's numbers. Expect real data to score *worse* on held-out R² than the old synthetic
  set (synthetic was uniform-random and trivially easy to fit) — that's expected, not a
  regression to chase.

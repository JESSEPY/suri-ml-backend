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

Three further BSWM/DA reference documents have since been supplied and change how fertilizer
*targets* are computed (previously an unresearched hand-built formula, now an official
lookup table — see "Nutrient targets" below):

- `suri/public/Nutrient Recommendation Guide.pdf` — the actual BSWM soil-test-band → fertilizer-rate
  reference table, covering dozens of crops.
- `suri/public/Fertilizer Computation Guide.pptx` — a worked-example walkthrough (by a BSWM
  Agriculturist II) of using that table, including two crops' worked numbers usable as
  transcription golden records (Coffee, Rice-Inbred).
- `suri/public/MODE OF APPLICATION (1).pdf` — per-crop application timing/method narrative
  (when/how to split-apply, not how much) — feeds a new advisory text field, not the model.

**Goal:** retire the synthetic generator and retrain the model on this real data instead —
real chemistry inputs, real (table-computed) targets, no fabricated numbers anywhere in the
corpus.

**Explicitly out of scope for this spec:** the `suri` web app's integration with this backend.
That work (three data-entry flows, the STK Color Vector CMS, report persistence) was designed
separately and is on hold until this backend's retrained contract (crop list, units, whether
`data_source` needs per-source training) is settled — this spec does not touch the `suri` repo.
The response-schema changes below (see "Schema") *are* part of that contract, even though
wiring `suri` to them is not.

## Non-goals

- Not building a real-outcome-learned model. `target_n`/`target_p2o5`/`target_k2o` are, and
  remain, computed by `ml/agronomy.py`'s `nutrient_requirements()` — now a **direct lookup**
  against the official BSWM Nutrient Recommendation Guide (soil-test band → rate), not a
  learned pattern and no longer our own hand-built baseline+adjustment formula either. The real
  spreadsheet has soil chemistry, not observed fertilizer-outcome data, so the model is still
  learning to interpolate a deterministic table — just a real, authoritative one now, instead of
  random synthetic inputs feeding an invented formula.
- Not adding real training signal for `stk` or `iot` `data_source` categories (see "Data source
  scope" below) — both remain valid, requestable values with zero dedicated training rows.
- Not adding a plant-age input for Banana — see "Nutrient targets" below; its guide rows split
  by plant part (Fruit/Pulp), not age, and don't fit the age-band mechanism the other three tree
  crops use. Banana stays a single flat rate.
- Not resolving Gap 6 or any other item from the (paused) web-integration gap analysis.
- Not reviving or feeding new documents into `suri`'s existing RAG pipeline
  (`suri/docs/11-rag-fertilizer-pipeline.md`) — that pipeline is already slated for retirement
  in favor of this ML backend (`suri/docs/12-ml-backend-integration-gaps.md`, Gap 5). Mode of
  Application becomes a static deterministic lookup in this repo instead (see below), keeping
  the same "no LLM synthesis, no fabrication risk" property the ML-backend-as-source-of-truth
  decision was made for.

## Architecture

```
Region 7-National Soil Database_2022-2025.xlsx
        │
        ▼
data/parse_real_soil_database.py   (new — cleans, normalizes, computes remarks/targets via table lookup)
        │            ▲
        │            │ reads
        │    data/nutrient_recommendation_guide.py   (new — transcribed BSWM guide table)
        │    data/mode_of_application.py             (new — transcribed advisory text per crop)
        ▼
suri_real_training_data.csv        (same shape as the old synthetic CSV, is_synthetic column dropped)
        │
        ▼
data/ingest_to_supabase.py         (unchanged) → ml_soil_training_data (Supabase)
        │
        ▼
ml/train.py                        (unchanged except target-scale handling, see "Model changes") → new model artifacts
```

`data/nutrient_recommendation_guide.py` and `data/mode_of_application.py` are also imported
directly by `ml/agronomy.py` and `main.py` respectively at prediction time (a live request's
targets and advisory text come from the same tables the training corpus was built from — not
duplicated logic).

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

## Cultivation method (Rice) and plant age (tree crops)

The Nutrient Recommendation Guide doesn't split every crop the same way our `variety` field
does — it introduces two more axes that our current data model and the `suri` web app don't
collect at all:

- **Rice** is split by **cultivation method** (`Upland` / `Lowland Wet` / `Lowland Dry`), not
  by seed variety — the guide has no Hybrid/Inbred distinction. Corn's split (`Native` /
  `Hybrid`) does map onto our existing `variety` field (`OPV`→`Native` row, `Hybrid`→`Hybrid`
  row), so Corn needs no new field.
- **Coconut, Cacao, and Mango** are split by **plant age band** (Coconut: `Below 1yr` /
  `1-5yrs` / `6-8yrs` / `9yrs & up`; Cacao: `NT` / `1yr` / `2yrs` / `3yrs` / `Over 7yrs`; Mango:
  `NT` / `1-3yrs` / `4-6yrs` / `7-10yrs & up` / `10yrs & up`) and reported **per tree**, not per
  hectare. **Banana's guide rows split by plant part (`Fruit`/`Pulp`), not age** — that doesn't
  fit this mechanism, so Banana is excluded from age-banding (see "Non-goals") and stays a
  single flat per-hectare rate.

Both new axes are added as **optional** request fields so nothing breaks for existing callers
and `suri` doesn't need to change yet:

- `SoilTestRequest.cultivation_method: str | None = None` — only consulted for Rice; defaults
  to `"Lowland Wet"` (the most common real-world case in Region 7) when Rice is requested
  without one.
- `SoilTestRequest.age_band: str | None = None` — only consulted for Coconut/Cacao/Mango. A
  request may pass the literal alias `"bearing"`, which is resolved **immediately, at the API
  boundary**, to that crop's specific mature guide-row label (Coconut `9yrs & up`, Cacao
  `Over 7yrs`, Mango `10yrs & up`) — the literal string `"bearing"` is never what gets stored,
  used as a feature category, or written to the training CSV; every downstream consumer
  (`nutrient_requirements()`, `ml/preprocessing.py`'s `age_band` category list, the training
  data) only ever sees the crop-specific resolved band string. Omitting `age_band` entirely is
  equivalent to passing `"bearing"` — same resolution happens either way. This is a genuine
  accuracy improvement over a single flat rate: the guide's rates swing as much as ~30× across a
  tree crop's age bands (e.g. Cacao ranges 0.003–0.10 kg/tree N by band), so a young-tree
  request defaulting to the mature rate would previously have been badly wrong with no way to
  correct it; now it can be corrected by passing an explicit `age_band`.

Training-data rows get their `cultivation_method`/`age_band` from the source spreadsheet's
`Crops` free-text column where it's stated (e.g. "Rice Lowland Wet"); most rows don't state
one, and get the same default used above (`"Lowland Wet"` / `"bearing"`) — consistent with how
`variety` already defaults to `"None"` for the same reason.

## Units

- `potassium_ppm` → **`potassium_cmol_kg`**, matching the real lab reports' native unit
  (`Exchangeable Potassium (K), cmol/kg`) — stored and reported as-is, not converted. This
  is what's literally printed on the lab reports web-side users will eventually transcribe.
- **Derived, not stored:** `k_ppm = potassium_cmol_kg × 391` (the standard cmol(+)/kg → mg/kg
  conversion for potassium: equivalent weight 39.1 g/mol ÷ charge 1 × 10). This derived value
  is computed on the fly, only where the guide's K bands or the existing K remark bands need a
  ppm value to classify against — it is not a new stored/training column. `ml/remarks.py`'s
  `_K_BANDS` stay in their original ppm scale (`DEFICIENT <60`, `SUFFICIENT 60–120`,
  `SUFFICIENT+ >120`) — the earlier plan's change to rescale those bands into cmol/kg is
  reverted; the conversion now happens on the input value instead, which is the more standard
  place for a unit conversion to live.
- `phosphorus_ppm` unchanged — the file already reports P natively as `mg/kg` (=ppm). The
  Nutrient Recommendation Guide's P bands are given in **both** Bray#1 and Olsen columns
  (different ppm cutoffs, same ordinal fertility classes). The source spreadsheet never states
  which method Region 7 labs use. **Assumption, not a verified fact:** Olsen — Region 7
  (Central Visayas) soils are predominantly limestone-derived/calcareous, and Olsen is the
  standard BSWM method for neutral-to-alkaline soils. This should be corrected in
  `data/nutrient_recommendation_guide.py` (a one-constant change) if that assumption turns out
  wrong.
- pH, `organic_matter_pct`, `total_nitrogen_pct` unchanged.

This is a backend-only, `suri-ml-backend`-internal contract change. The `suri` web side's
corresponding change is out of scope here (see "Explicitly out of scope").

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
`Hybrid`; `"OPV Corn"`/`"Corn OPV"` → crop `Corn`, variety `OPV`; etc.), and separately detect
cultivation-method/age-band text where present (e.g. `"Rice Upland"` → `cultivation_method`
`Upland`; `"Coconut 6-8yrs"` → `age_band` `6-8yrs`); keep only rows whose crop is in the
9-crop list.

**Row exclusion.** A row is dropped if: no crop listed, multi-crop, crop not in the 9-crop
list, or no real numeric pH (pH is the one required field). Missing OM/P/K/Total N are
tolerated (become nulls, handled by the model's existing missingness-indicator imputation) —
only pH is load-bearing enough to exclude on.

**Per-sheet layout differences.** The four source sheets have different column layouts for
the fields this design actually reads (Laboratory No./Province/Municipality/Barangay/
pH/OM/P/K/Total N/Crops) — confirmed by direct inspection, not assumed. The parser reads each
sheet's own header row to build its column map; no column index is shared across sheets.

## Nutrient targets: BSWM Nutrient Recommendation Guide table lookup

`ml/agronomy.py`'s `nutrient_requirements()` changes from "crop baseline + hand-built OM/P/K
adjustment formula" (the previous plan, sourced from a mix of general knowledge, web search,
and a user-provided figures table — never independently verified end-to-end) to a **direct
table lookup**: for a given crop (+ variety / cultivation_method / age_band where applicable),
three independent band lookups —

- `organic_matter_pct` → OM band → N rate
- P (via `phosphorus_ppm`, classified using the Olsen band boundaries — see "Units") → P band → P rate
- K (via the derived `k_ppm = potassium_cmol_kg × 391` — see "Units") → K band → K rate

— each read straight from `data/nutrient_recommendation_guide.py`, a hand-transcribed
structured version of `Nutrient Recommendation Guide.pdf`'s tables. This replaces guesswork
with the actual official reference; it does **not** make the targets "real outcome data" (see
"Non-goals") — it's still a deterministic table, just an authoritative one instead of an
invented one.

**Units differ by crop group**, driving the schema change below: field/row crops (Rice, Corn,
Cassava, Peanut, Tomato) report **kg/ha** straight from the guide's per-hectare rows.
Coconut/Cacao/Mango report **kg/tree** from the guide's per-tree, age-banded rows (see
"Cultivation method and plant age"). **Banana is also kg/tree** — its guide values are the same
small per-tree scale (`.16`, `.12`, ... with an explicit "576 trees/ha" planting-density note),
just split by plant part (`Fruit`/`Pulp`) instead of age; it uses the `Fruit` row as a flat
default (`Pulp` is not used — see "Non-goals") with no `age_band` involved.

**Known transcription risks — to be resolved by careful manual transcription plus golden-record
tests during implementation, not assumed correct here:**

- The guide's P section lists 6 sub-band boundaries for both Bray#1 and Olsen methods in its
  header, but most crop rows show only 5 P rate values — the exact column-to-band alignment
  needs to be nailed down by counting cells directly against the source PDF, not inferred from
  the header labels alone.
- The pptx's worked examples describe soil-test results using loose qualitative labels ("Low"
  OM, "Low" P, "Sufficient" K) rather than the guide's precise band boundaries or our own
  `PH_ORDER`/`OM_ORDER`/`P_ORDER`/`K_ORDER` remark vocabulary, and don't show the underlying raw
  pH/OM/P/K numbers — so matching a worked example's rate (e.g. Coffee Wet Season → 90/60/40)
  to a specific guide row-index requires inference, not a literal one-to-one field match.

**Golden-record validation plan:** hand-transcribe the guide into
`data/nutrient_recommendation_guide.py`, then add unit tests asserting the transcribed table
reproduces the pptx's worked examples once the same band is selected: Coffee (Wet → 90/60/40,
Dry → 100/60/40) — usable even though Coffee isn't one of our 9 supported crops, purely to
validate the transcription mechanism against a document-verified source — and Rice (Inbred Wet
→ 90/60/40, Inbred Dry → 100/60/40), which **is** one of our supported crops and should be
cross-checked against whichever guide row (`Lowland Wet`/`Lowland Dry`) the worked example's
implied OM/P/K band maps to.

`data/crop_baselines.py`'s old `CROP_BASELINES`/`VARIETY_BASELINES` dicts (and the
plan's previously-committed DA-ATI/HVCDP/PCAARRD-sourced figures for Cassava, Peanut, Mango,
Cacao, Banana, Coconut, and the two Rice/Corn variety overrides) are retired in favor of this
table — they were a reasonable stand-in before the official guide was available, but the guide
supersedes them for every crop it covers, which is all 9.

## Mode of Application

A new static, deterministic lookup — `data/mode_of_application.py`, hand-transcribed from
`MODE OF APPLICATION (1).pdf` — maps crop name to its split-application timing/method text
(e.g. Rice: "½ N + all P + K at 10-14 DAT; ¼ N at mid-tillering (23-27 DAT); remaining ¼ N +
any K at panicle initiation (33-37 DAT)"). Exposed as a new optional
`PredictionResponse.application_schedule: str | None` field — populated when the requested crop
has a transcribed entry, `None` otherwise. This is advisory text only: no training/model impact,
no LLM synthesis, not routed through `suri`'s RAG pipeline (see "Non-goals" — that pipeline is
being retired specifically in favor of deterministic backend logic like this).

## Golden test — retired

`ml/train.py`'s `GOLDEN_SCC417` fixture and the `_golden_test()` function validate against a
specific Sorghum record (`laboratory_code: "SCC-26-417-417"`, pH 8.17/OM 3.21%/P 18.96 ppm/
K 138 ppm → 50/40/60 kg/ha). Traced to its origin
(`docs/superpowers/specs/2026-08-03-suri-fertilizer-ml-pipeline-design.md:44`): a specific
pair of DA documents (Basmayor, Jampang, Argao, Cebu) the original spec author had direct
access to — not part of this bulk export. Every `"SCC 417"` actually present in this file
(checked all four sheets) has a different location and different chemistry values; the code
is reused per-year, not a unique identifier. Since Sorghum is also being dropped for
insufficient real coverage, this fixture is removed rather than chased down — replaced by the
Nutrient Recommendation Guide golden-record checks described above, which validate the table
transcription instead of one specific historical record.

## Schema

Training CSV / `ml/preprocessing.py`'s `TARGET_COLUMNS` / `main.py`'s `PredictionResponse`:

```
target_n_kg_ha, target_p2o5_kg_ha, target_k2o_kg_ha   →   target_n, target_p2o5, target_k2o, target_unit
```

`target_unit` is `"kg_ha"` (Rice, Corn, Cassava, Peanut, Tomato) or `"kg_tree"` (Coconut,
Cacao, Mango, Banana), set deterministically from the row's crop — not a model output.

Full training CSV schema, same `COLUMNS` as `data/generate_synthetic_data.py` produced, with
`is_synthetic` dropped (always `False` now — dead weight), `potassium_ppm` renamed
`potassium_cmol_kg`, the target columns renamed as above, and `cultivation_method`/`age_band`
added:

```
laboratory_code, data_source, sampling_site, date_of_sampling, crop, variety,
cultivation_method, age_band, farm_area_sqm,
ph, organic_matter_pct, total_nitrogen_pct, phosphorus_ppm, potassium_cmol_kg,
temp, moist, ec, sal,
ph_remark, om_remark, p_remark, k_remark,
target_n, target_p2o5, target_k2o, target_unit,
app1_formulation, app2_formulation
```

(`temp`/`moist`/`ec`/`sal` are always null — no `iot`-sourced rows in this corpus, consistent
with "Data source scope" above.)

## Model changes

- `ml/preprocessing.py`: `CONTINUOUS` — rename `potassium_ppm` → `potassium_cmol_kg`.
  `NOMINAL` gains `"variety"` (`None`/`Hybrid`/`Inbred`/`OPV`), `"cultivation_method"`
  (`None`/`Upland`/`Lowland Wet`/`Lowland Dry`), and `"age_band"` (`None` plus each tree crop's
  specific guide-row labels — never the literal `"bearing"` alias, which is resolved before this
  layer; see "Cultivation method and plant age"). `TARGET_COLUMNS` renamed per "Schema".
- `ml/agronomy.py`: `nutrient_requirements()` becomes a table lookup against
  `data/nutrient_recommendation_guide.py`, gaining `variety`, `cultivation_method`, and
  `age_band` parameters (all optional, defaults per "Cultivation method and plant age"); returns
  a rate plus its unit.
- `data/nutrient_recommendation_guide.py` (new): hand-transcribed guide table, replaces
  `data/crop_baselines.py`'s `CROP_BASELINES`/`VARIETY_BASELINES`.
- `data/mode_of_application.py` (new): hand-transcribed advisory text lookup.
- `ml/remarks.py`: `_K_BANDS` reverted to ppm scale (see "Units") — the input to
  `derive_k_remark()` is the derived `k_ppm`, not the stored `potassium_cmol_kg`.
- `main.py`: `SoilTestRequest` gains optional `variety`, `cultivation_method`, `age_band`.
  `PredictionResponse` fields renamed per "Schema", gains optional `application_schedule`.
- `ml/train.py`: `GOLDEN_SCC417`/`_golden_test()` removed.
- **New concern this pass introduces:** target scale now varies by ~2 orders of magnitude
  between kg/ha rows (roughly 20-300) and kg/tree rows (roughly 0.003-2 for Cacao/Coconut/Mango).
  A single shared regressor trained on `target_n`/`target_p2o5`/`target_k2o` across both groups
  risks the loss being dominated by the larger-scale field-crop rows, underfitting tree-crop
  predictions without it being visible in aggregate metrics. Mitigation, both small changes to
  `ml/train.py`: (1) log-transform targets before fitting (`log1p`/`expm1` round-trip — standard
  for this kind of scale spread), (2) add a `metrics.by_crop` breakdown (mirroring the existing
  `metrics.by_source`) to `metadata.json` so a regression here is visible in the test report
  rather than only in aggregate MAE.
- Grouped-by-`laboratory_code` splitting, the XGBoost+KRR ensemble, the classifier heads —
  otherwise unchanged.

## Testing

- Unit tests for `data/parse_real_soil_database.py`: sentinel-string exclusion (per-field, not
  per-row), crop/variety/cultivation_method/age_band normalization, lab-code namespacing and
  collision resolution, row-exclusion rules, per-sheet header-based column mapping.
- Unit tests for `data/nutrient_recommendation_guide.py`: the golden-record checks described
  above (Coffee, Rice-Inbred) plus boundary tests at each band edge for at least the crops this
  system actually supports.
- Unit tests for `data/mode_of_application.py`: lookup returns the transcribed text for a known
  crop, `None` for an unsupported one.
- A data-quality report as part of running the parser script: rows read per sheet, rows dropped
  and why (no crop / multi-crop / crop not in list / no pH), final per-crop/variety/
  cultivation_method/age_band row counts, lab-code collision count before/after namespacing.
- Retrain, compare new `metadata.json` (MAE/R² by source, and the new by-crop breakdown) against
  the last synthetic-trained model's numbers, with particular attention to the by-crop numbers
  for Coconut/Cacao/Mango given the target-scale concern above. Expect real data to score
  *worse* on held-out R² than the old synthetic set (synthetic was uniform-random and trivially
  easy to fit) — that's expected, not a regression to chase.

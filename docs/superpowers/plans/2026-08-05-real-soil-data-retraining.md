# Retraining on the Region 7 National Soil Database — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic training-data generator with a real one sourced from
`suri/public/Region 7-National Soil Database_2022-2025.xlsx`, and replace the hand-built
crop-baseline + adjustment-formula target computation with a direct lookup against the real
BSWM Nutrient Recommendation Guide, then retrain the model on the result.

**Architecture:** `data/nutrient_recommendation_guide.py` (new) holds a hand-transcribed,
band-indexed version of the official guide table and does the actual N/P/K rate lookup.
`ml/agronomy.py`'s `nutrient_requirements()` becomes a thin delegate to it. A new
`data/parse_real_soil_database.py` reads and cleans the xlsx, calls `nutrient_requirements()`
per row to compute real targets, and writes a CSV in the shape `ml/train.py` already expects
(via the unchanged `data/ingest_to_supabase.py` → `ml_soil_training_data` → `ml/train.py`
pipeline). `data/generate_synthetic_data.py` and `data/crop_baselines.py` are deleted once
nothing depends on them. `data/mode_of_application.py` (new) is a second, independent static
lookup for advisory application-timing text, wired into `main.py`'s response only — no
training/model impact.

**Tech Stack:** Python, pandas, openpyxl (new dependency), scikit-learn, XGBoost, FastAPI, pytest.

## Global Constraints

- Source file: `suri/public/Region 7-National Soil Database_2022-2025.xlsx` — a sibling repo
  (`suri`) checked out alongside this one at `../suri` relative to `suri-ml-backend`'s root. The
  parser takes this path as a CLI argument with that relative path as the default.
- Four usable sheets: `Laboratory Details2022`, `Laboratory Details2023`, `Laboratory Details2024`,
  `Soil Data 2025`. `Soil Data 2025 with coordinates` and `Sheet2` are excluded (empty / reference-only).
- Column positions are **not** consistent across sheets — confirmed by direct inspection, most
  surprisingly within the "Laboratory Details" family itself (2023's chemistry columns shift by
  one relative to 2022/2024 starting at Organic Matter). Every column must be located by matching
  its header text on that specific sheet, never by a hardcoded index shared across sheets.
- Final crop list (9): `Corn, Rice, Cassava, Peanut, Mango, Cacao, Banana, Tomato, Coconut`.
  `Cabbage`, `Sorghum`, `Sugarcane` are removed (each under the 20-real-sample floor).
- Variety is tracked separately from crop, only for Rice (`Hybrid`/`Inbred`) and Corn
  (`OPV`/`Hybrid`) — every other crop's variety is always `"None"`.
- **`nutrient_requirements()` no longer takes a `ph` argument and no longer applies any
  formula.** It is now a direct band lookup (OM% → N, P ppm → P, K ppm → K) against
  `data/nutrient_recommendation_guide.py`'s transcribed table. pH still gets its own
  `ph_remark` for display; it no longer numerically adjusts the P rate (the official guide has
  no pH axis — that was our own invented adjustment, now retired along with the rest of the old
  formula). This is a real behavior change, not just a rename.
- New optional dimensions beyond crop/variety: `cultivation_method` (Rice only — `Upland` /
  `Lowland Wet` / `Lowland Dry`, defaults to `Lowland Wet`) and `age_band` (Coconut/Cacao/Mango
  only — the request may pass the literal alias `"bearing"`, or omit the field, either of which
  resolves immediately to that crop's specific mature-tree guide row; the literal string
  `"bearing"` must never reach a model feature or the training CSV). Banana is **not**
  age-banded — its guide rows split by plant part, not age; it always uses the `Fruit` row.
- **Target unit varies by crop.** Rice/Corn/Cassava/Peanut/Tomato report `kg_ha`.
  Mango/Cacao/Coconut/Banana report `kg_tree` (the guide's own native unit for those rows — see
  Task 1). The API/training schema carries this explicitly via a new `target_unit` field —
  `target_n_kg_ha`/`target_p2o5_kg_ha`/`target_k2o_kg_ha` are renamed `target_n`/`target_p2o5`/
  `target_k2o` throughout (training CSV, `TARGET_COLUMNS`, `PredictionResponse`).
- `potassium_ppm` is renamed `potassium_cmol_kg` everywhere, stored and reported in the lab's
  native unit — **not converted at storage time.** A derived (never stored) `k_ppm =
  potassium_cmol_kg * 391` is computed on the fly, only at the two places a ppm value is needed:
  `data/nutrient_recommendation_guide.py`'s K-band lookup, and the existing call to
  `derive_k_remark()` (whose bands in `ml/remarks.py` need **no changes** — they are already
  expressed in ppm and stay that way; only callers change what they pass in).
  `phosphorus_ppm` is unchanged; P-band classification assumes the **Olsen** method (a named,
  isolated constant in `data/nutrient_recommendation_guide.py` — see Task 1 — since the source
  spreadsheet never states which method Region 7 labs use).
- Only `data_source="lab"` gets real training rows. `"stk"` and `"iot"` remain valid values
  accepted by the API and by `ml/preprocessing.py`'s encoder categories, with zero dedicated
  training rows — this is intentional, not a gap to fill later in this plan.
- The `GOLDEN_SCC417`/`SCC-26-417-417` fixture is retired throughout (Sorghum is no longer a
  supported crop, and the record's origin was traced to a document outside this dataset). It is
  **replaced**, not just deleted: `data/nutrient_recommendation_guide.py` gets its own
  golden-record checks against the two worked examples in `Fertilizer Computation Guide.pptx`
  (Coffee, Rice-Inbred) — see Task 1.
- **The transcribed guide table is a best-effort reading of a scanned, imperfectly-OCR'd PDF
  table and has at least one known unresolved discrepancy** (documented inline in Task 1): the
  Rice-Inbred golden check does not cleanly match a single guide band on all three nutrients
  simultaneously. Task 1 includes an explicit manual-verification step against the source PDF
  pages, and a golden-test failure on first run should be treated as a signal to re-examine that
  specific row against the PDF, not silently patched to force a pass.
- Target values now span roughly two orders of magnitude across the corpus (`kg_ha` rows
  ~20-300, `kg_tree` rows ~0.003-2). `ml/train.py` fits the regressor in log1p space via a new
  `LogTargetRegressor` wrapper (Task 4) specifically so this scale spread doesn't make the loss
  function ignore the smaller-scale tree-crop rows; `metadata.json` gains a `metrics.by_crop`
  breakdown (alongside the existing `by_source`) so this is visible in the training report.
- Reference spec: `docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md`.

---

## File Structure

New files:
- `data/nutrient_recommendation_guide.py` — transcribed BSWM guide table, band-lookup, unit/age/cultivation-method resolvers. Replaces `data/crop_baselines.py`.
- `data/mode_of_application.py` — transcribed advisory application-timing text per crop.
- `data/parse_real_soil_database.py` — reads, cleans, and normalizes the xlsx into training rows.
- `tests/test_nutrient_recommendation_guide.py`, `tests/test_mode_of_application.py`, `tests/test_parse_real_soil_database.py`.
- `tests/fixtures.py` — hermetic test-only DataFrame builder in the new schema, replacing `generate_synthetic_data.generate()` as a fixture source for tests that need a plausible multi-row frame but aren't testing the real parser itself.

Modified files:
- `ml/agronomy.py` — `nutrient_requirements()` becomes a thin delegate to `data/nutrient_recommendation_guide.py`; drops `ph_p_penalty`, the clamp constants, and the `CROP_BASELINES` import.
- `ml/model.py` — new `LogTargetRegressor` wrapper class.
- `ml/preprocessing.py` — `potassium_ppm` → `potassium_cmol_kg`; `NOMINAL` gains `"variety"`, `"cultivation_method"`, `"age_band"`; `TARGET_COLUMNS` renamed.
- `main.py` — `SoilTestRequest` gains `variety`/`cultivation_method`/`age_band`; `potassium_ppm` → `potassium_cmol_kg`; `PredictionResponse` fields renamed + gains `target_unit`/`application_schedule`.
- `ml/train.py` — `GOLDEN_SCC417`/`_golden_test()` removed; regressor wrapped in `LogTargetRegressor`; `evaluate()` gains a `by_crop` breakdown.
- `data/ingest_to_supabase.py` — points at the new parser's output CSV.
- `tests/test_agronomy.py`, `tests/test_model.py`, `tests/test_preprocessing.py`, `tests/test_api.py`, `tests/test_train.py`, `tests/test_ingest_to_supabase.py` — updated for the above.
- `requirements.txt` — add `openpyxl`.

Deleted files (Task 11):
- `data/crop_baselines.py`, `data/generate_synthetic_data.py`, `tests/test_generate_synthetic_data.py`.

---

### Task 1: `data/nutrient_recommendation_guide.py` — transcribed guide table and lookup

**Files:**
- Create: `data/nutrient_recommendation_guide.py`
- Create: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces: `CROPS: list[str]` (9 crops), `k_cmol_to_ppm(k_cmol_kg: float) -> float`,
  `target_unit_for_crop(crop: str) -> str`, `resolve_variety(crop, variety) -> str`,
  `resolve_cultivation_method(crop, cultivation_method) -> str`,
  `resolve_age_band(crop, age_band) -> str`,
  `nutrient_requirements(crop: str, om: float, p_ppm: float, k_cmol_kg: float, variety: str | None = None, cultivation_method: str | None = None, age_band: str | None = None) -> tuple[float, float, float]`.
  **Returns floats, not ints** — tree-crop rates (Cacao/Coconut/Mango/Banana) are legitimately
  sub-1 kg/tree; rounding to whole `int` would collapse most of them to `0`, silently corrupting
  both live predictions and every training label built from them (see Task 1 Step 3's fix
  history for the specific case this caught).
  Consumed by Task 2 (`ml/agronomy.py`), Task 5 (`ml/preprocessing.py`'s category lists), Task 6
  (`main.py`), Task 9 (parser).

**Context — read before writing the table:** the source PDF's table has at least one confirmed
internal inconsistency (row-to-row P-column counts vary between 5 and 6 in the raw extraction,
and the Rice-Inbred worked example from the pptx does not cleanly match a single guide band on
all three nutrients at once with the transcription below). The values below are a best-effort
transcription — **before this is used to train a production model, manually re-open the actual
PDF pages (not a text extraction) and check every row against this file.** The golden-record
test in Step 1 exists to make this checkable, not to prove it's already correct.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nutrient_recommendation_guide.py
import pytest

from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, target_unit_for_crop, resolve_variety,
    resolve_cultivation_method, resolve_age_band, nutrient_requirements,
)


class TestKConversion:
    def test_standard_conversion_factor(self):
        """1 cmol(+)/kg K = 391 mg/kg (ppm): equivalent weight 39.1 g/mol, charge 1."""
        assert k_cmol_to_ppm(1.0) == pytest.approx(391.0)

    def test_zero_stays_zero(self):
        assert k_cmol_to_ppm(0.0) == 0.0


class TestTargetUnit:
    @pytest.mark.parametrize("crop", ["Rice", "Corn", "Cassava", "Peanut", "Tomato"])
    def test_field_crops_are_kg_ha(self, crop):
        assert target_unit_for_crop(crop) == "kg_ha"

    @pytest.mark.parametrize("crop", ["Mango", "Cacao", "Coconut", "Banana"])
    def test_tree_crops_are_kg_tree(self, crop):
        assert target_unit_for_crop(crop) == "kg_tree"

    def test_every_supported_crop_has_a_unit(self):
        for crop in CROPS:
            assert target_unit_for_crop(crop) in ("kg_ha", "kg_tree")


class TestResolvers:
    def test_variety_none_becomes_the_string_none(self):
        assert resolve_variety("Cassava", None) == "None"

    def test_variety_passthrough(self):
        assert resolve_variety("Rice", "Hybrid") == "Hybrid"

    def test_cultivation_method_only_applies_to_rice(self):
        assert resolve_cultivation_method("Corn", "Upland") == "None"

    def test_cultivation_method_defaults_to_lowland_wet(self):
        assert resolve_cultivation_method("Rice", None) == "Lowland Wet"

    def test_cultivation_method_passthrough_for_known_value(self):
        assert resolve_cultivation_method("Rice", "Upland") == "Upland"

    def test_age_band_only_applies_to_coconut_cacao_mango(self):
        assert resolve_age_band("Tomato", "bearing") == "None"

    def test_age_band_bearing_alias_resolves_per_crop(self):
        assert resolve_age_band("Coconut", "bearing") == "9yrs & up"
        assert resolve_age_band("Cacao", "bearing") == "Over 7yrs"
        assert resolve_age_band("Mango", "bearing") == "10yrs & up"

    def test_age_band_omitted_is_the_same_as_bearing(self):
        assert resolve_age_band("Coconut", None) == resolve_age_band("Coconut", "bearing")

    def test_age_band_passthrough_for_known_value(self):
        assert resolve_age_band("Coconut", "1-5yrs") == "1-5yrs"


class TestNutrientRequirements:
    def test_returns_a_numeric_triple(self):
        n, p, k = nutrient_requirements("Corn", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert isinstance(n, float) and isinstance(p, float) and isinstance(k, float)

    def test_tree_crop_rates_are_not_rounded_to_zero(self):
        """The guide's Banana|Fruit rates are all sub-1 (e.g. 0.16, 0.05,
        0.33) — an int() cast anywhere in this path would silently collapse
        every real Banana recommendation to (0, 0, 0). This is the specific
        regression a review caught in the first implementation of this
        file: nutrient_requirements() rounded to int, and Banana always
        returned zeros no matter the input."""
        n, p, k = nutrient_requirements("Banana", om=1.0, p_ppm=3.0, k_cmol_kg=0.1)
        assert n > 0 and p > 0 and k > 0

    def test_boundary_value_lands_in_the_band_that_prints_it(self):
        """The guide prints bands as '0-2.0' then '2.1-3.5' — a 0.1 gap, not
        a continuous boundary. A value of exactly 2.0 belongs in the first
        band ('0-2.0'), not the second. This is the specific regression a
        review caught: the first implementation used a strict '<' test
        against the literal printed upper bounds, which puts an exact 2.0
        into the '2.1-3.5' band instead."""
        at_boundary = nutrient_requirements("Rice", om=2.0, p_ppm=3.0, k_cmol_kg=0.1, cultivation_method="Upland")[0]
        just_over = nutrient_requirements("Rice", om=2.1, p_ppm=3.0, k_cmol_kg=0.1, cultivation_method="Upland")[0]
        assert at_boundary == 45   # Rice|Upland's "0-2.0" band
        assert just_over == 40     # Rice|Upland's "2.1-3.5" band

    def test_lower_om_increases_nitrogen(self):
        """Guide bands are descending as soil fertility improves — this is a
        property of the real table, not a formula we chose."""
        low_om = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3)[0]
        high_om = nutrient_requirements("Corn", om=5.0, p_ppm=15.0, k_cmol_kg=0.3)[0]
        assert low_om > high_om

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)

    def test_variety_selects_a_different_guide_row(self):
        native = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="OPV")
        hybrid = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert native != hybrid

    def test_cultivation_method_selects_a_different_guide_row(self):
        wet = nutrient_requirements("Rice", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, cultivation_method="Lowland Wet")
        upland = nutrient_requirements("Rice", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, cultivation_method="Upland")
        assert wet != upland

    def test_age_band_selects_a_different_guide_row(self):
        young = nutrient_requirements("Coconut", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="Below 1yr")
        mature = nutrient_requirements("Coconut", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="9yrs & up")
        assert young != mature

    def test_no_age_band_defaults_to_bearing(self):
        default = nutrient_requirements("Cacao", om=1.0, p_ppm=10.0, k_cmol_kg=0.3)
        explicit_bearing = nutrient_requirements("Cacao", om=1.0, p_ppm=10.0, k_cmol_kg=0.3, age_band="bearing")
        assert default == explicit_bearing


class TestGoldenRecordRiceInbred:
    """Fertilizer Computation Guide.pptx worked example: Rice (Inbred), Wet
    Season -> N/P/K 90/60/40 kg/ha at soil-test bands described only as
    'Low'/'Low'/'Sufficient' (no raw pH/OM/P/K numbers given in the pptx).
    Wet-season N and P match this file's Lowland Wet band-0 (lowest OM/P)
    values exactly. K does not currently match any band at band-0's OM/P —
    a known, documented discrepancy (see Task 1's Context note and the
    plan's Global Constraints). This test asserts the two nutrients that do
    match; the K mismatch is intentionally not asserted here so this test
    suite doesn't silently start passing by weakening a real check — chasing
    the K discrepancy requires re-reading the source PDF, tracked as
    follow-up work, not papered over in this test."""

    def test_wet_season_low_om_low_p_matches_n_and_p(self):
        n, p, _k = nutrient_requirements(
            "Rice", om=1.0, p_ppm=3.0, k_cmol_kg=0.5, cultivation_method="Lowland Wet",
        )
        assert n == 90
        assert p == 60
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: FAIL — `data/nutrient_recommendation_guide.py` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# data/nutrient_recommendation_guide.py
"""Hand-transcribed BSWM Nutrient Recommendation Guide — the official
soil-test-band -> fertilizer-rate reference (Nutrient Recommendation
Guide.pdf), replacing the old data/crop_baselines.py baseline+formula
approach entirely.

Mechanism: for a crop (+ variety / cultivation_method / age_band where the
guide splits it), three INDEPENDENT band lookups -- %OM -> N rate,
Available P (ppm) -> P rate, Extractable K (ppm) -> K rate -- each an
ascending list of (upper_bound, rate) pairs using the same
[lower, upper) semantics as ml/remarks.py's bands. There is no pH axis in
the official guide; pH still gets its own qualitative remark elsewhere but
no longer numerically adjusts any rate here.

TRANSCRIPTION STATUS: best-effort, from a scanned/OCR'd source. See Task 1
of the implementation plan (docs/superpowers/plans/2026-08-05-real-soil-
data-retraining.md) for the known Rice-Inbred K-band discrepancy and the
required manual re-verification step against the actual PDF pages before
this drives a production model.
"""

from __future__ import annotations

from dataclasses import dataclass

# cmol(+)/kg -> mg/kg (ppm) for potassium: equivalent weight 39.1 g/mol,
# charge 1 -> 1 cmol/kg = 10 mmol/kg x 39.1 mg/mmol = 391 mg/kg.
K_CMOL_TO_PPM = 391.0

# The source spreadsheet never states which method Region 7 labs use for
# Available P. Assumed Olsen: Region 7 (Central Visayas) soils are
# predominantly limestone-derived/calcareous, the textbook Olsen use-case.
# Isolated here as a single named constant so it's a one-line fix if wrong
# -- flip to "bray1" and swap which _P_BANDS values are used per row if a
# lab confirms otherwise.
P_METHOD = "olsen"


def k_cmol_to_ppm(k_cmol_kg: float) -> float:
    return k_cmol_kg * K_CMOL_TO_PPM


@dataclass(frozen=True)
class GuideRow:
    unit: str  # "kg_ha" or "kg_tree"
    om_bands: tuple[tuple[float, float], ...]
    p_bands: tuple[tuple[float, float], ...]
    k_bands: tuple[tuple[float, float], ...]


def _band_value(value: float, bands: tuple[tuple[float, float], ...]) -> float:
    """bands is an ascending list of (upper_bound, rate). The guide prints
    ranges with a 0.1 gap ("0-2.0" then "2.1-3.5", or "0-35" then "36-55"
    for the integer-scaled K bands) — the printed upper bound is inclusive
    of its own band, not a shared continuous boundary. <=, not <, is what
    makes a value of exactly 2.0 land in "0-2.0" rather than "2.1-3.5"."""
    for upper, rate in bands:
        if value <= upper:
            return rate
    return bands[-1][1]


CROPS: list[str] = [
    "Corn", "Rice", "Cassava", "Peanut", "Mango", "Cacao", "Banana", "Tomato", "Coconut",
]

TARGET_UNIT: dict[str, str] = {
    "Rice": "kg_ha", "Corn": "kg_ha", "Cassava": "kg_ha", "Peanut": "kg_ha", "Tomato": "kg_ha",
    "Mango": "kg_tree", "Cacao": "kg_tree", "Coconut": "kg_tree", "Banana": "kg_tree",
}

# crop -> its guide row-label for the mature/fruit-bearing stage, used when
# a request passes the "bearing" alias (or omits age_band entirely).
BEARING_AGE_BAND: dict[str, str] = {
    "Coconut": "9yrs & up", "Cacao": "Over 7yrs", "Mango": "10yrs & up",
}

RICE_CULTIVATION_METHODS = ("Upland", "Lowland Wet", "Lowland Dry")

# Guide row key -> GuideRow. Keys: bare crop name for crops the guide
# doesn't further split (Cassava, Peanut, Tomato, Banana); "{crop}|{axis
# value}" for Rice (cultivation_method), Corn (Native/Hybrid), and the
# age-banded tree crops (Coconut/Cacao/Mango). "Coffee|..." rows are
# included solely for the pptx golden-record cross-check -- Coffee is not
# in CROPS and is never reachable through nutrient_requirements().
GUIDE_ROWS: dict[str, GuideRow] = {
    "Rice|Upland": GuideRow(
        "kg_ha",
        om_bands=((2.0, 45), (3.5, 40), (4.5, 35), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 20), (float("inf"), 0)),
    ),
    "Rice|Lowland Wet": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 75), (4.5, 60), (5.5, 45), (float("inf"), 20)),
        p_bands=((6.0, 60), (10.0, 45), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 20), (float("inf"), 0)),
    ),
    "Rice|Lowland Dry": GuideRow(
        "kg_ha",
        om_bands=((2.0, 120), (3.5, 75), (4.5, 60), (5.5, 45), (float("inf"), 20)),
        p_bands=((6.0, 60), (10.0, 45), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 20), (float("inf"), 0)),
    ),
    "Corn|Native": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 45), (5.5, 40), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 60), (15.0, 40), (20.0, 40), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 60), (250.0, 60), (300.0, 0), (float("inf"), 0)),
    ),
    "Corn|Hybrid": GuideRow(
        "kg_ha",
        om_bands=((2.0, 120), (3.5, 90), (4.5, 60), (5.5, 50), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 60), (15.0, 40), (20.0, 40), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 60), (250.0, 60), (300.0, 0), (float("inf"), 0)),
    ),
    "Cassava": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 40), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 45), (300.0, 30), (float("inf"), 0)),
    ),
    "Peanut": GuideRow(
        "kg_ha",
        om_bands=((2.0, 10), (3.5, 0), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 40), (10.0, 30), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 30), (250.0, 15), (300.0, 0), (float("inf"), 0)),
    ),
    "Tomato": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 80), (4.5, 70), (5.5, 60), (float("inf"), 0)),
        p_bands=((6.0, 120), (10.0, 90), (15.0, 60), (20.0, 30), (float("inf"), 0)),
        k_bands=((35.0, 150), (55.0, 120), (250.0, 90), (300.0, 60), (float("inf"), 0)),
    ),
    "Mango|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.01), (3.5, 0.01), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.01), (10.0, 0.01), (15.0, 0.008), (20.0, 0.005), (float("inf"), 0)),
        k_bands=((35.0, 0.15), (55.0, 0.009), (250.0, 0.003), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|1-3yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.05), (4.5, 0.02), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.05), (15.0, 0.03), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.15), (55.0, 0.09), (250.0, 0.06), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|4-6yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.15), (3.5, 0.10), (4.5, 0.05), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.03), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.09), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|7-10yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.15), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.10), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 0.45), (55.0, 0.30), (250.0, 0.15), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|10yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.15), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.003), (3.5, 0.002), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.004), (10.0, 0.003), (15.0, 0.002), (20.0, 0.001), (float("inf"), 0)),
        k_bands=((35.0, 0.003), (55.0, 0.002), (250.0, 0.001), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|1yr": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.055), (3.5, 0.035), (4.5, 0.02), (5.5, 0.005), (float("inf"), 0)),
        p_bands=((6.0, 0.11), (10.0, 0.07), (15.0, 0.04), (20.0, 0.015), (float("inf"), 0)),
        k_bands=((35.0, 0.075), (55.0, 0.05), (250.0, 0.024), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|2yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.065), (3.5, 0.04), (4.5, 0.025), (5.5, 0.01), (float("inf"), 0)),
        p_bands=((6.0, 0.13), (10.0, 0.08), (15.0, 0.05), (20.0, 0.025), (float("inf"), 0)),
        k_bands=((35.0, 0.08), (55.0, 0.06), (250.0, 0.03), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|3yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.075), (3.5, 0.045), (4.5, 0.03), (5.5, 0.015), (float("inf"), 0)),
        p_bands=((6.0, 0.15), (10.0, 0.09), (15.0, 0.055), (20.0, 0.025), (float("inf"), 0)),
        k_bands=((35.0, 0.10), (55.0, 0.075), (250.0, 0.036), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|Over 7yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.06), (4.5, 0.035), (5.5, 0.02), (float("inf"), 0)),
        p_bands=((6.0, 0.12), (10.0, 0.07), (15.0, 0.045), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.12), (55.0, 0.09), (250.0, 0.045), (300.0, 0), (float("inf"), 0)),
    ),
    "Coconut|Below 1yr": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.06), (3.5, 0.06), (4.5, 0.04), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.05), (10.0, 0.03), (15.0, 0.02), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.20), (55.0, 0.15), (250.0, 0.10), (300.0, 0), (float("inf"), 0)),
    ),
    "Coconut|1-5yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.20), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0.10), (float("inf"), 0)),
    ),
    "Coconut|6-8yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.10), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 1.20), (55.0, 0.90), (250.0, 0.60), (300.0, 0.30), (float("inf"), 0)),
    ),
    "Coconut|9yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.50), (3.5, 0.40), (4.5, 0.30), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 1.80), (55.0, 1.50), (250.0, 1.20), (300.0, 0.60), (float("inf"), 0)),
    ),
    "Banana|Fruit": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.16), (3.5, 0.12), (4.5, 0.09), (5.5, 0.04), (float("inf"), 0)),
        p_bands=((6.0, 0.05), (10.0, 0.045), (15.0, 0.04), (20.0, 0.035), (float("inf"), 0)),
        k_bands=((35.0, 0.33), (55.0, 0.23), (250.0, 0.13), (300.0, 0.08), (float("inf"), 0)),
    ),
    # Coffee: golden-record reference only (Fertilizer Computation Guide.pptx),
    # not a supported crop -- unreachable via nutrient_requirements(), whose
    # crop param is validated against CROPS well before any GUIDE_ROWS lookup.
    "Coffee|Over 15yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.165), (3.5, 0.10), (4.5, 0.07), (5.5, 0.05), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.065), (15.0, 0.04), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.20), (250.0, 0.06), (300.0, 0), (float("inf"), 0)),
    ),
}


def target_unit_for_crop(crop: str) -> str:
    return TARGET_UNIT[crop]


def resolve_variety(crop: str, variety: str | None) -> str:
    return variety or "None"


def resolve_cultivation_method(crop: str, cultivation_method: str | None) -> str:
    if crop != "Rice":
        return "None"
    if cultivation_method in RICE_CULTIVATION_METHODS:
        return cultivation_method
    return "Lowland Wet"


def resolve_age_band(crop: str, age_band: str | None) -> str:
    if crop not in BEARING_AGE_BAND:
        return "None"
    if age_band in (None, "bearing"):
        return BEARING_AGE_BAND[crop]
    return age_band


def _guide_row_key(
    crop: str, variety: str | None, cultivation_method: str | None, age_band: str | None,
) -> str:
    if crop == "Rice":
        return f"Rice|{resolve_cultivation_method(crop, cultivation_method)}"
    if crop == "Corn":
        resolved_variety = resolve_variety(crop, variety)
        return f"Corn|{'Hybrid' if resolved_variety == 'Hybrid' else 'Native'}"
    if crop in BEARING_AGE_BAND:
        return f"{crop}|{resolve_age_band(crop, age_band)}"
    if crop == "Banana":
        return "Banana|Fruit"
    return crop


def nutrient_requirements(
    crop: str, om: float, p_ppm: float, k_cmol_kg: float,
    variety: str | None = None, cultivation_method: str | None = None,
    age_band: str | None = None,
) -> tuple[float, float, float]:
    """Direct BSWM guide lookup: three independent band lookups (OM% -> N,
    P ppm -> P, K ppm, converted from cmol/kg -> K). Raises KeyError for an
    unknown crop or an unresolvable guide row (e.g. an age_band string this
    crop's guide rows don't have).

    Returns floats, not ints: tree-crop rates (Cacao/Coconut/Mango/Banana)
    are legitimately sub-1 kg/tree. Rounding to a whole int here would
    collapse most tree-crop rates to 0 -- verified concretely for
    Banana|Fruit, whose entire band set is < 1, so an int cast makes every
    possible Banana call return (0, 0, 0). Rounded to 3 decimal places,
    matching main.py's existing display rounding and the guide's own
    precision (its smallest printed values have 3 decimal digits)."""
    if crop not in CROPS:
        raise KeyError(f"Unknown crop {crop!r}")
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
    row = GUIDE_ROWS[key]
    k_ppm = k_cmol_to_ppm(k_cmol_kg)
    return (
        round(_band_value(om, row.om_bands), 3),
        round(_band_value(p_ppm, row.p_bands), 3),
        round(_band_value(k_ppm, row.k_bands), 3),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: PASS, all tests including `TestGoldenRecordRiceInbred`. If it fails, do not adjust the
test to make the number fit — re-open `Nutrient Recommendation Guide.pdf` and check the actual
`Rice | Lowland | Wet` row's OM/P values against what's transcribed above.

- [ ] **Step 5: Manually cross-check every row against the source PDF**

Open `suri/public/Nutrient Recommendation Guide.pdf` directly (not a text extraction) and check
each of this file's `GUIDE_ROWS` entries cell-by-cell against the actual table image, paying
particular attention to: (a) whether each crop row truly has 5 P-band values or 6 (the raw
OCR text disagreed row to row — see this file's module docstring), and (b) the Rice Lowland
Wet/Dry K-band values specifically, which are the source of the golden-record discrepancy noted
in `TestGoldenRecordRiceInbred`. Fix any row found to be wrong, and if the K discrepancy is
resolved, extend `TestGoldenRecordRiceInbred` to also assert on `k`.

- [ ] **Step 6: Commit**

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "feat: add transcribed BSWM Nutrient Recommendation Guide table and band lookup"
```

---

### Task 2: `ml/agronomy.py` — delegate to the guide table

**Files:**
- Modify: `ml/agronomy.py` (full rewrite)
- Modify: `tests/test_agronomy.py` (full rewrite)

**Interfaces:**
- Consumes: `nutrient_requirements`, `CROPS` (Task 1, `data.nutrient_recommendation_guide`).
- Produces: `nutrient_requirements(crop, om, p_ppm, k_cmol_kg, variety=None, cultivation_method=None, age_band=None) -> tuple[float, float, float]` (same name, now re-exported from `ml.agronomy` so every existing `from ml.agronomy import nutrient_requirements` call site keeps working), `total_nitrogen_from_om(om: float) -> float` (unchanged, still needed — the lab derives Total N from OM, unrelated to fertilizer targets). Consumed by Task 5 (`tests/fixtures.py`), Task 6 (`main.py`), Task 9 (parser).

**Context:** the old `ph_p_penalty()` function and the `N_MIN`/`N_MAX`/`P_MIN`/`P_MAX`/`K_MIN`/`K_MAX`/`_clamp` machinery are all removed — they existed to shape an invented formula's output into a plausible range; the guide's own table values are already realistic and need no clamping. `ph` is no longer a `nutrient_requirements()` parameter (see Global Constraints).

- [ ] **Step 1: Replace the tests first**

Replace `tests/test_agronomy.py` in full:

```python
import pytest

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from data.nutrient_recommendation_guide import CROPS


class TestTotalNitrogenFromOm:
    def test_derives_total_n_at_the_lab_s_five_percent_ratio(self):
        assert total_nitrogen_from_om(2.0) == pytest.approx(0.10)

    def test_zero_om_is_zero_n(self):
        assert total_nitrogen_from_om(0.0) == 0.0


class TestNutrientRequirements:
    def test_returns_a_numeric_triple(self):
        n, p, k = nutrient_requirements("Corn", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert isinstance(n, float) and isinstance(p, float) and isinstance(k, float)

    def test_tree_crop_rates_are_not_rounded_to_zero(self):
        """See data/nutrient_recommendation_guide.py's nutrient_requirements
        docstring — this must stay a float pass-through, not re-introduce
        an int cast at this delegation layer."""
        n, p, k = nutrient_requirements("Banana", om=1.0, p_ppm=3.0, k_cmol_kg=0.1)
        assert n > 0 and p > 0 and k > 0

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)

    def test_variety_with_no_matching_row_falls_back_gracefully(self):
        """Cassava has no variety-specific guide rows — passing one must not
        raise, since resolve_variety/(guide row keying) for Cassava ignores
        variety entirely."""
        without = nutrient_requirements("Cassava", om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        with_variety = nutrient_requirements("Cassava", om=2.5, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert without == with_variety

    def test_known_variety_changes_the_result(self):
        opv = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="OPV")
        hybrid = nutrient_requirements("Corn", om=1.0, p_ppm=15.0, k_cmol_kg=0.3, variety="Hybrid")
        assert opv != hybrid

    @pytest.mark.parametrize("crop", CROPS)
    def test_every_supported_crop_resolves_without_error(self, crop):
        n, p, k = nutrient_requirements(crop, om=2.5, p_ppm=15.0, k_cmol_kg=0.3)
        assert n >= 0 and p >= 0 and k >= 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_agronomy.py -v`
Expected: FAIL — `ml/agronomy.py` still has the old `ph`-taking signature and imports `CROP_BASELINES` from a module the new tests don't reference; `TypeError`/`ImportError` depending on which test runs first.

- [ ] **Step 3: Implement**

Replace `ml/agronomy.py` in full:

```python
"""Soil-test to fertilizer requirement lookup.

nutrient_requirements() is a thin delegate to
data/nutrient_recommendation_guide.py's transcribed BSWM table — see that
module for the actual band-lookup mechanism and its known transcription
caveats. This file's only remaining independent logic is
total_nitrogen_from_om(), which is unrelated to fertilizer targets: it's
how the lab itself derives a Total Nitrogen percentage when a report only
has Organic Matter.
"""

from __future__ import annotations

from data.nutrient_recommendation_guide import nutrient_requirements as _guide_lookup

# Total Nitrogen is calculated by the lab as OM x 0.05, never measured directly.
OM_TO_TOTAL_N = 0.05


def total_nitrogen_from_om(om: float) -> float:
    return om * OM_TO_TOTAL_N


def nutrient_requirements(
    crop: str, om: float, p_ppm: float, k_cmol_kg: float,
    variety: str | None = None, cultivation_method: str | None = None,
    age_band: str | None = None,
) -> tuple[float, float, float]:
    """Return (n, p2o5, k2o) — kg/ha for field crops, kg/tree for
    Coconut/Cacao/Mango/Banana (see
    data.nutrient_recommendation_guide.target_unit_for_crop). Floats, not
    ints: several tree-crop rates are sub-1 kg/tree. Raises KeyError for an
    unknown crop."""
    return _guide_lookup(
        crop, om, p_ppm, k_cmol_kg,
        variety=variety, cultivation_method=cultivation_method, age_band=age_band,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_agronomy.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/agronomy.py tests/test_agronomy.py
git commit -m "feat: delegate nutrient_requirements to the guide table, drop the invented formula"
```

---

### Task 3: `data/mode_of_application.py` — advisory application-timing text

**Files:**
- Create: `data/mode_of_application.py`
- Create: `tests/test_mode_of_application.py`

**Interfaces:**
- Produces: `application_schedule_for_crop(crop: str) -> str | None`. Consumed by Task 6 (`main.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mode_of_application.py
from data.nutrient_recommendation_guide import CROPS
from data.mode_of_application import application_schedule_for_crop


class TestApplicationScheduleForCrop:
    def test_known_crop_returns_text(self):
        text = application_schedule_for_crop("Rice")
        assert isinstance(text, str) and len(text) > 0

    def test_unknown_crop_returns_none(self):
        assert application_schedule_for_crop("Durian") is None

    def test_every_supported_crop_has_an_entry(self):
        """All 9 crops this system serves are covered by MODE OF
        APPLICATION.pdf under their own name or a listed alias."""
        for crop in CROPS:
            assert application_schedule_for_crop(crop) is not None, crop
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mode_of_application.py -v`
Expected: FAIL — `data/mode_of_application.py` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# data/mode_of_application.py
"""Hand-transcribed advisory application-timing/method text, from
MODE OF APPLICATION (1).pdf. Purely informational — no training or model
impact. Not routed through suri's RAG pipeline (suri/docs/11-rag-fertilizer
-pipeline.md), which is being retired in favor of deterministic backend
logic like this (suri/docs/12-ml-backend-integration-gaps.md, Gap 5).
"""

from __future__ import annotations

APPLICATION_SCHEDULE: dict[str, str] = {
    "Rice": (
        "Basal: apply half of the recommended N plus all P and K 10-14 days "
        "after transplanting (DAT). Topdress a quarter of the N at active "
        "tillering (23-27 DAT). Topdress the remaining quarter of N, and any "
        "remaining K, at panicle initiation (33-37 DAT)."
    ),
    "Corn": (
        "Wet season: apply half of the N and K plus all P in a band 5cm to "
        "the side and 5cm below the seed at planting; sidedress the "
        "remainder when plants are about 4 weeks old (knee-high), followed "
        "by hilling up and irrigation. Dry season: apply all fertilizer at "
        "planting, or up to 3 weeks after planting."
    ),
    "Cassava": (
        "Apply the P and K plus half of the N at planting. Sidedress the "
        "remaining N 2-3 months after planting, during hilling up."
    ),
    "Peanut": (
        "Apply the fertilizer in the furrow and cover with 2-3 inches of "
        "soil before sowing the seeds. Inoculating the seeds with "
        "nitrogen-fixing bacteria improves yield and quality."
    ),
    "Tomato": (
        "Apply half of the N 3 inches from the base of the plant and 3 "
        "inches deeper than the root system, immediately before or at "
        "planting. Sidedress the remaining N when plants start to bloom or "
        "small fruits appear. Avoid heavy N application — it delays "
        "ripening and produces soft fruit that splits easily."
    ),
    "Mango": (
        "Bearing trees: apply half of the recommended fertilizer at the "
        "start of the rainy season in 6-8 holes around the tree; apply the "
        "remaining half before the end of the rainy season, in 6-8 holes "
        "around the tree."
    ),
    "Cacao": (
        "Established crop: mix the recommended fertilizer and apply in 2 "
        "equal doses — the first at the start of the rainy season, the "
        "second before the end of the rainy season — in 5-6 holes bored 5cm "
        "deep around the base of the plant."
    ),
    "Coconut": (
        "Bearing trees: apply the recommended fertilizer annually, in at "
        "least 10 holes bored 8-16 inches deep at a 1-meter radius from the "
        "base of the tree."
    ),
    "Banana": (
        "Apply the recommended fertilizer in four equal splits: at "
        "planting, 3 months after transplanting, 6 months after "
        "transplanting, and at 9 months or just before/at initial "
        "flowering. Apply around the planting hole 30-60cm from the "
        "pseudostem, in a ring or band; on slopes, fork the fertilizer in "
        "on the uphill side of the plant."
    ),
}


def application_schedule_for_crop(crop: str) -> str | None:
    return APPLICATION_SCHEDULE.get(crop)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_mode_of_application.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add data/mode_of_application.py tests/test_mode_of_application.py
git commit -m "feat: add transcribed Mode of Application advisory text lookup"
```

---

### Task 4: `ml/model.py` — `LogTargetRegressor` wrapper for mixed-scale targets

**Files:**
- Modify: `ml/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `EnsembleFertilizerRegressor` (same file, unchanged).
- Produces: `LogTargetRegressor(base_regressor)` — `.fit(X, y)`/`.predict(X)`/`.models_` (delegated
  to the wrapped estimator), transparently fitting/predicting in `log1p` space. Consumed by
  Task 7 (`ml/train.py`).

**Context:** targets now span roughly two orders of magnitude — `kg_ha` rows (~20-300) and
`kg_tree` rows (~0.003-2) share the same three target columns. A regressor fit directly on that
mixture has its loss dominated by the larger-scale rows, silently starving the tree-crop
predictions of signal. `log1p`/`expm1` compress that spread onto a comparable scale for
fitting, then invert it back to real units on predict — entirely inside this wrapper, so nothing
downstream (`main.py`'s `ARTIFACTS["regressor"].predict(X)`, `ml/train.py`'s `evaluate()`) needs
to know it's happening.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
class TestLogTargetRegressor:
    def test_predictions_are_in_the_original_scale_not_log_scale(self, dataset):
        """The wrapped model is fit on log1p(y) internally, but predict()
        must return real, expm1'd values — a caller with no knowledge of
        the wrapper must see the same kind of numbers either way."""
        X, y = dataset
        from ml.model import LogTargetRegressor
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        predicted = wrapped.predict(X)
        assert predicted.shape == y.shape
        # y in this fixture ranges roughly 15-85; log1p of that range is
        # roughly 2.7-4.5 — a prediction still in log-space would be wildly
        # out of range, not just imprecise.
        assert predicted.min() > 5.0

    def test_handles_a_target_close_to_zero(self):
        """The smallest real targets in this system are ~0.003 (kg/tree,
        young Cacao) -- log1p/expm1 must round-trip near-zero values
        without producing negatives or NaNs."""
        from ml.model import LogTargetRegressor
        X = RNG.uniform(-1, 1, size=(60, 3))
        y = np.column_stack([
            np.full(60, 0.003), np.full(60, 0.01), np.full(60, 0.002),
        ])
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        predicted = wrapped.predict(X)
        assert np.all(predicted >= 0)
        assert not np.any(np.isnan(predicted))

    def test_exposes_the_wrapped_models_attribute(self, dataset):
        X, y = dataset
        from ml.model import LogTargetRegressor
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        assert len(wrapped.models_) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ml.model` has no `LogTargetRegressor`.

- [ ] **Step 3: Implement**

Append to `ml/model.py`:

```python
class LogTargetRegressor:
    """Wraps any (fit(X, y) / predict(X) / models_) regressor to fit and
    predict in log1p space, transparently inverse-transforming predictions
    back to the original scale via expm1.

    Needed once targets span both kg/ha (~20-300) and kg/tree (~0.003-2)
    scales in the same columns (Coconut/Cacao/Mango/Banana vs. the field
    crops) -- without this, a shared loss function is dominated by the
    larger-scale rows and silently starves the tree-crop predictions.
    """

    def __init__(self, base_regressor):
        self._base = base_regressor

    def fit(self, X, y):
        self._base.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self._base.predict(X))

    @property
    def models_(self):
        return self._base.models_
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/model.py tests/test_model.py
git commit -m "feat: add LogTargetRegressor to handle mixed kg/ha and kg/tree target scales"
```

---

### Task 5: Preprocessing — K rename, three new nominal features, target rename

**Files:**
- Modify: `ml/preprocessing.py` (full rewrite)
- Create: `tests/fixtures.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `CROPS` (Task 1, `data.nutrient_recommendation_guide`).
- Produces: `CONTINUOUS` (with `potassium_cmol_kg` replacing `potassium_ppm`), `NOMINAL = ["crop", "variety", "cultivation_method", "age_band", "data_source"]`, `FEATURE_COLUMNS`, `TARGET_COLUMNS = ["target_n", "target_p2o5", "target_k2o"]`, `VARIETIES`, `CULTIVATION_METHODS`, `AGE_BANDS` — consumed by Task 6 (`main.py`), Task 7 (`ml/train.py`, unchanged code but new columns), Task 9's parser (must produce matching columns).
- Produces (test-only): `tests/fixtures.py`'s `sample_training_frame(num_records, seed) -> pd.DataFrame` — consumed by Task 7 and Task 10's test updates.

- [ ] **Step 1: Create the test-only fixture builder**

```python
# tests/fixtures.py
"""Test-only DataFrame builder in the training-data schema.

NOT used by any production path (ml/train.py, data/ingest_to_supabase.py,
data/parse_real_soil_database.py never import this). Exists purely so tests
of generic pipeline mechanics — grouping, splitting, chunking, JSON
serialization — have a fast, hermetic multi-row frame to exercise, without
depending on the real xlsx file or asserting anything about real-world
accuracy.
"""

from __future__ import annotations

import random

import pandas as pd

from data.nutrient_recommendation_guide import (
    CROPS, target_unit_for_crop, resolve_variety, resolve_cultivation_method, resolve_age_band,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark
from data.nutrient_recommendation_guide import k_cmol_to_ppm

COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "cultivation_method", "age_band", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


def sample_training_frame(num_records: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    for index in range(num_records):
        crop = rng.choice(CROPS)
        ph = round(rng.uniform(4.5, 8.5), 2)
        om = round(rng.uniform(0.5, 5.0), 2)
        phosphorus = round(rng.uniform(5.0, 40.0), 2)
        potassium = round(rng.uniform(0.1, 1.0), 3)

        variety = resolve_variety(crop, None)
        cultivation_method = resolve_cultivation_method(crop, None)
        age_band = resolve_age_band(crop, None)

        n, p2o5, k2o = nutrient_requirements(crop, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o)

        rows.append({
            "laboratory_code": f"TEST-{index}",
            "data_source": "lab",
            "sampling_site": None,
            "date_of_sampling": None,
            "crop": crop,
            "variety": variety,
            "cultivation_method": cultivation_method,
            "age_band": age_band,
            "farm_area_sqm": round(rng.uniform(100, 10000), 2),
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
            "target_n": n, "target_p2o5": p2o5, "target_k2o": k2o,
            "target_unit": target_unit_for_crop(crop),
            "app1_formulation": app1, "app2_formulation": app2,
        })

    return pd.DataFrame(rows, columns=COLUMNS)
```

- [ ] **Step 2: Update the preprocessing tests**

Replace `tests/test_preprocessing.py` in full:

```python
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tests.fixtures import sample_training_frame
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS, VARIETIES, CULTIVATION_METHODS, AGE_BANDS,
)


@pytest.fixture(scope="module")
def frame():
    return sample_training_frame(num_records=100, seed=42)


@pytest.fixture(scope="module")
def fitted(frame):
    return build_preprocessor().fit(frame[FEATURE_COLUMNS])


class TestSchema:
    def test_target_columns_are_unit_agnostic_names(self):
        assert TARGET_COLUMNS == ["target_n", "target_p2o5", "target_k2o"]

    def test_potassium_column_is_cmol_kg(self):
        assert "potassium_cmol_kg" in CONTINUOUS
        assert "potassium_ppm" not in CONTINUOUS

    def test_nominal_includes_all_four_categorical_axes(self):
        assert NOMINAL == ["crop", "variety", "cultivation_method", "age_band", "data_source"]


class TestPreprocessor:
    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            build_preprocessor().transform(pd.DataFrame())

    def test_fit_transform_produces_a_dense_2d_array(self, frame):
        result = build_preprocessor().fit_transform(frame[FEATURE_COLUMNS])
        assert result.ndim == 2
        assert result.shape[0] == len(frame)

    def test_missing_continuous_value_does_not_raise(self, frame):
        degraded = frame.copy()
        degraded.loc[0, "phosphorus_ppm"] = None
        build_preprocessor().fit_transform(degraded[FEATURE_COLUMNS])


def _base_row() -> dict:
    return {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_cmol_kg": 0.35,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": "SLIGHTLY ACIDIC", "om_remark": "MODERATELY LOW",
        "p_remark": "MEDIUM", "k_remark": "SUFFICIENT",
        "crop": "Corn", "variety": "None", "cultivation_method": "None",
        "age_band": "None", "data_source": "lab",
    }


class TestVarietyEncoding:
    def test_variety_is_a_feature_column(self):
        assert "variety" in FEATURE_COLUMNS
        assert "variety" in NOMINAL

    def test_none_variety_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Cassava", "variety": "None"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_known_variety_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Rice", "variety": "Hybrid"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_unknown_variety_raises(self, fitted):
        row = pd.DataFrame([{**_base_row(), "variety": "Nonexistent"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)


class TestCultivationMethodEncoding:
    def test_cultivation_method_is_a_feature_column(self):
        assert "cultivation_method" in FEATURE_COLUMNS

    def test_known_cultivation_method_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Rice", "cultivation_method": "Upland"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_unknown_cultivation_method_raises(self, fitted):
        row = pd.DataFrame([{**_base_row(), "cultivation_method": "Nonexistent"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)


class TestAgeBandEncoding:
    def test_age_band_is_a_feature_column(self):
        assert "age_band" in FEATURE_COLUMNS

    def test_known_age_band_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Coconut", "age_band": "9yrs & up"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_bearing_alias_is_never_a_valid_category(self, fitted):
        """'bearing' must be resolved to a specific band before it ever
        reaches this layer — see data.nutrient_recommendation_guide.
        resolve_age_band. If the literal word leaks in here, that's a bug
        upstream, and this encoder must reject it like any other unknown
        category rather than silently accept it."""
        row = pd.DataFrame([{**_base_row(), "crop": "Coconut", "age_band": "bearing"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)
```

- [ ] **Step 3: Run to verify the changed/new tests fail**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ml.preprocessing` doesn't export `CULTIVATION_METHODS`/`AGE_BANDS` yet, `CONTINUOUS` still lists `potassium_ppm`, `TARGET_COLUMNS` still has the old `_kg_ha` names.

- [ ] **Step 4: Implement**

Replace `ml/preprocessing.py` in full:

```python
"""Feature preprocessing (spec section 7).

Continuous features are median-imputed with an explicit missingness indicator,
so the model learns "this was not measured" rather than silently trusting an
imputed guess — essential because each data source is missing different things.

Remarks are ordinal-encoded rather than one-hot because the bands are genuinely
ordered (LOW < MEDIUM < HIGH); one-hot would discard that ordering, which matters
for the Kernel Ridge half of the ensemble.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data.nutrient_recommendation_guide import CROPS, RICE_CULTIVATION_METHODS, BEARING_AGE_BAND, GUIDE_ROWS
from ml.remarks import K_ORDER, OM_ORDER, P_ORDER, PH_ORDER

CONTINUOUS = [
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
]

ORDERED = ["ph_remark", "om_remark", "p_remark", "k_remark"]

NOMINAL = ["crop", "variety", "cultivation_method", "age_band", "data_source"]

FEATURE_COLUMNS = CONTINUOUS + ORDERED + NOMINAL

TARGET_COLUMNS = ["target_n", "target_p2o5", "target_k2o"]

DATA_SOURCES = ["iot", "lab", "stk"]

# "None" (the string, not Python None) covers every crop/row that doesn't
# track this axis. See data/nutrient_recommendation_guide.py for how each
# is resolved from a request before it ever reaches this layer.
VARIETIES = ["None", "Hybrid", "Inbred", "OPV"]
CULTIVATION_METHODS = ["None", *RICE_CULTIVATION_METHODS]
AGE_BANDS = ["None"] + sorted({
    key.split("|", 1)[1] for key in GUIDE_ROWS if key.split("|", 1)[0] in BEARING_AGE_BAND
})

_ORDERED_CATEGORIES = [PH_ORDER, OM_ORDER, P_ORDER, K_ORDER]


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    ordered = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", OrdinalEncoder(
            categories=_ORDERED_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # handle_unknown="error", deliberately: an unrecognised crop, variety,
    # cultivation_method, or age_band must fail loudly rather than produce
    # an all-zeros vector and a confident-looking prediction built on no
    # real information. A farmer acting on that could over- or
    # under-fertilize a real field.
    nominal = OneHotEncoder(
        categories=[CROPS, VARIETIES, CULTIVATION_METHODS, AGE_BANDS, DATA_SOURCES],
        handle_unknown="error",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("cont", continuous, CONTINUOUS),
        ("ord", ordered, ORDERED),
        ("nom", nominal, NOMINAL),
    ])
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS, all tests including `TestCultivationMethodEncoding` and `TestAgeBandEncoding`.

- [ ] **Step 6: Commit**

```bash
git add ml/preprocessing.py tests/fixtures.py tests/test_preprocessing.py
git commit -m "feat: rename potassium_ppm to potassium_cmol_kg, add variety/cultivation_method/age_band features"
```

---

### Task 6: `main.py` — new request fields, response schema, K conversion, advisory text

**Files:**
- Modify: `main.py` (full rewrite)
- Modify: `tests/test_api.py` (full rewrite)

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` (Task 5, `ml.preprocessing`); `CROPS`, `k_cmol_to_ppm`,
  `target_unit_for_crop`, `resolve_variety`, `resolve_cultivation_method`, `resolve_age_band`
  (Task 1, `data.nutrient_recommendation_guide`); `application_schedule_for_crop` (Task 3,
  `data.mode_of_application`).
- Produces: `SoilTestRequest` gains `variety`/`cultivation_method`/`age_band` (all optional).
  `PredictionResponse` fields renamed `target_n`/`target_p2o5`/`target_k2o`, gains `target_unit`
  and `application_schedule`. `potassium_ppm` field renamed `potassium_cmol_kg`.

**Context — the ML model still does the prediction, not a direct table lookup.** `/predict`
calls the *trained regressor* for `target_n`/`target_p2o5`/`target_k2o` exactly as before (the
regressor learned from real, table-derived training labels — see Task 9 — but a live request is
scored by the model, not by re-running `nutrient_requirements()`). `target_unit` and
`application_schedule` are the two genuinely new *deterministic* lookups this task adds,
independent of the model. `farm_totals` (kg/ha rate × hectares) is only meaningful for `kg_ha`
crops — it's now `None` whenever `target_unit == "kg_tree"`, since a per-tree rate needs a tree
count (which this API doesn't collect) to total up, not a hectare figure.

- [ ] **Step 1: Replace the API tests first**

Replace `tests/test_api.py` in full:

```python
import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


SAMPLE_PAYLOAD = {
    "crop": "Corn", "data_source": "lab", "ph": 6.4,
    "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
    "potassium_cmol_kg": 0.35, "farm_area_sqm": 5000,
    "laboratory_code": "TEST-001",
}


class TestHealth:
    def test_reports_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_loaded"] is True
        assert body["model_version"]


class TestPredict:
    def test_returns_expected_shape(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        for field in ["target_n", "target_p2o5", "target_k2o", "target_unit",
                      "app1_formulation", "app2_formulation", "application_schedule",
                      "derived_remarks", "model_version", "source"]:
            assert field in body
        assert body["source"] == "ml-ensemble"
        assert body["target_unit"] == "kg_ha"

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["ph_remark"] == "SLIGHTLY ACIDIC"
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"
        assert body["derived_remarks"]["p_remark"] == "HIGH"
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_k_remark_uses_the_converted_ppm_value_not_the_raw_cmol_kg(self, client):
        """0.35 cmol/kg -> 136.85 ppm, which is SUFFICIENT+ on the (unchanged,
        ppm-scale) K bands. If a caller forgot the x391 conversion and fed
        0.35 straight into derive_k_remark, that would read as DEFICIENT
        (well under the 60 ppm floor) — this test exists to catch exactly
        that regression."""
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_application_schedule_present_for_a_covered_crop(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["application_schedule"] is not None
        assert len(body["application_schedule"]) > 0

    def test_farm_totals_scale_the_per_hectare_rate_for_kg_ha_crops(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        totals = body["farm_totals"]
        assert totals["area_ha"] == pytest.approx(0.5)
        assert totals["n_kg"] == pytest.approx(body["target_n"] * 0.5, rel=1e-3)

    def test_farm_totals_absent_when_area_not_given(self, client):
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "farm_area_sqm"}
        assert client.post("/predict", json=payload).json()["farm_totals"] is None

    def test_farm_totals_absent_for_kg_tree_crops(self, client):
        """A per-tree rate can't be scaled by hectares alone — this API
        doesn't collect a tree count, so totalling isn't attempted."""
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert body["farm_totals"] is None

    def test_minimal_payload_is_accepted(self, client):
        """Only crop, data_source and ph are required — everything else is
        imputed and flagged."""
        response = client.post("/predict", json={
            "crop": "Corn", "data_source": "iot", "ph": 6.4,
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("source", ["lab", "stk", "iot"])
    def test_every_data_source_is_servable(self, client, source):
        payload = {**SAMPLE_PAYLOAD, "data_source": source}
        assert client.post("/predict", json=payload).status_code == 200

    def test_variety_is_accepted_and_optional(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "variety": "Hybrid"}
        assert client.post("/predict", json=payload).status_code == 200
        without_variety = {**SAMPLE_PAYLOAD, "crop": "Rice"}
        assert client.post("/predict", json=without_variety).status_code == 200

    def test_cultivation_method_is_accepted_and_optional(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "cultivation_method": "Upland"}
        assert client.post("/predict", json=payload).status_code == 200
        without = {**SAMPLE_PAYLOAD, "crop": "Rice"}
        assert client.post("/predict", json=without).status_code == 200

    def test_age_band_bearing_alias_is_accepted(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_age_band_omitted_behaves_like_bearing(self, client):
        with_bearing = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        without = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        response_a = client.post("/predict", json=with_bearing).json()
        response_b = client.post("/predict", json=without).json()
        assert response_a["target_n"] == pytest.approx(response_b["target_n"])


class TestValidation:
    def test_ph_above_fourteen_is_rejected(self, client):
        assert client.post("/predict", json={**SAMPLE_PAYLOAD, "ph": 15.0}).status_code == 422

    def test_negative_ppm_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "phosphorus_ppm": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_data_source_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "data_source": "satellite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict", json={"crop": "Corn"}).status_code == 422

    def test_unknown_crop_is_rejected_with_the_valid_list(self, client):
        """A silent fallback would return a confident number computed from no
        crop information at all."""
        response = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Corn" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a recommendation."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `main.py`'s `SoilTestRequest` still has `potassium_ppm`, no `variety`/
`cultivation_method`/`age_band` fields, and `PredictionResponse` still has the old `_kg_ha`
field names.

- [ ] **Step 3: Implement**

Replace `main.py` in full:

```python
"""SURI fertilizer prediction service (spec section 11).

Run: uvicorn main:app --port 8001

Port 8001, not 8000: suri/features/flows/iot/IoTFlow.tsx already hard-codes 8000
for a different, unrelated crop-capability contract.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from data.mode_of_application import application_schedule_for_crop
from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, resolve_age_band, resolve_cultivation_method,
    resolve_variety, target_unit_for_crop,
)
from ml.agronomy import total_nitrogen_from_om
from ml.preprocessing import FEATURE_COLUMNS
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)
from ml.train import ARTIFACT_ROOT

logger = logging.getLogger("suri-ml")
logging.basicConfig(level=logging.INFO)

SQM_PER_HECTARE = 10_000

app = FastAPI(title="SURI Fertilizer ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SoilTestRequest(BaseModel):
    crop: str
    variety: str | None = None
    cultivation_method: str | None = None
    age_band: str | None = None
    data_source: Literal["lab", "stk", "iot"]
    ph: float = Field(ge=0, le=14)
    organic_matter_pct: float | None = Field(default=None, ge=0, le=100)
    total_nitrogen_pct: float | None = Field(default=None, ge=0)
    phosphorus_ppm: float | None = Field(default=None, ge=0)
    potassium_cmol_kg: float | None = Field(default=None, ge=0)
    temp: float | None = None
    moist: float | None = Field(default=None, ge=0, le=100)
    ec: float | None = Field(default=None, ge=0)
    sal: float | None = Field(default=None, ge=0)
    farm_area_sqm: float | None = Field(default=None, gt=0)
    laboratory_code: str | None = None

    @field_validator("crop")
    @classmethod
    def crop_must_be_known(cls, value: str) -> str:
        if value not in CROPS:
            raise ValueError(f"Unknown crop '{value}'. Valid crops: {', '.join(CROPS)}")
        return value


class FarmTotals(BaseModel):
    area_ha: float
    n_kg: float
    p2o5_kg: float
    k2o_kg: float


class PredictionResponse(BaseModel):
    target_n: float
    target_p2o5: float
    target_k2o: float
    target_unit: str
    app1_formulation: str
    app2_formulation: str
    application_schedule: str | None
    derived_remarks: dict[str, str]
    farm_totals: FarmTotals | None
    model_version: str
    source: str = "ml-ensemble"


def latest_version() -> str:
    if not os.path.isdir(ARTIFACT_ROOT):
        raise RuntimeError(
            f"No artifact directory at {ARTIFACT_ROOT}. Run: python -m ml.train --csv <path>"
        )
    versions = sorted(entry for entry in os.listdir(ARTIFACT_ROOT)
                      if os.path.isdir(os.path.join(ARTIFACT_ROOT, entry)))
    if not versions:
        raise RuntimeError(f"{ARTIFACT_ROOT} is empty. Run: python -m ml.train --csv <path>")
    return versions[-1]


def load_artifacts() -> dict:
    """Load at import time so a missing or corrupt model fails at boot rather
    than surfacing as a 500 on every request, or silently serving a stale model."""
    version = latest_version()
    directory = os.path.join(ARTIFACT_ROOT, version)
    return {
        "version": version,
        "preprocessor": joblib.load(os.path.join(directory, "preprocessor.joblib")),
        "regressor": joblib.load(os.path.join(directory, "regressor.joblib")),
        "clf_app1": joblib.load(os.path.join(directory, "clf_app1.joblib")),
        "clf_app2": joblib.load(os.path.join(directory, "clf_app2.joblib")),
        "encoders": joblib.load(os.path.join(directory, "label_encoders.joblib")),
    }


ARTIFACTS = load_artifacts()


def log_prediction(request: SoilTestRequest, response: PredictionResponse) -> None:
    from data.ingest_to_supabase import get_client
    get_client().table("ml_fertilizer_predictions").insert({
        "laboratory_code": request.laboratory_code,
        "data_source": request.data_source,
        "request_payload": request.model_dump(),
        "response_payload": response.model_dump(),
        "model_version": response.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def _build_feature_row(request: SoilTestRequest) -> pd.DataFrame:
    organic_matter = request.organic_matter_pct
    total_nitrogen = request.total_nitrogen_pct
    if total_nitrogen is None and organic_matter is not None:
        total_nitrogen = total_nitrogen_from_om(organic_matter)

    k_ppm = k_cmol_to_ppm(request.potassium_cmol_kg) if request.potassium_cmol_kg is not None else None

    row = {
        "ph": request.ph,
        "organic_matter_pct": organic_matter,
        "total_nitrogen_pct": total_nitrogen,
        "phosphorus_ppm": request.phosphorus_ppm,
        "potassium_cmol_kg": request.potassium_cmol_kg,
        "temp": request.temp, "moist": request.moist,
        "ec": request.ec, "sal": request.sal,
        "ph_remark": derive_ph_remark(request.ph),
        "om_remark": derive_om_remark(organic_matter),
        "p_remark": derive_p_remark(request.phosphorus_ppm),
        "k_remark": derive_k_remark(k_ppm),
        "crop": request.crop,
        "variety": resolve_variety(request.crop, request.variety),
        "cultivation_method": resolve_cultivation_method(request.crop, request.cultivation_method),
        "age_band": resolve_age_band(request.crop, request.age_band),
        "data_source": request.data_source,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.get("/health")
def health() -> dict:
    supabase_reachable = True
    try:
        from data.ingest_to_supabase import get_client
        get_client()
    except Exception:
        supabase_reachable = False

    return {
        "status": "ok",
        "model_version": ARTIFACTS["version"],
        "artifacts_loaded": True,
        "supabase_reachable": supabase_reachable,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SoilTestRequest) -> PredictionResponse:
    features = _build_feature_row(request)

    try:
        X = ARTIFACTS["preprocessor"].transform(features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    n, p2o5, k2o = ARTIFACTS["regressor"].predict(X)[0]
    n, p2o5, k2o = round(float(n), 3), round(float(p2o5), 3), round(float(k2o), 3)

    app1 = ARTIFACTS["encoders"]["app1_formulation"].inverse_transform(
        ARTIFACTS["clf_app1"].predict(X))[0]
    app2 = ARTIFACTS["encoders"]["app2_formulation"].inverse_transform(
        ARTIFACTS["clf_app2"].predict(X))[0]

    target_unit = target_unit_for_crop(request.crop)

    farm_totals = None
    if request.farm_area_sqm and target_unit == "kg_ha":
        # Derived from the rounded per-hectare rates, not the raw regressor
        # output, so the displayed totals always reconcile with the displayed
        # per-hectare figures a farmer would multiply by hand. Not attempted
        # for kg_tree crops — that needs a tree count, which this API
        # doesn't collect, not a hectare figure.
        hectares = request.farm_area_sqm / SQM_PER_HECTARE
        farm_totals = FarmTotals(
            area_ha=round(hectares, 4),
            n_kg=round(n * hectares, 3),
            p2o5_kg=round(p2o5 * hectares, 3),
            k2o_kg=round(k2o * hectares, 3),
        )

    response = PredictionResponse(
        target_n=n,
        target_p2o5=p2o5,
        target_k2o=k2o,
        target_unit=target_unit,
        app1_formulation=str(app1),
        app2_formulation=str(app2),
        application_schedule=application_schedule_for_crop(request.crop),
        derived_remarks={
            "ph_remark": features.at[0, "ph_remark"],
            "om_remark": features.at[0, "om_remark"],
            "p_remark": features.at[0, "p_remark"],
            "k_remark": features.at[0, "k_remark"],
        },
        farm_totals=farm_totals,
        model_version=ARTIFACTS["version"],
    )

    # An audit-log outage must never deny a farmer their recommendation.
    try:
        log_prediction(request, response)
    except Exception as error:
        logger.warning("Prediction logging failed (returning result anyway): %s", error)

    return response
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: add variety/cultivation_method/age_band, target_unit, application_schedule to the API"
```

---

### Task 7: `ml/train.py` — retire the golden test, wrap in `LogTargetRegressor`, add per-crop metrics

**Files:**
- Modify: `ml/train.py` (full rewrite)
- Modify: `tests/test_train.py` (full rewrite)

**Interfaces:**
- Consumes: `sample_training_frame` (Task 5, `tests.fixtures`); `LogTargetRegressor` (Task 4,
  `ml.model`); `TARGET_COLUMNS` (Task 5, `ml.preprocessing`, now `target_n`/`target_p2o5`/
  `target_k2o`).
- Produces: `train()`'s returned metadata has no `"golden_test"` key, gains
  `metrics["by_crop"]` (mirroring the existing `metrics["by_source"]`). The saved
  `regressor.joblib` is now a `LogTargetRegressor`-wrapped ensemble — `main.py`'s
  `ARTIFACTS["regressor"].predict(X)` needs no changes for this (see Task 4's docstring).

- [ ] **Step 1: Replace the tests first**

Replace `tests/test_train.py` in full:

```python
import json
import numpy as np
import pytest

from tests.fixtures import sample_training_frame
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import grouped_split, cv_weights, train


@pytest.fixture(scope="module")
def frame():
    return sample_training_frame(num_records=200, seed=42)


class TestGroupedSplit:
    def test_no_laboratory_code_spans_two_splits(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        train_codes = set(train_df.laboratory_code)
        val_codes = set(val_df.laboratory_code)
        test_codes = set(test_df.laboratory_code)
        assert not (train_codes & val_codes)
        assert not (train_codes & test_codes)
        assert not (val_codes & test_codes)

    def test_splits_cover_every_row_exactly_once(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        assert len(train_df) + len(val_df) + len(test_df) == len(frame)

    def test_split_is_roughly_seventy_fifteen_fifteen(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        assert 0.60 < len(train_df) / len(frame) < 0.80
        assert 0.08 < len(test_df) / len(frame) < 0.22

    def test_split_is_deterministic(self, frame):
        first = grouped_split(frame, seed=42)[0].laboratory_code.tolist()
        second = grouped_split(frame, seed=42)[0].laboratory_code.tolist()
        assert first == second

    def test_each_code_appears_exactly_once(self, frame):
        """Unlike the retired synthetic generator (which produced 3 rows per
        code, one per data_source), real training data is one row per
        laboratory_code — only "lab" has real training rows now."""
        train_df, _, _ = grouped_split(frame, seed=42)
        counts = train_df.groupby("laboratory_code").size()
        assert (counts == 1).all()


class TestCvWeights:
    def test_returns_one_weight_pair_per_target(self, frame):
        from ml.preprocessing import build_preprocessor
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(frame[FEATURE_COLUMNS])
        y = frame[TARGET_COLUMNS].to_numpy()
        weights = cv_weights(X, y, frame.laboratory_code.to_numpy())
        assert len(weights) == 3
        for xgb_weight, krr_weight in weights:
            assert xgb_weight >= 0 and krr_weight >= 0
            assert xgb_weight + krr_weight == pytest.approx(1.0)


class TestTrain:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        output = tmp_path_factory.mktemp("artifacts")
        return train(sample_training_frame(num_records=200, seed=42), output_dir=str(output)), output

    def test_writes_all_artifacts(self, result):
        _, output = result
        for name in ["preprocessor.joblib", "regressor.joblib", "clf_app1.joblib",
                     "clf_app2.joblib", "label_encoders.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 200
        assert "trained_at" in written
        assert "version" in written

    def test_reports_metrics_per_data_source(self, result):
        metadata, _ = result
        assert set(metadata["metrics"]["by_source"].keys()) == {"lab"}

    def test_reports_metrics_per_crop(self, result):
        """New this pass: with targets now spanning kg_ha and kg_tree
        scales in the same columns, aggregate metrics alone could hide a
        regression specific to the tree crops — this breakdown is how a
        retrain would actually catch that."""
        metadata, _ = result
        from data.nutrient_recommendation_guide import CROPS
        assert set(metadata["metrics"]["by_crop"].keys()) <= set(CROPS)
        assert len(metadata["metrics"]["by_crop"]) > 0

    def test_beats_the_mean_baseline_on_every_target(self, result):
        metadata, _ = result
        for target, scores in metadata["metrics"]["overall"].items():
            assert scores["mae"] < scores["dummy_mae"], target

    def test_no_golden_test_key(self, result):
        """Retired along with Sorghum — see the design spec's 'Golden test —
        retired' section."""
        metadata, _ = result
        assert "golden_test" not in metadata

    def test_predictions_are_non_negative(self, result):
        """A sanity floor independent of any specific golden record — every
        real target in this system (kg/ha or kg/tree) is >= 0; a
        LogTargetRegressor bug that let a negative slip through the expm1
        inverse would be a real, silent correctness break."""
        metadata, output = result
        import joblib
        import numpy as np
        regressor = joblib.load(output / "regressor.joblib")
        preprocessor = joblib.load(output / "preprocessor.joblib")
        frame = sample_training_frame(num_records=20, seed=99)
        X = preprocessor.transform(frame[FEATURE_COLUMNS])
        predicted = regressor.predict(X)
        assert np.all(predicted >= 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ml/train.py` still calls `_golden_test()` against `GOLDEN_SCC417`, which no
longer matches `FEATURE_COLUMNS` (missing `variety`/`cultivation_method`/`age_band`, and the
target columns it reads are still the old `_kg_ha` names) — expect a hard crash inside `train()`
itself, failing every test in `TestTrain`.

- [ ] **Step 3: Implement**

Replace `ml/train.py` in full:

```python
"""Training, evaluation, and artifact writing (spec section 10).

Every split and cross-validation fold is grouped on laboratory_code — each
physical sample must stay entirely within one split, since a random row
split would otherwise leak information across train/test.

The regressor is wrapped in LogTargetRegressor (ml/model.py) because
targets now span both kg/ha (~20-300, field crops) and kg/tree (~0.003-2,
Coconut/Cacao/Mango/Banana) scales in the same three columns — fitting
directly on that mixture would let the loss be dominated by the
larger-scale rows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from ml.model import EnsembleFertilizerRegressor, LogTargetRegressor, build_classifier, build_krr, build_xgb_regressor
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS, build_preprocessor

RANDOM_SEED = 42
ARTIFACT_ROOT = "ml/artifacts"
GROUP_COLUMN = "laboratory_code"
CLASSIFIER_TARGETS = ["app1_formulation", "app2_formulation"]


def grouped_split(frame: pd.DataFrame, seed: int = RANDOM_SEED):
    groups = frame[GROUP_COLUMN].to_numpy()

    holdout = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(holdout.split(frame, groups=groups))
    train_df = frame.iloc[train_idx]
    rest_df = frame.iloc[rest_idx]

    halve = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(halve.split(rest_df, groups=rest_df[GROUP_COLUMN].to_numpy()))
    return train_df, rest_df.iloc[val_idx], rest_df.iloc[test_idx]


def cv_weights(X, y, groups, n_splits: int = 5) -> list[tuple[float, float]]:
    """Set each target's (xgb, krr) voting weights from grouped cross-validated
    R2, so each nutrient favours whichever model actually performs better.

    Scored in log1p space, self-contained to this function — consistent
    with how the final model is actually fit (see LogTargetRegressor) —
    without requiring the caller to pre-transform y."""
    splitter = GroupKFold(n_splits=n_splits)
    y_log = np.log1p(y)
    weights: list[tuple[float, float]] = []

    for target in range(y_log.shape[1]):
        scores = {"xgb": [], "krr": []}
        for train_idx, test_idx in splitter.split(X, y_log[:, target], groups):
            for name, builder in (("xgb", build_xgb_regressor), ("krr", build_krr)):
                model = builder().fit(X[train_idx], y_log[train_idx, target])
                scores[name].append(r2_score(y_log[test_idx, target], model.predict(X[test_idx])))

        xgb_score = max(float(np.mean(scores["xgb"])), 0.0)
        krr_score = max(float(np.mean(scores["krr"])), 0.0)
        total = xgb_score + krr_score
        weights.append((0.5, 0.5) if total == 0 else (xgb_score / total, krr_score / total))

    return weights


def evaluate(model, X, y, sources, crops, X_train, y_train) -> dict:
    predicted = model.predict(X)

    overall = {}
    for index, name in enumerate(TARGET_COLUMNS):
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train[:, index])
        overall[name] = {
            "mae": float(mean_absolute_error(y[:, index], predicted[:, index])),
            "r2": float(r2_score(y[:, index], predicted[:, index])),
            "dummy_mae": float(mean_absolute_error(y[:, index], dummy.predict(X))),
        }

    def _grouped_metrics(labels: np.ndarray) -> dict:
        result = {}
        for label in np.unique(labels):
            mask = labels == label
            result[str(label)] = {
                name: {
                    "mae": float(mean_absolute_error(y[mask, index], predicted[mask, index])),
                    "r2": float(r2_score(y[mask, index], predicted[mask, index])),
                }
                for index, name in enumerate(TARGET_COLUMNS)
            }
        return result

    return {
        "overall": overall,
        "by_source": _grouped_metrics(sources),
        "by_crop": _grouped_metrics(crops),
    }


def train(frame: pd.DataFrame, output_dir: str | None = None, seed: int = RANDOM_SEED) -> dict:
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    output_dir = output_dir or os.path.join(ARTIFACT_ROOT, version)
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = grouped_split(frame, seed=seed)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    weights = cv_weights(X_train, y_train, train_df[GROUP_COLUMN].to_numpy())
    regressor = LogTargetRegressor(EnsembleFertilizerRegressor(weights=weights)).fit(X_train, y_train)

    metrics = evaluate(
        regressor, X_test, y_test,
        test_df.data_source.to_numpy(), test_df.crop.to_numpy(),
        X_train, y_train,
    )

    # Classification heads. Note these will score near-perfectly on covered
    # classes: the recipe is a deterministic function of the targets, so the
    # classifier only rediscovers arithmetic. That is not evidence of
    # real-world skill — it becomes meaningful only on real DA data where
    # agronomists exercise judgment.
    #
    # Formulation strings are near-unique per record (continuous kg amounts),
    # so the grouped train split cannot cover every class — the encoder is
    # fit on train_df only, matching exactly what XGBClassifier sees, and
    # held-out metrics are computed only over the subset of test labels the
    # model actually saw at train time (see test_coverage).
    encoders: dict[str, LabelEncoder] = {}
    classifiers = {}
    classification_metrics = {}
    for target in CLASSIFIER_TARGETS:
        encoder = LabelEncoder().fit(train_df[target])
        model = build_classifier().fit(X_train, encoder.transform(train_df[target]))

        known_mask = test_df[target].isin(encoder.classes_).to_numpy()
        coverage = float(known_mask.mean()) if len(test_df) else 0.0
        if known_mask.any():
            actual = encoder.transform(test_df[target][known_mask])
            predicted = model.predict(X_test[known_mask])
            accuracy = float(accuracy_score(actual, predicted))
            macro_f1 = float(f1_score(actual, predicted, average="macro"))
        else:
            accuracy = None
            macro_f1 = None

        encoders[target] = encoder
        classifiers[target] = model
        classification_metrics[target] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_classes": int(len(encoder.classes_)),
            "test_coverage": coverage,
            "note": "Deterministic given the targets — not a V1 success signal. "
                    "Formulations are near-unique per record; metrics cover only "
                    "the test_coverage fraction of test rows whose label was seen "
                    "in training.",
        }

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_rows": len(frame),
        "n_records": int(frame[GROUP_COLUMN].nunique()),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "voting_weights": [{"xgb": xgb, "krr": krr} for xgb, krr in weights],
        "metrics": metrics,
        "classification_metrics": classification_metrics,
        "recipe_vocabulary": {
            target: encoders[target].classes_.tolist() for target in CLASSIFIER_TARGETS
        },
    }

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(regressor, os.path.join(output_dir, "regressor.joblib"))
    joblib.dump(classifiers["app1_formulation"], os.path.join(output_dir, "clf_app1.joblib"))
    joblib.dump(classifiers["app2_formulation"], os.path.join(output_dir, "clf_app2.joblib"))
    joblib.dump(encoders, os.path.join(output_dir, "label_encoders.joblib"))
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def _load_frame(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)
    from data.ingest_to_supabase import TABLE, get_client
    rows = get_client().table(TABLE).select("*").execute().data
    return pd.DataFrame(rows)


def _print_summary(metadata: dict) -> None:
    print(f"\nModel version: {metadata['version']}")
    print(f"Records: {metadata['n_records']}  Rows: {metadata['n_rows']}")
    print("\nRegression (held-out test):")
    for name, scores in metadata["metrics"]["overall"].items():
        print(f"  {name:22s} MAE {scores['mae']:7.3f}  R2 {scores['r2']:6.3f}"
              f"  (mean baseline MAE {scores['dummy_mae']:7.3f})")
    print("\nMAE by data source:")
    for source, targets in metadata["metrics"]["by_source"].items():
        errors = "  ".join(f"{name}={scores['mae']:6.3f}"
                           for name, scores in targets.items())
        print(f"  {source:4s}  {errors}")
    print("\nMAE by crop:")
    for crop, targets in metadata["metrics"]["by_crop"].items():
        errors = "  ".join(f"{name}={scores['mae']:6.3f}"
                           for name, scores in targets.items())
        print(f"  {crop:10s}  {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SURI fertilizer ensemble.")
    parser.add_argument("--csv", help="Train from a local CSV instead of Supabase.")
    args = parser.parse_args()

    result = train(_load_frame(args.csv))
    _print_summary(result)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "feat: retire the golden SCC-26-417-417 fixture, wrap regressor in LogTargetRegressor, add per-crop metrics"
```

---

### Task 8: Parser foundations — cell parsing, lab-code/crop/cultivation-method/age-band normalization

**Files:**
- Create: `data/parse_real_soil_database.py`
- Create: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Consumes: `CROPS` (Task 1, `data.nutrient_recommendation_guide`).
- Produces: `parse_numeric_cell(value) -> float | None`, `normalize_lab_code(raw_code: str, year_label: str) -> str`, `normalize_crop(raw_text: str | None) -> tuple[str, str | None] | None`, `normalize_cultivation_method(crop: str, raw_text: str) -> str | None`, `normalize_age_band(crop: str, raw_text: str) -> str | None`, `find_column(header_row: tuple, target_text: str) -> int` — all pure functions, consumed by Task 9's sheet-reading logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parse_real_soil_database.py
import pytest

from data.parse_real_soil_database import (
    parse_numeric_cell, normalize_lab_code, normalize_crop,
    normalize_cultivation_method, normalize_age_band, find_column,
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
        """Crops below the real-sample floor (Cabbage/Sorghum/Sugarcane) or
        not in the 9-crop list at all must not silently pass through."""
        assert normalize_crop("Sorghum") is None
        assert normalize_crop("Ampalaya") is None


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_parse_real_soil_database.py -v`
Expected: FAIL — `data/parse_real_soil_database.py` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# data/parse_real_soil_database.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "feat: add parser foundations (cell parsing, lab-code/crop/cultivation-method/age-band normalization)"
```

---

### Task 9: Parser — per-sheet reading, row building, CSV output, data-quality report

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Modify: `tests/test_parse_real_soil_database.py`
- Modify: `requirements.txt` — add `openpyxl>=3.1.0` (confirmed present at the repo root; no other dependency file exists).

**Interfaces:**
- Consumes: `parse_numeric_cell`, `normalize_lab_code`, `normalize_crop`,
  `normalize_cultivation_method`, `normalize_age_band`, `find_column` (Task 8, same file);
  `nutrient_requirements` (Task 2, `ml.agronomy`); `target_unit_for_crop`, `k_cmol_to_ppm`
  (Task 1, `data.nutrient_recommendation_guide`); `total_nitrogen_from_om` (`ml.agronomy`,
  unchanged); `plan_applications` (`ml.recipes`, unchanged); `derive_ph_remark`/
  `derive_om_remark`/`derive_p_remark`/`derive_k_remark` (`ml.remarks`, unchanged).
- Produces: `SHEET_CONFIGS: list[dict]`, `parse_sheet(worksheet, config: dict) -> list[dict]`,
  `build_training_row(raw: dict) -> dict | None`, `parse_workbook(xlsx_path: str) -> tuple[list[dict], dict]`
  (rows, data-quality report), `OUTPUT_CSV: str` — `OUTPUT_CSV` consumed by Task 10
  (`data/ingest_to_supabase.py`).

**Context:** Exact, individually-verified per-sheet layout:

| Sheet | Main header row | Chemistry units row | Data starts | Lab code col | Crop col |
|---|---|---|---|---|---|
| Laboratory Details2022 | 3 | 5 | 6 | 1 | 14 |
| Laboratory Details2023 | 3 | 5 | 6 | 1 | 14 |
| Laboratory Details2024 | 3 | 5 | 6 | 1 | 14 |
| Soil Data 2025 | 2 | 2 | 4 | 0 | 9 |

The three "Laboratory Details" sheets share one chemistry-header vocabulary (`"pH (H2O, 1:1)"`
etc.) but at column positions that drift between 2022/2024 and 2023 specifically — this is
exactly why `find_column` (Task 8) resolves each sheet's columns independently rather than
trusting the table above for anything but row numbers. `Soil Data 2025` uses an entirely
different, shorter header vocabulary (`"pH"`, `"OM"`, etc.) at fixed early columns.

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_parse_real_soil_database.py

import openpyxl
from data.parse_real_soil_database import (
    build_training_row, parse_sheet, SHEET_CONFIGS, LAB_DETAIL_HEADERS,
    SOIL_DATA_2025_HEADERS,
)


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
        assert row["om_remark"] == "MODERATELY LOW"
        assert row["target_unit"] == "kg_ha"
        assert isinstance(row["target_n"], float)
        assert row["target_n"] > 0

    def test_k_remark_uses_the_converted_ppm_not_the_raw_cmol_kg(self):
        """0.35 cmol/kg -> 136.85 ppm -> SUFFICIENT+. Feeding the raw 0.35
        into derive_k_remark directly would misclassify it as DEFICIENT."""
        raw = {
            "laboratory_code": "2022-SCC417", "crop": "Corn", "variety": None,
            "cultivation_method": None, "age_band": None,
            "ph": 6.4, "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
            "potassium_cmol_kg": 0.35, "total_nitrogen_pct": None,
        }
        assert build_training_row(raw)["k_remark"] == "SUFFICIENT+"

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_parse_real_soil_database.py -v`
Expected: FAIL — `build_training_row`, `parse_sheet`, `SHEET_CONFIGS`, `LAB_DETAIL_HEADERS`,
`SOIL_DATA_2025_HEADERS` don't exist yet.

- [ ] **Step 3: Implement**

Append to `data/parse_real_soil_database.py`:

```python
import csv
import sys

import openpyxl

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark
from data.nutrient_recommendation_guide import (
    k_cmol_to_ppm, target_unit_for_crop,
    resolve_variety, resolve_cultivation_method, resolve_age_band,
)

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
    app1, app2 = plan_applications(n, p2o5, k2o)
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
```

Add `openpyxl>=3.1.0` as a new line to `requirements.txt`.

- [ ] **Step 4: Run to verify it passes**

Run: `pip install -r requirements.txt` (installs openpyxl), then `pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py requirements.txt
git commit -m "feat: complete the real soil database parser — sheet reading, row building, CSV output"
```

---

### Task 10: Point ingestion at the new parser's output

**Files:**
- Modify: `data/ingest_to_supabase.py:16`
- Modify: `tests/test_ingest_to_supabase.py`

**Interfaces:**
- Consumes: `OUTPUT_CSV` (Task 9, `data.parse_real_soil_database`); `sample_training_frame` (Task 5, `tests.fixtures`).

- [ ] **Step 1: Update the failing test first**

Replace `tests/test_ingest_to_supabase.py` in full:

```python
import math

import pytest

from data.ingest_to_supabase import chunked, frame_to_records
from tests.fixtures import sample_training_frame


class TestChunking:
    def test_splits_into_batches_of_the_given_size(self):
        batches = list(chunked(list(range(250)), 100))
        assert [len(batch) for batch in batches] == [100, 100, 50]

    def test_empty_input_yields_no_batches(self):
        assert list(chunked([], 100)) == []

    def test_exact_multiple_has_no_trailing_empty_batch(self):
        assert len(list(chunked(list(range(200)), 100))) == 2


class TestRecordConversion:
    @pytest.fixture(scope="class")
    def records(self):
        return frame_to_records(sample_training_frame(num_records=10, seed=42))

    def test_produces_one_record_per_row(self, records):
        assert len(records) == 10

    def test_nan_becomes_none_for_json(self, records):
        """Supabase rejects NaN — it is not valid JSON. Any NaN produced by
        pandas (e.g. from an unset optional field) must serialise as null."""
        for record in records:
            for value in record.values():
                assert not (isinstance(value, float) and math.isnan(value))

    def test_records_are_json_serialisable(self, records):
        import json
        json.dumps(records)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ingest_to_supabase.py -v`
Expected: FAIL — `data/ingest_to_supabase.py:16`'s `from data.generate_synthetic_data import
OUTPUT_CSV` still points at the old generator (import still succeeds, since that file isn't
deleted until Task 11 — the test failures here come purely from the schema/assertion changes
above being correct against the new fixture, not an import error).

- [ ] **Step 3: Implement**

In `data/ingest_to_supabase.py`, change line 16:

```python
from data.generate_synthetic_data import OUTPUT_CSV
```

to:

```python
from data.parse_real_soil_database import OUTPUT_CSV
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_ingest_to_supabase.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add data/ingest_to_supabase.py tests/test_ingest_to_supabase.py
git commit -m "feat: point ingestion at the real soil database parser's output"
```

---

### Task 11: Delete the retired synthetic generator and crop-baseline formula

**Files:**
- Delete: `data/generate_synthetic_data.py`
- Delete: `tests/test_generate_synthetic_data.py`
- Delete: `data/crop_baselines.py`
- Delete: `tests/test_crop_baselines.py` if it exists as a separate file (it does not in this
  codebase — baseline tests live inside `tests/test_agronomy.py`, already rewritten in Task 2 to
  no longer reference `data.crop_baselines`; nothing further to delete there).

**Interfaces:**
- Consumes: nothing — this task only runs after Tasks 1, 2, 5, 7, 9, 10 have removed every
  remaining import of `data.crop_baselines` and `data.generate_synthetic_data`.

- [ ] **Step 1: Confirm nothing still imports either retired module**

Run: `grep -rn "generate_synthetic_data\|crop_baselines" --include="*.py" .`
Expected: no matches outside `data/generate_synthetic_data.py` and
`tests/test_generate_synthetic_data.py` themselves (both about to be deleted) and
`data/crop_baselines.py` itself (also about to be deleted). If anything else matches, stop and
fix that reference first — do not delete out from under a live import.

- [ ] **Step 2: Delete all three files**

```bash
git rm data/generate_synthetic_data.py tests/test_generate_synthetic_data.py data/crop_baselines.py
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: PASS, every test in the repo — the first point in the plan the *entire* suite runs at
once, catching any straggler cross-file dependency the per-task runs above didn't surface.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove the synthetic data generator and the retired crop-baseline formula"
```

---

### Task 12: End-to-end run — parse, ingest, retrain, verify

This task has no code changes — it's the real-data equivalent of a smoke test, and the point
where the data-quality report (Task 9) gets its first real read.

- [ ] **Step 1: Confirm the source file is reachable**

Run: `ls "../suri/public/Region 7-National Soil Database_2022-2025.xlsx"`
Expected: file exists. If not, pass the correct path as an argument in the next step instead of
relying on the default.

- [ ] **Step 2: Run the parser against the real file**

Run: `python -m data.parse_real_soil_database`
Expected: prints total row count, per-sheet breakdown, and per-crop/variety/cultivation-method/
age-band breakdowns; writes `suri_real_training_data.csv`. Sanity-check: per-crop counts should
be in the same ballpark as this plan's Global Constraints table (Rice and Corn in the thousands;
the other 7 crops each at least in the dozens). Expect `rows_by_cultivation_method` and
`rows_by_age_band` to show mostly `"None"`/default values — the source spreadsheet's `Crops`
text rarely states these explicitly (see Task 8's `normalize_cultivation_method`/
`normalize_age_band`), so most rows fall back to the guide's default row, which is expected, not
a bug. A wildly different crop/variety distribution than expected means something upstream
(crop normalization, column mapping) regressed and needs investigating before proceeding.

- [ ] **Step 3: Ingest into Supabase**

Run: `python -m data.ingest_to_supabase`
Expected: prints batch-by-batch upsert progress and a final total matching Step 2's row count.
Requires `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` in `.env` (see
`data/ingest_to_supabase.py`'s `get_client()` for the exact error message if unset).

- [ ] **Step 4: Retrain**

Run: `python -m ml.train --csv suri_real_training_data.csv`
Expected: a summary table prints (MAE/R² per target, MAE by source — should show only `lab` —
and the new MAE by crop). Check the by-crop breakdown specifically for Coconut/Cacao/Mango/
Banana (the `kg_tree` crops) — if their MAE looks proportionally far worse than the `kg_ha`
crops even after the `LogTargetRegressor` change, that's the target-scale risk flagged in the
Global Constraints materializing and worth a closer look before shipping this model version.
Compare against the previous synthetic-trained model's numbers if still available — expect
real-data R² to be *lower* than the old synthetic numbers (synthetic was uniform-random and
trivially easy to fit); that's expected, not a regression to chase.

- [ ] **Step 5: Smoke-test the running service**

Run: `uvicorn main:app --port 8001` (separate terminal), then:

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Rice","variety":"Hybrid","cultivation_method":"Lowland Wet","data_source":"lab","ph":6.5,"organic_matter_pct":2.8,"phosphorus_ppm":18.0,"potassium_cmol_kg":0.35,"laboratory_code":"TEST-001"}' \
  | python -m json.tool
```

Expected: a `PredictionResponse` JSON body with `target_n`/`target_p2o5`/`target_k2o`/
`target_unit` (`"kg_ha"`)/`app1_formulation`/`app2_formulation`/`application_schedule`/
`derived_remarks`/`model_version` all present, `model_version` matching Step 4's newly-trained
version string. Then check a tree crop:

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Coconut","age_band":"bearing","data_source":"lab","ph":6.5,"organic_matter_pct":2.8,"phosphorus_ppm":18.0,"potassium_cmol_kg":0.35,"laboratory_code":"TEST-002"}' \
  | python -m json.tool
```

Expected: `target_unit` is `"kg_tree"` and `farm_totals` is `null` (see Task 6's context note).

No commit for this task — it produces a trained model artifact and a populated Supabase table,
not a code change.

# Retraining on the Region 7 National Soil Database — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic training-data generator with a real one sourced from
`suri/public/Region 7-National Soil Database_2022-2025.xlsx`, expand and correct the crop/variety
taxonomy and units to match what the real data supports, and retrain the model on it.

**Architecture:** One new script, `data/parse_real_soil_database.py`, reads and cleans the xlsx
and writes a CSV in the same shape `ml/train.py` already expects (via the unchanged
`data/ingest_to_supabase.py` → `ml_soil_training_data` → `ml/train.py` pipeline). `data/generate_synthetic_data.py`
is deleted once nothing depends on it. Small, isolated edits to `crop_baselines.py`, `ml/agronomy.py`,
`ml/remarks.py`, `ml/preprocessing.py`, and `main.py` carry the crop/variety/unit changes through to
inference.

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
- `potassium_ppm` is renamed `potassium_cmol_kg` everywhere (Pydantic model, feature columns,
  crop baselines are unaffected since baselines are N/P2O5/K2O output targets, not K input units).
  `phosphorus_ppm` is unchanged.
- Only `data_source="lab"` gets real training rows. `"stk"` and `"iot"` remain valid values
  accepted by the API and by `ml/preprocessing.py`'s encoder categories, with zero dedicated
  training rows — this is intentional, not a gap to fill later in this plan.
- The `GOLDEN_SCC417`/`SCC-26-417-417` fixture is retired throughout (Sorghum is no longer a
  supported crop, and the record's origin was traced to a document outside this dataset).
- Every new numeric baseline (6 new crops + 4 variety overrides) and the rescaled K-remark
  thresholds are flagged in-code as needing agronomist/DA RSL confirmation before production use —
  matching the convention `data/crop_baselines.py` and `ml/remarks.py` already use for their
  existing placeholder values.
- Reference spec: `docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md`.

---

## File Structure

New files:
- `data/parse_real_soil_database.py` — reads, cleans, and normalizes the xlsx into training rows.
- `tests/test_parse_real_soil_database.py` — unit tests for the parser.

Modified files:
- `data/crop_baselines.py` — 9-crop `CROP_BASELINES`, new `VARIETY_BASELINES`.
- `ml/agronomy.py` — `nutrient_requirements()` gains an optional `variety` parameter.
- `ml/remarks.py` — `_K_BANDS` rescaled from ppm to cmol/kg.
- `ml/preprocessing.py` — `potassium_ppm` → `potassium_cmol_kg`; `NOMINAL` gains `"variety"`.
- `main.py` — `SoilTestRequest` gains `variety`; `potassium_ppm` → `potassium_cmol_kg` throughout.
- `ml/train.py` — `GOLDEN_SCC417`/`_golden_test()` removed.
- `data/ingest_to_supabase.py` — points at the new parser's output CSV.
- `tests/test_agronomy.py`, `tests/test_remarks.py`, `tests/test_preprocessing.py`,
  `tests/test_api.py`, `tests/test_train.py`, `tests/test_ingest_to_supabase.py` — updated for
  the above.
- `tests/fixtures.py` (new) — a small, hermetic test-only DataFrame builder replacing
  `generate_synthetic_data.generate()` as a fixture source for tests that need a
  plausible multi-row frame but aren't testing the real parser itself (grouped-split
  mechanics, chunking, JSON serialization). This is test infrastructure, not training data —
  never referenced by `ml/train.py`, `data/ingest_to_supabase.py`, or any production path.

Deleted files (Task 10):
- `data/generate_synthetic_data.py`
- `tests/test_generate_synthetic_data.py`

---

### Task 1: Crop and variety baselines

**Files:**
- Modify: `data/crop_baselines.py`
- Modify: `tests/test_agronomy.py:83-88` (the `TestCropBaselines` class)

**Interfaces:**
- Produces: `CROP_BASELINES: dict[str, tuple[int,int,int]]` (9 keys), `VARIETY_BASELINES: dict[tuple[str,str], tuple[int,int,int]]` (4 keys), `CROPS: list[str]` — consumed by Task 2 (`ml/agronomy.py`), Task 4 (`ml/preprocessing.py`'s `NOMINAL` category list), Task 5 (`main.py`'s crop validator).

- [ ] **Step 1: Replace `data/crop_baselines.py` in full**

```python
"""Per-crop and per-variety baseline nutrient requirements in kg/ha, before
soil-test adjustment.

Sourced from published Philippine DA/BSWM/PhilRice general fertilizer
recommendation guidance. These are starting-point central tendencies, not lab
measurements — the source spreadsheet (Region 7 National Soil Database) has
soil chemistry, not fertilizer application outcomes, so there is no way to
derive these numbers from data alone. They need agronomist review before any
production use — which is why they are isolated here: correcting a value is
a single edit plus a retrain, with no code changes anywhere else.

The crop list is restricted to crops with at least 20 real sole-crop lab
samples in that source (Cabbage/Sorghum/Sugarcane were dropped for having
only 8/8/11 — too few to reliably train or validate against).
"""

from __future__ import annotations

# crop -> (base_n, base_p2o5, base_k2o) in kg/ha
CROP_BASELINES: dict[str, tuple[int, int, int]] = {
    "Corn": (90, 60, 60),
    "Rice": (90, 30, 30),
    "Cassava": (70, 40, 80),
    "Peanut": (30, 60, 60),
    "Mango": (120, 60, 120),
    "Cacao": (100, 50, 100),
    "Banana": (250, 80, 350),
    "Tomato": (100, 80, 80),
    "Coconut": (150, 50, 250),
}

# (crop, variety) -> (base_n, base_p2o5, base_k2o) in kg/ha, overriding the
# crop-level baseline above when both crop and variety are known. Hybrid
# varieties are consistently recommended more N than their OPV/inbred
# counterparts, reflecting their higher yield potential.
VARIETY_BASELINES: dict[tuple[str, str], tuple[int, int, int]] = {
    ("Rice", "Hybrid"): (130, 40, 40),
    ("Rice", "Inbred"): (100, 30, 30),
    ("Corn", "OPV"): (75, 55, 55),
    ("Corn", "Hybrid"): (130, 65, 65),
}

CROPS: list[str] = sorted(CROP_BASELINES.keys())
```

- [ ] **Step 2: Update the crop-baseline tests**

In `tests/test_agronomy.py`, replace the `TestCropBaselines` class (lines 83-88):

```python
class TestCropBaselines:
    def test_sorghum_anchored_to_real_report(self):
        assert CROP_BASELINES["Sorghum"] == (50, 40, 60)

    def test_crops_list_matches_baseline_keys(self):
        assert set(CROPS) == set(CROP_BASELINES.keys())
```

with:

```python
class TestCropBaselines:
    def test_crops_list_matches_baseline_keys(self):
        assert set(CROPS) == set(CROP_BASELINES.keys())

    def test_dropped_crops_are_absent(self):
        for crop in ("Sorghum", "Cabbage", "Sugarcane"):
            assert crop not in CROP_BASELINES
            assert crop not in CROPS

    def test_nine_crops_present(self):
        assert len(CROPS) == 9

    def test_variety_baselines_reference_known_crops(self):
        for crop, variety in VARIETY_BASELINES:
            assert crop in CROP_BASELINES, f"{crop!r} (variety {variety!r}) not in CROP_BASELINES"
```

Add `VARIETY_BASELINES` to the existing import line (`tests/test_agronomy.py:2`):

```python
from data.crop_baselines import CROP_BASELINES, CROPS, VARIETY_BASELINES
```

- [ ] **Step 3: Run the test suite for this file**

Run: `pytest tests/test_agronomy.py -v`
Expected: `TestCropBaselines` passes (4/4); the rest of the file (`TestTotalNitrogen`, `TestPhPhosphorusPenalty`, most of `TestNutrientRequirements`) still references `Sorghum` and will fail until Task 2 — that's expected at this point, not a regression to fix now.

- [ ] **Step 4: Commit**

```bash
git add data/crop_baselines.py tests/test_agronomy.py
git commit -m "feat: replace crop baselines with 9 real-data-backed crops plus variety overrides"
```

---

### Task 2: `nutrient_requirements()` variety parameter

**Files:**
- Modify: `ml/agronomy.py:42-57`
- Modify: `tests/test_agronomy.py` (the `TestNutrientRequirements` class, lines 33-81)

**Interfaces:**
- Consumes: `CROP_BASELINES`, `VARIETY_BASELINES` (Task 1, `data.crop_baselines`).
- Produces: `nutrient_requirements(crop: str, ph: float, om: float, p_ppm: float, k_ppm: float, variety: str | None = None) -> tuple[int, int, int]` — `variety` is a new, keyword-defaulted trailing parameter, so every existing positional call site keeps working unchanged. Consumed by Task 8's parser and `main.py` indirectly (via the trained model, not a direct call).

- [ ] **Step 1: Update the failing/changed tests first**

In `tests/test_agronomy.py`, the `TestNutrientRequirements` class currently has a Sorghum-anchored golden test (lines 34-42) that must be removed now that Sorghum is dropped, and new variety-specific tests added. Replace lines 33-81 (the full `TestNutrientRequirements` class) with:

```python
class TestNutrientRequirements:
    def test_returns_whole_numbers(self):
        n, p, k = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 120.0)
        assert isinstance(n, int) and isinstance(p, int) and isinstance(k, int)

    def test_high_om_reduces_nitrogen(self):
        low = nutrient_requirements("Corn", 6.5, 1.0, 15.0, 120.0)[0]
        high = nutrient_requirements("Corn", 6.5, 5.0, 15.0, 120.0)[0]
        assert high < low

    def test_high_phosphorus_reduces_p2o5(self):
        low = nutrient_requirements("Corn", 6.5, 2.5, 5.0, 120.0)[1]
        high = nutrient_requirements("Corn", 6.5, 2.5, 35.0, 120.0)[1]
        assert high < low

    def test_high_potassium_reduces_k2o(self):
        low = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 40.0)[2]
        high = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 240.0)[2]
        assert high < low

    def test_extreme_ph_increases_p2o5(self):
        """The non-linear pH penalty is the specific reason this pipeline needs
        a tree model rather than a linear one."""
        optimal = nutrient_requirements("Corn", 6.5, 2.5, 15.0, 120.0)[1]
        alkaline = nutrient_requirements("Corn", 8.5, 2.5, 15.0, 120.0)[1]
        assert alkaline > optimal

    @pytest.mark.parametrize("crop", CROPS)
    def test_all_crops_produce_clamped_values(self, crop):
        for ph, om, p, k in [(4.5, 0.5, 3.0, 30.0), (8.5, 6.0, 40.0, 250.0)]:
            n, p2o5, k2o = nutrient_requirements(crop, ph, om, p, k)
            assert 20 <= n <= 200
            assert 20 <= p2o5 <= 150
            assert 20 <= k2o <= 200

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            nutrient_requirements("Durian", 6.5, 2.5, 15.0, 120.0)

    def test_variety_with_no_override_falls_back_to_crop_baseline(self):
        """Cassava has no VARIETY_BASELINES entries at all — passing a variety
        for it must not raise and must match the plain crop-level call."""
        without_variety = nutrient_requirements("Cassava", 6.5, 2.5, 15.0, 120.0)
        with_variety = nutrient_requirements("Cassava", 6.5, 2.5, 15.0, 120.0, variety="Hybrid")
        assert without_variety == with_variety

    def test_known_variety_overrides_the_crop_baseline(self):
        """Hybrid rice's baseline (130, 40, 40) differs from plain Rice's
        (90, 30, 30) — at the same soil-test inputs, the two must diverge."""
        plain = nutrient_requirements("Rice", 6.5, 2.5, 15.0, 120.0)
        hybrid = nutrient_requirements("Rice", 6.5, 2.5, 15.0, 120.0, variety="Hybrid")
        assert plain != hybrid

    def test_unrecognised_variety_falls_back_to_crop_baseline(self):
        """A variety string with no matching VARIETY_BASELINES entry (e.g. a
        typo, or a variety this crop doesn't track) must degrade gracefully,
        not raise — only an unknown *crop* is an error."""
        plain = nutrient_requirements("Rice", 6.5, 2.5, 15.0, 120.0)
        unknown_variety = nutrient_requirements("Rice", 6.5, 2.5, 15.0, 120.0, variety="Upland")
        assert plain == unknown_variety
```

- [ ] **Step 2: Run to verify the new/changed tests fail correctly**

Run: `pytest tests/test_agronomy.py -k TestNutrientRequirements -v`
Expected: FAIL — `nutrient_requirements()` doesn't accept a `variety` keyword yet (`TypeError: nutrient_requirements() got an unexpected keyword argument 'variety'`).

- [ ] **Step 3: Implement**

In `ml/agronomy.py`, change the import (line 10) and the function (lines 42-57):

```python
from data.crop_baselines import CROP_BASELINES, VARIETY_BASELINES
```

```python
def nutrient_requirements(
    crop: str, ph: float, om: float, p_ppm: float, k_ppm: float,
    variety: str | None = None,
) -> tuple[int, int, int]:
    """Return (n, p2o5, k2o) in whole kg/ha, matching how the real reports
    present them. Raises KeyError for an unknown crop.

    `variety` selects a more specific baseline from VARIETY_BASELINES when
    one exists for this exact (crop, variety) pair (currently only Rice
    Hybrid/Inbred and Corn OPV/Hybrid). Any other crop, or a variety with no
    matching override, falls back to the crop-level baseline silently —
    only an unrecognised *crop* is an error; an unrecognised variety is not,
    since not every crop tracks varieties.
    """
    if variety is not None and (crop, variety) in VARIETY_BASELINES:
        base_n, base_p, base_k = VARIETY_BASELINES[(crop, variety)]
    else:
        base_n, base_p, base_k = CROP_BASELINES[crop]

    n = base_n * (1 - 0.35 * _clamp((om - 2.5) / 3.5, -1, 1))
    p = base_p * (1 - 0.40 * _clamp((p_ppm - 15) / 25, -1, 1)) * ph_p_penalty(ph)
    k = base_k * (1 - 0.40 * _clamp((k_ppm - 120) / 130, -1, 1))

    return (
        int(round(_clamp(n, N_MIN, N_MAX))),
        int(round(_clamp(p, P_MIN, P_MAX))),
        int(round(_clamp(k, K_MIN, K_MAX))),
    )
```

- [ ] **Step 4: Run the full file to verify it passes**

Run: `pytest tests/test_agronomy.py -v`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add ml/agronomy.py tests/test_agronomy.py
git commit -m "feat: add optional variety parameter to nutrient_requirements"
```

---

### Task 3: Rescale K remark bands from ppm to cmol/kg

**Files:**
- Modify: `ml/remarks.py:44-48`
- Modify: `tests/test_remarks.py`

**Interfaces:**
- Produces: `derive_k_remark(k_cmol_kg: float | None) -> str` — same function name and signature (the parameter is still a bare float, just interpreted on a different scale now), consumed by Task 4 (`ml/preprocessing.py`'s ordered-encoding pipeline, unchanged) and Task 8's parser.

**Context:** `_K_BANDS` (`ml/remarks.py:44-48`) currently reads `[(60.0, "DEFICIENT"), (120.0, "SUFFICIENT"), (float("inf"), "SUFFICIENT+")]` — thresholds calibrated to the old `potassium_ppm` scale (the golden record's K=138 ppm produced "SUFFICIENT+"). Real exchangeable K in this dataset is `cmol/kg`, typically in the 0.1-1.0 range (sampled directly from the source file) — the old thresholds would classify every real value as "DEFICIENT" (nothing reaches 60). New thresholds, `0.2`/`0.4`, are a standard general reference point for exchangeable K interpretation and produce a real spread across all three bands against the sampled real data — same "needs DA RSL confirmation" caveat as the file's existing OM bands already carry.

- [ ] **Step 1: Update the failing tests first**

In `tests/test_remarks.py`, replace `TestGoldenRecordSCC417` (lines 8-22) — the golden record no longer applies (Sorghum dropped, and K is a different unit now):

```python
class TestGoldenRecordSCC417:
    """The real DA Region 7 report SCC-26-417-417 is the only real labeled
    example available. Every derived remark must match it exactly."""

    def test_ph_8_17_is_moderately_alkaline(self):
        assert derive_ph_remark(8.17) == "MODERATELY ALKALINE"

    def test_om_3_21_is_moderately_low(self):
        assert derive_om_remark(3.21) == "MODERATELY LOW"

    def test_p_18_96_is_high(self):
        assert derive_p_remark(18.96) == "HIGH"

    def test_k_138_is_sufficient_plus(self):
        assert derive_k_remark(138.00) == "SUFFICIENT+"
```

with:

```python
class TestPhOmPRemarksAgainstKnownValues:
    """pH/OM/P are unaffected by the K unit change — retained as fixed
    reference points independent of any single dataset."""

    def test_ph_8_17_is_moderately_alkaline(self):
        assert derive_ph_remark(8.17) == "MODERATELY ALKALINE"

    def test_om_3_21_is_moderately_low(self):
        assert derive_om_remark(3.21) == "MODERATELY LOW"

    def test_p_18_96_is_high(self):
        assert derive_p_remark(18.96) == "HIGH"
```

Replace the `test_k_boundaries` parametrization inside `TestOtherBands` (lines 60-65):

```python
    @pytest.mark.parametrize("value,expected", [
        (30.0, "DEFICIENT"), (60.0, "SUFFICIENT"),
        (120.0, "SUFFICIENT+"), (250.0, "SUFFICIENT+"),
    ])
    def test_k_boundaries(self, value, expected):
        assert derive_k_remark(value) == expected
```

with:

```python
    @pytest.mark.parametrize("value,expected", [
        (0.1, "DEFICIENT"), (0.2, "SUFFICIENT"),
        (0.4, "SUFFICIENT+"), (1.5, "SUFFICIENT+"),
    ])
    def test_k_boundaries(self, value, expected):
        assert derive_k_remark(value) == expected
```

And the last line of `test_every_derived_remark_is_in_its_order_list` (line 91):

```python
        assert derive_k_remark(138.0) in K_ORDER
```

with:

```python
        assert derive_k_remark(0.35) in K_ORDER
```

- [ ] **Step 2: Run to verify the changed tests fail**

Run: `pytest tests/test_remarks.py -v`
Expected: FAIL on the new `test_k_boundaries` values — `derive_k_remark(0.1)` still returns `"DEFICIENT"` correctly by coincidence (below both old and new low thresholds), but `derive_k_remark(0.2)` currently returns `"DEFICIENT"` (still below the old 60.0 threshold), not the expected `"SUFFICIENT"`.

- [ ] **Step 3: Implement**

In `ml/remarks.py`, replace `_K_BANDS` (lines 44-48):

```python
_K_BANDS = [
    (60.0, "DEFICIENT"),
    (120.0, "SUFFICIENT"),
    (float("inf"), "SUFFICIENT+"),
]
```

with:

```python
# Exchangeable K thresholds in cmol/kg (not ppm — see potassium_cmol_kg).
# 0.2 / 0.4 cmol/kg are standard general reference points for exchangeable K
# interpretation; like the OM bands above, confirm against DA RSL's published
# chart before production use.
_K_BANDS = [
    (0.2, "DEFICIENT"),
    (0.4, "SUFFICIENT"),
    (float("inf"), "SUFFICIENT+"),
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_remarks.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/remarks.py tests/test_remarks.py
git commit -m "fix: rescale K remark thresholds from ppm to cmol/kg"
```

---

### Task 4: Preprocessing — K field rename and variety feature

**Files:**
- Modify: `ml/preprocessing.py`
- Create: `tests/fixtures.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `CONTINUOUS` (with `potassium_cmol_kg` replacing `potassium_ppm`), `NOMINAL = ["crop", "variety", "data_source"]`, `FEATURE_COLUMNS`, `VARIETIES: list[str]` — consumed by Task 5 (`main.py`), Task 6 (`ml/train.py`, unchanged but now sees the new columns), Task 8's parser (must produce a `variety` column).
- Produces (test-only): `tests/fixtures.py`'s `sample_training_frame(num_records: int, seed: int) -> pd.DataFrame` — a hermetic, non-production DataFrame builder in the new schema, consumed by Task 6 and Task 9's test updates.

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

from data.crop_baselines import CROPS
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
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

        n, p2o5, k2o = nutrient_requirements(crop, ph, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o)

        rows.append({
            "laboratory_code": f"TEST-{index}",
            "data_source": "lab",
            "sampling_site": None,
            "date_of_sampling": None,
            "crop": crop,
            "variety": "None",
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
            "k_remark": derive_k_remark(potassium),
            "target_n_kg_ha": n, "target_p2o5_kg_ha": p2o5, "target_k2o_kg_ha": k2o,
            "app1_formulation": app1, "app2_formulation": app2,
        })

    return pd.DataFrame(rows, columns=COLUMNS)
```

- [ ] **Step 2: Update the preprocessing tests**

In `tests/test_preprocessing.py`, replace the import block (lines 1-9):

```python
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from data.generate_synthetic_data import generate
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS,
)
```

with:

```python
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tests.fixtures import sample_training_frame
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS, VARIETIES,
)
```

Replace the `frame` fixture (lines 12-14):

```python
@pytest.fixture(scope="module")
def frame():
    return generate(num_records=100, seed=42)
```

with:

```python
@pytest.fixture(scope="module")
def frame():
    return sample_training_frame(num_records=100, seed=42)
```

Update `_base_row()` (lines 103-111) to the new schema:

```python
def _base_row() -> dict:
    return {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_cmol_kg": 0.35,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": "SLIGHTLY ACIDIC", "om_remark": "MODERATELY LOW",
        "p_remark": "MEDIUM", "k_remark": "SUFFICIENT",
        "crop": "Corn", "variety": "None", "data_source": "lab",
    }
```

Add a new test class after `TestEncoders` (after line 100, before `def _base_row()`):

```python
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

    def test_varieties_list_covers_the_baselines(self):
        from data.crop_baselines import VARIETY_BASELINES
        used_varieties = {variety for _, variety in VARIETY_BASELINES}
        assert used_varieties <= set(VARIETIES)
```

- [ ] **Step 3: Run to verify the changed/new tests fail**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ml.preprocessing` doesn't export `VARIETIES` yet, and `CONTINUOUS` still lists `potassium_ppm` not `potassium_cmol_kg` (import error and/or KeyError from the fixture referencing the new column name).

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

from data.crop_baselines import CROPS
from ml.remarks import K_ORDER, OM_ORDER, P_ORDER, PH_ORDER

CONTINUOUS = [
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
]

ORDERED = ["ph_remark", "om_remark", "p_remark", "k_remark"]

NOMINAL = ["crop", "variety", "data_source"]

FEATURE_COLUMNS = CONTINUOUS + ORDERED + NOMINAL

TARGET_COLUMNS = ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha"]

DATA_SOURCES = ["iot", "lab", "stk"]

# "None" (the string, not Python None) covers every crop that doesn't track
# varieties. Hybrid/Inbred/OPV cover Rice and Corn's real, floor-clearing
# variety splits — see data/crop_baselines.py's VARIETY_BASELINES.
VARIETIES = ["None", "Hybrid", "Inbred", "OPV"]

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

    # handle_unknown="error", deliberately: an unrecognised crop or variety
    # must fail loudly rather than produce an all-zeros vector and a
    # confident-looking prediction built on no crop information. A farmer
    # acting on that could over- or under-fertilize a real field.
    nominal = OneHotEncoder(
        categories=[CROPS, VARIETIES, DATA_SOURCES],
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
Expected: PASS, all tests including the new `TestVarietyEncoding` class.

- [ ] **Step 6: Commit**

```bash
git add ml/preprocessing.py tests/fixtures.py tests/test_preprocessing.py
git commit -m "feat: rename potassium_ppm to potassium_cmol_kg, add variety as a model feature"
```

---

### Task 5: `main.py` — SoilTestRequest variety field and K rename

**Files:**
- Modify: `main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` (Task 4, `ml.preprocessing`).
- Produces: `SoilTestRequest` gains `variety: str | None = None`; `potassium_ppm` field renamed `potassium_cmol_kg` (same `ge=0` constraint). `PredictionResponse` unchanged.

- [ ] **Step 1: Update the API tests first**

In `tests/test_api.py`, replace `GOLDEN_PAYLOAD` (lines 12-17) — Sorghum is dropped and the "golden" framing (an untraceable external record) no longer applies. Replace with a plain, clearly-labeled sample payload using a supported crop:

```python
SAMPLE_PAYLOAD = {
    "crop": "Corn", "data_source": "lab", "ph": 6.4,
    "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
    "potassium_cmol_kg": 0.35, "farm_area_sqm": 5000,
    "laboratory_code": "TEST-001",
}
```

Then update every reference to `GOLDEN_PAYLOAD` in the file to `SAMPLE_PAYLOAD`, and remove the two tests that specifically asserted against the old golden numbers (`test_golden_record_is_agronomically_sane`, lines 37-41, which asserted `target_n_kg_ha` within 15 of Sorghum's 50 — meaningless for Corn) and adjust `test_remarks_are_derived_from_the_numerics` (lines 43-48) to match `SAMPLE_PAYLOAD`'s actual derived remarks instead of the old hardcoded ones. The full updated file:

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
        for field in ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
                      "app1_formulation", "app2_formulation", "derived_remarks",
                      "model_version", "source"]:
            assert field in body
        assert body["source"] == "ml-ensemble"

    def test_targets_are_within_the_clamped_range(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert 20 <= body["target_n_kg_ha"] <= 200
        assert 20 <= body["target_p2o5_kg_ha"] <= 150
        assert 20 <= body["target_k2o_kg_ha"] <= 200

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["ph_remark"] == "SLIGHTLY ACIDIC"
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"
        assert body["derived_remarks"]["p_remark"] == "HIGH"
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT"

    def test_farm_totals_scale_the_per_hectare_rate(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        totals = body["farm_totals"]
        assert totals["area_ha"] == pytest.approx(0.5)
        assert totals["n_kg"] == pytest.approx(body["target_n_kg_ha"] * 0.5, rel=1e-3)

    def test_farm_totals_absent_when_area_not_given(self, client):
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "farm_area_sqm"}
        assert client.post("/predict", json=payload).json()["farm_totals"] is None

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

    def test_total_nitrogen_derived_from_organic_matter(self, client):
        """The caller need not send it — the lab itself calculates it."""
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"

    def test_variety_is_accepted_and_optional(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "variety": "Hybrid"}
        assert client.post("/predict", json=payload).status_code == 200
        without_variety = {**SAMPLE_PAYLOAD, "crop": "Rice"}
        assert client.post("/predict", json=without_variety).status_code == 200


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
Expected: FAIL — `main.py`'s `SoilTestRequest` still has `potassium_ppm`, not `potassium_cmol_kg`, so `SAMPLE_PAYLOAD`'s field is silently ignored by Pydantic (extra fields are dropped, not rejected, by default) and predictions run with `potassium_cmol_kg=None`, producing different remarks/targets than the test expects. `test_variety_is_accepted_and_optional` also fails since `variety` isn't a recognised field yet (silently dropped, but the underlying feature row still needs to build a `variety` column — this will actually crash inside `_build_feature_row`, since `ml.preprocessing.FEATURE_COLUMNS` now requires a `variety` column Task 4 added, which nothing in `main.py` supplies yet — expect a `KeyError` or similar from `main.py`'s `_build_feature_row`, not a clean 422).

- [ ] **Step 3: Implement**

In `main.py`, update `SoilTestRequest` (lines 44-64):

```python
class SoilTestRequest(BaseModel):
    crop: str
    variety: str | None = None
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
```

Update `_build_feature_row` (lines 128-149):

```python
def _build_feature_row(request: SoilTestRequest) -> pd.DataFrame:
    organic_matter = request.organic_matter_pct
    total_nitrogen = request.total_nitrogen_pct
    if total_nitrogen is None and organic_matter is not None:
        total_nitrogen = total_nitrogen_from_om(organic_matter)

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
        "k_remark": derive_k_remark(request.potassium_cmol_kg),
        "crop": request.crop,
        "variety": request.variety or "None",
        "data_source": request.data_source,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: add variety field and rename potassium_ppm to potassium_cmol_kg in the API"
```

---

### Task 6: Retire the golden test infrastructure in `ml/train.py`

**Files:**
- Modify: `ml/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `tests/fixtures.py`'s `sample_training_frame` (Task 4).
- Produces: `train()`'s returned metadata no longer has a `"golden_test"` key. `evaluate()`'s `by_source` dict now only ever contains `"lab"` (since no other source has training rows going forward) — this is a real, permanent change to `metadata.json`'s shape, not a temporary test artifact.

- [ ] **Step 1: Update the tests first**

In `tests/test_train.py`, replace the import block and fixture (lines 1-12):

```python
import json
import numpy as np
import pytest

from data.generate_synthetic_data import generate
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import grouped_split, cv_weights, train


@pytest.fixture(scope="module")
def frame():
    return generate(num_records=200, seed=42)
```

with:

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
```

Remove `test_all_three_views_of_a_code_land_together` (lines 42-45) — it asserted exactly 3 rows per `laboratory_code`, a property specific to the retired synthetic generator's one-record-three-views design. Real (and the new test fixture's) data is one row per `laboratory_code`. Replace it with:

```python
    def test_each_code_appears_exactly_once(self, frame):
        """Unlike the retired synthetic generator (which produced 3 rows per
        code, one per data_source), real training data is one row per
        laboratory_code — only "lab" has real training rows now."""
        train_df, _, _ = grouped_split(frame, seed=42)
        counts = train_df.groupby("laboratory_code").size()
        assert (counts == 1).all()
```

Update every `generate(num_records=200, seed=42)` call inside `TestTrain.result` (line 65) to `sample_training_frame(num_records=200, seed=42)`.

Update `test_metadata_records_provenance` (lines 73-79) — `n_rows` was `600` (200 records × 3 views); the fixture now produces exactly `num_records` rows:

```python
    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 200
        assert "trained_at" in written
        assert "version" in written
```

Replace `test_reports_metrics_per_data_source` (lines 81-86) — only `"lab"` will ever appear now:

```python
    def test_reports_metrics_per_data_source(self, result):
        metadata, _ = result
        assert set(metadata["metrics"]["by_source"].keys()) == {"lab"}
```

Remove `test_golden_record_scc417_is_within_tolerance` (lines 93-100) entirely — the fixture no longer produces a `golden_test` key in metadata.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `train()` still calls `_golden_test()` internally and writes a `"golden_test"` key that no test now asserts on directly, but `_golden_test()` calls `GOLDEN_SCC417`-derived `regressor.predict()` against `FEATURE_COLUMNS` that no longer match (missing `variety`) — expect a hard crash inside `train()` itself (`KeyError: 'variety'` from `_golden_test`'s `pd.DataFrame([GOLDEN_SCC417])[FEATURE_COLUMNS]`), failing every test in `TestTrain`.

- [ ] **Step 3: Implement**

In `ml/train.py`, remove `GOLDEN_SCC417`, `GOLDEN_ACTUAL`, `GOLDEN_TOLERANCE` (lines 32-41) and `_golden_test()` (lines 104-114) entirely. Remove the `"golden_test": _golden_test(preprocessor, regressor),` line from `metadata` (inside `train()`, originally around line 189). Remove the golden-test print block from `_print_summary()` (originally lines 226-231):

```python
    golden = metadata["golden_test"]
    status = "PASS" if golden["passed"] else "FAIL"
    print(f"\nGolden test SCC-26-417-417: {status}")
    print(f"  predicted {golden['predicted_n']:.1f}/{golden['predicted_p2o5']:.1f}"
          f"/{golden['predicted_k2o']:.1f}  actual 50/40/60"
          f"  (tolerance +/-{golden['tolerance_kg_ha']})")
```

Delete that block with no replacement — `_print_summary` ends after the "MAE by data source" loop.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "fix: retire the golden SCC-26-417-417 test fixture, now untraceable and Sorghum-based"
```

---

### Task 7: Parser foundations — cell parsing, lab-code normalization, crop/variety normalization

**Files:**
- Create: `data/parse_real_soil_database.py`
- Create: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Consumes: `CROPS` (Task 1, `data.crop_baselines`).
- Produces: `parse_numeric_cell(value) -> float | None`, `normalize_lab_code(raw_code: str, year_label: str) -> str`, `normalize_crop(raw_text: str | None) -> tuple[str, str | None] | None`, `find_column(header_row: tuple, target_text: str) -> int` — all pure functions, consumed by Task 8's sheet-reading logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parse_real_soil_database.py
import pytest

from data.parse_real_soil_database import (
    parse_numeric_cell, normalize_lab_code, normalize_crop, find_column,
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

from data.crop_baselines import CROPS

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
git commit -m "feat: add parser foundations for the real soil database (cell parsing, lab-code and crop normalization)"
```

---

### Task 8: Parser — per-sheet reading, row building, CSV output, data-quality report

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Modify: `tests/test_parse_real_soil_database.py`
- Modify: `pyproject.toml` or `requirements.txt` — add `openpyxl` (check which file this repo uses for dependencies before editing; `requirements.txt` was confirmed present at the repo root during planning — add `openpyxl>=3.1.0` as a new line there, no other dependency-file changes needed since none other exists).

**Interfaces:**
- Consumes: `parse_numeric_cell`, `normalize_lab_code`, `normalize_crop`, `find_column` (Task 7, same file); `nutrient_requirements` (Task 2, `ml.agronomy`); `total_nitrogen_from_om` (`ml.agronomy`, unchanged); `plan_applications` (`ml.recipes`, unchanged); `derive_ph_remark`/`derive_om_remark`/`derive_p_remark`/`derive_k_remark` (Task 3, `ml.remarks`).
- Produces: `SHEET_CONFIGS: list[dict]`, `parse_sheet(worksheet, config: dict) -> list[dict]`, `build_training_row(raw: dict) -> dict | None`, `parse_workbook(xlsx_path: str) -> tuple[list[dict], dict]` (rows, data-quality report), `OUTPUT_CSV: str` — `OUTPUT_CSV` consumed by Task 9 (`data/ingest_to_supabase.py`).

**Context:** Exact, individually-verified per-sheet layout (do not re-derive these — they were confirmed by direct inspection of the live file, including catching two real column-drift bugs along the way):

| Sheet | Main header row | Chemistry units row | Data starts | Lab code col | Crop col |
|---|---|---|---|---|---|
| Laboratory Details2022 | 3 | 5 | 6 | 1 | 14 |
| Laboratory Details2023 | 3 | 5 | 6 | 1 | 14 |
| Laboratory Details2024 | 3 | 5 | 6 | 1 | 14 |
| Soil Data 2025 | 2 | 2 | 4 | 0 | 9 |

The three "Laboratory Details" sheets share one chemistry-header vocabulary (`"pH (H2O, 1:1)"` etc.) but at column positions that drift between 2022/2024 and 2023 specifically — this is exactly why `find_column` (Task 7) resolves each sheet's columns independently rather than trusting the table above for anything but row numbers. `Soil Data 2025` uses an entirely different, shorter header vocabulary (`"pH"`, `"OM"`, etc.) at fixed early columns.

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
            "ph": 6.4, "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
            "potassium_cmol_kg": 0.35, "total_nitrogen_pct": None,
        }
        row = build_training_row(raw)
        assert row is not None
        assert row["data_source"] == "lab"
        assert row["crop"] == "Corn"
        assert row["variety"] == "None"
        assert row["om_remark"] == "MODERATELY LOW"
        assert isinstance(row["target_n_kg_ha"], int)
        assert 20 <= row["target_n_kg_ha"] <= 200

    def test_total_nitrogen_derived_when_absent(self):
        raw = {
            "laboratory_code": "2022-SCC1", "crop": "Rice", "variety": "Hybrid",
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
            "ph": 6.5, "organic_matter_pct": 2.5, "phosphorus_ppm": 15.0,
            "potassium_cmol_kg": 0.3, "total_nitrogen_pct": None,
        }
        for missing_field in ("organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg"):
            incomplete = {**base, missing_field: None}
            assert build_training_row(incomplete) is None

    def test_variety_none_becomes_the_string_none(self):
        raw = {
            "laboratory_code": "2022-SCC1", "crop": "Cassava", "variety": None,
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
        # Row 3: main header (only the columns this parser reads)
        ws.cell(row=3, column=2, value="Laboratory Code")
        ws.cell(row=3, column=15, value="Crops")
        # Row 5: chemistry units header
        ws.cell(row=5, column=17, value="pH (H2O, 1:1)")
        ws.cell(row=5, column=18, value="Available Phosphorus (Avail. P), mg/kg")
        ws.cell(row=5, column=19, value="Exchangeable Potassium (K), cmol/kg")
        ws.cell(row=5, column=20, value="Organic Matter (OM), %")
        ws.cell(row=5, column=21, value="Total Nitrogen (N), %")
        # Row 6+: data (1-based columns: B=lab code, O=crop, Q..U=chem)
        ws.cell(row=6, column=2, value="SCC 1")
        ws.cell(row=6, column=15, value="Corn")
        ws.cell(row=6, column=17, value=6.4)
        ws.cell(row=6, column=18, value=18.0)
        ws.cell(row=6, column=19, value=0.35)
        ws.cell(row=6, column=20, value=2.8)
        ws.cell(row=6, column=21, value=0.14)
        # A row that was field-tested instead of lab-tested for phosphorus
        ws.cell(row=7, column=2, value="SCC 2")
        ws.cell(row=7, column=15, value="Rice")
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
        config = {**SHEET_CONFIGS[0]}  # Laboratory Details2022's config
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_parse_real_soil_database.py -v`
Expected: FAIL — `build_training_row`, `parse_sheet`, `SHEET_CONFIGS`, `LAB_DETAIL_HEADERS`, `SOIL_DATA_2025_HEADERS` don't exist yet.

- [ ] **Step 3: Implement**

Append to `data/parse_real_soil_database.py`:

```python
import csv
import sys

import openpyxl

from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

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
# chem_header_row, since they drift between sheets (see Task 8's docstring
# table in the plan this was built from).
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
    "crop", "variety", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
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

        crop_variety = normalize_crop(row[config["crop_col"]])
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

    total_nitrogen = raw["total_nitrogen_pct"]
    if total_nitrogen is None:
        total_nitrogen = total_nitrogen_from_om(om)

    n, p2o5, k2o = nutrient_requirements(crop, ph, om, p, k, variety=variety)
    app1, app2 = plan_applications(n, p2o5, k2o)

    return {
        "laboratory_code": raw["laboratory_code"],
        "data_source": "lab",
        "sampling_site": None,
        "date_of_sampling": None,
        "crop": crop,
        "variety": variety or "None",
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
        "k_remark": derive_k_remark(k),
        "target_n_kg_ha": n, "target_p2o5_kg_ha": p2o5, "target_k2o_kg_ha": k2o,
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
    for row in training_rows:
        report["rows_by_crop"][row["crop"]] = report["rows_by_crop"].get(row["crop"], 0) + 1

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

### Task 9: Point ingestion at the new parser's output

**Files:**
- Modify: `data/ingest_to_supabase.py:16`
- Modify: `tests/test_ingest_to_supabase.py`

**Interfaces:**
- Consumes: `OUTPUT_CSV` (Task 8, `data.parse_real_soil_database`); `sample_training_frame` (Task 4, `tests.fixtures`).

- [ ] **Step 1: Update the failing test first**

In `tests/test_ingest_to_supabase.py`, replace the import (line 4) and the `records` fixture (lines 19-22):

```python
from data.generate_synthetic_data import generate
```

```python
from tests.fixtures import sample_training_frame
```

```python
    @pytest.fixture(scope="class")
    def records(self):
        return frame_to_records(generate(num_records=10, seed=42))
```

```python
    @pytest.fixture(scope="class")
    def records(self):
        return frame_to_records(sample_training_frame(num_records=10, seed=42))
```

Update `test_produces_one_record_per_row` (line 24-25) — the fixture is now one row per record, not three:

```python
    def test_produces_one_record_per_row(self, records):
        assert len(records) == 10
```

Update `test_nan_becomes_none_for_json` (lines 27-31) — the old test specifically checked IoT rows' missing `organic_matter_pct`; `sample_training_frame` only produces `"lab"` rows, all of which have real OM values, so this exact assertion no longer applies. Replace with a version that checks the mechanism generically instead of relying on a specific always-missing field:

```python
    def test_nan_becomes_none_for_json(self, records):
        """Supabase rejects NaN — it is not valid JSON. Any NaN produced by
        pandas (e.g. from an unset optional field) must serialise as null."""
        import math
        for record in records:
            for value in record.values():
                assert not (isinstance(value, float) and math.isnan(value))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ingest_to_supabase.py -v`
Expected: FAIL — `tests.fixtures` doesn't have the right shape mismatch initially only if run before Task 4, but since Task 4 is already done by this point in the plan, the actual failure here is `data/ingest_to_supabase.py:16`'s `from data.generate_synthetic_data import OUTPUT_CSV` still pointing at the old generator (import succeeds since that file still exists until Task 10, so this specific line won't fail yet — the test failures here are purely from the fixture/assertion changes above being correct against the new schema). Confirm via the run that only the intended assertions changed behavior, not an import error.

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

### Task 10: Delete the synthetic generator

**Files:**
- Delete: `data/generate_synthetic_data.py`
- Delete: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: nothing — this task only runs after Tasks 4, 6, 9 have removed every remaining import of `data.generate_synthetic_data` (`tests/test_preprocessing.py`, `tests/test_train.py`, `tests/test_ingest_to_supabase.py`, `data/ingest_to_supabase.py`).

- [ ] **Step 1: Confirm nothing still imports it**

Run: `grep -rn "generate_synthetic_data" --include="*.py" .`
Expected: no matches outside `data/generate_synthetic_data.py` and `tests/test_generate_synthetic_data.py` themselves. If anything else matches, stop and fix that reference first — do not delete out from under a live import.

- [ ] **Step 2: Delete both files**

```bash
git rm data/generate_synthetic_data.py tests/test_generate_synthetic_data.py
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: PASS, every test in the repo (this is the first point in the plan where the *entire* suite — not just one file — is run, so it also catches any straggler cross-file dependency the per-task runs above didn't surface).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove the synthetic training data generator, fully replaced by the real parser"
```

---

### Task 11: End-to-end run — parse, ingest, retrain, verify

This task has no code changes — it's the real-data equivalent of a smoke test, and the point where the data-quality report (Task 8) gets its first real read.

- [ ] **Step 1: Confirm the source file is reachable**

Run: `ls "../suri/public/Region 7-National Soil Database_2022-2025.xlsx"`
Expected: file exists. If not, pass the correct path as an argument in the next step instead of relying on the default.

- [ ] **Step 2: Run the parser against the real file**

Run: `python -m data.parse_real_soil_database`
Expected: prints total row count, per-sheet breakdown, and per-crop breakdown; writes `suri_real_training_data.csv`. Sanity-check the printed per-crop counts are in the same ballpark as this plan's Global Constraints table (Rice and Corn in the thousands; Cassava/Peanut/Mango/Cacao/Banana/Coconut/Tomato each at least in the dozens) — a wildly different distribution means something upstream (crop normalization, column mapping) regressed and needs investigating before proceeding.

- [ ] **Step 3: Ingest into Supabase**

Run: `python -m data.ingest_to_supabase`
Expected: prints batch-by-batch upsert progress and a final total matching Step 2's row count. Requires `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` in `.env` (see `data/ingest_to_supabase.py`'s `get_client()` for the exact error message if unset).

- [ ] **Step 4: Retrain**

Run: `python -m ml.train --csv suri_real_training_data.csv`
Expected: a summary table prints (MAE/R² per target, MAE by source — should show only `lab`). Compare against the previous synthetic-trained model's numbers (if still available from a prior artifact directory under `ml/artifacts/`) — expect real-data R² to be *lower* than the old synthetic numbers (synthetic was uniform-random and trivially easy to fit); that's expected, not a regression to chase, per the design spec's Testing section.

- [ ] **Step 5: Smoke-test the running service**

Run: `uvicorn main:app --port 8001` (separate terminal), then:

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Rice","variety":"Hybrid","data_source":"lab","ph":6.5,"organic_matter_pct":2.8,"phosphorus_ppm":18.0,"potassium_cmol_kg":0.35,"laboratory_code":"TEST-001"}' \
  | python -m json.tool
```

Expected: a `PredictionResponse` JSON body with `target_n_kg_ha`/`target_p2o5_kg_ha`/`target_k2o_kg_ha`/`app1_formulation`/`app2_formulation`/`derived_remarks`/`model_version` all present, `model_version` matching Step 4's newly-trained version string (visible under `ml/artifacts/`).

No commit for this task — it produces a trained model artifact and a populated Supabase table, not a code change.

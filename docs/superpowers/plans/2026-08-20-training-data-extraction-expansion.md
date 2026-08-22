# Training Data Extraction Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `data/parse_real_soil_database.py`'s crop-text recognition so the real lab rows already sitting in the Region 7 National Soil Database for the 35-crop vocabulary (not just the original 9) become extractable into training data — closing the gap the crop-coverage-expansion plan explicitly left open.

**Architecture:** `normalize_crop()` already has the right shape — a flat `CROP_VARIETY_MAP: dict[str, tuple[str, str | None]]` matched by exact/prefix text, feeding into a parser that already handles multi-sheet layout drift, missing-value sentinels, and target derivation correctly. Nothing structural needs to change. This plan is close to pure data: add real-world label variants (Filipino names, alternate spellings, word-order variants, common typos — all confirmed against the actual source spreadsheet, not guessed) as new `CROP_VARIETY_MAP` keys, each resolving to a crop already in `CANONICAL_CROPS`. Multi-crop (intercropped) rows and genuinely ambiguous/non-crop labels are explicitly out of scope — see "Deliberately Excluded" below.

**Tech Stack:** Python 3.12, openpyxl, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md` §2.1 (crop coverage), and the "Real Data Availability" section of `docs/superpowers/plans/2026-08-18-crop-coverage-expansion.md` (the row counts this plan acts on).

## Global Constraints

- **Every `CROP_VARIETY_MAP` value's crop name must be a canonical crop** (a member of `CANONICAL_CROPS`), never an alias name — resolve to canonical at the map level. This matches `data/training_upload.py::build_training_row`'s existing convention (`crop = resolve_crop(requested)`) and keeps `by_crop` metrics grouping (which `ml/train.py` and Task 7's `trained_crops()` guard both key by canonical name) consistent regardless of which code path produced a training row.
- **Variety defaults to `None`, never a hardcoded guess.** When the raw label doesn't explicitly state a variety (e.g. bare `"ginger"`, not `"improved ginger"`), the map entry's variety is `None` — exactly like the pre-existing `"corn": ("Corn", None)` entry — and `resolve_variety()`/`build_training_row()` apply the crop's real default downstream. Only map an explicit variety when the label states one (e.g. `"spring onion"` → `("Onion", "Spring")`).
- **Every new mapping is backed by a real, counted row in the actual source spreadsheet** — this plan's tables were built by directly reading `Region 7-National Soil Database_2022-2025.xlsx`'s crop-text column with openpyxl (exact text extraction, not OCR — this workbook, unlike the nutrient guide PDF, has a real text layer, so these counts are ground truth, not inferred). No label is added on the strength of "this crop probably appears somewhere."
- **Multi-crop (intercropped) rows stay out of scope.** ~1,979 rows have comma/slash-separated crop text (e.g. `"Cassava, Corn"`). One soil reading cannot be cleanly attributed to one crop's nutrient needs without a real design decision on how to split it. `normalize_crop()` already returns `None` for these and this plan does not change that.
- **A label with no defensible home in the current 35-crop registry is excluded, not force-mapped.** Several real labels (bell pepper family, ornamentals, forage grasses, herbs) don't match any crop this system currently supports — see "Deliberately Excluded."
- Commit after every task. Run `python -m pytest` from `suri-ml-backend/` before each commit.
- **This plan stops at CSV regeneration.** It does not ingest into Supabase, does not retrain, does not promote. Those remain deliberate, separately-triggered actions — same posture the two prior plans this session held throughout.

## Real Data This Plan Unlocks

Counted directly against the current `CROP_VARIETY_MAP` (before this plan) and the current 35-crop `CANONICAL_CROPS`/`CROP_ALIASES` (after the crop-coverage-expansion plan): of 4,375 single-crop-text rows whose text isn't already recognized or excluded as intercropped, **3,754 already resolve** (the original 9 crops' existing label variants), leaving **621 unmatched**. This plan's 60 new `CROP_VARIETY_MAP` entries (Tasks 2-5) recover **391 of those 621 rows** exactly — summed directly from each task's table, not estimated. Of the remaining 230: **146 rows** (27 distinct labels, each with count ≥ 2) are explicitly catalogued and excluded for a stated reason in "Deliberately Excluded" below; the remaining **84 rows** are a long tail of singleton (count = 1) labels never individually catalogued — too numerous and too low-value per label to vet one at a time here, left for a future pass if ever worth it.

Note that these are rows recognized at the crop-text stage only — some will still drop at the parser's existing `dropped_for_incomplete_chemistry` filter (missing OM/P/K), the same way a fraction of the original 9 crops' rows already do today. Task 6 reports the actual final training-row count after that filter runs.

## Deliberately Excluded

Labels with real row counts that this plan does **not** map, and why:

| Label(s) | Count | Reason |
|---|---|---|
| `vegetables`, `vegetable` | 28+2 | Too generic — names a category, not a crop. |
| `ornamentals`, `ornamental` | 10+4 | Not a food/fertilizer-recommendation crop this system supports. |
| `pinakbet` | 8 | A dish name (a vegetable medley), not a crop — a data-entry artifact, not a species. |
| `coffee` | 16 | Already deliberately deferred in the crop-coverage-expansion plan (guide-table anomaly, not data scarcity) — mapping it to training data ahead of the guide entry existing would let it silently reach the untrained-crop guard's table-lookup path with no real target formula backing it, since `nutrient_requirements("Coffee", ...)` still raises `KeyError`. |
| `dragonfruit`, `dragon fruit` | 11+4 | Same reason as Coffee — deliberately deferred, guide has blank cells for it. |
| `bell pepper`, `sweet pepper`, `bellpepper` | 8+4+2 | Botanically distinct from the guide's "Chili Pepper" (sili) grouping — no defensible guide-table home. Needs its own future transcription, not a forced fit into Garlic's rates. |
| `agarwood` | 5 | Not a crop this system's guide covers. |
| `bermuda grass` | 3 | Forage grass, not a supported crop. |
| `lemon grass` | 3 | Herb, no guide entry. |
| `narra` | 2 | Timber tree, not a fertilizer-recommendation crop. |
| `rosemary` | 2 | Herb, no guide entry. |
| `batong` | 6 | Unclear referent — not confidently identifiable as any supported crop. |
| `melon` (bare) | 2 | Ambiguous between muskmelon/cantaloupe/watermelon-family readings; `watermelon` itself is mapped (see Task 3) but bare "melon" isn't specific enough to guess. |
| `sikwa` | 2 | Possibly a Visayan term for a gourd/luffa type, not confidently identifiable. |
| `lemonsito` | 6 | Likely a small citrus, not confidently identifiable as a specific guide-covered species. |
| `masterlist`, `research`, `researh`, `high value crops`, `texture only`, `no crops`, `fishpond` | 4+3+3+2+2+2+2 | Housekeeping/non-crop spreadsheet artifacts, not species names. |

If any of these later gets a confident identification or a guide-table entry, adding it is a small follow-up to `CROP_VARIETY_MAP` — the same shape as every task below.

---

### Task 1: Tighten normalize_crop's canonical-only guard and add a self-consistency test

**Why first:** every later task adds entries to `CROP_VARIETY_MAP`. This task adds the guard that would catch a future entry accidentally targeting an alias or a typo'd crop name, before any new entries exist to test it against.

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Test: `tests/test_parse_real_soil_database.py` (create if it doesn't exist — check first)

**Interfaces:**
- Consumes: `data.nutrient_recommendation_guide.CANONICAL_CROPS`
- Produces: no new public interface — tightens `normalize_crop`'s internal validation

- [ ] **Step 1: Check whether a test file already exists**

Run: `ls tests/test_parse_real_soil_database.py 2>&1` (or the Glob/Read tool). This module has existed since before this session but may not have a dedicated test file yet — if `tests/test_parse_real_soil_database.py` doesn't exist, create it with this header:

```python
"""Tests for data/parse_real_soil_database.py's crop-text recognition.

CROP_VARIETY_MAP is the single source of truth for how free-text crop
labels in the Region 7 National Soil Database map to this system's
canonical crops. Every test here either pins an existing mapping's
behavior or proves a newly-added one resolves correctly."""

import pytest

from data.nutrient_recommendation_guide import CANONICAL_CROPS
from data.parse_real_soil_database import CROP_VARIETY_MAP, normalize_crop
```

If the file already exists, read it first and add to it rather than overwriting — match whatever import/structure convention is already there.

- [ ] **Step 2: Write the failing test**

Append:

```python
class TestCropVarietyMapConsistency:
    def test_every_mapped_crop_is_canonical(self):
        """A CROP_VARIETY_MAP entry targeting an alias (or a typo) would
        silently desynchronize by_crop training metrics from the canonical
        name Task 7's untrained-crop guard and ml/train.py's evaluation
        both key on."""
        for text, (crop, variety) in CROP_VARIETY_MAP.items():
            assert crop in CANONICAL_CROPS, f"{text!r} maps to non-canonical {crop!r}"

    def test_the_original_nine_crops_still_resolve(self):
        """Behavior-preservation pin for the pre-existing map — this task
        changes normalize_crop's validation, not its entries."""
        assert normalize_crop("Corn") == ("Corn", None)
        assert normalize_crop("OPV Corn") == ("Corn", "OPV")
        assert normalize_crop("Hybrid Rice") == ("Rice", "Hybrid")
        assert normalize_crop("Cassava") == ("Cassava", None)
```

- [ ] **Step 3: Run the test to verify it currently passes (this is a pin, not a new-feature test)**

Run: `python -m pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS — `CROP_VARIETY_MAP`'s 9 pre-existing entries are already canonical, so `test_every_mapped_crop_is_canonical` should already hold. If it fails, stop: that means one of the pre-existing 9 entries doesn't target a real canonical crop, which is a pre-existing bug this task's guard just surfaced — report it rather than silently adjusting the test to pass.

- [ ] **Step 4: Tighten the runtime guard to match**

In `data/parse_real_soil_database.py`, change the import and the check inside `normalize_crop`:

```python
from data.nutrient_recommendation_guide import (
    CANONICAL_CROPS, BEARING_AGE_BAND, k_cmol_to_ppm, target_unit_for_crop,
    resolve_variety, resolve_cultivation_method, resolve_age_band,
)
```

(replacing the existing `CROPS` import — check whether `CROPS` is used anywhere else in this file before removing the import entirely; if it is, keep both imports.)

In `normalize_crop`, change:

```python
    if mapped is None or mapped[0] not in CROPS:
        return None
```

to:

```python
    if mapped is None or mapped[0] not in CANONICAL_CROPS:
        return None
```

- [ ] **Step 5: Update the module's stale comment**

`normalize_crop`'s docstring currently says: `"None for multi-crop text, empty text, or any crop not in the 9-crop list (including the three dropped for insufficient real coverage)."` This is stale after the crop-coverage-expansion plan. Update it to something like: `"None for multi-crop text, empty text, or any crop CROP_VARIETY_MAP doesn't have an entry for."` (Don't restate the full 35-crop list here — that's what `CANONICAL_CROPS` is for.)

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "refactor: validate CROP_VARIETY_MAP entries against CANONICAL_CROPS directly"
```

---

### Task 2: Corn, Rice, and Sorghum label variants

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Test: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Produces: 10 new `CROP_VARIETY_MAP` entries

| Real label | Count | Maps to |
|---|---|---|
| `sweetcorn` | 14 | `("Corn", None)` |
| `sweet corn` | 8 | `("Corn", None)` |
| `corn hybrid` | 9 | `("Corn", "Hybrid")` |
| `corn hybrid yellow` | 2 | `("Corn", "Hybrid")` |
| `corn tinigib` | 10 | `("Corn", None)` |
| `corn tiniguib` | 3 | `("Corn", None)` (same word, alternate spelling) |
| `sorghum` | 8 | `("Corn", None)` (Sorghum is already a registered `CROP_ALIASES` entry → Corn; this maps the raw label directly to canonical, per this plan's convention) |
| `adlai` | 4 | `("Corn", "Hybrid")` — see note below |
| `upland rice` | 10 | `("Rice", None)` |
| `rice inbred` | 2 | `("Rice", "Inbred")` (reversed word order from the already-mapped `"inbred rice"`) |

**Note on Adlai:** the crop-coverage-expansion plan deliberately did *not* register Adlai as a `CROP_ALIASES` entry, because the guide only lists it on Corn's *Hybrid* row (`Corn /Sorghum/Adlai`), and a whole-crop alias would incorrectly resolve to Corn's *Native* default for any caller not specifying a variety. That problem is specific to the alias system (which has no per-alias variety override). `CROP_VARIETY_MAP` doesn't have that limitation — each entry already carries its own explicit variety — so mapping `"adlai"` directly to `("Corn", "Hybrid")` here is correct and sidesteps the exact issue the other plan flagged, without needing to touch `CROP_ALIASES` at all.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parse_real_soil_database.py`:

```python
class TestCornRiceVariants:
    @pytest.mark.parametrize("text,expected", [
        ("sweetcorn", ("Corn", None)),
        ("Sweet Corn", ("Corn", None)),
        ("Corn Hybrid", ("Corn", "Hybrid")),
        ("CORN HYBRID YELLOW", ("Corn", "Hybrid")),
        ("Corn Tinigib", ("Corn", None)),
        ("corn tiniguib", ("Corn", None)),
        ("Sorghum", ("Corn", None)),
        ("Adlai", ("Corn", "Hybrid")),
        ("Upland Rice", ("Rice", None)),
        ("Rice Inbred", ("Rice", "Inbred")),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

    def test_upland_rice_cultivation_method_is_still_detected_downstream(self):
        """normalize_crop only resolves the crop; normalize_cultivation_method
        runs separately against the same raw text and must still find
        'upland' regardless of which CROP_VARIETY_MAP key matched."""
        from data.parse_real_soil_database import normalize_cultivation_method

        assert normalize_cultivation_method("Rice", "Upland Rice") == "Upland"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_parse_real_soil_database.py::TestCornRiceVariants -v`
Expected: FAIL — `normalize_crop` returns `None` for all of these today.

- [ ] **Step 3: Add the entries**

In `data/parse_real_soil_database.py`, extend `CROP_VARIETY_MAP` (add these after the existing 9 entries, before the closing brace):

```python
    # --- Corn / Rice / Sorghum real-world label variants -----------------
    "sweetcorn": ("Corn", None),
    "sweet corn": ("Corn", None),
    "corn hybrid": ("Corn", "Hybrid"),
    "corn hybrid yellow": ("Corn", "Hybrid"),
    "corn tinigib": ("Corn", None),
    "corn tiniguib": ("Corn", None),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "feat: recognize Corn/Rice/Sorghum/Adlai real-world label variants"
```

---

### Task 3: Vegetable label variants

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Test: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Produces: 18 new `CROP_VARIETY_MAP` entries

| Real label | Count | Maps to |
|---|---|---|
| `lettuce` | 22 | `("Lettuce", None)` |
| `pechay` | 21 | `("Cabbage", None)` (Pechay is a registered alias → Cabbage; mapped directly per this plan's convention) |
| `ampalaya` | 18 | `("Ampalaya", None)` |
| `eggplant` | 18 | `("Eggplant", None)` |
| `talong` | 14 | `("Eggplant", None)` (Filipino for eggplant) |
| `okra` | 9 | `("Eggplant", None)` (Okra is a registered alias → Eggplant) |
| `sayote` | 9 | `("Chayote", None)` (Filipino spelling of Chayote) |
| `cabbage` | 8 | `("Cabbage", None)` |
| `squash` | 8 | `("Squash", None)` |
| `carrots` | 4 | `("Radish", None)` (Carrots is a registered alias → Radish) |
| `cauliflower` | 4 | `("Cauliflower", None)` |
| `radish` | 3 | `("Radish", None)` |
| `raddish` | 2 | `("Radish", None)` (typo of `radish`) |
| `cucumber` | 2 | `("Cucumber", None)` |
| `pipino` | 2 | `("Cucumber", None)` (Filipino for cucumber) |
| `watermelon` | 2 | `("Cucumber", None)` (Watermelon is a registered alias → Cucumber) |
| `red cabbage`, `repolyo` | 2+2 | `("Cabbage", None)` (`repolyo` is Filipino for cabbage; no red-cabbage distinction in the guide) |

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parse_real_soil_database.py`:

```python
class TestVegetableVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Lettuce", ("Lettuce", None)),
        ("Pechay", ("Cabbage", None)),
        ("Ampalaya", ("Ampalaya", None)),
        ("Eggplant", ("Eggplant", None)),
        ("Talong", ("Eggplant", None)),
        ("Okra", ("Eggplant", None)),
        ("Sayote", ("Chayote", None)),
        ("Cabbage", ("Cabbage", None)),
        ("Squash", ("Squash", None)),
        ("Carrots", ("Radish", None)),
        ("Cauliflower", ("Cauliflower", None)),
        ("Radish", ("Radish", None)),
        ("Raddish", ("Radish", None)),
        ("Cucumber", ("Cucumber", None)),
        ("Pipino", ("Cucumber", None)),
        ("Watermelon", ("Cucumber", None)),
        ("Red Cabbage", ("Cabbage", None)),
        ("Repolyo", ("Cabbage", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_parse_real_soil_database.py::TestVegetableVariants -v`
Expected: FAIL

- [ ] **Step 3: Add the entries**

Extend `CROP_VARIETY_MAP`:

```python
    # --- Vegetable real-world label variants ------------------------------
    "lettuce": ("Lettuce", None),
    "pechay": ("Cabbage", None),
    "ampalaya": ("Ampalaya", None),
    "eggplant": ("Eggplant", None),
    "talong": ("Eggplant", None),
    "okra": ("Eggplant", None),
    "sayote": ("Chayote", None),
    "cabbage": ("Cabbage", None),
    "red cabbage": ("Cabbage", None),
    "repolyo": ("Cabbage", None),
    "squash": ("Squash", None),
    "cauliflower": ("Cauliflower", None),
    "carrots": ("Radish", None),
    "radish": ("Radish", None),
    "raddish": ("Radish", None),
    "cucumber": ("Cucumber", None),
    "pipino": ("Cucumber", None),
    "watermelon": ("Cucumber", None),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "feat: recognize vegetable real-world label variants"
```

---

### Task 4: Condiment, rootcrop, legume, and sugarcane label variants

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Test: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Produces: 27 new `CROP_VARIETY_MAP` entries (19 condiment/rootcrop/legume/sugarcane + 8 sili/chili-family, listed in the two tables below)

| Real label | Count | Maps to |
|---|---|---|
| `sugarcane` | 42 | `("Sugarcane", None)` |
| `onions` | 7 | `("Onion", None)` |
| `onion` | 2 | `("Onion", None)` |
| `sibuyas` | 2 | `("Onion", None)` (Filipino for onion) |
| `spring onion` | 5 | `("Onion", "Spring")` (explicit variety stated) |
| `spring onions` | 2 | `("Onion", "Spring")` |
| `green onion` | 3 | `("Onion", "Spring")` (green/spring onion is the same guide row) |
| `ginger` | 6 | `("Ginger", None)` |
| `gabi` | 3 | `("Gabi", None)` |
| `ube` | 10 | `("Gabi", None)` (Ube is a registered alias → Gabi) |
| `ubi` | 8 | `("Gabi", None)` (alternate spelling of Ube) |
| `ube kinampay` | 2 | `("Gabi", None)` (Kinampay is an Ube variety name; no variety axis for Gabi) |
| `kamote` | 3 | `("Cassava", None)` (Camote is a registered alias → Cassava; alternate spelling) |
| `sweet potato` | 8 | `("Cassava", None)` (English name for Camote) |
| `beans` | 6 | `("Beans", None)` |
| `string beans` | 5 | `("Beans", None)` |
| `baguio beans` | 8 | `("Beans", None)` |
| `mung beans` | 2 | `("Peanut", None)` (Mongo is a registered alias → Peanut) |
| `mungbean` | 2 | `("Peanut", None)` |

**Note on the sili/chili family:** the guide's CONDIMENTS/SPICES section co-lists "Garlic/ Chili Pepper" on one row (already a registered `CROP_ALIASES` entry: `"Chili Pepper": "Garlic"`, from the crop-coverage-expansion plan). Real labels in the source data use "Sili" (Filipino for chili) far more than the English "Chili Pepper" — `sili haba` (7), `sili labuyo` (4), `sili espada` (3), `sili` (3), `sili atsal` (2), `finger pepper` (3), `hot pepper` (2), `green chili` (2) all name chili varieties, not a botanically distinct pepper. Map all of these to `("Garlic", None)`, honoring the same guide-row grouping the existing alias already encodes. **Do not** map `bell pepper`/`sweet pepper`/`bellpepper` here — those are a botanically distinct sweet pepper with no guide-table home; see "Deliberately Excluded."

| Real label | Count | Maps to |
|---|---|---|
| `sili haba` | 7 | `("Garlic", None)` |
| `sili labuyo` | 4 | `("Garlic", None)` |
| `sili espada` | 3 | `("Garlic", None)` |
| `sili` | 3 | `("Garlic", None)` |
| `sili atsal` | 2 | `("Garlic", None)` |
| `finger pepper` | 3 | `("Garlic", None)` |
| `hot pepper` | 2 | `("Garlic", None)` |
| `green chili` | 2 | `("Garlic", None)` |

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parse_real_soil_database.py`:

```python
class TestCondimentRootcropLegumeVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Sugarcane", ("Sugarcane", None)),
        ("Onions", ("Onion", None)),
        ("Onion", ("Onion", None)),
        ("Sibuyas", ("Onion", None)),
        ("Spring Onion", ("Onion", "Spring")),
        ("Spring Onions", ("Onion", "Spring")),
        ("Green Onion", ("Onion", "Spring")),
        ("Ginger", ("Ginger", None)),
        ("Gabi", ("Gabi", None)),
        ("Ube", ("Gabi", None)),
        ("Ubi", ("Gabi", None)),
        ("Ube Kinampay", ("Gabi", None)),
        ("Kamote", ("Cassava", None)),
        ("Sweet Potato", ("Cassava", None)),
        ("Beans", ("Beans", None)),
        ("String Beans", ("Beans", None)),
        ("Baguio Beans", ("Beans", None)),
        ("Mung Beans", ("Peanut", None)),
        ("Mungbean", ("Peanut", None)),
        ("Sili Haba", ("Garlic", None)),
        ("Sili Labuyo", ("Garlic", None)),
        ("Sili Espada", ("Garlic", None)),
        ("Sili", ("Garlic", None)),
        ("Sili Atsal", ("Garlic", None)),
        ("Finger Pepper", ("Garlic", None)),
        ("Hot Pepper", ("Garlic", None)),
        ("Green Chili", ("Garlic", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected

    def test_bell_pepper_is_not_mapped(self):
        """Botanically distinct from the guide's Chili Pepper/Garlic
        grouping -- deliberately excluded, not force-fit. See the plan's
        'Deliberately Excluded' table."""
        assert normalize_crop("Bell Pepper") is None
        assert normalize_crop("Sweet Pepper") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_parse_real_soil_database.py::TestCondimentRootcropLegumeVariants -v`
Expected: most FAIL (the two `test_bell_pepper_is_not_mapped` assertions already pass today, since nothing maps them yet — that's fine, it's a pin).

- [ ] **Step 3: Add the entries**

Extend `CROP_VARIETY_MAP`:

```python
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
    "kamote": ("Cassava", None),
    "sweet potato": ("Cassava", None),
    "beans": ("Beans", None),
    "string beans": ("Beans", None),
    "baguio beans": ("Beans", None),
    "mung beans": ("Peanut", None),
    "mungbean": ("Peanut", None),
    # Guide co-lists "Garlic/ Chili Pepper" on one row (CROP_ALIASES already
    # has "Chili Pepper" -> Garlic); real labels overwhelmingly use "Sili"
    # (Filipino for chili) rather than the English name.
    "sili haba": ("Garlic", None),
    "sili labuyo": ("Garlic", None),
    "sili espada": ("Garlic", None),
    "sili": ("Garlic", None),
    "sili atsal": ("Garlic", None),
    "finger pepper": ("Garlic", None),
    "hot pepper": ("Garlic", None),
    "green chili": ("Garlic", None),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "feat: recognize condiment/rootcrop/legume/sugarcane real-world label variants"
```

---

### Task 5: Tree and plantation crop label variants

**Files:**
- Modify: `data/parse_real_soil_database.py`
- Test: `tests/test_parse_real_soil_database.py`

**Interfaces:**
- Produces: 5 new `CROP_VARIETY_MAP` entries

| Real label | Count | Maps to |
|---|---|---|
| `calamansi` | 9 | `("Citrus", None)` (Calamansi is a registered alias → Citrus) |
| `lemon` | 2 | `("Citrus", None)` (Lemon is a registered alias → Citrus) |
| `avocado` | 4 | `("Avocado", None)` |
| `jackfruit` | 2 | `("Avocado", None)` (Jackfruit is a registered alias → Avocado) |
| `banana lakatan` | 2 | `("Banana", None)` (Lakatan is a Banana variety name; the guide has no Banana variety axis, so the variety detail is discarded — same as how the pre-existing map already handles crops with no variety split) |

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parse_real_soil_database.py`:

```python
class TestTreeAndPlantationVariants:
    @pytest.mark.parametrize("text,expected", [
        ("Calamansi", ("Citrus", None)),
        ("Lemon", ("Citrus", None)),
        ("Avocado", ("Avocado", None)),
        ("Jackfruit", ("Avocado", None)),
        ("Banana Lakatan", ("Banana", None)),
    ])
    def test_variant_resolves(self, text, expected):
        assert normalize_crop(text) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_parse_real_soil_database.py::TestTreeAndPlantationVariants -v`
Expected: FAIL

- [ ] **Step 3: Add the entries**

Extend `CROP_VARIETY_MAP`:

```python
    # --- Tree / plantation crop real-world label variants -----------------
    "calamansi": ("Citrus", None),
    "lemon": ("Citrus", None),
    "avocado": ("Avocado", None),
    "jackfruit": ("Avocado", None),
    "banana lakatan": ("Banana", None),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_parse_real_soil_database.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/parse_real_soil_database.py tests/test_parse_real_soil_database.py
git commit -m "feat: recognize tree and plantation crop real-world label variants"
```

---

### Task 6: Regenerate the training CSV and report the new totals

**Why this is last:** every prior task only changed recognition logic; nothing has touched the actual training data yet. This task re-runs the existing parser (unmodified) against the source workbook and confirms the new rows actually materialize — proving the whole chain end to end, not just unit-level label matching.

**Files:**
- No source changes — this task runs existing code and inspects its output.

- [ ] **Step 1: Confirm the source workbook is present**

This repo has a local copy at `Region 7-National Soil Database_2022-2025.xlsx` (root of `suri-ml-backend`, checksum-identical to `suri/public/Region 7-National Soil Database_2022-2025.xlsx` — confirmed when it was added). Confirm it's still there:

```bash
ls "Region 7-National Soil Database_2022-2025.xlsx"
```

- [ ] **Step 2: Run the parser and capture the report**

```bash
python -m data.parse_real_soil_database "Region 7-National Soil Database_2022-2025.xlsx"
```

This overwrites `suri_real_training_data.csv` in the repo root (gitignored — the same file the earlier training-data-pipeline plan already established as the canonical local training corpus).

Expected output: a row count noticeably higher than the pre-plan baseline of 3,076 (every task above added real, counted rows; expect roughly 3,076 + 600-700 depending on how many of the newly-recognized rows also have complete OM/P/K chemistry — some recognized-crop rows will still drop at the existing `dropped_for_incomplete_chemistry` stage, same as today's 9-crop rows do). Record the actual printed totals in your report — do not assume a specific number, verify it.

- [ ] **Step 3: Confirm coverage across the newly-recognized crops**

The script's own printed `Per-crop:` line reports `rows_by_crop`. Confirm it now includes non-zero counts for crops that had zero training rows before this plan (e.g. `Ampalaya`, `Cabbage`, `Sugarcane`, `Onion`, `Gabi`, `Citrus`, `Avocado` — cross-check against whichever specific crops you added mappings for in Tasks 2-5).

- [ ] **Step 4: Run the full test suite once more**

Run: `python -m pytest`
Expected: PASS — this task makes no source changes, so this is a final confirmation nothing regressed.

- [ ] **Step 5: Do NOT commit the regenerated CSV, and do NOT ingest/retrain/promote**

`suri_real_training_data.csv` is gitignored by design (`.gitignore`'s blanket `*.csv` rule) — leave it as a local file. This plan's job ends at "the data can be extracted"; whether and when to ingest it into Supabase, retrain, review the resulting metrics, and promote is a separate, deliberate decision for whoever is driving the pipeline at the time — consistent with how the training-data-pipeline and crop-coverage-expansion plans both treated ingestion/promotion as human-triggered actions, never something a plan performs on its own.

Report the final row count and per-crop breakdown in your task report so the controller/user has the real numbers before deciding on ingestion.

- [ ] **Step 6: Commit the workflow confirmation**

There's no code change in this task, so there's nothing to commit unless Step 2-4 surfaced something worth recording. If everything ran clean, note that in your report and skip the commit — an empty commit is not useful here.

---

## Self-Review Notes

- **Spec coverage**: every crop this plan adds mappings for is already a real, working canonical crop (added by the crop-coverage-expansion plan) — this plan only extends data *extraction*, never crop *support*, matching the "Real Data Availability" section's own framing of the gap.
- **No placeholders**: every task gives the exact dict entries and exact test code to add — nothing deferred to "similar to the above."
- **Type consistency**: every `CROP_VARIETY_MAP` value is a `tuple[str, str | None]` matching the pre-existing entries' shape exactly; no new types introduced.

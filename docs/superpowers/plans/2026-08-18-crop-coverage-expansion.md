# Crop Coverage Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the number of crops SURI recognizes from 9 to 30 by transcribing more rows of the official BSWM Nutrient Recommendation Guide, and make every future crop addition a pure data edit.

**Architecture:** The crop registry (`data/nutrient_recommendation_guide.py`) currently maintains `CROPS`, `TARGET_UNIT`, and a hardcoded `_guide_row_key()` branch per crop, all in parallel with `GUIDE_ROWS`. Tasks 1–2 invert that: `GUIDE_ROWS` becomes the single source and everything else derives from it, with a generalized sub-axis mechanism and a crop-alias map. Tasks 3–5 are then pure data additions. Tasks 6–8 sync the frontend, guard the model path for crops with no training data, and retrain.

**Tech Stack:** Python 3.12, scikit-learn/XGBoost, pytest; TypeScript/Next.js for the frontend crop list.

**Spec:** `suri-ml-backend/docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md` (§2.1)

## Global Constraints

- **Source of truth is `suri/public/Nutrient Recommendation Guide.pdf`.** Every number entered must be read off that PDF. The values printed in this plan were transcribed from a scan and **must be cross-checked cell-by-cell before being committed** — the same manual verification the existing 9 crops went through (see `data/nutrient_recommendation_guide.py`'s docstring, which records four real transcription errors that the re-check caught).
- **Band structure is fixed by the guide's columns:**
  - `om_bands` — 5 values for columns `0-2.0 / 2.1-3.5 / 3.6-4.5 / 4.6-5.5 / >5.6` → `((2.0,a),(3.5,b),(4.5,c),(5.5,d),(inf,e))`
  - `p_bands` — 6 Olsen columns `0-6.0 / 6.1-10.0 / 10.1-15.0 / 15.1-20.0 / 20.1-30.0 / >30.1`. Collapse to 5 entries **only when the 5th and 6th columns hold the same rate**: `((6.0,a),(10.0,b),(15.0,c),(20.0,d),(inf,e))`; otherwise keep all six with `(30.0, …)` before the terminal band.
  - `k_bands` — 5 values for columns `0-35 / 36-55 / 56-250 / 251-300 / >301` → `((35.0,a),(55.0,b),(250.0,c),(300.0,d),(inf,e))`
- **P values use the Olsen column labels**, per the existing `P_METHOD = "olsen"` constant.
- **Field crops are `kg_ha`.** Every crop added by this plan is a field/vegetable/root/legume crop measured per hectare. No `kg_tree` crop is added here (see "Deliberately Deferred" at the end).
- **`handle_unknown="error"` stays.** No crop may silently produce an all-zeros feature vector.
- Commit after every task. Run `python -m pytest` from `suri-ml-backend/` before each commit.

---

### Task 1: Derive the crop registry from GUIDE_ROWS

**Why:** `CROPS`, `TARGET_UNIT`, and `_guide_row_key()` each restate, by hand, facts that `GUIDE_ROWS` already encodes. Adding 21 crops against that structure means editing four places per crop and hoping they agree. After this task, adding a crop means adding its `GUIDE_ROWS` entries and nothing else. **No behavior changes** — the tests prove the derived values equal today's hardcoded ones exactly.

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Test: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces:
  - `data.nutrient_recommendation_guide.CANONICAL_CROPS: list[str]` — sorted, derived from `GUIDE_ROWS`
  - `data.nutrient_recommendation_guide.DEFAULT_VARIETY_ROW: dict[str, str]`
  - `TARGET_UNIT` and `CROPS` keep their names and types but become derived

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestDerivedRegistry:
    """The registry is derived from GUIDE_ROWS rather than restated by hand.
    These pin the derivation to exactly today's values, so the refactor is
    provably behaviour-preserving."""

    def test_canonical_crops_match_the_guide_rows(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, GUIDE_ROWS

        expected = sorted({key.split("|", 1)[0] for key in GUIDE_ROWS})
        assert CANONICAL_CROPS == expected

    def test_the_nine_original_crops_are_all_present(self):
        from data.nutrient_recommendation_guide import CROPS

        for crop in ["Corn", "Rice", "Cassava", "Peanut", "Mango",
                     "Cacao", "Banana", "Tomato", "Coconut"]:
            assert crop in CROPS

    def test_target_unit_is_derived_and_still_correct(self):
        from data.nutrient_recommendation_guide import TARGET_UNIT

        for crop in ["Rice", "Corn", "Cassava", "Peanut", "Tomato"]:
            assert TARGET_UNIT[crop] == "kg_ha"
        for crop in ["Mango", "Cacao", "Coconut", "Banana"]:
            assert TARGET_UNIT[crop] == "kg_tree"

    def test_every_crop_has_a_target_unit(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, TARGET_UNIT

        assert set(TARGET_UNIT) == set(CANONICAL_CROPS)

    def test_a_crop_whose_rows_disagree_on_unit_is_rejected(self):
        """A mixed-unit crop would make target_unit_for_crop() arbitrary."""
        from data.nutrient_recommendation_guide import GuideRow, _derive_target_units

        rows = {
            "Bogus|A": GuideRow("kg_ha", ((2.0, 1),), ((6.0, 1),), ((35.0, 1),)),
            "Bogus|B": GuideRow("kg_tree", ((2.0, 1),), ((6.0, 1),), ((35.0, 1),)),
        }
        with pytest.raises(ValueError, match="Bogus"):
            _derive_target_units(rows)


class TestGuideRowKeyGeneralization:
    """Corn's Native/Hybrid split is a variety axis, same shape as the
    sub-axes new crops need. Pinning it here guards the generalization."""

    def test_corn_defaults_to_the_native_row(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Corn", None, None, None) == "Corn|Native"
        assert _guide_row_key("Corn", "OPV", None, None) == "Corn|Native"
        assert _guide_row_key("Corn", "Hybrid", None, None) == "Corn|Hybrid"

    def test_rice_still_keys_on_cultivation_method(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Rice", None, None, None) == "Rice|Lowland Wet"
        assert _guide_row_key("Rice", None, "Upland", None) == "Rice|Upland"

    def test_tree_crops_still_key_on_age_band(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Coconut", None, None, None) == "Coconut|9yrs & up"
        assert _guide_row_key("Coconut", None, None, "1-5yrs") == "Coconut|1-5yrs"

    def test_unsplit_crops_key_on_the_bare_name(self):
        from data.nutrient_recommendation_guide import _guide_row_key

        assert _guide_row_key("Cassava", None, None, None) == "Cassava"
```

Add `import pytest` at the top of the file if it isn't already imported.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestDerivedRegistry -v`
Expected: FAIL with `ImportError: cannot import name 'CANONICAL_CROPS'`

- [ ] **Step 3: Replace the hand-maintained registry with derivation**

In `data/nutrient_recommendation_guide.py`, **delete** the hardcoded `CROPS` and `TARGET_UNIT` literals (currently at lines 99–106) and **move** the derivation below `GUIDE_ROWS` (it must come after, since it reads from it). Immediately after the `GUIDE_ROWS` dict's closing brace, insert:

```python
def _derive_target_units(rows: dict[str, GuideRow]) -> dict[str, str]:
    """Every row of one crop must agree on its unit -- a crop measured in
    kg/ha by one row and kg/tree by another would make target_unit_for_crop()
    depend on dict ordering."""
    units: dict[str, str] = {}
    for key, row in rows.items():
        crop = key.split("|", 1)[0]
        existing = units.setdefault(crop, row.unit)
        if existing != row.unit:
            raise ValueError(
                f"Crop {crop!r} has rows with conflicting units: {existing} and {row.unit}"
            )
    return units


# Derived from GUIDE_ROWS, never restated by hand: adding a crop means adding
# its guide row(s) and nothing else. Sorted so the one-hot encoder's category
# order is deterministic across processes.
CANONICAL_CROPS: list[str] = sorted({key.split("|", 1)[0] for key in GUIDE_ROWS})

TARGET_UNIT: dict[str, str] = _derive_target_units(GUIDE_ROWS)

CROPS: list[str] = list(CANONICAL_CROPS)

# For crops the guide splits on a non-age, non-cultivation axis, the row label
# used when the caller supplies nothing (or something this crop doesn't have).
# Corn is the existing case: its rows are Native/Hybrid, and an unset or "OPV"
# variety maps to Native.
DEFAULT_VARIETY_ROW: dict[str, str] = {
    "Corn": "Native",
}
```

Then replace `_guide_row_key` with the generalized version:

```python
def _guide_row_key(
    crop: str, variety: str | None, cultivation_method: str | None, age_band: str | None,
) -> str:
    if crop == "Rice":
        return f"Rice|{resolve_cultivation_method(crop, cultivation_method)}"
    if crop in DEFAULT_VARIETY_ROW:
        resolved = resolve_variety(crop, variety)
        return f"{crop}|{DEFAULT_VARIETY_ROW[crop] if resolved == 'None' else resolved}"
    if crop in BEARING_AGE_BAND:
        return f"{crop}|{resolve_age_band(crop, age_band)}"
    if crop == "Banana":
        return "Banana|Fruit"
    return crop
```

> Note: `VARIETIES_BY_CROP["Corn"]` is `("OPV", "Hybrid")`, so `resolve_variety("Corn", "OPV")` returns `"OPV"`, which is not a guide row label — `DEFAULT_VARIETY_ROW` only fires for `"None"`. Fix this by mapping OPV to Native explicitly: change the `Corn` branch condition to treat any resolved variety that isn't a real row label as the default. Concretely, replace the `DEFAULT_VARIETY_ROW` branch body with:
> ```python
>         resolved = resolve_variety(crop, variety)
>         label = resolved if f"{crop}|{resolved}" in GUIDE_ROWS else DEFAULT_VARIETY_ROW[crop]
>         return f"{crop}|{label}"
> ```
> This is the version to implement — it keeps `Corn|OPV` (no such row) resolving to `Corn|Native`, exactly as today.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS. Every pre-existing test in `tests/test_nutrient_recommendation_guide.py`, `tests/test_preprocessing.py`, and `tests/test_api.py` must still pass unchanged — that is the proof this refactor changed no behavior.

- [ ] **Step 5: Commit**

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "refactor: derive CROPS and TARGET_UNIT from GUIDE_ROWS"
```

---

### Task 2: Crop aliases

**Why:** Most guide rows cover several species under one set of numbers — `Cassava, Camote, Yacon`, `Mongo, Soybean, Peanut`, `Corn/Sorghum`, `Cabbage, Pechay, Mustard, Spinach, Broccoli`. A farmer growing pechay should be able to pick "Pechay", not learn that it lives under "Cabbage". One alias map serves both the co-listed species in new rows (Task 3) and the nine species already covered by existing rows (Task 5).

Aliases resolve to their canonical crop **before** the feature row is built, so the one-hot encoder only ever sees canonical names and the feature width stays tied to the number of guide rows, not the number of selectable names.

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Modify: `ml/preprocessing.py`
- Modify: `main.py`
- Test: `tests/test_nutrient_recommendation_guide.py`, `tests/test_api.py`

**Interfaces:**
- Produces:
  - `data.nutrient_recommendation_guide.CROP_ALIASES: dict[str, str]` — alias → canonical
  - `data.nutrient_recommendation_guide.resolve_crop(crop: str) -> str`
  - `CROPS` becomes `CANONICAL_CROPS + sorted(CROP_ALIASES)` (everything selectable)
  - `ml.preprocessing` one-hot categories switch from `CROPS` to `CANONICAL_CROPS`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestCropAliases:
    def test_an_alias_resolves_to_its_canonical_crop(self):
        from data.nutrient_recommendation_guide import CROP_ALIASES, resolve_crop

        for alias, canonical in CROP_ALIASES.items():
            assert resolve_crop(alias) == canonical

    def test_a_canonical_crop_resolves_to_itself(self):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop("Cassava") == "Cassava"

    def test_an_unknown_crop_raises(self):
        from data.nutrient_recommendation_guide import resolve_crop

        with pytest.raises(KeyError, match="Nonesuch"):
            resolve_crop("Nonesuch")

    def test_every_alias_points_at_a_real_canonical_crop(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROP_ALIASES

        for alias, canonical in CROP_ALIASES.items():
            assert canonical in CANONICAL_CROPS, alias

    def test_no_alias_shadows_a_canonical_crop(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROP_ALIASES

        assert not (set(CROP_ALIASES) & set(CANONICAL_CROPS))

    def test_crops_exposes_canonical_names_and_aliases(self):
        from data.nutrient_recommendation_guide import CANONICAL_CROPS, CROPS, CROP_ALIASES

        assert set(CROPS) == set(CANONICAL_CROPS) | set(CROP_ALIASES)

    def test_nutrient_requirements_accepts_an_alias(self):
        from data.nutrient_recommendation_guide import CROP_ALIASES, nutrient_requirements

        if not CROP_ALIASES:
            pytest.skip("no aliases registered yet")
        alias, canonical = next(iter(CROP_ALIASES.items()))
        assert nutrient_requirements(alias, 2.5, 15.0, 0.3) == \
               nutrient_requirements(canonical, 2.5, 15.0, 0.3)
```

Append to `tests/test_api.py`:

```python
class TestAliasCrops:
    def test_an_alias_crop_predicts_identically_to_its_canonical_crop(self, client):
        from data.nutrient_recommendation_guide import CROP_ALIASES

        if not CROP_ALIASES:
            pytest.skip("no aliases registered yet")
        alias, canonical = next(iter(CROP_ALIASES.items()))

        alias_body = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": alias}).json()
        canonical_body = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": canonical}).json()

        assert alias_body["target_n"] == canonical_body["target_n"]
        assert alias_body["target_p2o5"] == canonical_body["target_p2o5"]
        assert alias_body["target_k2o"] == canonical_body["target_k2o"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestCropAliases -v`
Expected: FAIL with `ImportError: cannot import name 'CROP_ALIASES'`

- [ ] **Step 3: Add the alias map and resolver**

In `data/nutrient_recommendation_guide.py`, after `CANONICAL_CROPS`/`TARGET_UNIT`, add:

```python
# Species the guide prints on the SAME row as a canonical crop, sharing its
# exact rates. Aliasing rather than duplicating keeps one row per set of
# numbers, so the one-hot feature width tracks guide rows (not selectable
# names) and a rate correction is still a single edit.
CROP_ALIASES: dict[str, str] = {}


def resolve_crop(crop: str) -> str:
    if crop in CROP_ALIASES:
        return CROP_ALIASES[crop]
    if crop in CANONICAL_CROPS:
        return crop
    raise KeyError(f"Unknown crop {crop!r}")
```

Change `CROPS` to include aliases:

```python
CROPS: list[str] = CANONICAL_CROPS + sorted(CROP_ALIASES)
```

> Ordering note: `CROPS` is now `sorted(canonical) + sorted(aliases)`, which is deterministic. It is no longer used as the one-hot category list (Step 4 switches that to `CANONICAL_CROPS`), so its order only affects the 422 message and the frontend list.

Then route both public lookups through the resolver. In `nutrient_requirements`, replace:

```python
    if crop not in CROPS:
        raise KeyError(f"Unknown crop {crop!r}")
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
```

with:

```python
    crop = resolve_crop(crop)
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
```

Apply the identical two-line change in `max_total_need`. Also change `target_unit_for_crop`:

```python
def target_unit_for_crop(crop: str) -> str:
    return TARGET_UNIT[resolve_crop(crop)]
```

and make the three axis resolvers alias-aware by resolving first:

```python
def resolve_variety(crop: str, variety: str | None) -> str:
    if variety in VARIETIES_BY_CROP.get(resolve_crop(crop), ()):
        return variety
    return "None"


def resolve_cultivation_method(crop: str, cultivation_method: str | None) -> str:
    if resolve_crop(crop) != "Rice":
        return "None"
    if cultivation_method in RICE_CULTIVATION_METHODS:
        return cultivation_method
    return "Lowland Wet"


def resolve_age_band(crop: str, age_band: str | None) -> str:
    canonical = resolve_crop(crop)
    if canonical not in BEARING_AGE_BAND:
        return "None"
    if age_band in (None, "bearing"):
        return BEARING_AGE_BAND[canonical]
    return age_band
```

- [ ] **Step 4: Point the encoder at canonical crops only**

In `ml/preprocessing.py`, change the import:

```python
from data.nutrient_recommendation_guide import (
    CANONICAL_CROPS, RICE_CULTIVATION_METHODS, BEARING_AGE_BAND, GUIDE_ROWS,
)
```

and the encoder's category list:

```python
    nominal = OneHotEncoder(
        categories=[CANONICAL_CROPS, VARIETIES, CULTIVATION_METHODS, AGE_BANDS, DATA_SOURCES],
        handle_unknown="error",
        sparse_output=False,
    )
```

Also derive `VARIETIES` from the registry instead of restating it, so a new crop's variety axis can never drift out of sync with the encoder:

```python
# Derived, not restated: a crop whose guide rows split on variety must have
# that variety in the encoder's vocabulary or transform() raises at serve time.
VARIETIES = ["None"] + sorted({
    variety for varieties in VARIETIES_BY_CROP.values() for variety in varieties
})
```

adding `VARIETIES_BY_CROP` to the same import.

- [ ] **Step 5: Resolve the alias before featurizing in main.py**

In `main.py`'s `_build_feature_row`, add the import `resolve_crop` to the existing `data.nutrient_recommendation_guide` import list, then change the crop line:

```python
        # Canonical name only: the encoder's vocabulary is CANONICAL_CROPS, so
        # an alias would trip handle_unknown="error".
        "crop": resolve_crop(request.crop),
```

`_compute_confidence` also looks metrics up by crop name; make it use the canonical name too, since `metrics.by_crop` keys come from training rows (which store canonical names):

```python
        confidence = _compute_confidence(bundle, resolve_crop(request.crop), X)
```

> If Plan A (training pipeline) has not been implemented yet, this line reads `_compute_confidence(resolve_crop(request.crop), X)` instead — match whichever signature is present.

Finally, in `data/training_upload.py` (created by Plan A), the `build_training_row` function must store the canonical crop so training data and the encoder agree. Change:

```python
    crop = _text(raw, "crop")
    if crop not in CROPS:
        raise ValueError(f"Unknown crop {crop!r}. Valid crops: {', '.join(CROPS)}")
```

to:

```python
    requested = _text(raw, "crop")
    if requested not in CROPS:
        raise ValueError(f"Unknown crop {requested!r}. Valid crops: {', '.join(CROPS)}")
    crop = resolve_crop(requested)
```

adding `resolve_crop` to that module's import list.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS. `CROP_ALIASES` is still empty, so the alias-specific tests skip; everything else proves the plumbing is behavior-preserving.

- [ ] **Step 7: Commit**

```bash
git add data/nutrient_recommendation_guide.py ml/preprocessing.py main.py data/training_upload.py tests/
git commit -m "feat: add crop aliases resolving to canonical guide rows"
```

---

### Task 3: Transcribe the vegetable rows

**Verification is part of this task, not a follow-up.** The values below were read from a scan of the guide. Before committing, open `suri/public/Nutrient Recommendation Guide.pdf` at page 1 and check every cell against the table. The existing crops' docstring records four errors a re-check caught — assume the same rate here.

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Test: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces 13 new `GUIDE_ROWS` entries and their aliases. All `kg_ha`, no sub-axis.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestVegetableRows:
    """One assertion per band edge per crop. Values are the guide's own;
    a failure here means either the transcription or the band arithmetic is
    wrong, and both are worth failing loudly on."""

    @pytest.mark.parametrize("crop,om,p,k,expected", [
        # (crop, %OM, Olsen P ppm, K ppm, (N, P2O5, K2O))
        ("Alugbati", 1.0, 3.0, 20.0, (90, 60, 60)),
        ("Alugbati", 6.0, 35.0, 400.0, (0, 0, 0)),
        ("Ampalaya", 1.0, 3.0, 20.0, (40, 80, 45)),
        ("Asparagus", 3.0, 8.0, 45.0, (60, 30, 45)),
        ("Cabbage", 1.0, 3.0, 20.0, (150, 80, 75)),
        ("Cabbage", 5.0, 12.0, 100.0, (100, 40, 30)),
        ("Cauliflower", 1.0, 3.0, 20.0, (90, 80, 90)),
        ("Celery", 1.0, 3.0, 20.0, (60, 90, 60)),
        ("Chayote", 1.0, 3.0, 20.0, (60, 80, 65)),
        ("Cucumber", 1.0, 3.0, 20.0, (60, 100, 45)),
        ("Eggplant", 1.0, 3.0, 20.0, (90, 80, 90)),
        ("Lettuce", 1.0, 3.0, 20.0, (60, 60, 75)),
        ("Radish", 1.0, 3.0, 20.0, (90, 90, 120)),
        ("Radish", 3.0, 8.0, 280.0, (60, 60, 30)),
        ("Squash", 1.0, 3.0, 20.0, (60, 80, 60)),
        ("Strawberry", 1.0, 3.0, 20.0, (45, 30, 60)),
    ])
    def test_band_lookups(self, crop, om, p, k, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(crop, om, p, k / K_CMOL_TO_PPM) == expected

    def test_all_vegetables_are_per_hectare(self):
        from data.nutrient_recommendation_guide import target_unit_for_crop

        for crop in ["Alugbati", "Ampalaya", "Asparagus", "Cabbage", "Cauliflower",
                     "Celery", "Chayote", "Cucumber", "Eggplant", "Lettuce",
                     "Radish", "Squash", "Strawberry"]:
            assert target_unit_for_crop(crop) == "kg_ha"

    @pytest.mark.parametrize("alias,canonical", [
        ("Kangkong", "Alugbati"), ("Pechay", "Cabbage"), ("Mustard", "Cabbage"),
        ("Spinach", "Cabbage"), ("Broccoli", "Cabbage"), ("Muskmelon", "Cucumber"),
        ("Watermelon", "Cucumber"), ("Okra", "Eggplant"), ("Malunggay", "Lettuce"),
        ("Carrots", "Radish"), ("Upo", "Squash"), ("Patola", "Squash"),
    ])
    def test_co_listed_species_alias_to_their_row(self, alias, canonical):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop(alias) == canonical
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestVegetableRows -v`
Expected: FAIL with `KeyError: "Unknown crop 'Alugbati'"`

- [ ] **Step 3: Cross-check the values against the PDF**

Open `suri/public/Nutrient Recommendation Guide.pdf`, page 1, the `VEGETABLES` block. For each of the 13 rows confirm all 16 printed cells (5 OM, 6 P, 5 K). **Two rows to check with particular care** — the scan was ambiguous:
- **Squash, Upo, Patola** — the 3rd and 4th OM cells both read as `30`. Confirm whether the 4th is `30` or `20`.
- **Strawberry** — the 2nd and 3rd P cells both read as `20`. Confirm the 2nd is `20` and not `25`.

Correct the table in Step 4 to match the PDF before entering it. If a value differs from what is written below, **the PDF wins** — update both the code and the test's expected tuple.

- [ ] **Step 4: Add the rows**

In `data/nutrient_recommendation_guide.py`, add these entries to `GUIDE_ROWS` (after the `Banana|Fruit` entry, before the closing brace):

```python
    # --- VEGETABLES (guide page 1) -------------------------------------
    "Alugbati": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 60), (4.5, 45), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Ampalaya": GuideRow(
        "kg_ha",
        om_bands=((2.0, 40), (3.5, 30), (4.5, 15), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 30), (250.0, 15), (300.0, 0), (float("inf"), 0)),
    ),
    "Asparagus": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 60), (4.5, 45), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 40), (10.0, 30), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Cabbage": GuideRow(
        "kg_ha",
        om_bands=((2.0, 150), (3.5, 140), (4.5, 120), (5.5, 100), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 75), (55.0, 60), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Cauliflower": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 60), (4.5, 30), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Celery": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 40), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 90), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Chayote": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 40), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 65), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Cucumber": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 40), (4.5, 20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 100), (10.0, 75), (15.0, 50), (20.0, 25), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 30), (250.0, 15), (300.0, 0), (float("inf"), 0)),
    ),
    "Eggplant": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 80), (4.5, 70), (5.5, 60), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 30), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Lettuce": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 40), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 75), (55.0, 60), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Radish": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 60), (4.5, 30), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 90), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 120), (55.0, 90), (250.0, 60), (300.0, 30), (float("inf"), 0)),
    ),
    "Squash": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 40), (4.5, 30), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Strawberry": GuideRow(
        "kg_ha",
        om_bands=((2.0, 45), (3.5, 30), (4.5, 20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 30), (10.0, 20), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 48), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
```

Then register the co-listed species in `CROP_ALIASES`:

```python
CROP_ALIASES: dict[str, str] = {
    # Species the guide prints on the same row as the canonical crop above.
    "Kangkong": "Alugbati",
    "Pechay": "Cabbage",
    "Mustard": "Cabbage",
    "Spinach": "Cabbage",
    "Broccoli": "Cabbage",
    "Muskmelon": "Cucumber",
    "Watermelon": "Cucumber",
    "Okra": "Eggplant",
    "Malunggay": "Lettuce",
    "Carrots": "Radish",
    "Upo": "Squash",
    "Patola": "Squash",
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "feat: add the guide's 13 vegetable rows"
```

---

### Task 4: Transcribe condiments, rootcrops, legumes, and sugarcane

Same verification requirement as Task 3 — these rows are on **page 2** of the PDF.

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Test: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces 8 new canonical crops across 10 guide rows. Ginger and Onion split on a variety axis, so both gain `VARIETIES_BY_CROP` and `DEFAULT_VARIETY_ROW` entries.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestCondimentRootcropAndLegumeRows:
    @pytest.mark.parametrize("crop,om,p,k,expected", [
        ("Garlic", 1.0, 3.0, 20.0, (120, 80, 150)),
        ("Garlic", 6.0, 35.0, 400.0, (0, 0, 0)),
        ("Onion", 1.0, 3.0, 20.0, (120, 80, 150)),
        ("Gabi", 1.0, 3.0, 20.0, (40, 60, 90)),
        ("Potato", 1.0, 3.0, 20.0, (120, 100, 120)),
        ("Beans", 1.0, 3.0, 20.0, (10, 60, 60)),
        ("Peas", 1.0, 3.0, 20.0, (20, 40, 45)),
        ("Sugarcane", 1.0, 3.0, 20.0, (180, 120, 160)),
        ("Sugarcane", 5.0, 18.0, 280.0, (75, 20, 60)),
    ])
    def test_band_lookups(self, crop, om, p, k, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(crop, om, p, k / K_CMOL_TO_PPM) == expected

    def test_ginger_defaults_to_the_improved_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Ginger", 1.0, 3.0, k) == (60, 60, 120)
        assert nutrient_requirements("Ginger", 1.0, 3.0, k, variety="Local") == (40, 50, 90)

    def test_onion_spring_is_a_separate_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Onion", 1.0, 3.0, k, variety="Spring") == (130, 60, 45)

    @pytest.mark.parametrize("alias,canonical", [
        ("Chili Pepper", "Garlic"), ("Camote", "Cassava"), ("Ube", "Gabi"),
        ("Yam", "Gabi"), ("Taro", "Gabi"), ("Singkamas", "Potato"),
        ("Sitao", "Beans"), ("Lima", "Beans"), ("Cowpea", "Peas"), ("Tapilan", "Peas"),
    ])
    def test_co_listed_species_alias_to_their_row(self, alias, canonical):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop(alias) == canonical
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestCondimentRootcropAndLegumeRows -v`
Expected: FAIL with `KeyError: "Unknown crop 'Garlic'"`

- [ ] **Step 3: Cross-check against the PDF**

Page 2 of the guide, the `CONDIMENTS/ SPICES`, `ROOTCROPS`, `LEGUMES` blocks and the `Sugarcane` row in `INDUSTRIAL CROPS`. Confirm all 16 cells per row. **Check with particular care:**
- **Onion Spring** — the OM row scanned as `130 120 70 60 0`; confirm the 3rd cell is `70`, not `10`.
- **Potato/Singkamas** — the K row scanned as `120 90 60 0 0`; confirm the 4th cell is `0` and not `30`.
- **Beans/Peas** — these rows are labeled `with` / `w/o` in the guide, referring to **rhizobium inoculation**. This plan treats them as two independent crops (Beans = the `with` row, Peas = the `w/o` row) because that is how the guide groups the species, **not** as an inoculation axis on one crop. Confirm this reading matches the PDF's row grouping before proceeding; if the guide intends `with`/`without` as an axis over all four legume groups, stop and re-scope this row set.

- [ ] **Step 4: Add the rows**

Add to `GUIDE_ROWS`:

```python
    # --- CONDIMENTS / SPICES (guide page 2) -----------------------------
    "Garlic": GuideRow(
        "kg_ha",
        om_bands=((2.0, 120), (3.5, 90), (4.5, 60), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 150), (55.0, 120), (250.0, 90), (300.0, 60), (float("inf"), 0)),
    ),
    "Ginger|Improved": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 40), (4.5, 20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 120), (55.0, 90), (250.0, 60), (300.0, 30), (float("inf"), 0)),
    ),
    "Ginger|Local": GuideRow(
        "kg_ha",
        om_bands=((2.0, 40), (3.5, 30), (4.5, 20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 50), (10.0, 40), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 45), (300.0, 30), (float("inf"), 0)),
    ),
    "Onion|Bulb": GuideRow(
        "kg_ha",
        om_bands=((2.0, 120), (3.5, 90), (4.5, 60), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 150), (55.0, 120), (250.0, 90), (300.0, 60), (float("inf"), 0)),
    ),
    "Onion|Spring": GuideRow(
        "kg_ha",
        om_bands=((2.0, 130), (3.5, 120), (4.5, 70), (5.5, 60), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 45), (15.0, 35), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 35), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    # --- ROOTCROPS (guide page 2) ---------------------------------------
    "Gabi": GuideRow(
        "kg_ha",
        om_bands=((2.0, 40), (3.5, 30), (4.5, 20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Potato": GuideRow(
        "kg_ha",
        om_bands=((2.0, 120), (3.5, 90), (4.5, 60), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 100), (10.0, 80), (15.0, 60), (20.0, 30), (float("inf"), 0)),
        k_bands=((35.0, 120), (55.0, 90), (250.0, 60), (300.0, 0), (float("inf"), 0)),
    ),
    # --- LEGUMES (guide page 2) -----------------------------------------
    "Beans": GuideRow(
        "kg_ha",
        om_bands=((2.0, 10), (3.5, 0), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Peas": GuideRow(
        "kg_ha",
        om_bands=((2.0, 20), (3.5, 10), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 40), (10.0, 30), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 30), (250.0, 15), (300.0, 0), (float("inf"), 0)),
    ),
    # --- INDUSTRIAL (guide page 2) --------------------------------------
    "Sugarcane": GuideRow(
        "kg_ha",
        om_bands=((2.0, 180), (3.5, 130), (4.5, 100), (5.5, 75), (float("inf"), 0)),
        p_bands=((6.0, 120), (10.0, 80), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 160), (55.0, 130), (250.0, 100), (300.0, 60), (float("inf"), 0)),
    ),
```

Register the two new variety axes:

```python
VARIETIES_BY_CROP: dict[str, tuple[str, ...]] = {
    "Rice": ("Hybrid", "Inbred"),
    "Corn": ("OPV", "Hybrid"),
    "Ginger": ("Improved", "Local"),
    "Onion": ("Bulb", "Spring"),
}
```

```python
DEFAULT_VARIETY_ROW: dict[str, str] = {
    "Corn": "Native",
    "Ginger": "Improved",
    "Onion": "Bulb",
}
```

And extend `CROP_ALIASES` with:

```python
    "Chili Pepper": "Garlic",
    "Ube": "Gabi",
    "Yam": "Gabi",
    "Taro": "Gabi",
    "Singkamas": "Potato",
    "Lima": "Beans",
    "Sitao": "Beans",
    "Tapilan": "Peas",
    "Cowpea": "Peas",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: PASS. The `("Camote", "Cassava")` case will still fail — it is registered in Task 5. Temporarily mark it `pytest.mark.xfail(reason="registered in Task 5")` or move that one parameter case into Task 5's test.

- [ ] **Step 6: Commit**

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "feat: add condiment, rootcrop, legume, and sugarcane guide rows"
```

---

### Task 5: Register species already covered by existing rows

**Why:** Nine species share a guide row with a crop already in the table, so they need no new numbers — only an alias entry. Sorghum in particular is printed on Corn's own row (`Corn /Sorghum`).

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Test: `tests/test_nutrient_recommendation_guide.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestFreeSpeciesAliases:
    @pytest.mark.parametrize("alias,canonical", [
        ("Sorghum", "Corn"), ("Camote", "Cassava"), ("Yacon", "Cassava"),
        ("Soybean", "Peanut"), ("Mongo", "Peanut"), ("Santol", "Mango"),
        ("Caimito", "Mango"), ("Durian", "Mango"), ("Marang", "Mango"),
    ])
    def test_alias_resolves_and_predicts_identically(self, alias, canonical):
        from data.nutrient_recommendation_guide import nutrient_requirements, resolve_crop

        assert resolve_crop(alias) == canonical
        assert nutrient_requirements(alias, 2.5, 15.0, 0.3) == \
               nutrient_requirements(canonical, 2.5, 15.0, 0.3)

    def test_tree_crop_aliases_keep_the_tree_unit(self):
        from data.nutrient_recommendation_guide import target_unit_for_crop

        for alias in ["Santol", "Caimito", "Durian", "Marang"]:
            assert target_unit_for_crop(alias) == "kg_tree"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestFreeSpeciesAliases -v`
Expected: FAIL with `KeyError: "Unknown crop 'Sorghum'"`

- [ ] **Step 3: Confirm the groupings against the PDF**

Verify on the PDF that each of these species is printed on the same row as its canonical crop: `Corn /Sorghum` (page 1), `Cassava, Camote, Yacon` and `Mongo, Soybean, Peanut` (page 2), `Mango, Santol, Caimito, Durian, Marang` (page 3). If any species is on its own separate row with different numbers, it is **not** an alias — drop it from this task and transcribe it as its own crop instead.

- [ ] **Step 4: Add the aliases**

Extend `CROP_ALIASES`:

```python
    # Species printed on an existing crop's own guide row -- no new rates.
    "Sorghum": "Corn",
    "Camote": "Cassava",
    "Yacon": "Cassava",
    "Soybean": "Peanut",
    "Mongo": "Peanut",
    "Santol": "Mango",
    "Caimito": "Mango",
    "Durian": "Mango",
    "Marang": "Mango",
```

Remove any `xfail` marker added in Task 4 Step 5.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "feat: register nine species already covered by existing guide rows"
```

---

### Task 6: Refuse model-path predictions for crops with no training data

**The problem this prevents:** `OneHotEncoder` is built with an explicit `categories=` list, so after a retrain every crop in `CANONICAL_CROPS` has its own column — including the 21 crops with zero training rows. Their column is never active during training, so the tree ensemble simply never splits on it and the model returns something near the global mean. That is a **confident-looking number computed from no crop-specific information**, which is precisely the failure mode `handle_unknown="error"` was chosen to prevent for unknown crops. The complete-data path is unaffected (it reads the guide table directly and is exact), so this guard applies only to the model path.

**Files:**
- Modify: `main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `main.trained_crops(bundle: dict) -> set[str]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
class TestUntrainedCropGuard:
    """A crop the model never saw must not get a model-path prediction --
    its one-hot column was never active in training, so the answer would be
    the global mean wearing a crop's name."""

    def test_incomplete_request_for_an_untrained_crop_is_refused(self, client, monkeypatch):
        import main as service

        monkeypatch.setattr(service, "trained_crops", lambda bundle: {"Corn"})
        response = client.post("/predict", json={
            "crop": "Cabbage", "data_source": "stk", "ph": 6.4,
        })
        assert response.status_code == 422
        body = response.json()
        assert "Cabbage" in body["detail"]
        assert "organic_matter_pct" in body["detail"]

    def test_complete_request_for_an_untrained_crop_still_succeeds(self, client, monkeypatch):
        """The guide table has the exact answer regardless of training data."""
        import main as service

        monkeypatch.setattr(service, "trained_crops", lambda bundle: {"Corn"})
        response = client.post("/predict", json={
            "crop": "Cabbage", "data_source": "lab", "ph": 6.4,
            "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0, "potassium_cmol_kg": 0.35,
        })
        assert response.status_code == 200
        assert response.json()["source"] == "table-lookup"

    def test_incomplete_request_for_a_trained_crop_still_succeeds(self, client):
        response = client.post("/predict", json={
            "crop": "Corn", "data_source": "iot", "ph": 6.4,
        })
        assert response.status_code == 200
        assert response.json()["source"] == "ml-ensemble"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_api.py::TestUntrainedCropGuard -v`
Expected: FAIL — `AttributeError: module 'main' has no attribute 'trained_crops'`, and the first test returns 200 rather than 422.

- [ ] **Step 3: Implement the guard**

In `main.py`, add near `_compute_confidence`:

```python
def trained_crops(bundle: dict) -> set[str]:
    """Crops the loaded model actually has held-out metrics for. Metrics are
    computed per crop over the test split, so presence here means the model
    both trained on and was evaluated for that crop."""
    return set(bundle["metadata"].get("metrics", {}).get("by_crop", {}))
```

In `predict()`, inside the `else` (model path) branch, before transforming:

```python
    else:
        canonical = resolve_crop(request.crop)
        known = trained_crops(bundle)
        if known and canonical not in known:
            # The guide table can still answer this crop exactly -- it just
            # needs the full soil panel. Say so, rather than returning a
            # number derived from no crop-specific training signal.
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No training data exists yet for {request.crop!r}, so an estimate from "
                    f"an incomplete soil reading is not available. Supply "
                    f"organic_matter_pct, phosphorus_ppm, and potassium_cmol_kg for an exact "
                    f"result from the official guide, or upload lab data for this crop and "
                    f"retrain."
                ),
            )
        try:
            X = bundle["preprocessor"].transform(features)
```

> The `known and` guard means an artifact with no `by_crop` metrics at all (an old or hand-built one) degrades to the previous behavior rather than refusing every request.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: refuse model-path predictions for crops with no training data"
```

---

### Task 7: Sync suri's crop lists

**Why:** `suri` keeps its own two crop literals — `shared/types/ml-fertilizer.ts`'s `ML_CROPS` and `shared/constants/crops.ts`'s `CROP_OPTIONS` — and `scripts/check-crop-list-sync.mjs` fails if they disagree. Neither knows about the backend's expansion, so without this task the new crops are unselectable in every flow.

**Files:**
- Modify: `suri/shared/types/ml-fertilizer.ts`
- Modify: `suri/shared/constants/crops.ts`

- [ ] **Step 1: Get the authoritative list from the backend**

Run from `suri-ml-backend/`:

```bash
python -c "from data.nutrient_recommendation_guide import CROPS; print('\n'.join(CROPS))"
```

Use exactly this output — do not retype it from the plan.

- [ ] **Step 2: Update `ML_CROPS`**

In `suri/shared/types/ml-fertilizer.ts`, replace the `ML_CROPS` array contents with every name printed in Step 1, keeping the `as const` suffix. Also update `VARIETIES_BY_CROP` in that file to match the backend's (which now includes Ginger and Onion):

```typescript
export const VARIETIES_BY_CROP: Partial<Record<MLCrop, readonly string[]>> = {
  Rice: ["Hybrid", "Inbred"],
  Corn: ["OPV", "Hybrid"],
  Ginger: ["Improved", "Local"],
  Onion: ["Bulb", "Spring"],
};
```

- [ ] **Step 3: Update `CROP_OPTIONS`**

In `suri/shared/constants/crops.ts`, add one `{ value, label }` entry per new crop. Keep the existing nine unchanged (their labels carry Filipino names, e.g. `"Rice (Palay)"`). For new crops, use the crop name as both value and label, except where a Filipino name aids recognition:

```typescript
  { value: "Alugbati", label: "Alugbati" },
  { value: "Kangkong", label: "Kangkong" },
  { value: "Ampalaya", label: "Ampalaya" },
  { value: "Asparagus", label: "Asparagus" },
  { value: "Cabbage", label: "Cabbage (Repolyo)" },
  { value: "Pechay", label: "Pechay" },
  { value: "Mustard", label: "Mustard (Mustasa)" },
  { value: "Spinach", label: "Spinach" },
  { value: "Broccoli", label: "Broccoli" },
  { value: "Cauliflower", label: "Cauliflower" },
  { value: "Celery", label: "Celery" },
  { value: "Chayote", label: "Chayote (Sayote)" },
  { value: "Cucumber", label: "Cucumber (Pipino)" },
  { value: "Muskmelon", label: "Muskmelon" },
  { value: "Watermelon", label: "Watermelon (Pakwan)" },
  { value: "Eggplant", label: "Eggplant (Talong)" },
  { value: "Okra", label: "Okra" },
  { value: "Lettuce", label: "Lettuce" },
  { value: "Malunggay", label: "Malunggay" },
  { value: "Radish", label: "Radish (Labanos)" },
  { value: "Carrots", label: "Carrots" },
  { value: "Squash", label: "Squash (Kalabasa)" },
  { value: "Upo", label: "Upo" },
  { value: "Patola", label: "Patola" },
  { value: "Strawberry", label: "Strawberry" },
  { value: "Garlic", label: "Garlic (Bawang)" },
  { value: "Chili Pepper", label: "Chili Pepper (Sili)" },
  { value: "Ginger", label: "Ginger (Luya)" },
  { value: "Onion", label: "Onion (Sibuyas)" },
  { value: "Gabi", label: "Gabi" },
  { value: "Ube", label: "Ube" },
  { value: "Yam", label: "Yam" },
  { value: "Taro", label: "Taro" },
  { value: "Potato", label: "Potato (Patatas)" },
  { value: "Singkamas", label: "Singkamas" },
  { value: "Beans", label: "Beans (String/Sitao)" },
  { value: "Lima", label: "Lima Beans" },
  { value: "Sitao", label: "Sitao" },
  { value: "Peas", label: "Peas" },
  { value: "Tapilan", label: "Tapilan" },
  { value: "Cowpea", label: "Cowpea" },
  { value: "Sugarcane", label: "Sugarcane (Tubo)" },
  { value: "Sorghum", label: "Sorghum" },
  { value: "Camote", label: "Camote" },
  { value: "Yacon", label: "Yacon" },
  { value: "Soybean", label: "Soybean" },
  { value: "Mongo", label: "Mongo" },
  { value: "Santol", label: "Santol" },
  { value: "Caimito", label: "Caimito" },
  { value: "Durian", label: "Durian" },
  { value: "Marang", label: "Marang" },
```

- [ ] **Step 4: Run the sync guard**

Run from `suri/`:

```bash
node scripts/check-crop-list-sync.mjs
```

Expected: the script's success output, no missing/extra entries reported. Then:

```bash
npx tsc --noEmit
```

Expected: no errors. `MLCrop` is derived from `ML_CROPS`, so any consumer that switches exhaustively on crop would fail here.

- [ ] **Step 5: Commit**

```bash
git add shared/types/ml-fertilizer.ts shared/constants/crops.ts
git commit -m "feat: expand suri's crop lists to match the backend"
```

---

### Task 8: Retrain and promote

**Why this is required, not optional:** the preprocessor's one-hot vocabulary is captured inside the pickled artifact. The currently-pinned artifact (`ml/artifacts/2026-08-06-055755/`) encodes 9 crops and 4 varieties. After Tasks 1–5 the code offers 30 crops and 6 varieties, so **the live artifact and the running code now disagree** — a request for any new crop reaches `transform()` and raises `ValueError`, surfacing as a 422. Retraining rebuilds the preprocessor against the new vocabulary.

**Files:**
- Modify: `.gitignore` (re-pin the artifact)

- [ ] **Step 1: Confirm the incompatibility is real before fixing it**

Run from `suri-ml-backend/`:

```bash
python -c "
from ml import model_state
from ml.train import ARTIFACT_ROOT
b = model_state.load_bundle(model_state.available_versions(ARTIFACT_ROOT)[-1], ARTIFACT_ROOT)
enc = b['preprocessor'].named_transformers_['nom']
print('crops in the live artifact:', len(enc.categories[0]))
"
```

Expected: `9` — proving the pinned artifact predates the expansion.

- [ ] **Step 2: Retrain**

```bash
python -m ml.train --csv suri_real_training_data.csv
```

Expected: a metrics summary and a new `ml/artifacts/<version>/` directory. Note the version string.

Confirm the new artifact carries the expanded vocabulary:

```bash
python -c "
import joblib, sys
enc = joblib.load(f'ml/artifacts/{sys.argv[1]}/preprocessor.joblib').named_transformers_['nom']
print('crops:', len(enc.categories[0]), 'varieties:', len(enc.categories[1]))
" <version>
```

Expected: `crops: 30 varieties: 6`.

> The by-crop metrics will still cover only the 9 crops that have training rows — that is expected and is exactly what Task 6's guard keys off.

- [ ] **Step 3: Run the full suite against the new artifact**

Run: `python -m pytest -v`
Expected: PASS, including `tests/test_api.py::TestUntrainedCropGuard` (which now exercises a real untrained crop) and `tests/test_train.py::test_beats_the_mean_baseline_on_every_target`.

- [ ] **Step 4: Re-pin the artifact for fresh deploys**

In `.gitignore`, replace the two pinned lines with the new version:

```
ml/artifacts/*
!ml/artifacts/<new-version>/
!ml/artifacts/<new-version>/**
```

- [ ] **Step 5: Verify a new crop end to end**

Start the service (`uvicorn main:app --port 8001`) and check both paths:

```bash
# Complete data -> exact guide lookup, works despite zero training rows
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"crop":"Cabbage","data_source":"lab","ph":6.4,"organic_matter_pct":1.5,
       "phosphorus_ppm":4.0,"potassium_cmol_kg":0.05}' | python -m json.tool
```
Expected: HTTP 200, `"source": "table-lookup"`, `target_n: 150`, `target_p2o5: 80`, `target_k2o: 75`.

```bash
# Incomplete data -> refused with a clear reason, not a fabricated number
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"crop":"Cabbage","data_source":"stk","ph":6.4}' | python -m json.tool
```
Expected: HTTP 422 explaining no training data exists for Cabbage yet.

```bash
# An alias behaves exactly like its canonical crop
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"crop":"Sorghum","data_source":"lab","ph":6.4,"organic_matter_pct":2.8,
       "phosphorus_ppm":18.0,"potassium_cmol_kg":0.35}' | python -m json.tool
```
Expected: HTTP 200, identical targets to the same request with `"crop":"Corn"`.

- [ ] **Step 6: Commit and promote**

```bash
git add .gitignore "ml/artifacts/<new-version>"
git commit -m "feat: retrain against the expanded 30-crop vocabulary"
```

If Plan A is already deployed, promote through the admin UI's Model Training → Version History tab (this doubles as a live rehearsal of the promotion flow). Otherwise push to `main` and let the existing Actions deploy carry it.

---

## Deliberately Deferred

**The 9 age-banded tree/plantation crops are not in this plan.** Coffee, Papaya, Citrus, Guava, Lanzones, the Avocado group, Dragon Fruit, Pineapple, and Abaca were in the spec's §2.1 list but are held back for a specific reason: their rows are on pages 2–3 of a **scanned** PDF and several cells could not be read with confidence — Coffee's `6-10 yrs` K row scans as a non-monotonic `.36 / .12 / .60`, Guava's third age band repeats the label `4-6 yrs`, and Dragon Fruit's row leaves most columns blank. Entering a guessed rate for a crop a farmer will fertilize a real field from is not an acceptable trade for a larger crop count.

They also need structural work this plan does not do: `BEARING_AGE_BAND` and `AGE_BANDS` would grow, Papaya's bands are in **months** (`2-6 mos.`) rather than years, and Pineapple's `Planted`/`Ratoon` and Abaca's `Pulp`/`Cordage` are crop-cycle axes rather than varieties or ages.

Adding them later is a data-plus-axis task against the structure Tasks 1–2 establish, and requires either a higher-quality scan or a DA-provided digital copy of the guide. That takes coverage from 30 to 39 crops. Everything else — the pipeline, the aliases, the untrained-crop guard — already supports them the moment their rows exist.

**One caveat carried forward:** the `P_METHOD = "olsen"` assumption applies to every row added here. If a Region 7 lab confirms Bray #1 is used instead, every `p_bands` boundary in the file (old and new) shifts to the Bray labels — a single documented constant, but a file-wide edit.

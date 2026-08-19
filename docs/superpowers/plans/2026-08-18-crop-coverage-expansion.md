# Crop Coverage Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the number of crops SURI recognizes from 9 to 36 by transcribing more rows of the official BSWM Nutrient Recommendation Guide, and make every future crop addition a pure data edit.

**Architecture:** The crop registry (`data/nutrient_recommendation_guide.py`) currently maintains `CROPS`, `TARGET_UNIT`, and a hardcoded `_guide_row_key()` branch per crop, all in parallel with `GUIDE_ROWS`. Tasks 1–2 invert that: `GUIDE_ROWS` becomes the single source and everything else derives from it, with a generalized sub-axis mechanism and a crop-alias map. Tasks 3–6 are then pure data additions — including the age-banded tree crops, which need no new structural code because Task 1's generalization already covers their axes. Tasks 7–9 guard the model path for crops with no training data, sync the frontend, and retrain.

**Two phases, one plan.** Phase 1 (Tasks 3–5) adds 21 field/vegetable crops whose values are verified by two independent transcriptions. Phase 2 (Task 6) adds 6 tree and plantation crops that carry documented source-document inferences — read that task's anomaly table before starting it. Three further crops (Coffee, Dragon Fruit, Abaca) remain out of scope; see "Deliberately Deferred".

**Tech Stack:** Python 3.12, scikit-learn/XGBoost, pytest; TypeScript/Next.js for the frontend crop list.

**Spec:** `suri-ml-backend/docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md` (§2.1)

## Global Constraints

- **Source of truth is `Nutrient Recommendation Guide.pdf`** (at this repo's root), with `Nutrient Recommendation Guide.md` as a full structured transcription of it. Enter values from the `.md`; treat the PDF as the tiebreaker if they ever disagree. Both files are byte-identical copies of `suri/public/Nutrient Recommendation Guide.{md,pdf}` (confirmed via checksum) — kept here too so tasks working in this repo don't need to reach into the sibling `suri` repo.
- **The `.md` transcription has been cross-checked and corroborates the existing code.** All 9 currently-coded crops match it band-for-band, and it independently confirms all four transcription errors recorded in `data/nutrient_recommendation_guide.py`'s docstring (Rice|Upland's K 251-300 = `0`; Rice|Lowland Wet's OM = `60/40/35`; Rice|Lowland Dry's OM >5.6 = `45`; Corn's K 251-300 = `20`). It also confirms the unresolved `TestGoldenRecordRiceInbred` discrepancy is real: no `40` appears anywhere in Rice's K column under any cultivation method. Two independent transcriptions agreeing is the strongest verification available short of a DA-issued digital copy — **so the per-crop PDF re-check that earlier drafts of this plan required is no longer a blocking step for Phase 1 crops.**
- **Read the `.md`'s "Transcription notes" section before entering any value.** It flags cells that are anomalous *in the source document itself* (missing decimal points, duplicate labels, blank cells, non-monotonic rows) — those are data-quality problems no re-scan will fix, and are why Phase 2 crops are deferred.
- **Band structure is fixed by the guide's columns:**
  - `om_bands` — 5 values for columns `0-2.0 / 2.1-3.5 / 3.6-4.5 / 4.6-5.5 / >5.6` → `((2.0,a),(3.5,b),(4.5,c),(5.5,d),(inf,e))`
  - `p_bands` — 6 Olsen columns `0-6.0 / 6.1-10.0 / 10.1-15.0 / 15.1-20.0 / 20.1-30.0 / >30.1`. Collapse to 5 entries **only when the 5th and 6th columns hold the same rate**: `((6.0,a),(10.0,b),(15.0,c),(20.0,d),(inf,e))`; otherwise keep all six with `(30.0, …)` before the terminal band.
  - `k_bands` — 5 values for columns `0-35 / 36-55 / 56-250 / 251-300 / >301` → `((35.0,a),(55.0,b),(250.0,c),(300.0,d),(inf,e))`
- **P values use the Olsen column labels**, per the existing `P_METHOD = "olsen"` constant.
- **Units follow the row's scale.** Phase 1's crops are all `kg_ha`. Phase 2 adds Pineapple (`kg_ha`) and four `kg_tree` fruit-tree groups. Note that `kg_ha` vs `kg_tree` is an **inference**, not printed in the guide — it states units in only two places (Roses `g/plant`, Banana `576 trees/ha`). Whole-number rows are read as kg/ha and decimal rows as kg/tree, which is the convention the existing 9 crops already follow.
- **Record every source-document inference in a code comment** next to the value it affects. Phase 2 has several (a duplicated age label, four missing decimal points); a future reader must be able to tell a transcribed value from a reconstructed one without re-deriving it.
- **`handle_unknown="error"` stays.** No crop may silently produce an all-zeros feature vector.
- Commit after every task. Run `python -m pytest` from `suri-ml-backend/` before each commit.

## Real Data Availability (checked 2026-08-19, against the actual Region 7 database)

**Methodology:** `suri/public/Region 7-National Soil Database_2022-2025.xlsx` (the same file `data/parse_real_soil_database.py` already parses to produce `suri_real_training_data.csv`) has **9,654 rows with a lab code** across its four sheets. Only 3,076 (31.9%) reached the training corpus. Counted the rest by crop-text label — a literal, case-insensitive match against the raw `crop` column, **not** run through `normalize_crop()` (see the gap below for why that distinction matters). 1,979 rows are intercropping entries (`"Cassava, Corn"`, `"Corn, Peanut, Monggo"`, etc.) — a genuinely separate problem, since one soil reading can't be cleanly attributed to one crop's nutrient needs without a design decision on how to split it, and they're excluded from every count below.

| Plan group | Real rows found, by literal label |
|---|---|
| Original 9 (already trained) | 3,056 (close to the 3,076 in the corpus; the gap is rows the actual parser recovers via patterns this simple text match doesn't try, e.g. `"OPV Corn"`, `"Hybrid Rice"` — see below) |
| Task 3: Vegetables | 121 — Lettuce 22, Pechay 21, Ampalaya 18, Eggplant 18, Cabbage 8, Okra 9, Squash 8, Carrots 4, Cauliflower 4, Radish 3, Broccoli 1, Cucumber 2, Watermelon 2, Upo 1 |
| Task 4: Condiments/Rootcrops/Legumes/Sugarcane | 71 — Sugarcane 42, Ube 10, Ginger 6, Beans 6, Gabi 3, Onion 2, Chili Pepper 1, Potato 1 |
| Task 5: Free-species aliases | 21 — Camote 11, Sorghum 8, Soybean 1, Durian 1 |
| Task 6: Tree/plantation | 21 — Calamansi 9, Avocado 4, Jackfruit 2, Lemon 2, Pineapple 1, Papaya 1, Guyabano 1, Rambutan 1 |
| Deferred (Coffee/Dragon Fruit) | 31 — Coffee 16, Dragon Fruit/Dragonfruit 15 |

Every crop this plan already scoped has real backing data. Coffee and Dragon Fruit specifically have 16 and 15 rows respectively — confirming those two are blocked by the guide's own printed anomalies (non-monotonic/blank cells), not by data scarcity; the moment DA RSL resolves those cells, real rows are waiting.

### A real gap this plan does not close

**`data/parse_real_soil_database.py::normalize_crop()` only recognizes the current 9 crops — confirmed directly, not assumed:**

```python
>>> normalize_crop("Ampalaya")
None
>>> normalize_crop("Cabbage")
None
>>> normalize_crop("Garlic")
None
```

It returns `None` even for an exact, cleanly-spelled literal match. None of this plan's 9 tasks touch that file. So the table above is a real, counted *availability* number — but **none of those rows are actually extractable into `suri_real_training_data.csv` until `normalize_crop()` (and its co-listed-species handling) is taught the new crop vocabulary.** This plan makes the model capable of learning 27 new crops; it does not build the pipeline that would feed them real historical rows. Task 9's retrain will run against the same 3,076-row corpus as today — the 27 new crops will have zero training rows and rely entirely on Task 7's guard (exact guide-table lookup for complete data, 422 refusal for incomplete data) until a follow-up extraction task exists.

Two things worth knowing before writing that follow-up:

- **`normalize_crop()` already handles variety-prefixed text well** for the crops it knows: `"OPV Corn"`, `"Hybrid Rice"`, `"Corn OPV"`, `"Hybrid Corn"` all resolve correctly today. Extending it for new crops is the same pattern, not new machinery.
- **Real-world labels don't always match this plan's clean crop/alias names.** Concretely: `"Talong"` (Filipino for eggplant) appears **14 times** as a raw label and is not covered by Task 3's `Okra → Eggplant` alias — it would need its own `"Talong" → "Eggplant"` alias for those rows to ever be recovered. Other unmatched labels worth a look when that task is written: `"Sweetcorn"`/`"Sweet Corn"` (≈26 rows, a Corn variant), `"Upland Rice"` (10 rows, likely `cultivation_method`, not a new crop), `"Sayote"` (9, already `Chayote`'s Filipino name but spelled differently in the sheet), `"Baguio Beans"` (8, a Beans variant), `"Ubi"` (8, likely the same as `"Ube"`).

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

**These values are verified.** Every cell below was independently transcribed twice — once for this plan and once into `Nutrient Recommendation Guide.md` — and the two agree exactly, including the four cells the first pass flagged as uncertain (Squash's 4th OM cell = `30`, Strawberry's 2nd P cell = `20`). Enter them as written; no per-cell re-check is required.

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

- [ ] **Step 3: Confirm the source rows**

Open `Nutrient Recommendation Guide.md` → `Table 3` → `### VEGETABLES`. Confirm the 13 rows there match the values in Step 4 (they were derived from it). One genuine source anomaly to be aware of, per that file's transcription note 6:

- **Celery's `K >301` cell is blank in the original**, not `0`. Step 4 encodes it as `0`, matching what every other vegetable prints in that column. This is a deliberate, documented inference — note it in the code comment rather than leaving it silent.

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
        # Terminal K band is INFERRED as 0: the guide leaves Celery's >301
        # cell blank, unlike every other vegetable row, which prints 0 there.
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

- [ ] **Step 3: Confirm the source rows**

Open `Nutrient Recommendation Guide.md` → `Table 3` → the `### CONDIMENTS/ SPICES`, `### ROOTCROPS`, `### LEGUMES` sections and Sugarcane's row under `### INDUSTRIAL CROPS`. Confirm the values match Step 4. Two earlier uncertainties are now settled and need no further checking: **Onion Spring's 3rd OM cell is `70`**, and **Potato's 4th K cell is `0`**.

**The legume structure is confirmed**, which resolves the open question from this plan's earlier draft. The `.md`'s `Inoculant` column shows the `with`/`w/o` label is a **per-row property, not an axis over the group**:

| Row | Inoculant |
|---|---|
| Beans (String), Lima, Sitao | with |
| Batao, Baguio Beans | w/o |
| Mongo, Soybean, Peanut | with |
| Peas, Tapilan, Cowpea | w/o |

So Beans and Peas are independent crops with their own rates, exactly as Step 4 encodes them — no inoculation axis is needed. (`Mongo, Soybean, Peanut` is the row `Peanut` already occupies, which is why Task 5 can alias Mongo and Soybean to it for free. `Batao, Baguio Beans` stays out of scope.)

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

### Task 6: Transcribe the Phase 2 tree and plantation crops

**No structural work is needed.** Task 1's generalized `_guide_row_key()` already routes any crop in `BEARING_AGE_BAND` through its age axis and any crop in `DEFAULT_VARIETY_ROW` through its variety axis, `AGE_BANDS` already derives itself from `GUIDE_ROWS`, and `TARGET_UNIT` already derives from each row's `unit`. This task is therefore a pure data addition — which is exactly what Tasks 1–2 were built to make possible.

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Test: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces 6 new canonical crops across 24 guide rows: Pineapple (`kg_ha`, variety axis) and Papaya, Avocado, Citrus, Guava, Lanzones (all `kg_tree`, age axis), plus 16 aliases.

#### Source anomalies that must be resolved before this task ships

Three cells cannot be read straight off the guide. Two are flagged in the `.md`'s own transcription notes; **one was found during this plan's cross-check and is not flagged anywhere else.** Each is encoded below with an explicit inference, and each needs a sign-off decision — do not let them pass silently.

| Crop / row | What the source prints | What this task encodes | Basis |
|---|---|---|---|
| **Guava**, 3rd age row | Label repeats `4-6 yrs` | `7 yrs & up` | Its values (`P .30/.20/.15/.10`, `K .60/.45/.30`) are identical to the Avocado group's `7 yrs & up` row, and ascend from Guava's own `4-6 yrs` row. Strong corroboration. |
| **Citrus**, `5 yrs` / `6-10 yrs` / `11-15 yrs` | `K 0-35` reads `.03`; `K 36-55` reads `15`; `K 0-35` reads `45`; `P 0-6.0` reads `40` | `.30`, `.15`, `.45`, `.40` | Missing decimal points. Restoring them makes each row monotonic and continues the exact progression of the surrounding age bands (`4 yrs` K `.15/.09/.03` → `5 yrs` `.30/.15/.06` → `6-10 yrs` `.45/.30/.09`). The `.md`'s note 3 flags three of these; **the `5 yrs` `K 0-35` cell (`.03` → `.30`) is a fourth, found here.** |
| **Papaya**, `1-2 yrs.` | `K` reads `.27 / .08 / .135 / .054` | **UNRESOLVED — see below** | Non-monotonic: `.08` sits below the `.135` that follows it. Not flagged in the `.md`'s notes; found during this cross-check. |

**Papaya is the one that cannot be inferred safely.** Papaya's other two age rows both follow the ratio pattern `1 : 0.7 : 0.5 : 0.2` across the K classes. Applying it to `1-2 yrs` gives `.27 / .189 / .135 / .054`, so the printed `.08` is most likely `.18` or `.19` — but "most likely" is doing real work there, and this is a potassium rate a farmer applies to a bearing papaya. **Do not guess it.**

- [ ] **Step 0: Resolve the Papaya K cell before writing any code**

Confirm the `1-2 yrs.` `K 36-55` value against a clean copy of the guide or with DA RSL. Then take **one** of these paths:

- **Confirmed value obtained** → use it in Step 3 and add the `Papaya` entries as written.
- **Cannot be confirmed in time** → omit Papaya entirely from this task (drop its `GUIDE_ROWS` entries, its `BEARING_AGE_BAND` entry, and its test cases). The other five crops are independent and ship fine without it; coverage lands at 35 canonical crops instead of 36.

Do not proceed to Step 1 until this is decided. Record the decision in the commit message.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nutrient_recommendation_guide.py`:

```python
class TestPhase2TreeAndPlantationCrops:
    """Age-banded tree crops plus Pineapple's Planted/Ratoon cycle axis.
    Every expectation reads the lowest band of each class (poorest soil), so
    a transposed column would fail loudly."""

    @pytest.mark.parametrize("crop,expected", [
        ("Pineapple", (300, 80, 180)),
        ("Avocado", (0.3, 0.3, 0.6)),
        ("Citrus", (0.5, 0.5, 0.75)),
        ("Guava", (0.05, 0.3, 0.6)),
        ("Lanzones", (0.4, 0.8, 1.2)),
    ])
    def test_poorest_soil_at_the_default_stage(self, crop, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(crop, 1.0, 3.0, 20.0 / K_CMOL_TO_PPM) == expected

    def test_pineapple_ratoon_is_a_separate_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Pineapple", 1.0, 3.0, k, variety="Ratoon") == (150, 40, 120)

    def test_pineapple_defaults_to_the_planted_row(self):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        k = 20.0 / K_CMOL_TO_PPM
        assert nutrient_requirements("Pineapple", 1.0, 3.0, k) == \
               nutrient_requirements("Pineapple", 1.0, 3.0, k, variety="Planted")

    def test_pineapple_is_the_only_per_hectare_crop_added_here(self):
        from data.nutrient_recommendation_guide import target_unit_for_crop

        assert target_unit_for_crop("Pineapple") == "kg_ha"
        for crop in ["Avocado", "Citrus", "Guava", "Lanzones"]:
            assert target_unit_for_crop(crop) == "kg_tree"

    @pytest.mark.parametrize("crop,age_band,expected", [
        ("Avocado", "NT", (0.01, 0.01, 0.015)),
        ("Avocado", "1-3 yrs", (0.1, 0.1, 0.3)),
        ("Lanzones", "NT", (0.01, 0.02, 0.08)),
        ("Guava", "1-3 yrs", (0.02, 0.1, 0.3)),
        ("Citrus", "NT", (0.01, 0.01, 0.015)),
    ])
    def test_explicit_age_bands(self, crop, age_band, expected):
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(
            crop, 1.0, 3.0, 20.0 / K_CMOL_TO_PPM, age_band=age_band
        ) == expected

    def test_citrus_five_year_row_uses_the_restored_decimals(self):
        """The source prints K 0-35 as '.03' and K 36-55 as '15'; both are
        missing decimal points. Restoring them keeps the row monotonic and
        continues the 4yrs -> 6-10yrs progression."""
        from data.nutrient_recommendation_guide import K_CMOL_TO_PPM, nutrient_requirements

        assert nutrient_requirements(
            "Citrus", 1.0, 3.0, 20.0 / K_CMOL_TO_PPM, age_band="5 yrs"
        ) == (0.15, 0.2, 0.3)
        assert nutrient_requirements(
            "Citrus", 1.0, 3.0, 45.0 / K_CMOL_TO_PPM, age_band="5 yrs"
        ) == (0.15, 0.2, 0.15)

    def test_guava_third_row_is_registered_as_seven_years_and_up(self):
        from data.nutrient_recommendation_guide import GUIDE_ROWS

        assert "Guava|7 yrs & up" in GUIDE_ROWS
        assert "Guava|4-6 yrs" in GUIDE_ROWS

    @pytest.mark.parametrize("alias,canonical", [
        ("Guyabano", "Avocado"), ("Atis", "Avocado"), ("Chico", "Avocado"),
        ("Jackfruit", "Avocado"), ("Cashew", "Avocado"),
        ("Calamansi", "Citrus"), ("Lime", "Citrus"), ("Orange", "Citrus"),
        ("Passion Fruit", "Citrus"),
        ("Rambutan", "Lanzones"), ("Mangosteen", "Lanzones"),
    ])
    def test_co_listed_species_alias_to_their_row(self, alias, canonical):
        from data.nutrient_recommendation_guide import resolve_crop

        assert resolve_crop(alias) == canonical

    def test_every_age_banded_crop_has_a_bearing_default(self):
        """resolve_age_band() falls back to BEARING_AGE_BAND when the caller
        omits age_band; a crop with age rows but no entry there would key to
        'None' and raise KeyError on every request."""
        from data.nutrient_recommendation_guide import BEARING_AGE_BAND, GUIDE_ROWS

        for crop in ["Papaya", "Avocado", "Citrus", "Guava", "Lanzones"]:
            if not any(key.startswith(f"{crop}|") for key in GUIDE_ROWS):
                continue  # Papaya may have been omitted -- see Step 0
            assert crop in BEARING_AGE_BAND
            assert f"{crop}|{BEARING_AGE_BAND[crop]}" in GUIDE_ROWS
```

If Papaya was omitted at Step 0, also drop the Papaya case from `test_every_age_banded_crop_has_a_bearing_default`'s list (the `continue` guard already handles it, so no edit is strictly required).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py::TestPhase2TreeAndPlantationCrops -v`
Expected: FAIL with `KeyError: "Unknown crop 'Pineapple'"`

- [ ] **Step 3: Add the rows**

Add to `GUIDE_ROWS`:

```python
    # --- PLANTATION / FRUIT TREES (guide pages 2-3) ---------------------
    # Pineapple's Planted/Ratoon is a crop-cycle axis, carried on the
    # variety field (same mechanism as Ginger's Improved/Local).
    "Pineapple|Planted": GuideRow(
        "kg_ha",
        om_bands=((2.0, 300), (3.5, 200), (4.5, 100), (5.5, 80), (float("inf"), 0)),
        p_bands=((6.0, 80), (10.0, 60), (15.0, 40), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 180), (55.0, 120), (250.0, 60), (300.0, 30), (float("inf"), 0)),
    ),
    "Pineapple|Ratoon": GuideRow(
        "kg_ha",
        om_bands=((2.0, 150), (3.5, 120), (4.5, 80), (5.5, 50), (float("inf"), 0)),
        p_bands=((6.0, 40), (10.0, 30), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 120), (55.0, 75), (250.0, 45), (300.0, 30), (float("inf"), 0)),
    ),
    # Papaya -- OMIT THIS BLOCK ENTIRELY if Step 0 could not confirm the
    # 1-2 yrs K 36-55 value. The placeholder below is deliberately left as
    # the source's own printed .08 so that shipping an unconfirmed guess is
    # impossible without someone editing this line on purpose.
    "Papaya|2-6 mos": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.024), (3.5, 0.016), (4.5, 0.012), (5.5, 0.006), (float("inf"), 0)),
        p_bands=((6.0, 0.048), (10.0, 0.032), (15.0, 0.024), (20.0, 0.012), (float("inf"), 0)),
        k_bands=((35.0, 0.03), (55.0, 0.021), (250.0, 0.015), (300.0, 0.006), (float("inf"), 0)),
    ),
    "Papaya|7-12 mos": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.08), (3.5, 0.05), (4.5, 0.04), (5.5, 0.02), (float("inf"), 0)),
        p_bands=((6.0, 0.16), (10.0, 0.10), (15.0, 0.075), (20.0, 0.035), (float("inf"), 0)),
        k_bands=((35.0, 0.105), (55.0, 0.09), (250.0, 0.06), (300.0, 0.024), (float("inf"), 0)),
    ),
    "Papaya|1-2 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.20), (3.5, 0.15), (4.5, 0.10), (5.5, 0.05), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.25), (15.0, 0.18), (20.0, 0.09), (float("inf"), 0)),
        # <-- Step 0's confirmed value replaces 0.08 here.
        k_bands=((35.0, 0.27), (55.0, 0.08), (250.0, 0.135), (300.0, 0.054), (float("inf"), 0)),
    ),
    "Avocado|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.01), (3.5, 0.01), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.01), (10.0, 0.01), (15.0, 0.01), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.015), (55.0, 0.015), (250.0, 0.015), (300.0, 0), (float("inf"), 0)),
    ),
    "Avocado|1-3 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.05), (4.5, 0.03), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.05), (15.0, 0.03), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.05), (300.0, 0), (float("inf"), 0)),
    ),
    "Avocado|4-6 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.20), (3.5, 0.10), (4.5, 0.06), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.08), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 0.45), (55.0, 0.30), (250.0, 0.15), (300.0, 0), (float("inf"), 0)),
    ),
    "Avocado|7 yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.20), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.15), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.01), (3.5, 0.01), (4.5, 0.005), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.01), (10.0, 0.01), (15.0, 0.008), (20.0, 0.005), (float("inf"), 0)),
        k_bands=((35.0, 0.015), (55.0, 0.009), (250.0, 0.003), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|1 yr": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.04), (3.5, 0.02), (4.5, 0.008), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.04), (10.0, 0.02), (15.0, 0.01), (20.0, 0.008), (float("inf"), 0)),
        k_bands=((35.0, 0.03), (55.0, 0.015), (250.0, 0.015), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|2 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.05), (3.5, 0.03), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.05), (10.0, 0.03), (15.0, 0.015), (20.0, 0.010), (float("inf"), 0)),
        k_bands=((35.0, 0.06), (55.0, 0.03), (250.0, 0.009), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|3 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.075), (3.5, 0.04), (4.5, 0.015), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.075), (10.0, 0.04), (15.0, 0.02), (20.0, 0.015), (float("inf"), 0)),
        k_bands=((35.0, 0.09), (55.0, 0.06), (250.0, 0.015), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|4 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.06), (4.5, 0.02), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.05), (15.0, 0.03), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.15), (55.0, 0.09), (250.0, 0.03), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|5 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.15), (3.5, 0.10), (4.5, 0.03), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.03), (float("inf"), 0)),
        # Decimal points RESTORED: the source prints '.03' and '15' for the
        # first two K classes. Both are missing a decimal -- .30/.15/.06 keeps
        # the row monotonic and continues the 4yrs (.15/.09/.03) to 6-10yrs
        # (.45/.30/.09) progression.
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.06), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|6-10 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.20), (4.5, 0.05), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.10), (20.0, 0.05), (float("inf"), 0)),
        # Decimal point RESTORED: source prints '45' for K 0-35.
        k_bands=((35.0, 0.45), (55.0, 0.30), (250.0, 0.09), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|11-15 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        # Decimal point RESTORED: source prints '40' for P 0-6.0.
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.15), (300.0, 0), (float("inf"), 0)),
    ),
    "Citrus|15 yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.50), (3.5, 0.40), (4.5, 0.20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.50), (10.0, 0.40), (15.0, 0.30), (20.0, 0.20), (float("inf"), 0)),
        k_bands=((35.0, 0.75), (55.0, 0.60), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Guava|1-3 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.02), (3.5, 0.01), (4.5, 0.005), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.05), (15.0, 0.03), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.05), (300.0, 0), (float("inf"), 0)),
    ),
    "Guava|4-6 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.04), (3.5, 0.02), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.08), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 0.45), (55.0, 0.30), (250.0, 0.15), (300.0, 0), (float("inf"), 0)),
    ),
    # Label INFERRED: the guide prints '4-6 yrs' twice. These values ascend
    # from the row above and match the Avocado group's '7 yrs & up' row
    # exactly, so this is that band.
    "Guava|7 yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.05), (3.5, 0.03), (4.5, 0.02), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.15), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Lanzones|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.01), (3.5, 0.01), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.02), (10.0, 0.02), (15.0, 0.02), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.08), (55.0, 0.05), (250.0, 0.03), (300.0, 0), (float("inf"), 0)),
    ),
    "Lanzones|1-3 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.05), (4.5, 0.03), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.03), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.09), (300.0, 0), (float("inf"), 0)),
    ),
    "Lanzones|4-6 yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.20), (3.5, 0.10), (4.5, 0.05), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.90), (55.0, 0.60), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Lanzones|7 yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.80), (10.0, 0.60), (15.0, 0.40), (20.0, 0.20), (float("inf"), 0)),
        k_bands=((35.0, 1.2), (55.0, 0.90), (250.0, 0.60), (300.0, 0), (float("inf"), 0)),
    ),
```

Extend `BEARING_AGE_BAND` — the mature/fruit-bearing row each crop falls back to when the caller omits `age_band`:

```python
BEARING_AGE_BAND: dict[str, str] = {
    "Coconut": "9yrs & up", "Cacao": "Over 7yrs", "Mango": "10yrs & up",
    "Papaya": "1-2 yrs", "Avocado": "7 yrs & up", "Citrus": "15 yrs & up",
    "Guava": "7 yrs & up", "Lanzones": "7 yrs & up",
}
```

Register Pineapple's cycle axis:

```python
VARIETIES_BY_CROP: dict[str, tuple[str, ...]] = {
    "Rice": ("Hybrid", "Inbred"),
    "Corn": ("OPV", "Hybrid"),
    "Ginger": ("Improved", "Local"),
    "Onion": ("Bulb", "Spring"),
    "Pineapple": ("Planted", "Ratoon"),
}
```

```python
DEFAULT_VARIETY_ROW: dict[str, str] = {
    "Corn": "Native",
    "Ginger": "Improved",
    "Onion": "Bulb",
    "Pineapple": "Planted",
}
```

And extend `CROP_ALIASES` with the co-listed species:

```python
    # Fruit-tree groups: every species below shares its canonical crop's row.
    "Chico": "Avocado",
    "Atis": "Avocado",
    "Guyabano": "Avocado",
    "Apple": "Avocado",
    "Cashew": "Avocado",
    "Jackfruit": "Avocado",
    "Macopa": "Avocado",
    "Tambis": "Avocado",
    "Balimbing": "Avocado",
    "Lime": "Citrus",
    "Lemon": "Citrus",
    "Orange": "Citrus",
    "Calamansi": "Citrus",
    "Passion Fruit": "Citrus",
    "Rambutan": "Lanzones",
    "Mangosteen": "Lanzones",
```

> The guide prints `Calamnsi`; that is a typo in the source. Register the correct spelling `Calamansi` — a farmer-facing dropdown should not reproduce it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "feat: add Pineapple and four fruit-tree guide row groups

Records two source-document inferences: Guava's duplicated age label read
as '7 yrs & up', and four missing decimal points restored in the Citrus
rows. Papaya: <state the Step 0 decision here>."
```

---

### Task 7: Refuse model-path predictions for crops with no training data

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

### Task 8: Sync suri's crop lists

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
  Pineapple: ["Planted", "Ratoon"],
};
```

If that file also carries an age-band list mirroring the backend's, extend it with the Phase 2 bands (`Papaya`: `2-6 mos` / `7-12 mos` / `1-2 yrs`; `Avocado`, `Guava`, `Lanzones`: `NT` / `1-3 yrs` / `4-6 yrs` / `7 yrs & up`; `Citrus`: `NT` / `1 yr` / `2 yrs` / `3 yrs` / `4 yrs` / `5 yrs` / `6-10 yrs` / `11-15 yrs` / `15 yrs & up`). Confirm against the backend with:

```bash
python -c "from ml.preprocessing import AGE_BANDS; print(AGE_BANDS)"
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
  { value: "Pineapple", label: "Pineapple (Pinya)" },
  { value: "Papaya", label: "Papaya" },
  { value: "Avocado", label: "Avocado" },
  { value: "Chico", label: "Chico" },
  { value: "Atis", label: "Atis" },
  { value: "Guyabano", label: "Guyabano" },
  { value: "Apple", label: "Apple" },
  { value: "Cashew", label: "Cashew (Kasoy)" },
  { value: "Jackfruit", label: "Jackfruit (Langka)" },
  { value: "Macopa", label: "Macopa" },
  { value: "Tambis", label: "Tambis" },
  { value: "Balimbing", label: "Balimbing" },
  { value: "Citrus", label: "Citrus" },
  { value: "Lime", label: "Lime (Dayap)" },
  { value: "Lemon", label: "Lemon" },
  { value: "Orange", label: "Orange" },
  { value: "Calamansi", label: "Calamansi" },
  { value: "Passion Fruit", label: "Passion Fruit" },
  { value: "Guava", label: "Guava (Bayabas)" },
  { value: "Lanzones", label: "Lanzones" },
  { value: "Rambutan", label: "Rambutan" },
  { value: "Mangosteen", label: "Mangosteen" },
```

Omit the `Papaya` entry if it was dropped at Task 6 Step 0. The sync check in Step 4 will catch it either way.

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

### Task 9: Retrain and promote

**Why this is required, not optional:** the preprocessor's one-hot vocabulary is captured inside the pickled artifact. The currently-pinned artifact (`ml/artifacts/2026-08-06-055755/`) encodes 9 crops and 4 varieties. After Tasks 1–6 the code offers 36 crops and 10 varieties, so **the live artifact and the running code now disagree** — a request for any new crop reaches `transform()` and raises `ValueError`, surfacing as a 422. Retraining rebuilds the preprocessor against the new vocabulary.

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

Expected: `crops: 36 varieties: 10` — or `crops: 35` if Papaya was omitted at Task 6 Step 0.

> Counts explained: 9 original + 21 Phase 1 + 6 Phase 2 = 36 canonical crops (aliases do **not** get their own column — they resolve to a canonical name before featurizing, which is the whole point of Task 2). Varieties are `None` plus the union of `VARIETIES_BY_CROP`: Hybrid, Inbred, OPV, Improved, Local, Bulb, Spring, Planted, Ratoon = 10.

> The by-crop metrics will still cover only the 9 crops that have training rows — that is expected and is exactly what Task 7's guard keys off.

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

```bash
# A Phase 2 tree crop: per-tree unit, and the bearing age band by default
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"crop":"Lanzones","data_source":"lab","ph":6.4,"organic_matter_pct":1.5,
       "phosphorus_ppm":4.0,"potassium_cmol_kg":0.05}' | python -m json.tool
```
Expected: HTTP 200, `"target_unit": "kg_tree"`, `target_n: 0.4`, `target_p2o5: 0.8`, `target_k2o: 1.2`, and `farm_totals: null` (per-tree rates need a tree count, which this API doesn't collect).

```bash
# Pineapple's crop-cycle axis rides on `variety`
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"crop":"Pineapple","variety":"Ratoon","data_source":"lab","ph":6.4,
       "organic_matter_pct":1.5,"phosphorus_ppm":4.0,"potassium_cmol_kg":0.05}' \
  | python -m json.tool
```
Expected: HTTP 200, `target_n: 150` (the Ratoon row); omitting `variety` gives `300` from the Planted row.

- [ ] **Step 6: Commit and promote**

```bash
git add .gitignore "ml/artifacts/<new-version>"
git commit -m "feat: retrain against the expanded 36-crop vocabulary"
```

If Plan A is already deployed, promote through the admin UI's Model Training → Version History tab (this doubles as a live rehearsal of the promotion flow). Otherwise push to `main` and let the existing Actions deploy carry it.

---

## Deliberately Deferred

**Six of the nine age-banded crops are now in the plan** as Task 6 — Pineapple, Papaya, Avocado, Citrus, Guava, and Lanzones. Task 1's generalized row-key mechanism meant they needed no structural work, only data, and the full transcription in `Nutrient Recommendation Guide.md` supplied values clean enough to use (with the inferences that task documents). Papaya carries one unresolved cell and may be dropped at Task 6 Step 0, which would land coverage at 35 rather than 36.

**Three remain out of scope**, blocked on anomalies *in the printed document* that no better scan fixes:
- **Coffee `6-10 yrs`** — K reads `.36 / .12 / .60`, non-monotonic where every other row decreases across K classes.
- **Dragon Fruit** (and Basil) — all columns but the first are blank in the original, so no bands can be constructed at all.
- **Abaca** — K reads `.045 / .03 / .015 / .05`, with the `251-300` value above the band below it. (Not flagged in the `.md`'s notes; caught during this plan's cross-check.)

These three need an authoritative answer from DA RSL, not a re-scan. Adding them afterwards is pure data — Abaca's `Pulp`/`Cordage` cycle axis is handled by the same `DEFAULT_VARIETY_ROW` mechanism Pineapple uses, and Coffee's and Dragon Fruit's age bands by the same `BEARING_AGE_BAND` mechanism the other tree crops use. That would take coverage from 36 to **39**.

This isn't blocked by a lack of real data, either — see "Real Data Availability" above: the Region 7 database has 16 real Coffee rows and 15 real Dragon Fruit rows waiting, once DA RSL resolves the guide cells and (separately) `normalize_crop()` is extended to recognize them.

Everything else — the pipeline, the alias mechanism, the untrained-crop guard — already supports them the moment their rows exist.

### Smaller findings from the full transcription, not acted on here

- **`Adlai` appears only on Corn's *Hybrid* row** (`Corn /Sorghum/Adlai`), while `Sorghum` is on both Native and Hybrid. Aliases map whole-crop, so `Adlai → Corn` would resolve to the Native default — subtly wrong. Adlai is therefore **not** registered in Task 5. Supporting it properly needs a per-crop default-row override, which is more machinery than one species justifies today.
- **Banana has a second guide row, `Pulp`**, alongside the `Fruit` row the code uses. `Fruit` remains the correct default for a farmer-facing tool; recorded only so a future reader doesn't think the row was missed.
- **`kg_ha` vs `kg_tree` is an inference, not printed.** The guide states units in exactly two places — Roses (`g/plant`) and Banana (`576 trees/ha`). Everything else is read as kg/ha for whole-number rows and kg/tree for decimal rows. The existing code already relies on this; the `.md`'s note 8 is now the written justification for it.
- **Whole-number `kg_ha` rows exist for several excluded crops** (Castor Oil, Cotton, Jute/Kenaf, Ramie/Rattan, Tangan-Tangan, Tobacco Virginia, Pasture Grasses, Legume Green Manure, Mint Herb). They remain out of scope as non-Region-7-farmer crops, but they are cleanly transcribable if that judgment ever changes.

**One caveat carried forward:** the `P_METHOD = "olsen"` assumption applies to every row added here. The `.md` confirms the guide's P columns are dual-labeled (Bray #1 on top, Olsen beneath) with identical rates — only the band boundaries differ by method. If a Region 7 lab confirms Bray #1, every `p_bands` boundary in the file shifts to the Bray labels: a single documented constant, but a file-wide edit.


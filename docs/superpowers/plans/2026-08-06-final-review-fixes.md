# Final Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 3 Critical + 5 Important findings from the final whole-branch review of the
real-soil-data retraining branch (`docs/superpowers/specs/2026-08-06-real-soil-data-retraining-final-review-findings.md`),
so the ML backend is genuinely safe to serve.

**Architecture:** Two of the three Critical findings collapse into one change: add a completeness
router to `main.py`'s `/predict` — when a request's `organic_matter_pct`/`phosphorus_ppm`/
`potassium_cmol_kg` are all present, call `data/nutrient_recommendation_guide.py`'s
`nutrient_requirements()` directly (exact, no training-coverage dependency) instead of the trained
model; otherwise keep using the model exactly as today. Fertilizer-bag formulation
(`app1_formulation`/`app2_formulation`) stops being a trained classifier entirely and becomes a
deterministic function of `(n, p2o5, k2o, target_unit)` — this fixes the kg/ha-hardcoded-regardless-of-unit
bug at its root and removes the corrupted classifier training labels as a concern, for either
serving path. The remaining findings (a broken migration, an ungated resolver, a missing test, a
test suite writing to production, stale docs) are independent, smaller fixes.

**Tech Stack:** Python, pandas, scikit-learn, FastAPI, pytest, PostgreSQL/Supabase.

## Global Constraints

- Findings doc (source of truth for every reproduction/number cited below):
  `docs/superpowers/specs/2026-08-06-real-soil-data-retraining-final-review-findings.md`.
- Work directly on `master` — no worktree (matches how the rest of this session's work was done,
  with the user's explicit consent).
- **Router completeness rule**: a request is "complete" when `organic_matter_pct`,
  `phosphorus_ppm`, and `potassium_cmol_kg` are all not `None`. This is deliberately the same rule
  `data/parse_real_soil_database.py`'s `build_training_row()` already uses to decide whether a row
  is trainable (`if om is None or p is None or k is None: return None`) — the router and the
  training-inclusion rule become the same check, not two independently-maintained ones.
- **`plan_applications()` gains a required `target_unit` parameter** (`"kg_ha"` or `"kg_tree"`,
  from `data.nutrient_recommendation_guide.target_unit_for_crop`) and drops its trained-classifier
  role entirely — every call site (parser, fixtures, and the new direct call in `main.py`) passes
  it explicitly. `kg_tree` formulations are never rounded to whole `int` (real amounts are sub-1)
  and are labeled `"kg/tree"`, not `"kg/ha"`.
- **The classifier heads (`clf_app1`, `clf_app2`, their `LabelEncoder`s, `recipe_vocabulary`,
  `classification_metrics`) are removed from `ml/train.py`'s output and `main.py`'s loaded
  artifacts entirely** — not kept alongside the new deterministic path. They were trained on
  labels `plan_applications()` itself produced with the unit bug already baked in; keeping them
  around unused is dead, misleading code.
- **`PredictionResponse.source`** becomes `"table-lookup"` or `"ml-ensemble"`, set by which path
  actually answered the request — not hardcoded.
- Every existing passing test must keep passing unless this plan explicitly says its expected
  value changes (and says what the new value is and why).
- No placeholder code, no `TODO`s. Every step's code is the actual, final code for that step.

---

## File Structure

Modified files:
- `ml/recipes.py` — `plan_applications()` gains `target_unit`; docstring's stale SCC-417 claim corrected.
- `data/parse_real_soil_database.py` — its `plan_applications()` call passes `target_unit`.
- `tests/fixtures.py` — same.
- `tests/test_recipes.py` — updated for the new signature and unit-aware formatting; stale `TestGoldenRecordSCC417` naming/docstring corrected.
- `ml/train.py` — classifier-head training/artifacts removed.
- `tests/test_train.py` — updated for the smaller artifact/metadata shape.
- `main.py` — completeness router, direct `plan_applications()` call, classifier-artifact loading removed, `source` varies.
- `tests/test_api.py` — new router-behavior tests; `log_prediction` monkeypatched for the whole file (Important #7).
- `supabase/migrations/20260806_align_ml_training_schema.sql` — synthetic-row delete step added; K-unit handling made explicit.
- `data/nutrient_recommendation_guide.py` — `resolve_variety()` crop-gated; dead `"Coffee|Over 15yrs"` `GUIDE_ROWS` entry removed.
- `tests/test_nutrient_recommendation_guide.py` — new crop-gating tests; `TestGoldenRecordRiceInbred`'s surrounding context updated to note the Coffee cross-check was abandoned (not silently dropped).
- `ml/remarks.py`, `data/ingest_to_supabase.py`, `ml/model.py` — stale docstrings corrected.
- `docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md` — the Global Constraint promising a Coffee golden-record test amended to state why it was abandoned.
- `README.md` — full rewrite against the current API contract.

No new files.

---

### Task 1: `ml/recipes.py` — unit-aware, non-classifier formulation

**Files:**
- Modify: `ml/recipes.py` (full rewrite)
- Modify: `data/parse_real_soil_database.py:23,276`
- Modify: `tests/fixtures.py:21,53`
- Modify: `tests/test_recipes.py` (full rewrite)

**Interfaces:**
- Produces: `plan_applications(n: float, p2o5: float, k2o: float, target_unit: str) -> tuple[str, str]`
  (signature change: gains `target_unit`, `n`/`p2o5`/`k2o` typed `float` not `int` — matches what
  every real caller already passes since `nutrient_requirements()` returns floats). Consumed by
  Task 3 (`main.py`, new direct call replacing the classifier heads).
- Consumes: nothing new.

**Context:** `_format_terms`'s zero-threshold (`int(round(kg)) > 0`) silently drops every `kg_tree`
term, since real `kg_tree` amounts are sub-1 and round to `0`. The single-vs-double-application
thresholds (`_SINGLE_APPLICATION_MAX_N=30` etc.) are calibrated to the `kg_ha` scale (~20-300) and
don't transfer to `kg_tree` (~0.003-2) — every `kg_tree` request would trivially satisfy all three
and always get a single application, which doesn't match real guidance
(`data/mode_of_application.py`'s text for every one of the four `kg_tree` crops describes at least
two applications per year regardless of amount). `kg_tree` therefore always splits into two.

- [ ] **Step 1: Replace the tests first**

Replace `tests/test_recipes.py` in full:

```python
import pytest
from ml.recipes import plan_applications, NO_APPLICATION


class TestKgHaFormatting:
    """Working backward from real DA-report-style numbers must reproduce the
    same kind of application strings this system has always produced for
    field crops — the kg_ha formatting itself hasn't changed, only that it's
    now explicit about which unit it's formatting for."""

    def test_reproduces_expected_kg_ha_formulation(self):
        app1, app2 = plan_applications(50, 40, 60, "kg_ha")
        expected = "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)"
        assert app1 == expected
        assert app2 == expected

    def test_terms_are_joined_with_plus(self):
        app1, _ = plan_applications(90, 60, 60, "kg_ha")
        assert " + " in app1

    def test_each_term_has_kg_ha_and_grade(self):
        app1, _ = plan_applications(90, 60, 60, "kg_ha")
        for term in app1.split(" + "):
            assert " kg/ha (" in term
            assert term.endswith(")")

    def test_zero_quantity_terms_are_omitted(self):
        app1, _ = plan_applications(20, 150, 20, "kg_ha")
        assert "21-0-0" not in app1
        assert "0-0-60" not in app1
        assert "14-14-14" in app1

    def test_low_requirement_gives_no_second_application(self):
        _, app2 = plan_applications(20, 20, 20, "kg_ha")
        assert app2 == NO_APPLICATION

    def test_high_requirement_gives_a_second_application(self):
        _, app2 = plan_applications(120, 90, 90, "kg_ha")
        assert app2 != NO_APPLICATION

    def test_quantities_are_whole_numbers(self):
        app1, _ = plan_applications(93, 61, 57, "kg_ha")
        for term in app1.split(" + "):
            quantity = term.split(" kg/ha")[0]
            assert quantity.isdigit()


class TestKgTreeFormatting:
    """The regression this class exists to catch: kg_tree amounts are
    legitimately sub-1 and were previously formatted as '0 kg/ha (...)' and
    dropped entirely (rounding to int(round(kg)) == 0), or — worse, at
    serving time via the old trained classifier — shown with the wrong unit
    label at a wildly wrong scale."""

    def test_sub_one_amount_is_not_dropped(self):
        app1, _ = plan_applications(0.947, 1.166, 2.111, "kg_tree")
        assert app1 != NO_APPLICATION
        assert "14-14-14" in app1

    def test_uses_kg_tree_label_not_kg_ha(self):
        app1, _ = plan_applications(0.947, 1.166, 2.111, "kg_tree")
        assert "kg/tree" in app1
        assert "kg/ha" not in app1

    def test_kg_tree_always_gets_a_second_application(self):
        """Every kg_tree crop's Mode of Application guidance describes at
        least two splits per year regardless of amount — the kg_ha
        single-application thresholds (calibrated to a ~20-300 scale) would
        otherwise always trivially apply at kg_tree's ~0.003-2 scale and
        wrongly collapse every kg_tree recommendation to one application."""
        _, app2 = plan_applications(0.06, 0.05, 0.15, "kg_tree")
        assert app2 != NO_APPLICATION

    def test_tiny_amount_is_not_rounded_to_zero_and_dropped(self):
        """0.003 kg/tree (the smallest real target in this system, young
        Cacao) must still appear as a non-zero term, not vanish the way
        int(round(0.003)) == 0 would have dropped it."""
        app1, _ = plan_applications(0.003, 0.002, 0.001, "kg_tree")
        assert app1 != NO_APPLICATION


class TestVocabularyStability:
    def test_same_input_gives_same_output(self):
        assert plan_applications(90, 60, 60, "kg_ha") == plan_applications(90, 60, 60, "kg_ha")

    def test_vocabulary_is_finite_over_the_kg_ha_target_space(self):
        seen = set()
        for n in range(20, 201, 10):
            for p in range(20, 151, 10):
                for k in range(20, 201, 10):
                    seen.update(plan_applications(n, p, k, "kg_ha"))
        assert len(seen) < 5000
        assert all(isinstance(label, str) and label for label in seen)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_recipes.py -v`
Expected: FAIL — `plan_applications()` doesn't accept a 4th positional argument yet.

- [ ] **Step 3: Implement**

Replace `ml/recipes.py` in full:

```python
"""Fertilizer product catalog and application planner.

Deterministic — not a trained classifier. Converts a known nutrient target
into product-bag amounts by covering the P2O5 requirement with a Complete
(14-14-14) base, then topping up nitrogen with Ammosul and potassium with
Muriate of Potash, splitting the total across a basal application at
planting and a topdress one month later — the structure of a real DA
Fertilizer Recommendation Report. This file's own docstring previously
claimed provenance from a specific real report (SCC-26-417-417); that
record was traced to a document outside this system's real dataset and
retired (see docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md's
"Golden test — retired" section) — the arithmetic here doesn't depend on
that record, so the claim is removed rather than corrected to a new one.
"""

from __future__ import annotations

NO_APPLICATION = "None"

# Standard Philippine fertilizer grades as (N%, P2O5%, K2O%).
COMPLETE = ("14-14-14", 0.14, 0.14, 0.14)
AMMOSUL = ("21-0-0", 0.21, 0.0, 0.0)
MOP = ("0-0-60", 0.0, 0.0, 0.60)

# Below these combined kg/ha requirements a single basal application is
# sufficient. Calibrated to the kg_ha scale only -- see plan_applications
# for why kg_tree always splits into two instead of using a scaled version
# of these.
_SINGLE_APPLICATION_MAX_N = 30
_SINGLE_APPLICATION_MAX_P = 25
_SINGLE_APPLICATION_MAX_K = 30

_NUM_APPLICATIONS = 2

_UNIT_LABEL = {"kg_ha": "kg/ha", "kg_tree": "kg/tree"}


def _format_terms(terms: list[tuple[str, float]], target_unit: str) -> str:
    """Render [(grade, kg)] as '143 kg/ha (14-14-14) + 24 kg/ha (21-0-0)' for
    kg_ha, or '0.947 kg/tree (14-14-14)' for kg_tree -- kg_ha amounts round
    to whole numbers (matching the real DA report format this mirrors);
    kg_tree amounts stay at 3-decimal precision (matching
    nutrient_requirements()'s own rounding) since real kg_tree amounts are
    legitimately sub-1 and a whole-number round would drop every one of
    them. Terms are dropped only when they round to exactly zero at their
    own unit's precision, not the other unit's."""
    label = _UNIT_LABEL[target_unit]
    parts = []
    for grade, kg in terms:
        if target_unit == "kg_ha":
            amount = int(round(kg))
        else:
            amount = round(kg, 3)
        if amount > 0:
            parts.append(f"{amount} {label} ({grade})")
    return " + ".join(parts) if parts else NO_APPLICATION


def plan_applications(n: float, p2o5: float, k2o: float, target_unit: str) -> tuple[str, str]:
    """Return (app1_formulation, app2_formulation) for the given nutrient
    targets. target_unit ("kg_ha" or "kg_tree", from
    data.nutrient_recommendation_guide.target_unit_for_crop) controls both
    the formatted unit label and the application-count logic."""
    complete_grade, complete_n, _, complete_k = COMPLETE

    kg_complete = p2o5 / COMPLETE[2]
    n_supplied = kg_complete * complete_n
    k_supplied = kg_complete * complete_k

    kg_ammosul = max(0.0, n - n_supplied) / AMMOSUL[1]
    kg_mop = max(0.0, k2o - k_supplied) / MOP[3]

    if target_unit == "kg_ha":
        single_application = (
            n <= _SINGLE_APPLICATION_MAX_N
            and p2o5 <= _SINGLE_APPLICATION_MAX_P
            and k2o <= _SINGLE_APPLICATION_MAX_K
        )
    else:
        # kg_tree crops (Coconut/Cacao/Mango/Banana): every crop's Mode of
        # Application guidance (data/mode_of_application.py) describes at
        # least two applications per year regardless of amount -- the
        # kg_ha thresholds above don't transfer to this scale (every real
        # kg_tree value is far under all three), so always split.
        single_application = False
    splits = 1 if single_application else _NUM_APPLICATIONS

    per_application = [
        (complete_grade, kg_complete / splits),
        (AMMOSUL[0], kg_ammosul / splits),
        (MOP[0], kg_mop / splits),
    ]
    formulation = _format_terms(per_application, target_unit)

    if single_application:
        return formulation, NO_APPLICATION
    return formulation, formulation
```

In `data/parse_real_soil_database.py`, update the import and call site. Find:
```python
from ml.recipes import plan_applications
```
(unchanged — same import). Find the call around line 276:
```python
    app1, app2 = plan_applications(n, p2o5, k2o)
```
Replace with:
```python
    app1, app2 = plan_applications(n, p2o5, k2o, target_unit_for_crop(crop))
```
(`target_unit_for_crop` is already imported in this file from Task 1 of the original retraining plan — confirm the import line already reads
`from data.nutrient_recommendation_guide import (k_cmol_to_ppm, target_unit_for_crop, resolve_variety, resolve_cultivation_method, resolve_age_band)`
and add `target_unit_for_crop` to it if it isn't already there.)

In `tests/fixtures.py`, update the call site around line 53:
```python
        n, p2o5, k2o = nutrient_requirements(crop, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o)
```
Replace with:
```python
        n, p2o5, k2o = nutrient_requirements(crop, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o, target_unit_for_crop(crop))
```
(`target_unit_for_crop` is already imported in this file from
`data.nutrient_recommendation_guide` — confirm and add to the import line if not already present.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_recipes.py -v`
Expected: PASS, all tests.

Run: `pytest tests/test_parse_real_soil_database.py tests/test_preprocessing.py tests/test_train.py -v`
Expected: PASS — these exercise `tests/fixtures.py` and the parser, both of which now call
`plan_applications` with the new signature.

- [ ] **Step 5: Commit**

```bash
git add ml/recipes.py data/parse_real_soil_database.py tests/fixtures.py tests/test_recipes.py
git commit -m "fix: make plan_applications unit-aware, stop rounding kg_tree amounts to zero"
```

---

### Task 2: `ml/train.py` — remove the classifier heads

**Files:**
- Modify: `ml/train.py` (full rewrite)
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train()`'s returned metadata has no `classification_metrics` or `recipe_vocabulary`
  keys. `train()` no longer writes `clf_app1.joblib`, `clf_app2.joblib`, or
  `label_encoders.joblib`. Consumed by Task 3 (`main.py` stops loading those files).

**Context:** these classifiers were trained to predict `app1_formulation`/`app2_formulation`
strings — labels that `plan_applications()` itself produces deterministically from
`(n, p2o5, k2o, target_unit)` (Task 1). A classifier trained to reproduce a deterministic
function's output adds nothing a direct call doesn't already give exactly, and — per the findings
doc — was trained on labels that had the unit bug baked in for every `kg_tree` row. Task 3 replaces
the served formulation with a direct `plan_applications()` call; this task stops training and
persisting the now-unused classifiers so the artifact directory doesn't carry dead files.

- [ ] **Step 1: Update the tests first**

In `tests/test_train.py`, remove `test_writes_all_artifacts`'s classifier-file assertions and any
test referencing `classification_metrics`/`recipe_vocabulary`. Replace the whole file's `TestTrain`
class with:

```python
class TestTrain:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        output = tmp_path_factory.mktemp("artifacts")
        return train(sample_training_frame(num_records=200, seed=42), output_dir=str(output)), output

    def test_writes_all_artifacts(self, result):
        _, output = result
        for name in ["preprocessor.joblib", "regressor.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_no_longer_writes_classifier_artifacts(self, result):
        """The classifier heads were removed -- app1/app2_formulation is now
        computed directly by ml.recipes.plan_applications(), not predicted.
        This test exists so a future change can't silently reintroduce the
        classifier-training path without a deliberate decision to do so."""
        _, output = result
        for name in ["clf_app1.joblib", "clf_app2.joblib", "label_encoders.joblib"]:
            assert not (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 200
        assert "trained_at" in written
        assert "version" in written

    def test_metadata_has_no_classification_keys(self, result):
        metadata, _ = result
        assert "classification_metrics" not in metadata
        assert "recipe_vocabulary" not in metadata

    def test_reports_metrics_per_data_source(self, result):
        metadata, _ = result
        assert set(metadata["metrics"]["by_source"].keys()) == {"lab"}

    def test_reports_metrics_per_crop(self, result):
        metadata, _ = result
        from data.nutrient_recommendation_guide import CROPS
        assert set(metadata["metrics"]["by_crop"].keys()) <= set(CROPS)
        assert len(metadata["metrics"]["by_crop"]) > 0

    def test_beats_the_mean_baseline_on_every_target(self, result):
        metadata, _ = result
        for target, scores in metadata["metrics"]["overall"].items():
            assert scores["mae"] < scores["dummy_mae"], target

    def test_no_golden_test_key(self, result):
        metadata, _ = result
        assert "golden_test" not in metadata

    def test_predictions_are_non_negative(self, result):
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

(Keep `TestGroupedSplit` and `TestCvWeights` from the existing file unchanged — this task doesn't
touch either.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `test_no_longer_writes_classifier_artifacts` and `test_metadata_has_no_classification_keys`
fail because `train()` still writes those files/keys.

- [ ] **Step 3: Implement**

In `ml/train.py`, remove the `CLASSIFIER_TARGETS` constant (line 35) and the `from sklearn.metrics
import accuracy_score, f1_score` names (keep `mean_absolute_error, r2_score`) and the `from
sklearn.preprocessing import LabelEncoder` import. Remove the entire classifier-training block
inside `train()` (the `# Classification heads.` comment through `classification_metrics[target] =
{...}` loop) and its two `joblib.dump` calls for `clf_app1`/`clf_app2`/`label_encoders`. Remove
`"classification_metrics"` and `"recipe_vocabulary"` from the `metadata` dict. The corrected
`train()` function:

```python
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

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_rows": len(frame),
        "n_records": int(frame[GROUP_COLUMN].nunique()),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "voting_weights": [{"xgb": xgb, "krr": krr} for xgb, krr in weights],
        "metrics": metrics,
    }

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(regressor, os.path.join(output_dir, "regressor.joblib"))
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata
```

`_print_summary` needs no changes (it never referenced classification metrics).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "refactor: remove the classifier heads for app1/app2_formulation, now computed directly"
```

---

### Task 3: `main.py` — completeness router, direct formulation, `source` varies

**Files:**
- Modify: `main.py` (full rewrite)
- Modify: `tests/test_api.py` (full rewrite — also folds in Important #7's `log_prediction` fix)

**Interfaces:**
- Consumes: `nutrient_requirements` (Task 1 of the original retraining plan,
  `data.nutrient_recommendation_guide`); `plan_applications(n, p2o5, k2o, target_unit)` (Task 1);
  the trained artifacts no longer include `clf_app1`/`clf_app2`/`encoders` (Task 2).
- Produces: `PredictionResponse.source` is `"table-lookup"` or `"ml-ensemble"`.

**Context — this is the fix for Critical #1 and Critical #2 together.** A request is "complete"
when `organic_matter_pct`, `phosphorus_ppm`, and `potassium_cmol_kg` are all present — the same
rule the parser already uses to decide if a row is trainable. Complete requests get the exact
guide-table answer regardless of whether the specific `(crop, variety, cultivation_method,
age_band)` combination has any training coverage at all, since it's a computation, not a learned
pattern. Incomplete requests (the genuine `stk`/`iot`/partial-`lab` case) keep using the trained
model exactly as before. Either way, `plan_applications()` is now called directly with whatever
`n`/`p2o5`/`k2o` the chosen path produced — formulation is never itself model-predicted.

- [ ] **Step 1: Replace the API tests first**

Replace `tests/test_api.py` in full:

```python
import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(autouse=True)
def _no_real_supabase_writes(monkeypatch):
    """Every /predict call in this file would otherwise insert a real row
    into the live ml_fertilizer_predictions table in production Supabase --
    confirmed happening (HTTP 201) before this fixture existed. Autouse so
    every test in this file is covered without each one remembering to opt
    in. TestLoggingIsNonFatal overrides this per-test (see below) since it
    specifically needs log_prediction to raise, not silently succeed."""
    monkeypatch.setattr(service, "log_prediction", lambda *args, **kwargs: None)


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
        assert body["target_unit"] == "kg_ha"

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["ph_remark"] == "SLIGHTLY ACIDIC"
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"
        assert body["derived_remarks"]["p_remark"] == "HIGH"
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_k_remark_uses_the_converted_ppm_value_not_the_raw_cmol_kg(self, client):
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
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert body["farm_totals"] is None

    def test_minimal_payload_is_accepted(self, client):
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


class TestCompletenessRouter:
    """The fix for Critical #1/#2: a request with complete OM/P/K is
    answered by the exact guide table, not the model -- regardless of
    whether the specific crop/variety/cultivation_method/age_band
    combination has any training coverage."""

    def test_complete_request_reports_table_lookup_source(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["source"] == "table-lookup"

    def test_incomplete_request_reports_ml_ensemble_source(self, client):
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "phosphorus_ppm"}
        body = client.post("/predict", json=payload).json()
        assert body["source"] == "ml-ensemble"

    def test_complete_request_matches_nutrient_requirements_exactly(self, client):
        """The whole point of the router: for a complete request, /predict's
        answer must be bit-for-bit what the deterministic table says, not a
        model approximation of it."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        expected_n, expected_p, expected_k = nutrient_requirements(
            "Corn", om=2.8, p_ppm=18.0, k_cmol_kg=0.35,
        )
        assert body["target_n"] == pytest.approx(expected_n)
        assert body["target_p2o5"] == pytest.approx(expected_p)
        assert body["target_k2o"] == pytest.approx(expected_k)

    def test_uncovered_cultivation_method_is_still_exact(self, client):
        """Before this fix: Rice + Upland (near-zero real training coverage)
        returned a model guess far from the guide's actual answer. Now it
        must match the table exactly, the same as any other combination."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "cultivation_method": "Upland",
                   "organic_matter_pct": 1.5, "phosphorus_ppm": 5.0, "potassium_cmol_kg": 0.1}
        body = client.post("/predict", json=payload).json()
        expected_n, _, _ = nutrient_requirements(
            "Rice", om=1.5, p_ppm=5.0, k_cmol_kg=0.1, cultivation_method="Upland",
        )
        assert body["target_n"] == pytest.approx(expected_n)

    def test_uncovered_age_band_is_still_exact(self, client):
        """Before this fix: Coconut + Below 1yr (zero real training
        coverage) returned a model guess ~18x the guide's actual answer."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "Below 1yr",
                   "organic_matter_pct": 1.5, "phosphorus_ppm": 5.0, "potassium_cmol_kg": 0.1}
        body = client.post("/predict", json=payload).json()
        expected_n, _, _ = nutrient_requirements(
            "Coconut", om=1.5, p_ppm=5.0, k_cmol_kg=0.1, age_band="Below 1yr",
        )
        assert body["target_n"] == pytest.approx(expected_n)

    def test_kg_tree_formulation_uses_the_correct_unit_and_scale(self, client):
        """Before this fix: a Coconut request's app1_formulation was a
        classifier-predicted string like '214 kg/ha (14-14-14) + 71 kg/ha
        (21-0-0)' -- kg/ha, at field-crop scale, for a kg_tree crop. Now the
        formulation is computed directly from the same exact target the
        table just produced."""
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert "kg/tree" in body["app1_formulation"]
        assert "kg/ha" not in body["app1_formulation"]


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
        response = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Corn" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a
        recommendation. This overrides the module's autouse no-op fixture
        with a raising one, specifically to prove /predict survives it."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `main.py` still always calls the model and always sets `source="ml-ensemble"`;
`TestCompletenessRouter` fails throughout, and `load_artifacts()` still tries to load
`clf_app1.joblib` (Task 2 stopped writing it, so this will actually raise `FileNotFoundError` at
import time against a freshly-trained artifact directory — if testing against an older,
pre-Task-2 artifact directory it will import fine but the router tests still fail).

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
    CROPS, k_cmol_to_ppm, nutrient_requirements, resolve_age_band,
    resolve_cultivation_method, resolve_variety, target_unit_for_crop,
)
from ml.agronomy import total_nitrogen_from_om
from ml.preprocessing import FEATURE_COLUMNS
from ml.recipes import plan_applications
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

    def is_complete(self) -> bool:
        """A request nutrient_requirements() can answer exactly -- the same
        completeness rule data/parse_real_soil_database.py's
        build_training_row() uses to decide whether a row is trainable."""
        return (
            self.organic_matter_pct is not None
            and self.phosphorus_ppm is not None
            and self.potassium_cmol_kg is not None
        )


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
    source: str


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

    if request.is_complete():
        # Exact: the guide table already has the right answer for any
        # (crop, variety, cultivation_method, age_band) combination,
        # regardless of whether the model was ever trained on it.
        n, p2o5, k2o = nutrient_requirements(
            request.crop, request.organic_matter_pct, request.phosphorus_ppm,
            request.potassium_cmol_kg, variety=request.variety,
            cultivation_method=request.cultivation_method, age_band=request.age_band,
        )
        source = "table-lookup"
    else:
        try:
            X = ARTIFACTS["preprocessor"].transform(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        n, p2o5, k2o = ARTIFACTS["regressor"].predict(X)[0]
        n, p2o5, k2o = float(n), float(p2o5), float(k2o)
        source = "ml-ensemble"

    n, p2o5, k2o = round(n, 3), round(p2o5, 3), round(k2o, 3)
    target_unit = target_unit_for_crop(request.crop)
    app1, app2 = plan_applications(n, p2o5, k2o, target_unit)

    farm_totals = None
    if request.farm_area_sqm and target_unit == "kg_ha":
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
        app1_formulation=app1,
        app2_formulation=app2,
        application_schedule=application_schedule_for_crop(request.crop),
        derived_remarks={
            "ph_remark": features.at[0, "ph_remark"],
            "om_remark": features.at[0, "om_remark"],
            "p_remark": features.at[0, "p_remark"],
            "k_remark": features.at[0, "k_remark"],
        },
        farm_totals=farm_totals,
        model_version=ARTIFACTS["version"],
        source=source,
    )

    # An audit-log outage must never deny a farmer their recommendation.
    try:
        log_prediction(request, response)
    except Exception as error:
        logger.warning("Prediction logging failed (returning result anyway): %s", error)

    return response
```

Note `_build_feature_row` is still called for the table-lookup path too — it's still needed to
compute `derived_remarks`, which the response always includes regardless of which path answered.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS, all tests, including every `TestCompletenessRouter` test.

Run: `pytest -v` (full suite)
Expected: PASS. If any test still references `clf_app1`/`clf_app2`/`ARTIFACTS["encoders"]`
anywhere outside this task's own files, fix that reference — Task 2 already removed the artifacts
those names pointed to.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "fix: route complete requests to the exact guide table, compute formulation directly, stop writing test predictions to production Supabase"
```

---

### Task 4: Fix the Supabase migration (Critical #3)

**Files:**
- Modify: `supabase/migrations/20260806_align_ml_training_schema.sql`

**Interfaces:** none (SQL only, not imported by any Python code).

**Context:** the migration as written loses K-unit correctness on any pre-existing row (renaming
`potassium_ppm`→`potassium_cmol_kg` doesn't convert the stored number) and leaves old synthetic
rows in place after dropping the only column that distinguished them from real ones. Given this
project's whole point this session was retiring synthetic data, and the table has never
successfully received real data yet (the Task 12 ingestion attempt failed before any row was
written — see the results doc), the only defensible resolution is: **delete every row that
predates this migration**, not attempt to convert or preserve any of it. There is no real data in
this table to lose.

- [ ] **Step 1: Add the delete step and remove the now-unnecessary K-unit ambiguity**

Replace the file's final section (from the `is_synthetic` comment to the end) — and add a new
delete step before the column renames, since renaming `potassium_ppm` before deleting old rows
would otherwise (briefly, mid-migration) leave old ppm-scale values sitting in a column already
declared `cmol/kg`:

Replace the full file with:

```sql
-- Aligns ml_soil_training_data with the real-data retraining schema
-- (docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md).
-- The synthetic generator is fully retired; every row ingested from here on
-- is real, lab-sourced data via data/parse_real_soil_database.py.
--
-- This table (ml_soil_training_data only -- NOT ml_fertilizer_predictions,
-- the separate JSONB-based audit log, which needs no schema change and
-- whose historical records remain valid regardless) has never successfully
-- received real data (the first real ingestion attempt failed before this
-- migration existed -- see
-- docs/superpowers/specs/2026-08-06-real-soil-data-retraining-results.md).
-- Any row already in this table predates the real pipeline entirely, so
-- rather than attempt to convert old values into the new schema (the old
-- potassium_ppm numbers can't become correct potassium_cmol_kg values
-- without knowing which specific conversion, if any, they need), this
-- migration deletes every pre-existing row outright. There is no real data
-- here to lose.

DELETE FROM public.ml_soil_training_data;

-- potassium is now stored in its lab-native unit (cmol/kg), not converted.
-- Safe now that no pre-existing row survives to carry a stale ppm value
-- under the new column name.
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN potassium_ppm TO potassium_cmol_kg;

-- Target columns lose their _kg_ha suffix: unit now varies by crop
-- (kg/ha for field crops, kg/tree for Coconut/Cacao/Mango/Banana) and is
-- carried explicitly in the new target_unit column below.
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_n_kg_ha TO target_n;
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_p2o5_kg_ha TO target_p2o5;
ALTER TABLE public.ml_soil_training_data
  RENAME COLUMN target_k2o_kg_ha TO target_k2o;

ALTER TABLE public.ml_soil_training_data
  ADD COLUMN variety             TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN cultivation_method  TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN age_band            TEXT NOT NULL DEFAULT 'None',
  ADD COLUMN target_unit         TEXT NOT NULL DEFAULT 'kg_ha';

-- No pre-existing rows survive the DELETE above, so no backfill UPDATE is
-- needed for the four columns just added -- every future row is written by
-- data/ingest_to_supabase.py, which always supplies all four explicitly.

-- is_synthetic is dead weight now that the synthetic generator is retired
-- and every pre-existing row (synthetic or not) has just been deleted.
DROP INDEX IF EXISTS idx_ml_training_synthetic;
ALTER TABLE public.ml_soil_training_data
  DROP COLUMN is_synthetic;
```

- [ ] **Step 2: Commit**

No automated test for SQL in this repo — commit directly.

```bash
git add supabase/migrations/20260806_align_ml_training_schema.sql
git commit -m "fix: delete pre-existing rows in the migration instead of silently mis-uniting them"
```

---

### Task 5: `resolve_variety()` crop-gating (Important #5)

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Modify: `tests/test_nutrient_recommendation_guide.py`

**Interfaces:**
- Produces: `resolve_variety(crop, variety) -> str` (same signature, now crop-gated like its two
  siblings).

**Context:** only Rice and Corn track variety at all (per `VARIETIES` in `ml/preprocessing.py` and
the fact that `_guide_row_key` only branches on `variety` for those two crops). Every other crop
passing a `variety` value currently gets it stored as a real feature category anyway, producing an
unseen one-hot combination the model has never trained on (measured: Cassava + Hybrid predicted
37% different from plain Cassava, despite the guide ignoring variety for Cassava entirely).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nutrient_recommendation_guide.py`'s `TestResolvers` class:

```python
    def test_variety_only_applies_to_rice_and_corn(self):
        assert resolve_variety("Cassava", "Hybrid") == "None"
        assert resolve_variety("Tomato", "Hybrid") == "None"
        assert resolve_variety("Mango", "Hybrid") == "None"

    def test_variety_still_passes_through_for_rice_and_corn(self):
        assert resolve_variety("Rice", "Hybrid") == "Hybrid"
        assert resolve_variety("Corn", "Hybrid") == "Hybrid"

    def test_variety_not_tracked_by_this_crop_falls_back_to_none(self):
        """Corn doesn't track Inbred (a Rice-only variety) -- passing it for
        Corn must not silently accept an axis value this crop's guide rows
        don't distinguish."""
        assert resolve_variety("Corn", "Inbred") == "None"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: FAIL — `resolve_variety("Cassava", "Hybrid")` currently returns `"Hybrid"`, not `"None"`.

- [ ] **Step 3: Implement**

In `data/nutrient_recommendation_guide.py`, add a new module-level constant near
`RICE_CULTIVATION_METHODS` and `BEARING_AGE_BAND`:

```python
# crop -> the variety strings its guide rows actually distinguish. Every
# other crop's variety is always "None" -- the guide has no per-variety
# rows for them, so accepting an arbitrary variety string would just be an
# unseen, meaningless one-hot combination at training/serving time.
VARIETIES_BY_CROP: dict[str, tuple[str, ...]] = {
    "Rice": ("Hybrid", "Inbred"),
    "Corn": ("OPV", "Hybrid"),
}
```

Replace `resolve_variety`:

```python
def resolve_variety(crop: str, variety: str | None) -> str:
    if variety in VARIETIES_BY_CROP.get(crop, ()):
        return variety
    return "None"
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_nutrient_recommendation_guide.py -v`
Expected: PASS, all tests.

Run: `pytest -v` (full suite)
Expected: PASS — check specifically that no other test asserted the old (ungated) behavior for a
non-Rice/Corn crop with a variety set; none should have, since `tests/fixtures.py` only ever
generates default (`None`) variety values and `main.py`'s tests only exercise Rice with variety.

- [ ] **Step 5: Commit**

```bash
git add data/nutrient_recommendation_guide.py tests/test_nutrient_recommendation_guide.py
git commit -m "fix: crop-gate resolve_variety like its two siblings already do"
```

---

### Task 6: Resolve the missing Coffee golden-record test (Important #6)

**Files:**
- Modify: `data/nutrient_recommendation_guide.py`
- Modify: `tests/test_nutrient_recommendation_guide.py`
- Modify: `docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md`

**Interfaces:** none new.

**Context:** the original plan's Global Constraint promised the retired SCC-417 fixture would be
"replaced, not just deleted" by two golden-record checks — Coffee and Rice-Inbred — against
`Fertilizer Computation Guide.pptx`'s worked examples. Only Rice-Inbred was ever written. The
`GUIDE_ROWS["Coffee|Over 15yrs"]` entry transcribed from the guide PDF is in `kg_tree` units
(0.165/0.10/0.30), but the pptx's Coffee worked example states 90/60/40 in what its own text
labels `kg/ha` — reconciling the two requires an assumed per-hectare planting density for coffee
trees that was never established anywhere in this project (a coffee-density figure like
~1,600 trees/ha would put the table's kg/tree numbers in roughly the right order of magnitude, but
"roughly the right order of magnitude with an assumed, unsourced density" is not a golden-record
check — it doesn't verify the transcription, it just picks a density that makes it look
approximately right). This is a materially different, weaker kind of ambiguity than the Rice-Inbred
K discrepancy (same crop, same unit, one nutrient off) and isn't resolvable without a real source
for coffee planting density this project doesn't have. Remove the dead entry and amend the
constraint honestly rather than force a check that wouldn't actually verify anything.

- [ ] **Step 1: Remove the dead `GUIDE_ROWS` entry**

In `data/nutrient_recommendation_guide.py`, delete the `"Coffee|Over 15yrs": GuideRow(...)` block
(with its preceding `# Coffee: golden-record reference only...` comment) from `GUIDE_ROWS`.

Update the module docstring's final paragraph (the one describing the Rice-Inbred K discrepancy) to
add a note about Coffee. Replace:

```
The TestGoldenRecordRiceInbred K-band discrepancy (pptx says K=40 for the
Rice-Inbred wet-season worked example; this table has no "40" anywhere in
Rice's K column under any cultivation method -- Upland/Wet/Dry all use the
same 60/45/30/{0 or 20}/0 rate set) was NOT resolved by this re-check: the
re-verified PDF values confirm 40 genuinely does not appear in the source
table, so the discrepancy is real and unexplained by transcription error,
not an artifact of a bad transcription. See that test's docstring.
```

with:

```
The TestGoldenRecordRiceInbred K-band discrepancy (pptx says K=40 for the
Rice-Inbred wet-season worked example; this table has no "40" anywhere in
Rice's K column under any cultivation method -- Upland/Wet/Dry all use the
same 60/45/30/{0 or 20}/0 rate set) was NOT resolved by this re-check: the
re-verified PDF values confirm 40 genuinely does not appear in the source
table, so the discrepancy is real and unexplained by transcription error,
not an artifact of a bad transcription. See that test's docstring.

A second planned golden-record check, against the pptx's Coffee worked
example (90/60/40, labeled kg/ha), was abandoned rather than forced: this
guide's Coffee row is in kg/tree (Coffee isn't a supported crop and its row
existed solely for this cross-check), and reconciling kg/tree against a
kg/ha-labeled worked example requires an assumed coffee planting density
this project never sourced. A check built on an assumed, unsourced density
wouldn't actually verify the transcription, so the dead row was removed
instead of kept around unused. See
docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md's amended
Global Constraint on this point.
```

- [ ] **Step 2: Remove the now-invalid reference in the test file's module context**

`tests/test_nutrient_recommendation_guide.py` has no `Coffee` test to remove (it was never
written), so no test file change is needed beyond confirming this with a grep.

- [ ] **Step 3: Verify nothing else references the removed row**

Run: `grep -rn "Coffee" --include="*.py" .`
Expected: no matches (the only prior reference was the now-removed `GUIDE_ROWS` entry and its
comment).

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: PASS — removing an unreferenced dict entry can't break anything that imports this module.

- [ ] **Step 5: Amend the original plan's Global Constraint**

In `docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md`, find the Global Constraints
bullet describing the golden-record replacement (containing the text `"replaced, not just
deleted"` and mentioning Coffee and Rice-Inbred). Add a parenthetical after it:

Find:
```
- The `GOLDEN_SCC417`/`SCC-26-417-417` fixture is retired throughout (Sorghum is no longer a
  supported crop, and the record's origin was traced to a document outside this dataset). It is
  **replaced**, not just deleted: `data/nutrient_recommendation_guide.py` gets its own
  golden-record checks against the two worked examples in `Fertilizer Computation Guide.pptx`
  (Coffee, Rice-Inbred) — see Task 1.
```

Replace with:

```
- The `GOLDEN_SCC417`/`SCC-26-417-417` fixture is retired throughout (Sorghum is no longer a
  supported crop, and the record's origin was traced to a document outside this dataset). It is
  **replaced**, not just deleted: `data/nutrient_recommendation_guide.py` gets its own
  golden-record checks against the worked examples in `Fertilizer Computation Guide.pptx` — see
  Task 1. **Amended after the final whole-branch review** (see
  `docs/superpowers/plans/2026-08-06-final-review-fixes.md` Task 6): only the Rice-Inbred check was
  ultimately implemented. The planned Coffee cross-check was abandoned — Coffee's transcribed guide
  row is in kg/tree, the pptx's worked example is labeled kg/ha, and reconciling the two needs an
  assumed coffee planting density this project never sourced; forcing a check on an unsourced
  assumption wouldn't have verified anything.
```

- [ ] **Step 6: Commit**

```bash
git add data/nutrient_recommendation_guide.py docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md
git commit -m "docs: remove the dead Coffee guide row, document why that golden-record check was abandoned"
```

---

### Task 7: Stale docstrings (Important #8)

**Files:**
- Modify: `ml/remarks.py`
- Modify: `data/ingest_to_supabase.py`
- Modify: `ml/model.py`

**Interfaces:** none — docstring-only changes.

**Context:** `ml/recipes.py` and `tests/test_recipes.py`'s stale SCC-417 claims were already fixed
in Task 1 (the recipes rewrite). This task covers the three remaining files the findings doc names.

- [ ] **Step 1: Fix `ml/remarks.py`'s docstring**

Find:
```
Provenance: the pH classes are the standard USDA/BSWM soil reaction classes and
the Olsen-P classes are the standard Olsen interpretation; both independently
reproduce the real SCC-417 report without having been fitted to it. The OM and K
bands are calibrated to that single real example and should be confirmed against
DA RSL's published interpretation chart before production use. Changing any
boundary is a one-constant edit here plus a retrain.
```

Replace with:
```
Provenance: the pH classes are the standard USDA/BSWM soil reaction classes and
the Olsen-P classes are the standard Olsen interpretation. The OM and K bands
were originally calibrated against a specific real DA report
(laboratory_code SCC-26-417-417); that record was later traced to a document
outside this project's real dataset and retired as a training/golden-test
fixture (see docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md's
"Golden test — retired" section) — these band boundaries were never revisited
against an independent source afterward and should be confirmed against DA
RSL's published interpretation chart before production use. Changing any
boundary is a one-constant edit here plus a retrain.
```

- [ ] **Step 2: Fix `data/ingest_to_supabase.py`'s module docstring**

Find:
```python
"""Load the synthetic training CSV into Supabase (spec section 9).

Upserts on the (laboratory_code, data_source) primary key so re-running is
idempotent rather than duplicating the corpus.
"""
```

Replace with:
```python
"""Load the real training CSV (data/parse_real_soil_database.py's output)
into Supabase.

Upserts on the (laboratory_code, data_source) primary key so re-running is
idempotent rather than duplicating the corpus.
"""
```

- [ ] **Step 3: Fix `ml/model.py`'s module docstring**

Find:
```python
"""Ensemble model definitions (spec section 8).

XGBoost captures the non-linear pH/phosphorus interaction and categorical splits;
Kernel Ridge Regression contributes an L2-regularised smooth fit that resists
overfitting the synthetic set. sklearn's VotingRegressor does not support
multi-output, so EnsembleFertilizerRegressor holds one per target and exposes a
single multi-output fit/predict.
"""
```

Replace with:
```python
"""Ensemble model definitions (spec section 8).

XGBoost captures non-linear interactions and categorical splits; Kernel Ridge
Regression contributes an L2-regularised smooth fit that resists overfitting
a corpus this size (a few thousand real rows). sklearn's VotingRegressor does
not support multi-output, so EnsembleFertilizerRegressor holds one per target
and exposes a single multi-output fit/predict.
"""
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: PASS — docstring-only changes.

- [ ] **Step 5: Commit**

```bash
git add ml/remarks.py data/ingest_to_supabase.py ml/model.py
git commit -m "docs: correct stale SCC-417/synthetic-data provenance claims"
```

---

### Task 8: Rewrite `README.md` against the current API (Important #4)

**Files:**
- Modify: `README.md` (full rewrite)

**Interfaces:** none — documentation only.

**Context:** the README still documents the deleted `data/generate_synthetic_data.py` workflow,
`potassium_ppm`/`target_n_kg_ha` field names, and a stale 6-crop list. This task must run after
Tasks 1-6 land (it documents the post-fix API shape, including the completeness-router `source`
values and the removal of the classifier heads).

- [ ] **Step 1: Replace `README.md` in full**

```markdown
# SURI Fertilizer ML Backend

Predicts fertilizer nutrient requirements (N, P₂O₅, K₂O) and application formulations from soil
test data, for the Department of Agriculture Region 7.

Two answer paths, chosen automatically per request:
- **Complete soil chemistry** (organic matter, phosphorus, and potassium all present) is answered
  by a direct lookup against the official BSWM Nutrient Recommendation Guide — exact, not a model
  approximation.
- **Incomplete soil chemistry** (the real case for IoT probe and Rapid Soil Test Kit readings,
  which cannot measure organic matter at all) is answered by a trained ensemble model, using
  missingness indicators so it knows what wasn't measured rather than trusting an imputed guess.

The response's `source` field says which path answered: `"table-lookup"` or `"ml-ensemble"`.

**Design docs:**
- `docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md` — the real-data retraining design.
- `docs/superpowers/specs/2026-08-06-real-soil-data-retraining-results.md` — real training results.
- `docs/superpowers/specs/2026-08-06-real-soil-data-retraining-final-review-findings.md` — what
  this session's Critical/Important fixes addressed, and why.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in from suri/.env.local
```

The service shares `suri/`'s Supabase project. Copy `NEXT_PUBLIC_SUPABASE_URL` into `SUPABASE_URL`
and `SUPABASE_SERVICE_ROLE_KEY` as-is.

Apply every migration under `supabase/migrations/` (in filename order) in the Supabase SQL editor
before ingesting.

## Parse, ingest, and train

```bash
python -m data.parse_real_soil_database    # reads ../suri/public/Region 7-National Soil Database_2022-2025.xlsx
                                            # writes suri_real_training_data.csv
python -m data.ingest_to_supabase          # upserts into Supabase
python -m ml.train --csv suri_real_training_data.csv
```

`ml/train.py` reads from Supabase when `--csv` is omitted. It writes versioned artifacts to
`ml/artifacts/<version>/` and prints a metrics summary (MAE/R² overall, by data source, and by
crop).

## Serve

```bash
uvicorn main:app --port 8001
```

Port 8001 — **not** 8000, which `suri/features/flows/iot/IoTFlow.tsx` already uses for an
unrelated contract.

## API

### `POST /predict`

Only `crop`, `data_source`, and `ph` are required. Everything else is optional; the model path
imputes missing values (median, with an explicit missingness flag), while the table-lookup path
requires `organic_matter_pct`, `phosphorus_ppm`, and `potassium_cmol_kg` to activate at all.

```json
{
  "crop": "Rice",
  "variety": "Hybrid",
  "cultivation_method": "Lowland Wet",
  "data_source": "lab",
  "ph": 6.5,
  "organic_matter_pct": 2.8,
  "phosphorus_ppm": 18.0,
  "potassium_cmol_kg": 0.35,
  "farm_area_sqm": 5000,
  "laboratory_code": "SCC-2024-0417"
}
```

| Field | Required | Notes |
|---|---|---|
| `crop` | yes | One of: Banana, Cacao, Cassava, Coconut, Corn, Mango, Peanut, Rice, Tomato. Unknown crops are rejected with 422. |
| `variety` | no | Rice: `Hybrid`/`Inbred`. Corn: `Hybrid`/`OPV`. Ignored (treated as unset) for every other crop. |
| `cultivation_method` | no | Rice only: `Upland`/`Lowland Wet`/`Lowland Dry`. Defaults to `Lowland Wet` when Rice is requested without one. |
| `age_band` | no | Coconut/Cacao/Mango only. Pass `"bearing"` (or omit) for the mature/fruit-bearing default, or a specific band (e.g. Coconut: `Below 1yr`/`1-5yrs`/`6-8yrs`/`9yrs & up`). |
| `data_source` | yes | `lab`, `stk`, or `iot` |
| `ph` | yes | 0–14 |
| `organic_matter_pct` | no | 0–100. Required (along with `phosphorus_ppm`/`potassium_cmol_kg`) for the exact table-lookup path — IoT/STK readings structurally can't supply this, so those requests always use the model path instead. |
| `total_nitrogen_pct` | no | Derived as `organic_matter_pct × 0.05` when omitted |
| `phosphorus_ppm` | no | ≥ 0 |
| `potassium_cmol_kg` | no | ≥ 0, the lab's native unit (not ppm) |
| `temp`, `moist`, `ec`, `sal` | no | IoT probe readings |
| `farm_area_sqm` | no | Enables `farm_totals` in the response, only for `kg_ha` crops |
| `laboratory_code` | no | Recorded in the prediction audit log |

Response (a complete request, answered by the exact table):

```json
{
  "target_n": 90.0,
  "target_p2o5": 60.0,
  "target_k2o": 40.0,
  "target_unit": "kg_ha",
  "app1_formulation": "214 kg/ha (14-14-14) + 65 kg/ha (21-0-0)",
  "app2_formulation": "214 kg/ha (14-14-14) + 65 kg/ha (21-0-0)",
  "application_schedule": "Basal: apply half of the recommended N plus all P and K 10-14 days after transplanting (DAT)...",
  "derived_remarks": {
    "ph_remark": "SLIGHTLY ACIDIC",
    "om_remark": "MODERATELY LOW",
    "p_remark": "HIGH",
    "k_remark": "SUFFICIENT+"
  },
  "farm_totals": { "area_ha": 0.5, "n_kg": 45.0, "p2o5_kg": 30.0, "k2o_kg": 20.0 },
  "model_version": "2026-08-06-011035",
  "source": "table-lookup"
}
```

`target_unit` is `"kg_ha"` for Rice/Corn/Cassava/Peanut/Tomato and `"kg_tree"` for
Coconut/Cacao/Mango/Banana — tree-crop rates are legitimately sub-1. `farm_totals` is only computed
for `kg_ha` crops (a per-tree rate needs a tree count, which this API doesn't collect) and is
`null` when `farm_area_sqm` is omitted or the crop is `kg_tree`.

### `GET /health`

Returns `{status, model_version, artifacts_loaded, supabase_reachable}`.

## Tests

```bash
python -m pytest -v
```

## Known limitations

- The table-lookup path's Rice cultivation-method P-band transcription passed a manual PDF
  cross-check but one specific discrepancy (the Rice-Inbred worked example's K value) remains
  unresolved — see `tests/test_nutrient_recommendation_guide.py::TestGoldenRecordRiceInbred`'s
  docstring.
- P-band classification assumes the Olsen method (`data/nutrient_recommendation_guide.py`'s
  `P_METHOD`) — the source spreadsheet never states which method Region 7 labs actually use.
- The model path's accuracy varies significantly by crop sample size — see
  `docs/superpowers/specs/2026-08-06-real-soil-data-retraining-results.md` for the full by-crop
  breakdown; Tomato (20 real rows) is a known weak spot.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README against the current router/table-lookup API contract"
```

---

### Task 9: Regenerate training data, retrain, and expand the smoke test

This task has no code changes — it's the operational verification that Tasks 1-8 actually work
together against real data, the same kind of check the original plan's Task 12 was.

- [ ] **Step 1: Regenerate the training CSV**

Run: `python -m data.parse_real_soil_database`
Expected: same row counts as before (Task 1's fix doesn't change which rows are included, only
how `app1_formulation`/`app2_formulation` are computed for `kg_tree` rows) — 3,076 total rows. Spot
check: open `suri_real_training_data.csv` and confirm a `kg_tree`-crop row (e.g. a Coconut row) now
has `"kg/tree"` in its `app1_formulation` column, not `"kg/ha"`, and that value is no longer the
literal string `"None"` for rows where it previously was.

- [ ] **Step 2: Retrain**

Run: `python -m ml.train --csv suri_real_training_data.csv`
Expected: completes with no error, prints MAE/R² overall/by-source/by-crop (same shape as before,
minus any classification-metrics section, which no longer exists per Task 2). The new artifact
directory should NOT contain `clf_app1.joblib`/`clf_app2.joblib`/`label_encoders.joblib` — confirm
with `ls ml/artifacts/<new-version>/`.

- [ ] **Step 3: Smoke-test the router explicitly**

Run: `uvicorn main:app --port 8001` (separate terminal), then:

```bash
# Complete request -> table-lookup, exact
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Rice","variety":"Hybrid","cultivation_method":"Upland","data_source":"lab","ph":6.5,"organic_matter_pct":1.5,"phosphorus_ppm":5.0,"potassium_cmol_kg":0.1,"laboratory_code":"TEST-001"}' \
  | python -m json.tool
```
Expected: `"source": "table-lookup"`. Cross-check `target_n` against
`python -c "from data.nutrient_recommendation_guide import nutrient_requirements; print(nutrient_requirements('Rice', 1.5, 5.0, 0.1, cultivation_method='Upland'))"` — must match exactly.
This is the previously-18x-wrong Coconut/Below-1yr case's sibling for Rice/Upland; both were
uncovered-combination cases before this fix.

```bash
# Incomplete request -> ml-ensemble
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Rice","data_source":"stk","ph":6.5}' \
  | python -m json.tool
```
Expected: `"source": "ml-ensemble"`.

```bash
# kg_tree formulation
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Coconut","age_band":"bearing","data_source":"lab","ph":6.5,"organic_matter_pct":2.8,"phosphorus_ppm":18.0,"potassium_cmol_kg":0.35,"laboratory_code":"TEST-002"}' \
  | python -m json.tool
```
Expected: `"target_unit": "kg_tree"`, `app1_formulation` contains `"kg/tree"` and does not contain
`"kg/ha"`, `"farm_totals": null`.

Stop uvicorn when done.

- [ ] **Step 4: Full suite**

Run: `pytest -v`
Expected: every test passes.

- [ ] **Step 5: Re-run the final whole-branch review**

Use `superpowers:subagent-driven-development`'s final-review process again (most capable available
model), against the same base commit the original review used
(`3179c2e642b13c555fb068ce1df29c26f3054029`) through the current HEAD, to confirm all 8 findings
are actually resolved and nothing new was introduced. This is the gate for updating
`docs/superpowers/specs/2026-08-06-real-soil-data-retraining-final-review-findings.md`'s status
header away from "NOT production-ready."

No commit for this task — it's verification, and Step 5 produces its own follow-up (either a clean
bill or a new findings round).

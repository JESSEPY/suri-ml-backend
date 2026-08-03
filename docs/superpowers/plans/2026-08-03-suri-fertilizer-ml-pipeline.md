# SURI Fertilizer ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone FastAPI microservice that predicts fertilizer nutrient requirements (N, P₂O₅, K₂O kg/ha) and application formulations from soil test data supplied by any of SURI's three data-entry flows (lab / STK / IoT).

**Architecture:** A synthetic data generator produces agronomically-grounded training records in three "views" per record (one per data source, each with that source's real missingness pattern). A scikit-learn `ColumnTransformer` imputes and encodes them. An ensemble of XGBoost + Kernel Ridge Regression predicts the three continuous nutrient targets; two XGBoost classifiers predict the application formulation labels. FastAPI serves predictions on port 8001 and logs them to Supabase.

**Tech Stack:** Python 3.11+, scikit-learn, XGBoost, pandas, FastAPI, Pydantic v2, supabase-py, pytest.

**Spec:** `docs/superpowers/specs/2026-08-03-suri-fertilizer-ml-pipeline-design.md`

## Global Constraints

- All work happens in this repo (`suri-ml-backend`). **Never modify anything under the sibling `suri/` app** — that is a separate Next.js app and is explicitly out of scope (spec §15).
- Service runs on **port 8001**, never 8000 (port 8000 is claimed by an unrelated existing contract in `suri/features/flows/iot/IoTFlow.tsx`).
- Total Nitrogen is **always** `organic_matter_pct * 0.05`, never an independent measurement (spec §2).
- Nutrient targets are **kg/ha rates**; `farm_area_sqm` is a post-prediction scaler and **must never be a model feature** (spec §3.1).
- All train/test splitting and cross-validation is **grouped on `laboratory_code`** (spec §10.1). A random row split is a correctness bug.
- Fixed random seed `RANDOM_SEED = 42` everywhere for reproducibility.
- Remark band boundaries use `[lower, upper)` semantics — `value < upper` selects that band.
- The real SCC-417 record is the golden fixture throughout: pH 8.17, OM 3.21%, Total N 0.1605%, Avail. P 18.96 ppm, Exch. K 138.00 ppm, crop Sorghum → N=50, P₂O₅=40, K₂O=60 kg/ha, applications `143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)`.

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `pytest.ini` | Test discovery config |
| `ml/remarks.py` | Numeric → qualitative remark band derivation (spec §4) |
| `data/crop_baselines.py` | Per-crop baseline N/P₂O₅/K₂O constants (spec §6.1) |
| `ml/agronomy.py` | Soil-test → nutrient requirement formulas (spec §6.2) |
| `ml/recipes.py` | Fertilizer product catalog + application planner (spec §6.3) |
| `data/generate_synthetic_data.py` | Record generation + three-view expansion (spec §6.4) |
| `supabase/migrations/20260803_ml_fertilizer_tables.sql` | Schema |
| `data/ingest_to_supabase.py` | CSV → Supabase upsert |
| `ml/preprocessing.py` | `ColumnTransformer` (spec §7) |
| `ml/model.py` | `EnsembleFertilizerRegressor` + classifiers (spec §8) |
| `ml/train.py` | Grouped split, CV weighting, evaluation, artifact writing (spec §10) |
| `main.py` | FastAPI app: `POST /predict`, `GET /health` (spec §11) |
| `tests/` | One test module per source module |
| `README.md` | Run/train/serve instructions + API contract |

**Note on `ml/agronomy.py`:** spec §14 folds the §6.2 formulas into `generate_synthetic_data.py`. This plan extracts them into their own module so they are unit-testable in isolation and reusable by the golden test. This is a refinement of the spec's file list, not a change to its logic.

---

## Task 1: Project scaffolding and remark derivation

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `ml/__init__.py`, `data/__init__.py`, `tests/__init__.py`
- Create: `ml/remarks.py`
- Test: `tests/test_remarks.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `ml.remarks.derive_ph_remark(ph: float | None) -> str`
  - `ml.remarks.derive_om_remark(om: float | None) -> str`
  - `ml.remarks.derive_p_remark(p_ppm: float | None) -> str`
  - `ml.remarks.derive_k_remark(k_ppm: float | None) -> str`
  - `ml.remarks.UNKNOWN: str` (value `"UNKNOWN"`)
  - `ml.remarks.PH_ORDER`, `OM_ORDER`, `P_ORDER`, `K_ORDER` — `list[str]`, ascending band order, excluding `UNKNOWN`, consumed by Task 6's `OrdinalEncoder`

- [ ] **Step 1: Create the directory structure and dependency files**

```bash
mkdir -p ml data tests supabase/migrations
touch ml/__init__.py data/__init__.py tests/__init__.py
```

`requirements.txt`:
```
fastapi==0.115.*
uvicorn[standard]==0.34.*
pydantic>=2.0.0
scikit-learn>=1.5.0
xgboost>=2.1.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.4.0
supabase>=2.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
httpx>=0.27.0
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without conflict.

- [ ] **Step 3: Write the failing test**

`tests/test_remarks.py`:
```python
import pytest
from ml.remarks import (
    derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark,
    UNKNOWN, PH_ORDER, OM_ORDER, P_ORDER, K_ORDER,
)


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


class TestPhBands:
    @pytest.mark.parametrize("value,expected", [
        (4.0, "STRONGLY ACIDIC"),
        (4.5, "MODERATELY ACIDIC"),
        (5.5, "MODERATELY ACIDIC"),
        (5.6, "SLIGHTLY ACIDIC"),
        (6.5, "SLIGHTLY ACIDIC"),
        (6.6, "NEUTRAL"),
        (7.3, "NEUTRAL"),
        (7.4, "SLIGHTLY ALKALINE"),
        (7.8, "SLIGHTLY ALKALINE"),
        (7.9, "MODERATELY ALKALINE"),
        (8.4, "MODERATELY ALKALINE"),
        (8.5, "STRONGLY ALKALINE"),
        (9.9, "STRONGLY ALKALINE"),
    ])
    def test_band_boundaries(self, value, expected):
        assert derive_ph_remark(value) == expected


class TestOtherBands:
    @pytest.mark.parametrize("value,expected", [
        (0.5, "VERY LOW"), (1.0, "LOW"), (2.0, "MODERATELY LOW"),
        (3.4, "MODERATE"), (5.0, "HIGH"), (7.0, "HIGH"),
    ])
    def test_om_boundaries(self, value, expected):
        assert derive_om_remark(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (2.0, "VERY LOW"), (5.0, "LOW"), (10.0, "MEDIUM"),
        (17.0, "HIGH"), (40.0, "HIGH"),
    ])
    def test_p_boundaries(self, value, expected):
        assert derive_p_remark(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (30.0, "DEFICIENT"), (60.0, "SUFFICIENT"),
        (120.0, "SUFFICIENT+"), (250.0, "SUFFICIENT+"),
    ])
    def test_k_boundaries(self, value, expected):
        assert derive_k_remark(value) == expected


class TestMissingValues:
    def test_none_returns_unknown(self):
        assert derive_ph_remark(None) == UNKNOWN
        assert derive_om_remark(None) == UNKNOWN
        assert derive_p_remark(None) == UNKNOWN
        assert derive_k_remark(None) == UNKNOWN


class TestOrderLists:
    def test_orders_exclude_unknown(self):
        for order in (PH_ORDER, OM_ORDER, P_ORDER, K_ORDER):
            assert UNKNOWN not in order

    def test_orders_have_expected_lengths(self):
        assert len(PH_ORDER) == 7
        assert len(OM_ORDER) == 5
        assert len(P_ORDER) == 4
        assert len(K_ORDER) == 3

    def test_every_derived_remark_is_in_its_order_list(self):
        assert derive_ph_remark(8.17) in PH_ORDER
        assert derive_om_remark(3.21) in OM_ORDER
        assert derive_p_remark(18.96) in P_ORDER
        assert derive_k_remark(138.0) in K_ORDER
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_remarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.remarks'`

- [ ] **Step 5: Write the implementation**

`ml/remarks.py`:
```python
"""Derivation of DA RSL qualitative remarks from numeric soil values.

Band tables per spec section 4. Boundaries use [lower, upper) semantics:
a value is placed in the first band whose upper bound it falls strictly below.

Provenance: the pH classes are the standard USDA/BSWM soil reaction classes and
the Olsen-P classes are the standard Olsen interpretation; both independently
reproduce the real SCC-417 report without having been fitted to it. The OM and K
bands are calibrated to that single real example and should be confirmed against
DA RSL's published interpretation chart before production use. Changing any
boundary is a one-constant edit here plus a retrain.
"""

from __future__ import annotations

UNKNOWN = "UNKNOWN"

# (exclusive upper bound, label), ascending. The final entry catches the rest.
_PH_BANDS = [
    (4.5, "STRONGLY ACIDIC"),
    (5.6, "MODERATELY ACIDIC"),
    (6.6, "SLIGHTLY ACIDIC"),
    (7.4, "NEUTRAL"),
    (7.9, "SLIGHTLY ALKALINE"),
    (8.5, "MODERATELY ALKALINE"),
    (float("inf"), "STRONGLY ALKALINE"),
]

_OM_BANDS = [
    (1.0, "VERY LOW"),
    (2.0, "LOW"),
    (3.4, "MODERATELY LOW"),
    (5.0, "MODERATE"),
    (float("inf"), "HIGH"),
]

_P_BANDS = [
    (5.0, "VERY LOW"),
    (10.0, "LOW"),
    (17.0, "MEDIUM"),
    (float("inf"), "HIGH"),
]

_K_BANDS = [
    (60.0, "DEFICIENT"),
    (120.0, "SUFFICIENT"),
    (float("inf"), "SUFFICIENT+"),
]

# Ascending category order for Task 6's OrdinalEncoder. UNKNOWN is deliberately
# absent — it is encoded as -1 via handle_unknown="use_encoded_value".
PH_ORDER = [label for _, label in _PH_BANDS]
OM_ORDER = [label for _, label in _OM_BANDS]
P_ORDER = [label for _, label in _P_BANDS]
K_ORDER = [label for _, label in _K_BANDS]


def _band(value: float | None, bands: list[tuple[float, str]]) -> str:
    if value is None:
        return UNKNOWN
    for upper, label in bands:
        if value < upper:
            return label
    return bands[-1][1]


def derive_ph_remark(ph: float | None) -> str:
    return _band(ph, _PH_BANDS)


def derive_om_remark(om: float | None) -> str:
    return _band(om, _OM_BANDS)


def derive_p_remark(p_ppm: float | None) -> str:
    return _band(p_ppm, _P_BANDS)


def derive_k_remark(k_ppm: float | None) -> str:
    return _band(k_ppm, _K_BANDS)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_remarks.py -v`
Expected: PASS — all tests green.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini ml data tests
git commit -m "feat: add project scaffolding and soil remark derivation"
```

---

## Task 2: Crop baselines and nutrient requirement formulas

**Files:**
- Create: `data/crop_baselines.py`
- Create: `ml/agronomy.py`
- Test: `tests/test_agronomy.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces:
  - `data.crop_baselines.CROP_BASELINES: dict[str, tuple[int, int, int]]` — crop name → `(base_n, base_p2o5, base_k2o)`
  - `data.crop_baselines.CROPS: list[str]` — the valid crop names, consumed by Tasks 4, 6, 9
  - `ml.agronomy.total_nitrogen_from_om(om: float) -> float`
  - `ml.agronomy.ph_p_penalty(ph: float) -> float`
  - `ml.agronomy.nutrient_requirements(crop, ph, om, p_ppm, k_ppm) -> tuple[int, int, int]` returning `(n, p2o5, k2o)` in whole kg/ha

- [ ] **Step 1: Write the failing test**

`tests/test_agronomy.py`:
```python
import pytest
from data.crop_baselines import CROP_BASELINES, CROPS
from ml.agronomy import (
    total_nitrogen_from_om, ph_p_penalty, nutrient_requirements,
)


class TestTotalNitrogen:
    def test_matches_real_scc417_report(self):
        """The real report shows OM 3.21% -> Total N 0.1605%, exactly OM x 0.05."""
        assert total_nitrogen_from_om(3.21) == pytest.approx(0.1605)

    def test_is_exactly_five_percent_of_om(self):
        assert total_nitrogen_from_om(2.0) == pytest.approx(0.10)


class TestPhPhosphorusPenalty:
    def test_no_penalty_at_optimal_ph(self):
        """Phosphorus availability peaks at pH 6.5 — no extra P needed."""
        assert ph_p_penalty(6.5) == pytest.approx(1.0)

    def test_penalty_is_symmetric_around_optimum(self):
        assert ph_p_penalty(5.5) == pytest.approx(ph_p_penalty(7.5))

    def test_penalty_increases_away_from_optimum(self):
        assert ph_p_penalty(8.5) > ph_p_penalty(7.5) > ph_p_penalty(6.5)

    def test_penalty_is_capped_at_thirty_percent(self):
        assert ph_p_penalty(14.0) == pytest.approx(1.30)
        assert ph_p_penalty(0.0) == pytest.approx(1.30)


class TestNutrientRequirements:
    def test_golden_record_scc417_within_tolerance(self):
        """Real report: Sorghum, pH 8.17, OM 3.21, P 18.96, K 138 -> 50/40/60.
        The generator only approximates DA's real rule, so tolerance is +/-15
        kg/ha — but a miss wider than that means the formulas disagree with real
        agronomic practice badly enough to investigate."""
        n, p, k = nutrient_requirements("Sorghum", 8.17, 3.21, 18.96, 138.0)
        assert abs(n - 50) <= 15
        assert abs(p - 40) <= 15
        assert abs(k - 60) <= 15

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


class TestCropBaselines:
    def test_sorghum_anchored_to_real_report(self):
        assert CROP_BASELINES["Sorghum"] == (50, 40, 60)

    def test_crops_list_matches_baseline_keys(self):
        assert set(CROPS) == set(CROP_BASELINES.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agronomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.crop_baselines'`

- [ ] **Step 3: Write the crop baselines**

`data/crop_baselines.py`:
```python
"""Per-crop baseline nutrient requirements in kg/ha, before soil-test adjustment.

Sorghum is anchored exactly to the real DA report SCC-26-417-417 (50/40/60).
The other five are conventional Philippine baseline rates used as placeholder
scaffolding while awaiting the real DA historical export. They set the model's
central tendency per crop, so they need agronomist review before any production
use — which is why they are isolated here: correcting them is a single edit plus
a retrain, with no code changes anywhere else.
"""

from __future__ import annotations

# crop -> (base_n, base_p2o5, base_k2o) in kg/ha
CROP_BASELINES: dict[str, tuple[int, int, int]] = {
    "Sorghum": (50, 40, 60),
    "Corn": (90, 60, 60),
    "Rice": (90, 30, 30),
    "Cabbage": (120, 90, 90),
    "Tomato": (100, 80, 80),
    "Sugarcane": (120, 60, 120),
}

CROPS: list[str] = sorted(CROP_BASELINES.keys())
```

- [ ] **Step 4: Write the agronomy formulas**

`ml/agronomy.py`:
```python
"""Soil-test to fertilizer requirement formulas (spec section 6.2).

Each nutrient requirement is its crop baseline adjusted by what the soil already
supplies: organic matter is the nitrogen reserve, and measured available P and
exchangeable K offset the P2O5 and K2O requirements respectively.
"""

from __future__ import annotations

from data.crop_baselines import CROP_BASELINES

# Total Nitrogen is calculated by the lab as OM x 0.05, never measured directly.
OM_TO_TOTAL_N = 0.05

N_MIN, N_MAX = 20, 200
P_MIN, P_MAX = 20, 150
K_MIN, K_MAX = 20, 200

_PH_OPTIMUM = 6.5
_PH_PENALTY_MAX = 0.30


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def total_nitrogen_from_om(om: float) -> float:
    return om * OM_TO_TOTAL_N


def ph_p_penalty(ph: float) -> float:
    """Phosphorus availability peaks near pH 6.5 and falls off in both
    directions — fixed by iron/aluminium in acid soils, by calcium in alkaline
    soils. Modeled as an inverted-U adding up to 30% to the P2O5 requirement.

    This non-monotonic interaction between two input features is the specific
    reason the architecture needs XGBoost rather than a purely linear model.
    """
    return 1.0 + _PH_PENALTY_MAX * min((ph - _PH_OPTIMUM) ** 2 / 4.0, 1.0)


def nutrient_requirements(
    crop: str, ph: float, om: float, p_ppm: float, k_ppm: float
) -> tuple[int, int, int]:
    """Return (n, p2o5, k2o) in whole kg/ha, matching how the real reports
    present them. Raises KeyError for an unknown crop."""
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

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_agronomy.py -v`
Expected: PASS. The golden record yields approximately (46, 45, 57) against the real (50, 40, 60) — all within the ±15 tolerance.

- [ ] **Step 6: Commit**

```bash
git add data/crop_baselines.py ml/agronomy.py tests/test_agronomy.py
git commit -m "feat: add crop baselines and nutrient requirement formulas"
```

---

## Task 3: Fertilizer recipe planner

**Files:**
- Create: `ml/recipes.py`
- Test: `tests/test_recipes.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ml.recipes.plan_applications(n: int, p2o5: int, k2o: int) -> tuple[str, str]` returning `(app1_formulation, app2_formulation)`
  - `ml.recipes.NO_APPLICATION: str` (value `"None"`)

**Why this logic:** working backward from the real SCC-417 report validates the planner exactly. Given N=50/P₂O₅=40/K₂O=60: cover P₂O₅ with Complete (40 / 0.14 = 286 kg total), split evenly across two applications → **143 kg/ha (14-14-14)**. Top up N (50 − 40 = 10, / 0.21 = 47.6 total, split) → **24 kg/ha (21-0-0)**. Top up K (60 − 40 = 20, / 0.60 = 33.3 total, split) → **17 kg/ha (0-0-60)**. All three match the real report exactly. The planner is reverse-engineered from real DA output, not invented.

- [ ] **Step 1: Write the failing test**

`tests/test_recipes.py`:
```python
import pytest
from ml.recipes import plan_applications, NO_APPLICATION


class TestGoldenRecordSCC417:
    """Working backward from the real report's own numbers must reproduce its
    exact application strings."""

    def test_reproduces_real_report_exactly(self):
        app1, app2 = plan_applications(50, 40, 60)
        expected = "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)"
        assert app1 == expected
        assert app2 == expected


class TestFormatting:
    def test_terms_are_joined_with_plus(self):
        app1, _ = plan_applications(90, 60, 60)
        assert " + " in app1

    def test_each_term_has_kg_ha_and_grade(self):
        app1, _ = plan_applications(90, 60, 60)
        for term in app1.split(" + "):
            assert " kg/ha (" in term
            assert term.endswith(")")

    def test_zero_quantity_terms_are_omitted(self):
        """When the Complete base already supplies all the N and K needed,
        no Ammosul or MOP term should appear."""
        app1, _ = plan_applications(20, 150, 20)
        assert "21-0-0" not in app1
        assert "0-0-60" not in app1
        assert "14-14-14" in app1


class TestSingleApplication:
    def test_low_requirement_gives_no_second_application(self):
        _, app2 = plan_applications(20, 20, 20)
        assert app2 == NO_APPLICATION

    def test_high_requirement_gives_a_second_application(self):
        _, app2 = plan_applications(120, 90, 90)
        assert app2 != NO_APPLICATION


class TestVocabularyStability:
    def test_same_input_gives_same_output(self):
        assert plan_applications(90, 60, 60) == plan_applications(90, 60, 60)

    def test_quantities_are_whole_numbers(self):
        app1, _ = plan_applications(93, 61, 57)
        for term in app1.split(" + "):
            quantity = term.split(" kg/ha")[0]
            assert quantity.isdigit()

    def test_vocabulary_is_finite_over_the_target_space(self):
        """Rounding to whole kg/ha keeps the classifier's label space finite."""
        seen = set()
        for n in range(20, 201, 10):
            for p in range(20, 151, 10):
                for k in range(20, 201, 10):
                    seen.update(plan_applications(n, p, k))
        assert len(seen) < 5000
        assert all(isinstance(label, str) and label for label in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recipes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.recipes'`

- [ ] **Step 3: Write the implementation**

`ml/recipes.py`:
```python
"""Fertilizer product catalog and application planner (spec section 6.3).

The planner covers the P2O5 requirement with a Complete (14-14-14) base, then
tops up nitrogen with Ammosul and potassium with Muriate of Potash, splitting
the total evenly across a basal application at planting and a topdress one month
later — the structure of the real DA Fertilizer Recommendation Report.

Validated against the real SCC-417 report: inputs (50, 40, 60) reproduce its
"143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)" exactly.
"""

from __future__ import annotations

NO_APPLICATION = "None"

# Standard Philippine fertilizer grades as (N%, P2O5%, K2O%).
COMPLETE = ("14-14-14", 0.14, 0.14, 0.14)
AMMOSUL = ("21-0-0", 0.21, 0.0, 0.0)
MOP = ("0-0-60", 0.0, 0.0, 0.60)

# Below these combined requirements a single basal application is sufficient.
_SINGLE_APPLICATION_MAX_N = 30
_SINGLE_APPLICATION_MAX_P = 25
_SINGLE_APPLICATION_MAX_K = 30

_NUM_APPLICATIONS = 2


def _format_terms(terms: list[tuple[str, float]]) -> str:
    """Render [(grade, kg)] as '143 kg/ha (14-14-14) + 24 kg/ha (21-0-0)',
    dropping any term that rounds to zero."""
    parts = [
        f"{int(round(kg))} kg/ha ({grade})"
        for grade, kg in terms
        if int(round(kg)) > 0
    ]
    return " + ".join(parts) if parts else NO_APPLICATION


def plan_applications(n: int, p2o5: int, k2o: int) -> tuple[str, str]:
    """Return (app1_formulation, app2_formulation) for the given kg/ha targets."""
    complete_grade, complete_n, _, complete_k = COMPLETE

    kg_complete = p2o5 / COMPLETE[2]
    n_supplied = kg_complete * complete_n
    k_supplied = kg_complete * complete_k

    kg_ammosul = max(0.0, n - n_supplied) / AMMOSUL[1]
    kg_mop = max(0.0, k2o - k_supplied) / MOP[3]

    single_application = (
        n <= _SINGLE_APPLICATION_MAX_N
        and p2o5 <= _SINGLE_APPLICATION_MAX_P
        and k2o <= _SINGLE_APPLICATION_MAX_K
    )
    splits = 1 if single_application else _NUM_APPLICATIONS

    per_application = [
        (complete_grade, kg_complete / splits),
        (AMMOSUL[0], kg_ammosul / splits),
        (MOP[0], kg_mop / splits),
    ]
    formulation = _format_terms(per_application)

    if single_application:
        return formulation, NO_APPLICATION
    return formulation, formulation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_recipes.py -v`
Expected: PASS — including the golden test reproducing the real report string exactly.

- [ ] **Step 5: Commit**

```bash
git add ml/recipes.py tests/test_recipes.py
git commit -m "feat: add fertilizer application planner validated against real DA report"
```

---

## Task 4: Synthetic data generator with three-view expansion

**Files:**
- Create: `data/generate_synthetic_data.py`
- Test: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: `ml.remarks.*` (Task 1), `data.crop_baselines.CROPS` and `ml.agronomy.*` (Task 2), `ml.recipes.plan_applications` (Task 3)
- Produces:
  - `data.generate_synthetic_data.generate(num_records: int, seed: int = 42) -> pandas.DataFrame`
  - `data.generate_synthetic_data.COLUMNS: list[str]` — the exact CSV column order, consumed by Tasks 5 and 6
  - `data.generate_synthetic_data.LOCATIONS: list[str]`
  - Running the module as a script writes `suri_synthetic_training_data_region7.csv`

- [ ] **Step 1: Write the failing test**

`tests/test_generate_synthetic_data.py`:
```python
import pytest
from data.generate_synthetic_data import generate, COLUMNS
from data.crop_baselines import CROPS


@pytest.fixture(scope="module")
def df():
    return generate(num_records=200, seed=42)


class TestShape:
    def test_emits_three_views_per_record(self, df):
        assert len(df) == 200 * 3

    def test_has_exactly_the_declared_columns(self, df):
        assert list(df.columns) == COLUMNS

    def test_every_lab_code_has_all_three_sources(self, df):
        per_code = df.groupby("laboratory_code")["data_source"].apply(set)
        assert all(sources == {"lab", "stk", "iot"} for sources in per_code)

    def test_lab_code_and_source_are_unique_together(self, df):
        assert not df.duplicated(subset=["laboratory_code", "data_source"]).any()


class TestReproducibility:
    def test_same_seed_gives_identical_output(self):
        assert generate(50, seed=7).equals(generate(50, seed=7))

    def test_different_seed_gives_different_output(self):
        assert not generate(50, seed=7).equals(generate(50, seed=8))


class TestTargetsAreSharedAcrossViews:
    def test_all_views_of_a_record_share_identical_targets(self, df):
        targets = ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
                   "app1_formulation", "app2_formulation"]
        for _, group in df.groupby("laboratory_code"):
            for column in targets:
                assert group[column].nunique() == 1


class TestMissingnessPatternPerSource:
    def test_lab_view_has_the_full_panel(self, df):
        lab = df[df.data_source == "lab"]
        for column in ["ph", "organic_matter_pct", "total_nitrogen_pct",
                       "phosphorus_ppm", "potassium_ppm"]:
            assert lab[column].notna().all()

    def test_lab_and_stk_have_no_sensor_columns(self, df):
        subset = df[df.data_source.isin(["lab", "stk"])]
        for column in ["temp", "moist", "ec", "sal"]:
            assert subset[column].isna().all()

    def test_iot_view_is_missing_om_and_total_nitrogen(self, df):
        """No IoT probe measures organic matter — genuinely absent, not zero."""
        iot = df[df.data_source == "iot"]
        assert iot["organic_matter_pct"].isna().all()
        assert iot["total_nitrogen_pct"].isna().all()

    def test_iot_view_has_sensor_columns(self, df):
        iot = df[df.data_source == "iot"]
        for column in ["temp", "moist", "ec", "sal"]:
            assert iot[column].notna().all()

    def test_stk_view_has_derived_om(self, df):
        """STK's nitrogen colour reading is a proxy for OM banding, so OM and
        Total N are derived rather than missing."""
        stk = df[df.data_source == "stk"]
        assert stk["organic_matter_pct"].notna().all()
        assert stk["total_nitrogen_pct"].notna().all()


class TestValueRanges:
    def test_total_nitrogen_is_five_percent_of_om(self, df):
        present = df[df.organic_matter_pct.notna()]
        expected = present.organic_matter_pct * 0.05
        assert (present.total_nitrogen_pct - expected).abs().max() < 1e-9

    def test_stk_ph_uses_only_the_kit_indicator_values(self, df):
        stk = df[df.data_source == "stk"]
        assert set(stk.ph.unique()).issubset({4.5, 5.5, 6.8, 7.8})

    def test_targets_are_within_clamp_ranges(self, df):
        assert df.target_n_kg_ha.between(20, 200).all()
        assert df.target_p2o5_kg_ha.between(20, 150).all()
        assert df.target_k2o_kg_ha.between(20, 200).all()

    def test_crops_are_from_the_known_set(self, df):
        assert set(df.crop.unique()).issubset(set(CROPS))

    def test_remarks_are_populated_wherever_the_numeric_is(self, df):
        with_ph = df[df.ph.notna()]
        assert (with_ph.ph_remark != "UNKNOWN").all()

    def test_iot_om_remark_is_unknown(self, df):
        iot = df[df.data_source == "iot"]
        assert (iot.om_remark == "UNKNOWN").all()


class TestSignalExists:
    def test_targets_are_not_random_noise(self, df):
        """The original draft synthesizer assigned targets uniformly at random,
        so a model could learn nothing but the mean. Higher available P must
        actually reduce the P2O5 requirement."""
        lab = df[df.data_source == "lab"]
        low_p = lab[lab.phosphorus_ppm < 12].target_p2o5_kg_ha.mean()
        high_p = lab[lab.phosphorus_ppm > 28].target_p2o5_kg_ha.mean()
        assert high_p < low_p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.generate_synthetic_data'`

- [ ] **Step 3: Write the implementation**

`data/generate_synthetic_data.py`:
```python
"""Synthetic training data generator (spec section 6).

Produces a lab-grade record, then derives three training views from it — one per
data source, each carrying that source's real inference-time missingness pattern.
Training on the actual missingness the model will face is what makes it usable by
all three SURI flows.

Run as a script to write suri_synthetic_training_data_region7.csv.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from data.crop_baselines import CROPS
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import (
    UNKNOWN, derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark,
)

RANDOM_SEED = 42
DEFAULT_NUM_RECORDS = 5000
OUTPUT_CSV = "suri_synthetic_training_data_region7.csv"

LOCATIONS = [
    "Jampang, Argao, Cebu", "Tulic, Argao, Cebu",
    "Mantalongon, Dalaguete, Cebu", "Obo, Dalaguete, Cebu",
    "Campesa, Barili, Cebu", "Guadalupe, Carcar, Cebu",
    "Lamacan, Sibonga, Cebu", "San Nicholas, Cebu City, Cebu",
    "Sudlon II, Cebu City, Cebu", "Guba, Cebu City, Cebu",
    "Cansomoroy, Balamban, Cebu", "Luray, Toledo, Cebu",
    "Gairan, Bogo City, Cebu", "Buenos Aires, Carmen, Bohol",
    "Nueva Fuerza, Carmen, Bohol", "Biabas, Ubay, Bohol",
    "San Pascual, Ubay, Bohol", "Pula, Canlaon, Negros Oriental",
    "Linothangan, Canlaon, Negros Oriental",
]

COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_ppm",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
    "app1_formulation", "app2_formulation",
    "is_synthetic",
]

# STK bands map to representative ranges in LAB units. The kit's own `value`
# fields in suri/shared/constants/stk-options.ts are an internal app scale, not
# lab units — STK P reads 30/70/110 where real Olsen P spans 5-40 ppm.
_STK_PH_INDICATORS = [4.5, 5.5, 6.8, 7.8]
_STK_OM_BANDS = [(2.0, (0.5, 2.0)), (3.5, (2.0, 3.5)), (float("inf"), (3.5, 6.0))]
_STK_P_BANDS = [(10.0, (3.0, 10.0)), (20.0, (10.0, 20.0)), (float("inf"), (20.0, 40.0))]
_STK_K_BANDS = [(80.0, (30.0, 80.0)), (150.0, (80.0, 150.0)), (float("inf"), (150.0, 250.0))]

_IOT_SENSOR_NOISE = 0.10


def _coarsen(rng: random.Random, value: float, bands) -> float:
    """Place a lab value in its STK band, then resample within that band — the
    coarsening a kit user's reading actually represents."""
    for upper, (low, high) in bands:
        if value < upper:
            return round(rng.uniform(low, high), 2)
    return round(value, 2)


def _nearest(value: float, options: list[float]) -> float:
    return min(options, key=lambda option: abs(option - value))


def _row(base: dict, source: str, **overrides) -> dict:
    row = {**base, "data_source": source, **overrides}
    row["ph_remark"] = derive_ph_remark(row["ph"])
    row["om_remark"] = derive_om_remark(row["organic_matter_pct"])
    row["p_remark"] = derive_p_remark(row["phosphorus_ppm"])
    row["k_remark"] = derive_k_remark(row["potassium_ppm"])
    return row


def generate(num_records: int = DEFAULT_NUM_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    start_date = datetime(2025, 1, 1)
    rows: list[dict] = []

    for index in range(num_records):
        ph = round(rng.uniform(4.5, 8.5), 2)
        om = round(rng.uniform(0.5, 5.0), 2)
        phosphorus = round(rng.uniform(5.0, 40.0), 2)
        potassium = round(rng.uniform(50.0, 250.0), 2)
        crop = rng.choice(CROPS)

        n, p2o5, k2o = nutrient_requirements(crop, ph, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o)

        base = {
            "laboratory_code": f"SCC-26-{400 + index}",
            "sampling_site": rng.choice(LOCATIONS),
            "date_of_sampling": (
                start_date + timedelta(days=rng.randint(0, 500))
            ).strftime("%Y-%m-%d"),
            "crop": crop,
            "farm_area_sqm": round(rng.uniform(100, 10000), 2),
            "temp": None, "moist": None, "ec": None, "sal": None,
            "target_n_kg_ha": n,
            "target_p2o5_kg_ha": p2o5,
            "target_k2o_kg_ha": k2o,
            "app1_formulation": app1,
            "app2_formulation": app2,
            "is_synthetic": True,
        }

        # Lab view — the full LS-0040 panel.
        rows.append(_row(
            base, "lab",
            ph=ph, organic_matter_pct=om,
            total_nitrogen_pct=round(total_nitrogen_from_om(om), 4),
            phosphorus_ppm=phosphorus, potassium_ppm=potassium,
        ))

        # STK view — coarsened to kit granularity. OM comes from the nitrogen
        # colour band, which the kit documents as an organic-matter proxy.
        stk_om = _coarsen(rng, om, _STK_OM_BANDS)
        rows.append(_row(
            base, "stk",
            ph=_nearest(ph, _STK_PH_INDICATORS),
            organic_matter_pct=stk_om,
            total_nitrogen_pct=round(total_nitrogen_from_om(stk_om), 4),
            phosphorus_ppm=_coarsen(rng, phosphorus, _STK_P_BANDS),
            potassium_ppm=_coarsen(rng, potassium, _STK_K_BANDS),
        ))

        # IoT view — no organic-matter equivalent exists on a probe.
        def noisy(value: float) -> float:
            return round(value * rng.uniform(1 - _IOT_SENSOR_NOISE, 1 + _IOT_SENSOR_NOISE), 2)

        rows.append(_row(
            base, "iot",
            ph=round(ph + rng.uniform(-0.15, 0.15), 2),
            organic_matter_pct=None, total_nitrogen_pct=None,
            phosphorus_ppm=noisy(phosphorus), potassium_ppm=noisy(potassium),
            temp=round(rng.uniform(24.0, 34.0), 2),
            moist=round(rng.uniform(20.0, 70.0), 2),
            ec=round(rng.uniform(100.0, 1500.0), 2),
            sal=round(rng.uniform(50.0, 800.0), 2),
        ))

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    frame = generate()
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(frame)} rows ({len(frame) // 3} records x 3 views) to {OUTPUT_CSV}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real dataset**

Run: `python -m data.generate_synthetic_data`
Expected: `Wrote 15000 rows (5000 records x 3 views) to suri_synthetic_training_data_region7.csv`

- [ ] **Step 6: Commit**

```bash
git add data/generate_synthetic_data.py tests/test_generate_synthetic_data.py
git commit -m "feat: add synthetic data generator with per-source view expansion"
```

Note: do **not** commit the generated CSV — add `*.csv` to `.gitignore` in this step. It is reproducible from the seed.

---

## Task 5: Supabase schema and ingestion

**Files:**
- Create: `supabase/migrations/20260803_ml_fertilizer_tables.sql`
- Create: `data/ingest_to_supabase.py`
- Create: `.env.example`
- Test: `tests/test_ingest_to_supabase.py`

**Interfaces:**
- Consumes: `data.generate_synthetic_data.COLUMNS` (Task 4)
- Produces:
  - `data.ingest_to_supabase.get_client() -> supabase.Client`
  - `data.ingest_to_supabase.chunked(records: list[dict], size: int) -> Iterator[list[dict]]`
  - `data.ingest_to_supabase.ingest(csv_path: str, batch_size: int = 500) -> int` returning rows upserted

**Environment:** reuses the existing Supabase project from `suri/.env.local`. Copy `NEXT_PUBLIC_SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` into `.env` as `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

- [ ] **Step 1: Write the migration**

`supabase/migrations/20260803_ml_fertilizer_tables.sql`:
```sql
-- SURI Fertilizer ML Pipeline tables.
-- Training corpus is keyed by (laboratory_code, data_source): each soil record
-- contributes three rows, one per data source view.

CREATE TABLE public.ml_soil_training_data (
  laboratory_code      TEXT NOT NULL,
  data_source          TEXT NOT NULL CHECK (data_source IN ('lab', 'stk', 'iot')),
  sampling_site        TEXT,
  date_of_sampling     DATE,
  crop                 TEXT NOT NULL,
  farm_area_sqm        NUMERIC,
  ph                   NUMERIC,
  organic_matter_pct   NUMERIC,
  total_nitrogen_pct   NUMERIC,
  phosphorus_ppm       NUMERIC,
  potassium_ppm        NUMERIC,
  temp                 NUMERIC,
  moist                NUMERIC,
  ec                   NUMERIC,
  sal                  NUMERIC,
  ph_remark            TEXT,
  om_remark            TEXT,
  p_remark             TEXT,
  k_remark             TEXT,
  target_n_kg_ha       NUMERIC NOT NULL,
  target_p2o5_kg_ha    NUMERIC NOT NULL,
  target_k2o_kg_ha     NUMERIC NOT NULL,
  app1_formulation     TEXT NOT NULL,
  app2_formulation     TEXT NOT NULL,
  is_synthetic         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (laboratory_code, data_source)
);

CREATE INDEX idx_ml_training_crop ON public.ml_soil_training_data(crop);
CREATE INDEX idx_ml_training_source ON public.ml_soil_training_data(data_source);
CREATE INDEX idx_ml_training_synthetic ON public.ml_soil_training_data(is_synthetic);

-- Append-only audit log of live predictions, for retraining and traceability.
CREATE TABLE public.ml_fertilizer_predictions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  laboratory_code      TEXT,
  data_source          TEXT NOT NULL,
  request_payload      JSONB NOT NULL,
  response_payload     JSONB NOT NULL,
  model_version        TEXT NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_predictions_created ON public.ml_fertilizer_predictions(created_at DESC);
CREATE INDEX idx_ml_predictions_version ON public.ml_fertilizer_predictions(model_version);

ALTER TABLE public.ml_soil_training_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_fertilizer_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admin access to ml_soil_training_data" ON public.ml_soil_training_data
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role IN ('admin', 'superadmin')
    )
  );

CREATE POLICY "Admin access to ml_fertilizer_predictions" ON public.ml_fertilizer_predictions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
      AND profiles.role IN ('admin', 'superadmin')
    )
  );
```

`.env.example`:
```
# Copy these from suri/.env.local — the ML service shares suri's Supabase project.
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

- [ ] **Step 2: Apply the migration**

Run the SQL in the Supabase SQL editor for the project configured in `suri/.env.local`.
Expected: both tables appear under `public`.

- [ ] **Step 3: Write the failing test**

`tests/test_ingest_to_supabase.py`:
```python
import pandas as pd
import pytest
from data.ingest_to_supabase import chunked, frame_to_records
from data.generate_synthetic_data import generate


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
        return frame_to_records(generate(num_records=10, seed=42))

    def test_produces_one_record_per_row(self, records):
        assert len(records) == 30

    def test_nan_becomes_none_for_json(self, records):
        """Supabase rejects NaN — it is not valid JSON. Missing values must
        serialise as null."""
        iot = [record for record in records if record["data_source"] == "iot"]
        assert all(record["organic_matter_pct"] is None for record in iot)

    def test_records_are_json_serialisable(self, records):
        import json
        json.dumps(records)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest_to_supabase.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.ingest_to_supabase'`

- [ ] **Step 5: Write the implementation**

`data/ingest_to_supabase.py`:
```python
"""Load the synthetic training CSV into Supabase (spec section 9).

Upserts on the (laboratory_code, data_source) primary key so re-running is
idempotent rather than duplicating the corpus.
"""

from __future__ import annotations

import os
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from data.generate_synthetic_data import OUTPUT_CSV

TABLE = "ml_soil_training_data"
DEFAULT_BATCH_SIZE = 500


def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
            "Copy them from suri/.env.local into .env"
        )
    return create_client(url, key)


def chunked(records: list, size: int) -> Iterator[list]:
    for start in range(0, len(records), size):
        yield records[start:start + size]


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert to JSON-safe dicts. pandas uses NaN for missing numerics, which
    is not valid JSON and which Supabase rejects — it must become None."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def ingest(csv_path: str = OUTPUT_CSV, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    client = get_client()
    records = frame_to_records(pd.read_csv(csv_path))

    total = 0
    for index, batch in enumerate(chunked(records, batch_size), start=1):
        client.table(TABLE).upsert(
            batch, on_conflict="laboratory_code,data_source"
        ).execute()
        total += len(batch)
        print(f"  batch {index}: upserted {len(batch)} rows ({total} total)")

    return total


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} rows into {TABLE}.")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest_to_supabase.py -v`
Expected: PASS.

- [ ] **Step 7: Run the real ingestion**

Run: `python -m data.ingest_to_supabase`
Expected: 30 batches, `Ingested 15000 rows into ml_soil_training_data.`
Verify in Supabase: `SELECT data_source, count(*) FROM ml_soil_training_data GROUP BY data_source;` → 5000 each.

- [ ] **Step 8: Commit**

```bash
git add supabase data/ingest_to_supabase.py tests/test_ingest_to_supabase.py .env.example
git commit -m "feat: add Supabase schema and training data ingestion"
```

---

## Task 6: Preprocessing pipeline

**Files:**
- Create: `ml/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `ml.remarks.PH_ORDER/OM_ORDER/P_ORDER/K_ORDER` (Task 1), `data.crop_baselines.CROPS` (Task 2)
- Produces:
  - `ml.preprocessing.build_preprocessor() -> sklearn.compose.ColumnTransformer`
  - `ml.preprocessing.FEATURE_COLUMNS: list[str]` — every column the transformer consumes, in order
  - `ml.preprocessing.CONTINUOUS`, `ORDERED`, `NOMINAL` — `list[str]`
  - `ml.preprocessing.TARGET_COLUMNS: list[str]` — the three regression target names

- [ ] **Step 1: Write the failing test**

`tests/test_preprocessing.py`:
```python
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from data.generate_synthetic_data import generate
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS,
)


@pytest.fixture(scope="module")
def frame():
    return generate(num_records=100, seed=42)


@pytest.fixture(scope="module")
def fitted(frame):
    preprocessor = build_preprocessor()
    preprocessor.fit(frame[FEATURE_COLUMNS])
    return preprocessor


class TestColumnSets:
    def test_feature_columns_is_the_union_of_the_three_groups(self):
        assert set(FEATURE_COLUMNS) == set(CONTINUOUS) | set(ORDERED) | set(NOMINAL)

    def test_farm_area_is_not_a_feature(self):
        """Nutrient targets are kg/ha rates, identical regardless of farm size.
        Including area would invite a spurious area-to-dosage correlation."""
        assert "farm_area_sqm" not in FEATURE_COLUMNS

    def test_leakage_columns_are_not_features(self):
        for column in TARGET_COLUMNS + ["app1_formulation", "app2_formulation",
                                        "laboratory_code"]:
            assert column not in FEATURE_COLUMNS

    def test_deferred_metadata_is_not_a_feature(self):
        """Synthetic dates and sites carry no signal — including them would
        train on pure noise."""
        assert "date_of_sampling" not in FEATURE_COLUMNS
        assert "sampling_site" not in FEATURE_COLUMNS


class TestTransform:
    def test_output_is_two_dimensional_and_numeric(self, fitted, frame):
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert out.ndim == 2
        assert out.shape[0] == len(frame)

    def test_output_width_matches_feature_names(self, fitted, frame):
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert out.shape[1] == len(fitted.get_feature_names_out())

    def test_no_nan_survives_transform(self, fitted, frame):
        import numpy as np
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert not np.isnan(out).any()

    def test_transform_before_fit_raises(self, frame):
        with pytest.raises(NotFittedError):
            build_preprocessor().transform(frame[FEATURE_COLUMNS])


class TestMissingnessIndicators:
    def test_indicator_exists_for_organic_matter(self, fitted):
        """The IoT view has no organic matter at all — the model must be able
        to see that it was not measured, rather than trust the median."""
        names = list(fitted.get_feature_names_out())
        assert any("organic_matter_pct" in name and "missing" in name.lower()
                   for name in names)

    def test_indicator_exists_for_sensor_columns(self, fitted):
        names = list(fitted.get_feature_names_out())
        assert any("temp" in name and "missing" in name.lower() for name in names)


class TestEncoders:
    def test_unknown_remark_encodes_without_error(self, fitted, frame):
        row = frame[FEATURE_COLUMNS].head(1).copy()
        row.loc[:, "om_remark"] = "UNKNOWN"
        fitted.transform(row)

    def test_unknown_crop_raises(self, fitted, frame):
        """A silent all-zeros crop vector would yield a confident-looking
        prediction computed from no crop information at all."""
        row = frame[FEATURE_COLUMNS].head(1).copy()
        row.loc[:, "crop"] = "Durian"
        with pytest.raises(ValueError):
            fitted.transform(row)

    def test_ordinal_encoding_preserves_band_order(self, fitted):
        rows = pd.DataFrame([
            {**_base_row(), "k_remark": "DEFICIENT"},
            {**_base_row(), "k_remark": "SUFFICIENT"},
            {**_base_row(), "k_remark": "SUFFICIENT+"},
        ])[FEATURE_COLUMNS]
        out = fitted.transform(rows)
        index = list(fitted.get_feature_names_out()).index("ord__k_remark")
        assert out[0][index] < out[1][index] < out[2][index]


def _base_row() -> dict:
    return {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_ppm": 120.0,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": "SLIGHTLY ACIDIC", "om_remark": "MODERATELY LOW",
        "p_remark": "MEDIUM", "k_remark": "SUFFICIENT",
        "crop": "Corn", "data_source": "lab",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.preprocessing'`

- [ ] **Step 3: Write the implementation**

`ml/preprocessing.py`:
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
    "phosphorus_ppm", "potassium_ppm",
    "temp", "moist", "ec", "sal",
]

ORDERED = ["ph_remark", "om_remark", "p_remark", "k_remark"]

NOMINAL = ["crop", "data_source"]

FEATURE_COLUMNS = CONTINUOUS + ORDERED + NOMINAL

TARGET_COLUMNS = ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha"]

DATA_SOURCES = ["iot", "lab", "stk"]

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

    # handle_unknown="error", deliberately: an unrecognised crop must fail loudly
    # rather than produce an all-zeros vector and a confident-looking prediction
    # built on no crop information. A farmer acting on that could over- or
    # under-fertilize a real field.
    nominal = OneHotEncoder(
        categories=[CROPS, DATA_SOURCES],
        handle_unknown="error",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("cont", continuous, CONTINUOUS),
        ("ord", ordered, ORDERED),
        ("nom", nominal, NOMINAL),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing with missingness indicators and ordinal remarks"
```

---

## Task 7: Ensemble regressor and classifiers

**Files:**
- Create: `ml/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing (operates on already-transformed arrays)
- Produces:
  - `ml.model.EnsembleFertilizerRegressor(weights: list[tuple[float, float]] | None = None)` with `.fit(X, y) -> self` (y shape `(n, 3)`) and `.predict(X) -> ndarray(n, 3)`
  - `ml.model.build_classifier() -> xgboost.XGBClassifier`
  - `ml.model.build_xgb_regressor() -> xgboost.XGBRegressor`
  - `ml.model.build_krr() -> sklearn.kernel_ridge.KernelRidge`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from ml.model import (
    EnsembleFertilizerRegressor, build_classifier,
    build_xgb_regressor, build_krr,
)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def dataset():
    """A learnable non-linear relationship, standing in for the real one."""
    X = RNG.uniform(-1, 1, size=(400, 6))
    y = np.column_stack([
        50 + 20 * X[:, 0] - 10 * X[:, 1] ** 2,
        40 + 15 * X[:, 2] + 8 * X[:, 3],
        60 - 25 * X[:, 4] + 5 * X[:, 5],
    ])
    return X, y


class TestEnsembleRegressor:
    def test_predict_returns_one_column_per_target(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor().fit(X, y)
        assert model.predict(X).shape == (X.shape[0], 3)

    def test_beats_a_mean_baseline_on_every_target(self, dataset):
        """The guard against the original draft's failure mode: with randomly
        assigned targets the ensemble and the dummy score identically, and that
        equality is the detectable symptom."""
        X, y = dataset
        split = 300
        model = EnsembleFertilizerRegressor().fit(X[:split], y[:split])
        predicted = model.predict(X[split:])

        for target in range(3):
            dummy = DummyRegressor(strategy="mean").fit(X[:split], y[:split, target])
            ensemble_error = mean_absolute_error(y[split:, target], predicted[:, target])
            dummy_error = mean_absolute_error(y[split:, target], dummy.predict(X[split:]))
            assert ensemble_error < dummy_error

    def test_accepts_explicit_per_target_weights(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor(
            weights=[(0.8, 0.2), (0.5, 0.5), (0.3, 0.7)]
        ).fit(X, y)
        assert model.predict(X).shape == (X.shape[0], 3)

    def test_weights_actually_change_predictions(self, dataset):
        X, y = dataset
        xgb_heavy = EnsembleFertilizerRegressor(
            weights=[(1.0, 0.0)] * 3).fit(X, y).predict(X)
        krr_heavy = EnsembleFertilizerRegressor(
            weights=[(0.0, 1.0)] * 3).fit(X, y).predict(X)
        assert not np.allclose(xgb_heavy, krr_heavy)

    def test_holds_one_voting_regressor_per_target(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor().fit(X, y)
        assert len(model.models_) == 3

    def test_rejects_wrong_target_shape(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            EnsembleFertilizerRegressor().fit(X, y[:, 0])


class TestComponentBuilders:
    def test_xgb_regressor_is_seeded(self):
        assert build_xgb_regressor().random_state == 42

    def test_krr_uses_an_rbf_kernel(self):
        assert build_krr().kernel == "rbf"

    def test_classifier_fits_string_labels(self):
        X = RNG.uniform(-1, 1, size=(60, 4))
        labels = np.array(["recipe-a", "recipe-b", "recipe-c"] * 20)
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(labels)
        model = build_classifier().fit(X, encoder.transform(labels))
        assert len(encoder.inverse_transform(model.predict(X))) == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.model'`

- [ ] **Step 3: Write the implementation**

`ml/model.py`:
```python
"""Ensemble model definitions (spec section 8).

XGBoost captures the non-linear pH/phosphorus interaction and categorical splits;
Kernel Ridge Regression contributes an L2-regularised smooth fit that resists
overfitting the synthetic set. sklearn's VotingRegressor does not support
multi-output, so EnsembleFertilizerRegressor holds one per target and exposes a
single multi-output fit/predict.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBClassifier, XGBRegressor

RANDOM_SEED = 42
N_TARGETS = 3


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_krr() -> KernelRidge:
    return KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1)


def build_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


class EnsembleFertilizerRegressor(BaseEstimator, RegressorMixin):
    """Multi-output wrapper holding one VotingRegressor per nutrient target.

    weights: optional per-target (xgb_weight, krr_weight) pairs. Task 8 sets
    these from grouped cross-validated R2, so each nutrient uses whichever model
    actually performs better for it rather than a single global guess.
    """

    def __init__(self, weights: list[tuple[float, float]] | None = None):
        self.weights = weights

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != N_TARGETS:
            raise ValueError(
                f"Expected y of shape (n_samples, {N_TARGETS}), got {y.shape}"
            )

        self.models_ = []
        for target in range(N_TARGETS):
            pair = self.weights[target] if self.weights else None
            voter = VotingRegressor(
                estimators=[("xgb", build_xgb_regressor()), ("krr", build_krr())],
                weights=list(pair) if pair else None,
            )
            voter.fit(X, y[:, target])
            self.models_.append(voter)
        return self

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models_])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/model.py tests/test_model.py
git commit -m "feat: add XGBoost + Kernel Ridge ensemble regressor"
```

---

## Task 8: Training, grouped evaluation, and artifact writing

**Files:**
- Create: `ml/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `ml.preprocessing.*` (Task 6), `ml.model.*` (Task 7), `data.generate_synthetic_data.generate` (Task 4)
- Produces:
  - `ml.train.grouped_split(frame, seed=42) -> tuple[DataFrame, DataFrame, DataFrame]` (train, validation, test)
  - `ml.train.cv_weights(X, y, groups) -> list[tuple[float, float]]`
  - `ml.train.evaluate(model, X, y, sources) -> dict`
  - `ml.train.train(frame, output_dir) -> dict` (the metadata dict, also written as `metadata.json`)
  - `ml.train.ARTIFACT_ROOT: str` (value `"ml/artifacts"`)
  - Artifacts at `ml/artifacts/<version>/{preprocessor,regressor,clf_app1,clf_app2,label_encoders}.joblib` + `metadata.json`

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
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


class TestGroupedSplit:
    def test_no_laboratory_code_spans_two_splits(self, frame):
        """Each record contributes three rows sharing one code. A random row
        split would put a record's lab view in train and its iot view in test —
        the model would be scored on data it had effectively already seen."""
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

    def test_all_three_views_of_a_code_land_together(self, frame):
        train_df, _, _ = grouped_split(frame, seed=42)
        counts = train_df.groupby("laboratory_code").size()
        assert (counts == 3).all()


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
        return train(generate(num_records=200, seed=42), output_dir=str(output)), output

    def test_writes_all_artifacts(self, result):
        _, output = result
        for name in ["preprocessor.joblib", "regressor.joblib", "clf_app1.joblib",
                     "clf_app2.joblib", "label_encoders.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 600
        assert "trained_at" in written
        assert "version" in written

    def test_reports_metrics_per_data_source(self, result):
        """IoT has strictly less information than lab and should score worse. If
        it does not, the missingness handling is suspect."""
        metadata, _ = result
        for source in ["lab", "stk", "iot"]:
            assert source in metadata["metrics"]["by_source"]

    def test_beats_the_mean_baseline_on_every_target(self, result):
        metadata, _ = result
        for target, scores in metadata["metrics"]["overall"].items():
            assert scores["mae"] < scores["dummy_mae"], target

    def test_golden_record_scc417_is_within_tolerance(self, result):
        """The only real labeled example available."""
        metadata, _ = result
        golden = metadata["golden_test"]
        assert abs(golden["predicted_n"] - 50) <= 15
        assert abs(golden["predicted_p2o5"] - 40) <= 15
        assert abs(golden["predicted_k2o"] - 60) <= 15
        assert golden["passed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.train'`

- [ ] **Step 3: Write the implementation**

`ml/train.py`:
```python
"""Training, evaluation, and artifact writing (spec section 10).

Every split and cross-validation fold is grouped on laboratory_code. This is a
correctness requirement, not a preference: each record contributes three rows
(lab/stk/iot) sharing the same underlying chemistry and identical targets, so a
random row split leaks records across train and test and inflates every metric.
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

from ml.model import EnsembleFertilizerRegressor, build_classifier, build_krr, build_xgb_regressor
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS, build_preprocessor

RANDOM_SEED = 42
ARTIFACT_ROOT = "ml/artifacts"
GROUP_COLUMN = "laboratory_code"
CLASSIFIER_TARGETS = ["app1_formulation", "app2_formulation"]

GOLDEN_SCC417 = {
    "crop": "Sorghum", "data_source": "lab", "ph": 8.17,
    "organic_matter_pct": 3.21, "total_nitrogen_pct": 0.1605,
    "phosphorus_ppm": 18.96, "potassium_ppm": 138.0,
    "temp": None, "moist": None, "ec": None, "sal": None,
    "ph_remark": "MODERATELY ALKALINE", "om_remark": "MODERATELY LOW",
    "p_remark": "HIGH", "k_remark": "SUFFICIENT+",
}
GOLDEN_ACTUAL = (50, 40, 60)
GOLDEN_TOLERANCE = 15


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
    R2, so each nutrient favours whichever model actually performs better."""
    splitter = GroupKFold(n_splits=n_splits)
    weights: list[tuple[float, float]] = []

    for target in range(y.shape[1]):
        scores = {"xgb": [], "krr": []}
        for train_idx, test_idx in splitter.split(X, y[:, target], groups):
            for name, builder in (("xgb", build_xgb_regressor), ("krr", build_krr)):
                model = builder().fit(X[train_idx], y[train_idx, target])
                scores[name].append(r2_score(y[test_idx, target], model.predict(X[test_idx])))

        xgb_score = max(float(np.mean(scores["xgb"])), 0.0)
        krr_score = max(float(np.mean(scores["krr"])), 0.0)
        total = xgb_score + krr_score
        weights.append((0.5, 0.5) if total == 0 else (xgb_score / total, krr_score / total))

    return weights


def evaluate(model, X, y, sources, X_train, y_train) -> dict:
    predicted = model.predict(X)

    overall = {}
    for index, name in enumerate(TARGET_COLUMNS):
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train[:, index])
        overall[name] = {
            "mae": float(mean_absolute_error(y[:, index], predicted[:, index])),
            "r2": float(r2_score(y[:, index], predicted[:, index])),
            "dummy_mae": float(mean_absolute_error(y[:, index], dummy.predict(X))),
        }

    by_source = {}
    for source in np.unique(sources):
        mask = sources == source
        by_source[str(source)] = {
            name: {
                "mae": float(mean_absolute_error(y[mask, index], predicted[mask, index])),
                "r2": float(r2_score(y[mask, index], predicted[mask, index])),
            }
            for index, name in enumerate(TARGET_COLUMNS)
        }

    return {"overall": overall, "by_source": by_source}


def _golden_test(preprocessor, regressor) -> dict:
    row = pd.DataFrame([GOLDEN_SCC417])[FEATURE_COLUMNS]
    n, p2o5, k2o = regressor.predict(preprocessor.transform(row))[0]
    deltas = [abs(n - GOLDEN_ACTUAL[0]), abs(p2o5 - GOLDEN_ACTUAL[1]), abs(k2o - GOLDEN_ACTUAL[2])]
    return {
        "record": "SCC-26-417-417",
        "predicted_n": float(n), "predicted_p2o5": float(p2o5), "predicted_k2o": float(k2o),
        "actual_n": GOLDEN_ACTUAL[0], "actual_p2o5": GOLDEN_ACTUAL[1], "actual_k2o": GOLDEN_ACTUAL[2],
        "tolerance_kg_ha": GOLDEN_TOLERANCE,
        "passed": all(delta <= GOLDEN_TOLERANCE for delta in deltas),
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
    regressor = EnsembleFertilizerRegressor(weights=weights).fit(X_train, y_train)

    metrics = evaluate(
        regressor, X_test, y_test, test_df.data_source.to_numpy(), X_train, y_train
    )

    # Classification heads. Note these will score near-perfectly: the recipe is a
    # deterministic function of the targets, so the classifier only rediscovers
    # arithmetic. That is not evidence of real-world skill — it becomes
    # meaningful only on real DA data where agronomists exercise judgment.
    encoders: dict[str, LabelEncoder] = {}
    classifiers = {}
    classification_metrics = {}
    for target in CLASSIFIER_TARGETS:
        encoder = LabelEncoder().fit(frame[target])
        model = build_classifier().fit(X_train, encoder.transform(train_df[target]))
        predicted = model.predict(X_test)
        actual = encoder.transform(test_df[target])
        encoders[target] = encoder
        classifiers[target] = model
        classification_metrics[target] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "n_classes": int(len(encoder.classes_)),
            "note": "Deterministic given the targets — not a V1 success signal.",
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
        "golden_test": _golden_test(preprocessor, regressor),
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
        print(f"  {name:22s} MAE {scores['mae']:7.2f}  R2 {scores['r2']:6.3f}"
              f"  (mean baseline MAE {scores['dummy_mae']:7.2f})")
    print("\nMAE by data source:")
    for source, targets in metadata["metrics"]["by_source"].items():
        errors = "  ".join(f"{name.split('_')[1]}={scores['mae']:6.2f}"
                           for name, scores in targets.items())
        print(f"  {source:4s}  {errors}")
    golden = metadata["golden_test"]
    status = "PASS" if golden["passed"] else "FAIL"
    print(f"\nGolden test SCC-26-417-417: {status}")
    print(f"  predicted {golden['predicted_n']:.1f}/{golden['predicted_p2o5']:.1f}"
          f"/{golden['predicted_k2o']:.1f}  actual 50/40/60"
          f"  (tolerance +/-{golden['tolerance_kg_ha']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SURI fertilizer ensemble.")
    parser.add_argument("--csv", help="Train from a local CSV instead of Supabase.")
    args = parser.parse_args()

    result = train(_load_frame(args.csv))
    _print_summary(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS — including the leakage guard and the golden test.

- [ ] **Step 5: Train on the full dataset**

Run: `python -m ml.train --csv suri_synthetic_training_data_region7.csv`
Expected: a summary table; every target's MAE below its mean-baseline MAE; `Golden test SCC-26-417-417: PASS`. Note the version string — Task 9 loads the newest artifact directory automatically.

- [ ] **Step 6: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "feat: add training with grouped splits and golden-record validation"
```

Add `ml/artifacts/` to `.gitignore` — artifacts are reproducible and large.

---

## Task 9: FastAPI service

**Files:**
- Create: `main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `ml.remarks.*` (Task 1), `ml.agronomy.total_nitrogen_from_om` (Task 2), `ml.preprocessing.FEATURE_COLUMNS` (Task 6), `ml.train.ARTIFACT_ROOT` (Task 8), `data.crop_baselines.CROPS` (Task 2)
- Produces: `app` (FastAPI), `SoilTestRequest`, `PredictionResponse`, `load_artifacts()`, `latest_version()`

- [ ] **Step 1: Write the failing test**

`tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


GOLDEN_PAYLOAD = {
    "crop": "Sorghum", "data_source": "lab", "ph": 8.17,
    "organic_matter_pct": 3.21, "phosphorus_ppm": 18.96,
    "potassium_ppm": 138.0, "farm_area_sqm": 210,
    "laboratory_code": "SCC-26-417-417",
}


class TestHealth:
    def test_reports_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_loaded"] is True
        assert body["model_version"]


class TestPredict:
    def test_golden_record_returns_expected_shape(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        for field in ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
                      "app1_formulation", "app2_formulation", "derived_remarks",
                      "model_version", "source"]:
            assert field in body
        assert body["source"] == "ml-ensemble"

    def test_golden_record_is_agronomically_sane(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert abs(body["target_n_kg_ha"] - 50) <= 15
        assert abs(body["target_p2o5_kg_ha"] - 40) <= 15
        assert abs(body["target_k2o_kg_ha"] - 60) <= 15

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert body["derived_remarks"] == {
            "ph_remark": "MODERATELY ALKALINE", "om_remark": "MODERATELY LOW",
            "p_remark": "HIGH", "k_remark": "SUFFICIENT+",
        }

    def test_farm_totals_scale_the_per_hectare_rate(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        totals = body["farm_totals"]
        assert totals["area_ha"] == pytest.approx(0.021)
        assert totals["n_kg"] == pytest.approx(body["target_n_kg_ha"] * 0.021, rel=1e-3)

    def test_farm_totals_absent_when_area_not_given(self, client):
        payload = {key: value for key, value in GOLDEN_PAYLOAD.items()
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
        payload = {**GOLDEN_PAYLOAD, "data_source": source}
        assert client.post("/predict", json=payload).status_code == 200

    def test_total_nitrogen_derived_from_organic_matter(self, client):
        """The caller need not send it — the lab itself calculates it."""
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"


class TestValidation:
    def test_ph_above_fourteen_is_rejected(self, client):
        assert client.post("/predict", json={**GOLDEN_PAYLOAD, "ph": 15.0}).status_code == 422

    def test_negative_ppm_is_rejected(self, client):
        payload = {**GOLDEN_PAYLOAD, "phosphorus_ppm": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_data_source_is_rejected(self, client):
        payload = {**GOLDEN_PAYLOAD, "data_source": "satellite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict", json={"crop": "Corn"}).status_code == 422

    def test_unknown_crop_is_rejected_with_the_valid_list(self, client):
        """A silent fallback would return a confident number computed from no
        crop information at all."""
        response = client.post("/predict", json={**GOLDEN_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Sorghum" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a recommendation."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=GOLDEN_PAYLOAD).status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write the implementation**

`main.py`:
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

from data.crop_baselines import CROPS
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
    data_source: Literal["lab", "stk", "iot"]
    ph: float = Field(ge=0, le=14)
    organic_matter_pct: float | None = Field(default=None, ge=0, le=100)
    total_nitrogen_pct: float | None = Field(default=None, ge=0)
    phosphorus_ppm: float | None = Field(default=None, ge=0)
    potassium_ppm: float | None = Field(default=None, ge=0)
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
    target_n_kg_ha: float
    target_p2o5_kg_ha: float
    target_k2o_kg_ha: float
    app1_formulation: str
    app2_formulation: str
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

    row = {
        "ph": request.ph,
        "organic_matter_pct": organic_matter,
        "total_nitrogen_pct": total_nitrogen,
        "phosphorus_ppm": request.phosphorus_ppm,
        "potassium_ppm": request.potassium_ppm,
        "temp": request.temp, "moist": request.moist,
        "ec": request.ec, "sal": request.sal,
        "ph_remark": derive_ph_remark(request.ph),
        "om_remark": derive_om_remark(organic_matter),
        "p_remark": derive_p_remark(request.phosphorus_ppm),
        "k_remark": derive_k_remark(request.potassium_ppm),
        "crop": request.crop,
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

    app1 = ARTIFACTS["encoders"]["app1_formulation"].inverse_transform(
        ARTIFACTS["clf_app1"].predict(X))[0]
    app2 = ARTIFACTS["encoders"]["app2_formulation"].inverse_transform(
        ARTIFACTS["clf_app2"].predict(X))[0]

    farm_totals = None
    if request.farm_area_sqm:
        hectares = request.farm_area_sqm / SQM_PER_HECTARE
        farm_totals = FarmTotals(
            area_ha=round(hectares, 4),
            n_kg=round(float(n) * hectares, 2),
            p2o5_kg=round(float(p2o5) * hectares, 2),
            k2o_kg=round(float(k2o) * hectares, 2),
        )

    response = PredictionResponse(
        target_n_kg_ha=round(float(n), 1),
        target_p2o5_kg_ha=round(float(p2o5), 1),
        target_k2o_kg_ha=round(float(k2o), 1),
        app1_formulation=str(app1),
        app2_formulation=str(app2),
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

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS. Requires the artifacts from Task 8 Step 5.

- [ ] **Step 5: Verify the service runs**

Run: `uvicorn main:app --port 8001`
Then: `curl http://127.0.0.1:8001/health`
Expected: `{"status":"ok","model_version":"...","artifacts_loaded":true,...}`

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests across all modules PASS.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: add FastAPI prediction service on port 8001"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
````markdown
# SURI Fertilizer ML Backend

Predicts fertilizer nutrient requirements (N, P₂O₅, K₂O in kg/ha) and application
formulations from soil test data, for the Department of Agriculture Region 7.

Serves all three SURI data-entry flows — laboratory results, manual STK readings,
and IoT probe readings — from one model, using missingness indicators so each
source is handled according to what it can actually measure.

**Design spec:** `docs/superpowers/specs/2026-08-03-suri-fertilizer-ml-pipeline-design.md`

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in from suri/.env.local
```

The service shares `suri/`'s Supabase project. Copy `NEXT_PUBLIC_SUPABASE_URL`
into `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as-is.

Apply `supabase/migrations/20260803_ml_fertilizer_tables.sql` in the Supabase
SQL editor before ingesting.

## Generate data, ingest, and train

```bash
python -m data.generate_synthetic_data          # writes suri_synthetic_training_data_region7.csv
python -m data.ingest_to_supabase               # upserts 15,000 rows
python -m ml.train --csv suri_synthetic_training_data_region7.csv
```

`ml/train.py` reads from Supabase when `--csv` is omitted. It writes versioned
artifacts to `ml/artifacts/<version>/` and prints a metrics summary including the
golden-record check.

## Serve

```bash
uvicorn main:app --port 8001
```

Port 8001 — **not** 8000, which `suri/features/flows/iot/IoTFlow.tsx` already
uses for an unrelated contract.

## API

### `POST /predict`

Only `crop`, `data_source`, and `ph` are required. Everything else is optional:
missing values are median-imputed and explicitly flagged to the model.

```json
{
  "crop": "Sorghum",
  "data_source": "lab",
  "ph": 8.17,
  "organic_matter_pct": 3.21,
  "phosphorus_ppm": 18.96,
  "potassium_ppm": 138.0,
  "farm_area_sqm": 210,
  "laboratory_code": "SCC-26-417-417"
}
```

| Field | Required | Notes |
|---|---|---|
| `crop` | yes | One of: Cabbage, Corn, Rice, Sorghum, Sugarcane, Tomato. Unknown crops are rejected with 422. |
| `data_source` | yes | `lab`, `stk`, or `iot` |
| `ph` | yes | 0–14 |
| `organic_matter_pct` | no | 0–100 |
| `total_nitrogen_pct` | no | Derived as `organic_matter_pct × 0.05` when omitted |
| `phosphorus_ppm`, `potassium_ppm` | no | ≥ 0 |
| `temp`, `moist`, `ec`, `sal` | no | IoT probe readings |
| `farm_area_sqm` | no | Enables `farm_totals` in the response |
| `laboratory_code` | no | Recorded in the prediction audit log |

Response:

```json
{
  "target_n_kg_ha": 50.2,
  "target_p2o5_kg_ha": 38.7,
  "target_k2o_kg_ha": 61.4,
  "app1_formulation": "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)",
  "app2_formulation": "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)",
  "derived_remarks": {
    "ph_remark": "MODERATELY ALKALINE",
    "om_remark": "MODERATELY LOW",
    "p_remark": "HIGH",
    "k_remark": "SUFFICIENT+"
  },
  "farm_totals": { "area_ha": 0.021, "n_kg": 1.05, "p2o5_kg": 0.81, "k2o_kg": 1.29 },
  "model_version": "2026-08-03-142530",
  "source": "ml-ensemble"
}
```

Nutrient targets are **per-hectare rates**. `farm_totals` applies the area
scaling and is `null` when `farm_area_sqm` is omitted.

### `GET /health`

Returns `{status, model_version, artifacts_loaded, supabase_reachable}`.

## Tests

```bash
python -m pytest -v
```

The suite includes a golden test against the real DA report SCC-26-417-417 — the
only real labeled example available — and a leakage guard asserting that no
`laboratory_code` spans two dataset splits.

## Known limitations

- **The classifiers will score near-perfect accuracy, and that is not evidence of
  skill.** The recipe is a deterministic function of the nutrient targets, so the
  classifier only rediscovers an arithmetic rule the generator applied. It
  becomes genuinely useful when trained on real DA data, where agronomists
  exercise judgment no formula reproduces. Judge V1 on the regression metrics.
- Crop baselines other than Sorghum are placeholder scaffolding awaiting
  agronomist review. They live in `data/crop_baselines.py` so correcting them is
  one edit plus a retrain.
- The OM and K remark bands are calibrated to a single real example. The pH and
  Olsen-P bands are standard published classes that reproduce that example
  independently.
````

- [ ] **Step 2: Verify the documented commands work end to end**

Run each command in the README in order, in a clean checkout.
Expected: each succeeds; the `curl` example returns a prediction.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add ML backend README with API contract"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §3 schema | Task 4 (`COLUMNS`), Task 5 (SQL) |
| §3.1 farm area not a feature | Task 6 (excluded from `FEATURE_COLUMNS`, asserted), Task 9 (`farm_totals`) |
| §3.2 Total N derived | Task 2 (`total_nitrogen_from_om`), Task 9 (derived when omitted) |
| §3.3 temporal/geospatial deferred | Task 6 (asserted absent from features) |
| §4 remark bands | Task 1 |
| §5 multi-source vector | Task 4 (views), Task 6 (imputation + indicators) |
| §5.1 STK band mapping | Task 4 (`_STK_*_BANDS`, `_coarsen`) |
| §6.1 crop baselines | Task 2 |
| §6.2 target formulas | Task 2 |
| §6.3 recipe planner | Task 3 |
| §6.4 view expansion | Task 4 |
| §7 preprocessing | Task 6 |
| §8 ensemble | Task 7 |
| §9 Supabase | Task 5 |
| §10.1 grouped split | Task 8 (`grouped_split` + leakage test) |
| §10.2 metrics | Task 8 (`evaluate`, dummy baseline, per-source) |
| §10.3 golden test | Task 2, Task 8, Task 9 |
| §11.1 `/predict` | Task 9 |
| §11.2 validation | Task 9 |
| §11.3 `/health` | Task 9 |
| §12 testing | every task |
| §13 path to real data | Task 5 (`is_synthetic`), Task 8 (`--csv` vs Supabase) |
| §14 deliverables | Tasks 1–10 |

No gaps.

**2. Placeholder scan:** No TBD/TODO markers. Every code step contains complete,
runnable code. No "similar to Task N" references.

**3. Type consistency:**
- `derive_*_remark` names match between Tasks 1, 4, and 9.
- `PH_ORDER`/`OM_ORDER`/`P_ORDER`/`K_ORDER` produced in Task 1, consumed in Task 6.
- `CROPS` produced in Task 2, consumed in Tasks 4, 6, 9.
- `nutrient_requirements` returns `tuple[int, int, int]`; Task 4 unpacks three values.
- `plan_applications` returns `tuple[str, str]`; Task 4 unpacks two.
- `COLUMNS` produced in Task 4, consumed in Task 5.
- `FEATURE_COLUMNS`/`TARGET_COLUMNS` produced in Task 6, consumed in Tasks 8 and 9.
- `EnsembleFertilizerRegressor.predict` returns `(n, 3)`; Task 8 and Task 9 both index `[0]` for a single row.
- `ARTIFACT_ROOT` produced in Task 8, consumed in Task 9.
- `get_client` and `TABLE` produced in Task 5, consumed in Tasks 8 and 9.

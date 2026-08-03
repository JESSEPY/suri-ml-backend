# SURI Fertilizer ML Pipeline — Design Spec

## 1. Purpose & Scope

Build a standalone machine learning microservice, `suri-ml-backend/`, that predicts fertilizer
nutrient requirements (N, P₂O₅, K₂O in kg/ha) and standard application formulations from soil
test data. This is a **new, separate capability** from the existing BSWM Fertilizer Guide Map RAG
pipeline (`suri/docs/11-rag-fertilizer-pipeline.md`) — the RAG pipeline retrieves recommendations
from published crop/region guide tables; this service predicts directly from **measured soil
chemistry**.

**Eventual goal** (not built this session): all three of `suri/`'s manual data-entry flows — IoT
(`features/flows/iot/IoTFlow.tsx`), Manual/STK (`features/flows/manual/ManualFlow.tsx`), and Lab
Results (`features/flows/lab-results/LabResultsFlow.tsx`) — call this service and treat its output
as the primary fertilizer score, with the RAG pipeline supplying supplementary crop/region context.

**This session's scope:** the standalone service only — data synthesis, Supabase schema,
preprocessing, ensemble model, training script, evaluation, and the FastAPI `/predict` contract.
No changes to `suri/` frontend or backend code. No OCR/photo ingestion of lab reports (structured
JSON only).

**Service topology:** new FastAPI app in `suri-ml-backend/`, its own `requirements.txt`, runs on
**port 8001**. This deliberately avoids port 8000, which `IoTFlow.tsx` already hard-codes for a
*different*, currently-unimplemented contract (`{ml_capability_score, recommendations: crop[]}`,
a crop-suitability classifier from raw sensor readings) — unrelated to fertilizer prediction and
out of scope here.

## 2. Grounding: What the Real Documents Actually Say

Reference material found in `suri-ml-backend/`:

- `SURI Machine Learning Data Requirements Matrix V2.pdf` — the authoritative schema spec (read
  in full; supersedes assumptions from the master prompt where they conflict).
- `LS-0040-SOIL-ANALYSIS-REQUEST-FORM-rev11.pdf` — blank BSWM lab *request* form (test methods +
  fees), not a filled report. Confirms exact test methods: pH via Potentiometric (1:1); Available
  P via Olsen/Bray-1; Exchangeable K via Ammonium Acetate pH 7; OM via Walkley-Black; **Total N is
  calculated as OM × 0.05, not independently measured** (confirmed exactly by the real report
  below: 3.21% OM → 0.1605% N).
- `LS-0042-FERTILIZER-ANALYSIS-REQUEST-FORM-rev9.pdf` — **not** the target-schema source, despite
  the master prompt's claim. It's a request form for testing a *fertilizer product's own*
  guaranteed composition (e.g. verifying a bag labeled "14-14-14" actually contains 14% N/P₂O₅/K₂O)
  — unrelated to per-farm application recommendations.
- Three real DA Region 7 Regional Soils Laboratory documents for the same farmer (Feyme R.
  Basmayor, Jampang, Argao, Cebu; **Lab Code SCC 417**, paired across two report types):
  - **Soil Chemical Test Report**: pH 8.17 → `MODERATELY ALKALINE`; OM 3.21% → `MODERATELY LOW`;
    Avail. P 18.96 ppm → `HIGH`; Exch. K 138.00 ppm → `SUFFICIENT+`; Total N 0.1605% (calculated,
    no remark).
  - **Fertilizer Recommendation Report** (a distinct report type, generated *from* the soil test,
    explicitly cross-referencing Test Report Number SCC-26-417-417): Crop Sorghum, Farm Area 210
    sq.m, Nutrient Requirement N=50/P₂O₅=40/K₂O=60 kg/ha, First Application = `143 kg/ha (14-14-14)
    + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)` at planting, Second Application = same three products
    at 1 month after planting, Optional = `10-20 bags/ha Organic Fertilizer`.

The existing draft `lab-results-synthesizer.py` already used this exact lab code and these exact
blend strings — its author was clearly working from this real report, just hadn't connected the
target-generation *logic* to the inputs (targets were assigned uniformly at random).

**This SCC-417 record is the only real labeled example available.** It is used as a golden test
fixture (§10.3, §12) and as the anchor for the Sorghum baseline (§6.1).

## 3. Data Schema (Core Routine Set, per the Matrix)

One record = one paired (Soil Chemical Test Report, Fertilizer Recommendation Report), joined by
**Laboratory Code** — the Matrix's own pairing rule: "An orphaned soil test without its
corresponding recommendation cannot be used to train the model."

| Field | Type | Role |
|---|---|---|
| `laboratory_code` | text | join key + **grouping key for train/test splits** (§10.1) |
| `data_source` | enum `lab`\|`stk`\|`iot` | **input feature** (one-hot) |
| `sampling_site`, `date_of_sampling` | text, date | metadata, **not** model features in V1 (§3.3) |
| `crop` | categorical | **input feature** (one-hot) |
| `farm_area_sqm` | float | **stored, not a model feature** (§3.1) |
| `ph`, `organic_matter_pct`, `phosphorus_ppm`, `potassium_ppm` | float | **input features** |
| `total_nitrogen_pct` | float | **derived** = `organic_matter_pct * 0.05` (§3.2) |
| `ph_remark`, `om_remark`, `p_remark`, `k_remark` | ordered categorical | **input features** (ordinal-encoded, §7). No remark tier exists for N — it's calculated, not rated. |
| `target_n_kg_ha`, `target_p2o5_kg_ha`, `target_k2o_kg_ha` | float | **regression targets** |
| `app1_formulation`, `app2_formulation` | categorical | **classification targets** — one label each from a finite recipe vocabulary (§6.3), matching the Matrix's explicit "Multi-class Target" framing |

`laboratory_code` is unique per *record* but **not** unique per row — each record produces three
rows (one per `data_source` view, §5). The table's primary key is `(laboratory_code, data_source)`.

### 3.1 Farm Area is a post-prediction scaler, not a model feature

The real Fertilizer Recommendation Report gives N/P₂O₅/K₂O requirement and both applications as
**kg/ha rates**, identical regardless of farm size. Farm area's role (per the Matrix: "Scaling
Feature: Normalizes target fertilizer volumes") is to convert the model's per-hectare output into
a total-product-needed number for a specific farm (`rate_kg_ha * farm_area_sqm / 10000`) — a
downstream multiplication, not something the regressor should condition on. Feeding it as a
feature would invite the model to learn a spurious area↔dosage correlation that does not exist in
the real reports. Stored on every record for the scaling calculation; excluded from the feature
vector.

### 3.2 Total Nitrogen is collinear with OM by construction

Because Total N ≡ OM × 0.05 exactly, the two columns are perfectly correlated wherever both are
present. Decision: **keep both**, because (a) the Matrix lists both as distinct input features,
(b) tree models are unaffected by the redundancy, and (c) their *missingness patterns differ* for
the IoT view, where neither is present but the missingness indicator still carries signal. The
`StandardScaler` + KRR path is mildly affected by collinearity; the L2 (ridge) penalty in
`KernelRidge` is precisely the mechanism that handles it, which is one of the stated reasons for
choosing KRR.

### 3.3 Temporal and geospatial features are deferred, deliberately

The Matrix lists `date_of_sampling` as a "Temporal Feature" and `sampling_site` as a "Geospatial
Feature". Both are excluded from V1's feature vector for one concrete reason: **in synthetic data
they carry no signal**. The generator assigns dates and sites at random, uncorrelated with soil
chemistry, so including them would train the model on pure noise and risk it memorizing
site-specific artifacts that do not generalize. Both columns are stored on every record so they
become available the moment real historical DA lab data arrives (§13), at which point seasonality
and regional soil-profile effects become learnable and worth revisiting.

## 4. Remark Derivation Bands

Remarks are a deterministic function of their numeric parameter. Rather than treating them as
independently-supplied categorical inputs, the service **derives them from the numeric values**
whenever the numeric is present, for all three data sources. A caller-supplied remark is accepted
and used as-is when the corresponding numeric is absent; when both are present, the derived value
wins (the lab's own numeric is more trustworthy than a transcribed label). Only when both are
missing does the feature fall back to an `UNKNOWN` category.

This resolves an inconsistency in an earlier draft, which would have left STK and IoT records with
`UNKNOWN` for all four remarks purely because those flows don't display remark text — even though
their numeric values determine the remark perfectly well.

| Parameter | Band | Remark |
|---|---|---|
| pH | < 4.5 | `STRONGLY ACIDIC` |
| | 4.5 – 5.5 | `MODERATELY ACIDIC` |
| | 5.6 – 6.5 | `SLIGHTLY ACIDIC` |
| | 6.6 – 7.3 | `NEUTRAL` |
| | 7.4 – 7.8 | `SLIGHTLY ALKALINE` |
| | 7.9 – 8.4 | `MODERATELY ALKALINE` |
| | > 8.4 | `STRONGLY ALKALINE` |
| OM (%) | < 1.0 | `VERY LOW` |
| | 1.0 – 2.0 | `LOW` |
| | 2.0 – 3.4 | `MODERATELY LOW` |
| | 3.4 – 5.0 | `MODERATE` |
| | > 5.0 | `HIGH` |
| Avail. P (ppm, Olsen) | < 5 | `VERY LOW` |
| | 5 – 10 | `LOW` |
| | 10 – 17 | `MEDIUM` |
| | > 17 | `HIGH` |
| Exch. K (ppm) | < 60 | `DEFICIENT` |
| | 60 – 120 | `SUFFICIENT` |
| | > 120 | `SUFFICIENT+` |

**Provenance and confidence — important.** The pH classes are the standard USDA/BSWM soil reaction
classes, and the Olsen-P classes are the standard Olsen interpretation; both **independently
reproduce the real SCC-417 report** (8.17 → `MODERATELY ALKALINE`, 18.96 → `HIGH`) without having
been fitted to it, which is meaningful corroboration. The OM and K bands are **calibrated to be
consistent with that single real example** (3.21 → `MODERATELY LOW`, 138.00 → `SUFFICIENT+`) and
are therefore lower-confidence: they reproduce one observation and are otherwise conventional.
Before production use, all four tables should be confirmed against DA RSL's published
interpretation chart. A band-boundary change is a one-constant edit in `ml/remarks.py` plus a
retrain — cheap to correct, which is why it is not worth blocking V1 on.

## 5. Multi-Source Feature Vector (Lab / STK / IoT)

The three real `suri/` flows supply genuinely different measurements, not just "the same fields,
sometimes null":

| Flow | Supplies | Missing vs. lab panel |
|---|---|---|
| **Lab Results** (`LabResultsFlow.tsx`) | pH, OM%, Total N%, P ppm, K ppm — full lab panel, matches `LS-0040` exactly | nothing |
| **Manual/STK** (`ManualFlow.tsx`, `stk-options.ts`) | pH, plus qualitative P and K bands | Total N% and OM% are **not directly measured** — the STK's "N" reading is a color proxy for **OM banding** (its own hint text reads `< 4% / 4-8% / > 8% organic matter`), from which OM and then Total N are derived |
| **IoT** (`IoTFlow.tsx`) | pH, sensor N/P/K (raw capacitive probe units), temp, moist, EC, salinity | OM% and Total N% — no lab-equivalent measurement exists; genuinely missing and imputed |

### 5.1 STK values are qualitative bands, not lab units

An earlier draft said the STK view should "snap values to the nearest STK bucket." That is wrong
and would have corrupted the STK training view. Checking `stk-options.ts` against real lab ranges:
STK P options carry values `30 / 70 / 110` while real Olsen P spans roughly 5–40 ppm, and the STK
N options `50 / 100 / 150` are nowhere near a Total-N percentage. **The STK `value` fields are an
internal app scale, not lab units.**

The correct treatment: each STK band maps to a *representative range in lab units*, and the
generator samples within that range.

| STK selection | Maps to |
|---|---|
| N color `Orange/Brown` (Low) | OM ~ U(0.5, 2.0) % → Total N = OM × 0.05 |
| N color `Olive Green` (Medium) | OM ~ U(2.0, 3.5) % |
| N color `Navy Blue` (High) | OM ~ U(3.5, 6.0) % |
| P `Light Pastel Blue` (Low) | Avail. P ~ U(3, 10) ppm |
| P `Standard Blue` (Medium) | Avail. P ~ U(10, 20) ppm |
| P `Deep Navy Blue` (High) | Avail. P ~ U(20, 40) ppm |
| K `Slightly Cloudy Orange` (Low) | Exch. K ~ U(30, 80) ppm |
| K `Moderately Cloudy Orange` (Medium) | Exch. K ~ U(80, 150) ppm |
| K `Opaque Milky Orange` (High) | Exch. K ~ U(150, 250) ppm |
| pH indicator | used directly — the STK pH options (4.5 / 5.5 / 6.8 / 7.8) *are* on the real pH scale |

The STK bands deliberately overlap the lab distribution, so the STK view is a **coarsened** version
of the lab view — exactly the real-world relationship, and what the model must learn to handle.

### 5.2 Design

A canonical superset feature vector — `ph`, `organic_matter_pct`, `total_nitrogen_pct`,
`phosphorus_ppm`, `potassium_ppm`, plus optional `temp`, `moist`, `ec`, `sal` — with a required
`data_source` field. Preprocessing uses `SimpleImputer(strategy="median", add_indicator=True)` per
continuous field, producing both an imputed value *and* a missingness-flag column, so the model
explicitly learns "this wasn't measured" rather than trusting a silent guess. `data_source` is
one-hot encoded so the model can learn source-specific reliability.

**Training data must match real inference-time missingness.** The generator produces the full
lab-grade record, then derives three training views from it (§6.4): `lab` (full panel), `stk`
(coarsened per §5.1, OM derived from N-band), `iot` (OM/Total N missing, P/K with sensor noise,
temp/moist/EC/sal present). All three views share the record's `laboratory_code` and its targets.

## 6. Synthetic Data Generation

Revise `lab-results-synthesizer.py` into `data/generate_synthetic_data.py`. The current draft
assigns `target_req_n/p/k` and `app1/app2` **uniformly at random**, disconnected from the inputs —
a model trained on that learns nothing but the mean. This section specifies the actual generative
logic, which is the single largest determinant of whether the resulting model has any skill.

Fixed random seed for reproducibility. 5,000 base records, 15,000 rows after view expansion.
Region 7 municipality list and crop list retained from the draft script.

### 6.1 Crop baseline requirements (kg/ha)

| Crop | N | P₂O₅ | K₂O |
|---|---|---|---|
| Sorghum | 50 | 40 | 60 |
| Corn | 90 | 60 | 60 |
| Rice | 90 | 30 | 30 |
| Cabbage | 120 | 90 | 90 |
| Tomato | 100 | 80 | 80 |
| Sugarcane | 120 | 60 | 120 |

Sorghum is anchored exactly to the real SCC-417 report (50/40/60). The other five are conventional
Philippine baseline rates, used **as placeholder scaffolding** — V1 is a working template awaiting
the real DA export (§13), so these are not blocking on agronomist sign-off. They set the model's
central tendency per crop, so they will need review before any production use; they live in one
constant table (`data/crop_baselines.py`) so that review is a single edit plus a retrain, with no
code changes anywhere else.

### 6.2 Soil-test adjustments to baseline

Each nutrient's requirement is its crop baseline adjusted by what the soil already supplies. All
results clamped to sane ranges (N/K₂O ∈ [20, 200], P₂O₅ ∈ [20, 150]) and rounded to whole kg/ha,
matching how the real reports present them.

- **N** — soil OM is the nitrogen reserve, so higher OM reduces applied N:
  `n = base_n * (1 - 0.35 * clamp((om - 2.5) / 3.5, -1, 1))`
- **P₂O₅** — higher available P reduces applied P, *and* a pH availability penalty applies:
  `p = base_p * (1 - 0.40 * clamp((p_ppm - 15) / 25, -1, 1)) * ph_p_penalty(ph)`
- **K₂O** — higher exchangeable K reduces applied K:
  `k = base_k * (1 - 0.40 * clamp((k_ppm - 120) / 130, -1, 1))`

**The pH penalty on phosphorus is intentionally non-linear.** Phosphorus availability genuinely
peaks near pH 6.5 and falls off in both directions (fixed by iron/aluminium in acid soils, by
calcium in alkaline soils). Modeled as an inverted-U:
`ph_p_penalty(ph) = 1 + 0.30 * min((ph - 6.5)² / 4.0, 1.0)` — up to +30% more P₂O₅ required at pH
extremes. This is real agronomy, and it is also the specific reason the architecture needs XGBoost
rather than a purely linear model: a genuine non-monotonic interaction between two input features
that a linear regressor cannot represent and a tree ensemble can.

Small Gaussian noise (σ = 3% of value) is added to each target so the mapping is not perfectly
deterministic — otherwise the models would fit it exactly and every metric would be meaninglessly
perfect.

### 6.3 Recipe planner and the classification vocabulary

Given the rounded N/P₂O₅/K₂O targets, a rule-based planner selects products from a standard
Philippine fertilizer catalog and emits the compound recipe string:

| Product | Grade |
|---|---|
| Complete | 14-14-14 |
| Ammosul | 21-0-0 |
| Urea | 46-0-0 |
| Muriate of Potash | 0-0-60 |
| Ammophos | 16-20-0 |
| Solophos | 0-18-0 |

The planner covers the P₂O₅ requirement with a Complete/Ammophos base, tops up K with MOP and N
with Urea/Ammosul, splits the result across **First Application** (basal, at planting) and
**Second Application** (topdress, 1 month after planting), and formats each as a `+`-joined string
of `"<kg> kg/ha (<grade>)"` terms — the exact format of the real report. The App-2 vocabulary
additionally includes `"None"` for low-requirement cases (the draft script's `APP2_BLENDS` already
anticipated this). The classification vocabulary is the set of distinct strings the planner emits
across all 5,000 records; quantities are rounded to whole kg/ha to keep that set finite and each
class populated.

**Known limitation, stated plainly.** Because the planner is deterministic given the targets, the
classifiers are learning a deterministic function and will score near-perfect accuracy. **That is
not evidence of real-world skill** — it measures only that the model rediscovered an arithmetic
rule the generator applied. The classification head becomes genuinely useful only when trained on
real DA data, where human agronomists exercise judgment (product availability, farmer budget, local
practice) that no formula reproduces. §10.2 reports classifier accuracy but explicitly does not
treat it as a V1 success signal; the regression metrics are what matter. This limitation is
inherent to bootstrapping from synthetic data, not a defect in the design, and it resolves itself
on the first real data import.

### 6.4 View expansion

Each base record emits three rows sharing one `laboratory_code` and identical targets: `lab` (full
panel), `stk` (§5.1 coarsening), `iot` (OM/Total-N dropped, P/K with ~10% sensor noise,
temp/moist/EC/sal sampled from plausible Region 7 field ranges).

## 7. Preprocessing

`ml/preprocessing.py`, one `ColumnTransformer`:

- **Continuous** (ph, organic_matter_pct, total_nitrogen_pct, phosphorus_ppm, potassium_ppm, temp,
  moist, ec, sal): `SimpleImputer(strategy="median", add_indicator=True)` → `StandardScaler`
- **Ordered categorical** (ph_remark, om_remark, p_remark, k_remark): `OrdinalEncoder` with
  explicitly specified category order, `handle_unknown="use_encoded_value"`, `unknown_value=-1`.
  Ordinal rather than one-hot because these bands are genuinely ordered (`LOW` < `MEDIUM` <
  `HIGH`) — one-hot discards that ordering, which matters for the KRR half of the ensemble.
- **Nominal categorical** (crop, data_source): `OneHotEncoder(handle_unknown="error")` — see §11.2
  for why `error` and not `ignore`.

## 8. Model Architecture

Two heads sharing the preprocessor output.

**Regression** (3 continuous targets). sklearn's `VotingRegressor` does not support multi-output,
so `ml/model.py` defines an `EnsembleFertilizerRegressor` wrapper holding one `VotingRegressor` per
target, each combining:
- `XGBRegressor` — captures the non-linear pH/P interaction (§6.2) and categorical splits
- `KernelRidge(kernel="rbf")` — L2-regularized smooth fit, resists overfitting the synthetic set

The wrapper exposes a single `fit(X, y)` / `predict(X) -> ndarray(n, 3)`. Voting weights are set
per target from `GroupKFold` cross-validated R² (§10.1), so each nutrient uses whichever model
actually performs better for it rather than a global guess.

**Classification** (app1_formulation, app2_formulation): two independent `XGBClassifier`s, one per
application slot, each over its own recipe vocabulary (§6.3).

Artifacts are versioned — `ml/artifacts/<version>/{preprocessor,regressor,clf_app1,clf_app2}.joblib`
plus `metadata.json` recording training date, row count, seed, metrics, and the recipe vocabulary.
`/health` reports the loaded version (§11.3).

## 9. Supabase

**Confirmed: the same Supabase project as `suri/`**, reusing `NEXT_PUBLIC_SUPABASE_URL` /
`SUPABASE_SERVICE_ROLE_KEY` from `suri/.env.local`. Two new
tables, migration in `supabase/migrations/20260803_ml_fertilizer_tables.sql`, following the
existing convention in `suri/supabase/migrations/` (RLS enabled, admin-role policies):

- `ml_soil_training_data` — the training corpus, schema per §3, PK `(laboratory_code, data_source)`.
  Holds synthetic rows now and real DA historical exports later (§13).
- `ml_fertilizer_predictions` — append-only log of live `/predict` requests + responses + artifact
  version, for audit and future retraining.

`data/ingest_to_supabase.py` loads the CSV and upserts into `ml_soil_training_data` via
`supabase-py` in batches, idempotent on the composite PK so re-running does not duplicate.

## 10. Training & Evaluation

`ml/train.py` — reads from Supabase (or `--csv` for local runs), trains, evaluates, writes versioned
artifacts.

### 10.1 Splitting — group by laboratory_code, never by row

**This is a correctness requirement, not a preference.** Each record contributes three rows
(lab/stk/iot) sharing the same underlying soil chemistry and identical targets. A random row split
would place the `lab` view of a record in train and its `iot` view in test — the model would be
evaluated on a record it had effectively already seen, inflating every metric and hiding real
generalization failure.

All splitting and cross-validation is grouped on `laboratory_code`: `GroupShuffleSplit` for the
70/15/15 train/val/test split, `GroupKFold(n_splits=5)` for the CV that sets ensemble weights.

### 10.2 Metrics

- **Regression** (the metrics that matter for V1): MAE and R² per target, reported **both overall
  and broken down per `data_source`**. The per-source breakdown is the point — `iot` has strictly
  less information than `lab` and *should* score worse; if it doesn't, the missingness handling is
  suspect. Sanity gate: MAE < 10 kg/ha per target on the held-out test set.
- **Baseline comparison**: `DummyRegressor(strategy="mean")` must be clearly beaten on every
  target. This is the specific guard against the original draft's failure mode — with randomly
  assigned targets the ensemble and the dummy score identically, and that equality is the
  detectable symptom.
- **Classification**: accuracy and macro-F1 per slot, plus per-class support to surface rare
  recipes. Reported for completeness but explicitly **not** a V1 success signal (§6.3).

Metrics are written to `metadata.json` alongside the artifacts and printed as a summary table.

### 10.3 Golden test — the real SCC-417 record

The single most valuable check available: feed the real soil values (pH 8.17, OM 3.21%, Total N
0.1605%, P 18.96 ppm, K 138.00 ppm, crop Sorghum) and assert the predicted N/P₂O₅/K₂O falls within
tolerance of the real lab's answer (50/40/60 kg/ha). Tolerance ±15 kg/ha initially — loose, because
it is one data point and the synthetic generator only approximates DA's real rule, but a prediction
outside that band means §6.2's generative logic disagrees with real agronomic practice badly enough
to investigate before shipping.

## 11. API

`main.py`, port 8001. CORS configured for `http://localhost:3000` (matching the existing
`suri/backend/main.py` convention) so the Next.js app can call it once wiring begins.

### 11.1 `POST /predict`

Request — only `crop`, `data_source`, and `ph` are required; all else optional and imputed+flagged:
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
`total_nitrogen_pct` is derived from OM when omitted (§3.2); remarks are derived from numerics
(§4); `temp`/`moist`/`ec`/`sal` accepted for the `iot` source.

Response:
```json
{
  "target_n_kg_ha": 50.2,
  "target_p2o5_kg_ha": 38.7,
  "target_k2o_kg_ha": 61.4,
  "app1_formulation": "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)",
  "app2_formulation": "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)",
  "derived_remarks": {
    "ph_remark": "MODERATELY ALKALINE", "om_remark": "MODERATELY LOW",
    "p_remark": "HIGH", "k_remark": "SUFFICIENT+"
  },
  "farm_totals": { "n_kg": 1.05, "p2o5_kg": 0.81, "k2o_kg": 1.29, "area_ha": 0.021 },
  "model_version": "2026-08-03-01",
  "source": "ml-ensemble"
}
```
`farm_totals` applies the §3.1 scaling and is present only when `farm_area_sqm` is supplied.
`derived_remarks` is returned so callers can display the same agronomic labels a farmer would see
on a real lab report — useful for all three flows, and it makes the model's interpretation of the
inputs visible rather than hidden.

### 11.2 Validation and error handling

- Pydantic bounds: `ph` ∈ [0, 14], `organic_matter_pct` ∈ [0, 100], ppm values ≥ 0,
  `farm_area_sqm` > 0, `data_source` ∈ enum. Violations → 422 naming the offending field.
- **Unknown `crop` → 422 listing valid crops**, not a silent fallback. This is why §7 specifies
  `handle_unknown="error"` on the crop encoder: `"ignore"` would emit an all-zeros crop vector and
  return a confident-looking prediction computed from no crop information at all. A farmer acting
  on that number could over- or under-fertilize a real field. Failing loudly is the only safe
  behavior.
- **Missing/corrupt artifacts at startup → the service refuses to start**, logging the missing
  path. Failing at boot beats serving 500s per request or, worse, silently loading a stale model.
- **Supabase logging failure must never fail the prediction.** The write to
  `ml_fertilizer_predictions` is wrapped and, on failure, logs a warning and returns the prediction
  anyway — the same graceful-degrade pattern the existing RAG pipeline uses for non-essential
  paths. An audit-log outage is not a reason to deny a farmer their recommendation.

### 11.3 `GET /health`

Returns `{status, model_version, artifacts_loaded, supabase_reachable}` — enough for a deployment
check to distinguish "process up" from "process up and actually able to serve".

## 12. Testing

- **Unit — remark derivation** (`ml/remarks.py`): every band boundary, including exact-boundary
  values, plus the real SCC-417 values as a fixture (8.17 → `MODERATELY ALKALINE`, 3.21 →
  `MODERATELY LOW`, 18.96 → `HIGH`, 138.00 → `SUFFICIENT+`).
- **Unit — derivations**: `total_nitrogen_pct == organic_matter_pct * 0.05` (asserted against the
  real 3.21 → 0.1605); farm-area scaling arithmetic.
- **Unit — recipe planner**: emits well-formed strings, positive quantities, vocabulary finite and
  stable across runs for a fixed seed.
- **Unit — preprocessing**: expected output column count; missingness indicators fire exactly for
  absent inputs; ordinal encoders preserve the specified band order.
- **Integration — training**: train on a small synthetic sample end-to-end; assert prediction shape
  `(n, 3)`, values within clamp ranges, and that the ensemble beats `DummyRegressor` (§10.2).
- **Integration — leakage guard**: assert no `laboratory_code` appears in more than one split. A
  direct regression test for the §10.1 bug.
- **API**: happy path per `data_source`; minimal-payload path (only the 3 required fields);
  validation failures (bad pH, unknown crop); Supabase-unavailable path returns 200 with a logged
  warning; `/health` shape.
- **Golden test**: §10.3, run as part of the suite against the trained artifacts.

## 13. Path to Real Data

The Matrix is explicit that the real corpus should be requested from DA RSL as **flat CSV/Excel,
not PDFs**, with soil test and fertilizer recommendation strictly paired on Laboratory Code. When
that export arrives:

1. It loads into the same `ml_soil_training_data` table with `data_source = "lab"` — no schema
   change, because §3 was designed against the real report structure.
2. Synthetic rows stay distinguishable (an `is_synthetic` boolean column) so training can weight
   real records higher or drop synthetic ones entirely once volume permits.
3. The classification head becomes genuinely meaningful for the first time (§6.3).
4. The §3.3 temporal/geospatial features become worth adding.

## 14. Deliverables

```
suri-ml-backend/
  data/generate_synthetic_data.py     # §6 — agronomically grounded, 3 views/record, seeded
  data/crop_baselines.py              # §6.1 — isolated for agronomist correction
  data/ingest_to_supabase.py          # §9
  supabase/migrations/20260803_ml_fertilizer_tables.sql
  ml/remarks.py                       # §4 — band tables + derivation
  ml/recipes.py                       # §6.3 — product catalog + planner
  ml/preprocessing.py                 # §7 — ColumnTransformer
  ml/model.py                         # §8 — EnsembleFertilizerRegressor + classifiers
  ml/train.py                         # §10 — training, grouped eval, versioned artifacts
  ml/artifacts/<version>/             # joblib artifacts + metadata.json
  main.py                             # §11 — FastAPI, POST /predict, GET /health, port 8001
  tests/                              # §12
  requirements.txt
  README.md                           # run/train/serve + exact request/response contract
```

## 15. Explicitly Out of Scope

- Wiring `suri/` frontend flows (IoT/Manual/Lab Results) to call this service
- Modifying `lib/ai/fertilizer-synthesis.ts` or any Next.js route
- OCR/photo parsing of lab report images
- Matrix §2 extended parameters (Physical, Salinity/Alkalinity, Micronutrients) — explicitly
  optional in the Matrix and contingent on specialized datasets that do not exist yet. The schema
  and `ColumnTransformer` stay additive-friendly so these can be appended without restructuring.
- Resolving the pre-existing, unrelated `IoTFlow.tsx` → `:8000/predict` crop-capability contract
- Model retraining automation / scheduled jobs

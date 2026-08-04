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

Response (real output, from the SCC-26-417-417 record above):

```json
{
  "target_n_kg_ha": 46.4,
  "target_p2o5_kg_ha": 44.9,
  "target_k2o_kg_ha": 56.5,
  "app1_formulation": "164 kg/ha (14-14-14) + 10 kg/ha (21-0-0) + 15 kg/ha (0-0-60)",
  "app2_formulation": "164 kg/ha (14-14-14) + 10 kg/ha (21-0-0) + 15 kg/ha (0-0-60)",
  "derived_remarks": {
    "ph_remark": "MODERATELY ALKALINE",
    "om_remark": "MODERATELY LOW",
    "p_remark": "HIGH",
    "k_remark": "SUFFICIENT+"
  },
  "farm_totals": { "area_ha": 0.021, "n_kg": 0.974, "p2o5_kg": 0.943, "k2o_kg": 1.187 },
  "model_version": "2026-08-04-011242",
  "source": "ml-ensemble"
}
```

Nutrient targets are **per-hectare rates**. `farm_totals` applies the area
scaling — from the same rounded per-hectare figures shown above, so the totals
always reconcile with what's displayed — and is `null` when `farm_area_sqm` is
omitted.

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

- **The regression targets (N/P₂O₅/K₂O kg/ha) are the model's real signal.**
  Trained on the full 5,000-record synthetic set, they hold R² 0.90–0.97 against
  a mean-baseline dummy regressor and pass the golden test against the one real
  labeled example available (SCC-26-417-417: predicted 46.4/44.9/56.5 vs. actual
  50/40/60, tolerance ±15 kg/ha). Judge V1 on these.
- **The recipe classifiers (`app1_formulation`/`app2_formulation`) are not yet
  meaningful and should not be trusted for held-out records.** The recipe string
  is a deterministic function of continuous kg amounts, so it is near-unique per
  record (5,000 records → 3,144 distinct classes) — no train/test split can cover
  every class, and at full scale the classifier only recognizes ~15% of held-out
  labels from training, scoring ~4% accuracy on that subset (see
  `test_coverage`/`accuracy` in a run's `metadata.json`). The API still returns a
  best-effort formulation string on every request; it becomes genuinely useful
  once trained on real DA data, where formulations cluster around agronomist
  judgment rather than continuous per-record arithmetic.
- Crop baselines other than Sorghum are placeholder scaffolding awaiting
  agronomist review. They live in `data/crop_baselines.py` so correcting them is
  one edit plus a retrain.
- The OM and K remark bands are calibrated to a single real example. The pH and
  Olsen-P bands are standard published classes that reproduce that example
  independently.

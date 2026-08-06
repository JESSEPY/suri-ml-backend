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

## Try it yourself

With the server running (`uvicorn main:app --port 8001`), open
**http://localhost:8001/docs** in a browser. FastAPI generates an interactive page there
automatically — expand `POST /predict`, click "Try it out", paste one of the request bodies
below into the box, and click "Execute" to see the real response. No separate frontend needed.

**Complete request, field crop (`kg_ha`) — answered by the exact table, `source: "table-lookup"`:**
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
  "laboratory_code": "TEST-001"
}
```

**Complete request, tree crop (`kg_tree`) — same exact path, but the amounts and formulation are
per-tree, and `farm_totals` stays `null` (no tree count collected):**
```json
{
  "crop": "Coconut",
  "age_band": "bearing",
  "data_source": "lab",
  "ph": 6.5,
  "organic_matter_pct": 2.8,
  "phosphorus_ppm": 18.0,
  "potassium_cmol_kg": 0.35,
  "laboratory_code": "TEST-002"
}
```

**Incomplete request (the real IoT/STK case — no organic matter reading) — falls back to the
trained model, `source: "ml-ensemble"`:**
```json
{
  "crop": "Rice",
  "data_source": "stk",
  "ph": 6.5
}
```

Prefer the terminal? Any of the above works as a `curl` body too:
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Rice","data_source":"stk","ph":6.5}' \
  | python -m json.tool
```

`GET /health` (also browsable at `/docs`) is the quickest way to confirm the server is up and
which model version it loaded.

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

# Training Data Upload & Retraining Pipeline — Design Spec

## 1. Purpose & Scope

Today, adding new training data to the fertilizer ML model and getting a retrained model live is
a fully manual, CLI-only process on one developer's machine: run `data.parse_real_soil_database`
against a specific Excel workbook, run `data.ingest_to_supabase`, run `ml.train`, then
git-commit the (normally gitignored) new artifact directory and push to `main` to trigger a full
GitHub Actions redeploy of the single EC2 instance. There is no UI, no review step separate from
reading printed metrics, and no path for anyone without dev/git/AWS access to add data or trigger
a retrain. This gap was explicitly called out and deferred in the original pipeline spec
(`2026-08-03-suri-fertilizer-ml-pipeline-design.md`, §15: "Model retraining automation /
scheduled jobs").

**This spec builds a self-service loop**, exposed as a new admin panel in `suri`: upload new
soil-test data for a crop the system already supports → review a validation report → commit it to
the training corpus → trigger a retrain → review its metrics (including a comparison against the
currently-live model) → promote it live, or roll back, with no CLI, git, or CI involved for any of
those steps.

**Also in scope: expanding crop coverage by transcription, not by building a CMS.** Target values
(N/P₂O₅/K₂O, formulations) for every training row are derived from a hand-transcribed official
BSWM guide table (`data/nutrient_recommendation_guide.py`, sourced from
`suri/public/Nutrient Recommendation Guide.pdf`) keyed by crop — a crop with no entry there has no
target to derive at all, for either the table-lookup or ML-ensemble path. The PDF itself already
contains official rates for roughly 61 crops; only 9 are transcribed into the guide table so far.
Since the data already exists (it doesn't need to be authored, just copied over — the same manual
process already used for the current 9, cross-checked row-by-row against the PDF), this spec
includes transcribing a further **30 rows covering 30 additional crop names, plus 9 "free" species
that already share a row with an already-coded crop** (§2.1) — taking recognized coverage from 9
to 39. This is a data task (editing constants in `data/nutrient_recommendation_guide.py` and the
API's crop enum), not new architecture, and not an admin-facing CMS for authoring arbitrary crop
data — an actual admin UI to *add or edit* guide entries through the browser (as opposed to a
developer transcribing from the PDF) stays a separate, not-built-here project (the CMS item
deferred earlier in `2026-08-16-consultant-feedback-backlog.md` #7).

**Forward-compatibility for whatever's still missing**: the upload validator must read its
accepted-crop list from the existing single source of truth (`data.nutrient_recommendation_guide.
CROPS`, already imported by `main.py`'s own crop validator) rather than hardcoding a separate copy.
This costs nothing extra to build correctly and means that any crop still not transcribed after
this pass — or any crop a future CMS project adds — is supported by this exact pipeline the moment
it's added to that list, with zero code changes here. This is the actual scalability answer: DA
providing real data for any of the ~22 remaining PDF crops (or a wholly new one) only ever requires
a one-time guide-table registration step; the upload → retrain → promote loop already handles the
rest.

## 2. Current State (for context — not being rebuilt, being reused)

- `data/parse_real_soil_database.py` — bespoke to one specific multi-sheet DA Excel workbook
  shape; **not reused directly** by the new upload path (see §3).
- `data/ingest_to_supabase.py` — idempotent upsert into `ml_soil_training_data` on the
  `(laboratory_code, data_source)` primary key. **Reused as-is** by the new commit step.
- `ml/agronomy.py` (`nutrient_requirements`, `total_nitrogen_from_om`), `ml/remarks.py`
  (`derive_*_remark`), `ml/recipes.py` (`plan_applications`) — the deterministic derivation
  functions that turn raw soil chemistry into remarks/targets/formulations. **Reused as-is** by
  the new upload validator, so an admin never manually supplies (or can typo) a derived value.
- `ml/train.py` — grouped-split training + evaluation, versioned artifact output. **Reused as-is**
  as the body of the new background retrain job.
- `main.py` — loads one pinned artifact version at import time (`latest_version()` picks whichever
  local folder name sorts latest). **Changed**: model loading becomes swappable at runtime (§5).
- Real training data today: ~3,000 rows / 696 KB (`suri_real_training_data.csv`) — small enough
  that a full retrain is expected to take well under a minute.

### 2.1 Crop Coverage Expansion

Transcribed from `suri/public/Nutrient Recommendation Guide.pdf` into
`data/nutrient_recommendation_guide.py` (rates, OM/P/K bands) and `main.py`'s crop enum, following
the exact cross-check process already used and commented for the current 9 crops. Currently coded:
Banana, Cacao, Cassava, Coconut, Corn, Mango, Peanut, Rice, Tomato.

**30 new rows to transcribe** (relevant to Region 7 farmer-facing agriculture; ornamental,
forestry, pasture, and fiber-crop rows in the PDF are excluded as out of scope for this tool),
split into two phases by transcription confidence — see
`docs/superpowers/plans/2026-08-18-crop-coverage-expansion.md`:

**Phase 1 — 21 rows, all `kg_ha`, no age axis, cleanly legible in the source scan:**

- **Vegetables**: Alugbati/Kangkong; Ampalaya; Asparagus; Cabbage/Pechay/Mustard/Spinach/Broccoli;
  Cauliflower; Celery; Chayote; Cucumber/Muskmelon/Watermelon; Eggplant/Okra; Lettuce/Malunggay;
  Radish/Carrots; Squash/Upo/Patola; Strawberry
- **Condiments**: Garlic/Chili Pepper; Ginger (Improved/Local); Onion (Bulb/Spring)
- **Rootcrops**: Gabi/Ube/Yam/Taro; Potato/Singkamas
- **Legumes**: Beans (String)/Lima/Sitao; Peas/Tapilan/Cowpea
- **Industrial**: Sugarcane

**Phase 2 — 9 age-banded tree/plantation rows, deferred:** Coffee; Papaya; Pineapple;
Avocado/Guyabano/Atis/Apple/Cashew/Jackfruit/Macopa/Tambis/Balimbing;
Citrus/Lime/Lemon/Orange/Calamansi/Passion Fruit; Guava; Lanzones/Rambutan/Mangosteen; Abaca;
Dragon Fruit. Held back because several cells are unreadable in the available scan (Coffee's
`6-10 yrs` K row scans non-monotonically; Guava repeats an age-band label; Dragon Fruit's row is
mostly blank), and because they need axis work Phase 1 doesn't: Papaya's bands are in months, and
Pineapple's `Planted`/`Ratoon` and Abaca's `Pulp`/`Cordage` are crop-cycle axes rather than ages
or varieties. Entering a guessed fertilizer rate is not an acceptable trade for a larger crop
count; these need a better scan or a DA-provided digital copy of the guide.

**9 free species** — already covered by an existing coded crop's guide row, so recognizing them
needs only a new crop-enum/alias entry, no new numbers to transcribe: Sorghum (shares Corn's row),
Camote and Yacon (share Cassava's row), Soybean and Mongo (share Peanut's row), Santol, Caimito,
Durian, and Marang (share Mango's row).

**Left out of this pass** (present in the PDF, not transcribed here — still reachable later via
the same forward-compatible `CROPS` list, §1): Fishpond algae, Roses, Rubber, Black Pepper, Lemon
Grass/Centronella, Batao/Baguio Beans, Grapes, Grapevine, Forest/Pine trees, Castor Oil, Cotton,
Jute/Kenaf, Legume Green Manure, Pasture Grasses, Ramie/Rattan, Tangan-Tangan, Tobacco Virginia,
Palm Date/Oil, Herbs/Bamboo/Lagundi/Sambong, Basil, Mint Herb.

Net effect: recognized crop coverage goes from 9 → **30** after Phase 1 (→ 39 once Phase 2's tree
crops land), before any new training data is even uploaded. This is a transcription task,
independent of and prior to the upload/retrain pipeline itself — it just means the pipeline has 30
usable crops to receive data for on day one instead of 9.

**Important consequence, handled in the plan:** a newly recognized crop is immediately usable on
the *complete-data* path (the guide lookup is exact and needs no training), but its *model* path
would otherwise return a confident-looking number derived from no crop-specific signal — the
one-hot encoder is built from an explicit category list, so an untrained crop gets a column that
was simply never active during training. The plan therefore adds a guard that refuses an
incomplete-data request for a crop with no training rows, pointing the caller at the exact path
instead. This is what makes "upload data for that crop, then retrain" a real prerequisite rather
than a silent quality cliff.

## 3. Upload Template & Validation

**Template**: a normalized CSV (Excel accepted, same columns) containing only *raw measurements*,
one row per soil test:

| Column | Required | Notes |
|---|---|---|
| `laboratory_code` | yes | Unique key; rejected if it already exists in `ml_soil_training_data` |
| `crop` | yes | Must be in `data.nutrient_recommendation_guide.CROPS` (§1 forward-compat note) |
| `variety`, `cultivation_method`, `age_band` | no | Crop-dependent, same rules `main.py` already applies via `resolve_variety`/`resolve_cultivation_method`/`resolve_age_band` |
| `ph` | yes | 0–14 |
| `organic_matter_pct`, `phosphorus_ppm`, `potassium_cmol_kg` | yes | All three required — without them `nutrient_requirements()` cannot derive a target, so a row missing any one is rejected, not silently imputed (imputation is an inference-time fallback for incomplete STK/IoT requests, not something training data should ever be built on) |
| `farm_area_sqm`, `sampling_site`, `date_of_sampling` | no | Stored, not model features (matches §3.1/§3.3 of the original pipeline spec) |

`data_source` is not a template column — every admin-uploaded row is real lab data, so the backend
sets it to `"lab"` unconditionally, matching what `parse_real_soil_database.py` already does.

**`POST /training-data/validate`** (multipart upload) parses the file, and per row:
1. Checks required fields present and in-range (reusing `main.py`'s existing Pydantic bounds).
2. Checks `crop` against `CROPS`; unknown crop → rejected with the same "Unknown crop, valid
   crops: ..." message style `/predict` already uses.
3. Checks `laboratory_code` isn't a duplicate (existing row or a duplicate within the same file).
4. For every row that passes, computes `total_nitrogen_pct`, all four remarks, and
   `target_n/p2o5/k2o` + both formulations via the existing derivation functions, and includes
   them in the response — this is what the admin actually reviews, since it's the first visibility
   into what the row will contribute to training.

Response: `{accepted: [...], rejected: [{row, reason}, ...]}`. **Nothing is written to Supabase at
this step** — it's a dry run.

**`POST /training-data/commit`** — takes the previously-validated accepted rows (re-sent, or
referenced by a short-lived validation-batch token — implementation detail for the plan) and
upserts them via the existing `ingest_to_supabase` logic.

## 4. Retrain Job

**`POST /retrain`** starts `ml/train.py`'s existing logic as a FastAPI `BackgroundTask` — not a
request-blocking call — and returns `{job_id}` immediately. `/predict` keeps serving throughout;
training reads from Supabase and writes to a new, separate artifact directory, never touching the
currently-loaded model.

**`GET /retrain/{job_id}`** — `{status: "running"|"done"|"failed", metrics?, error?}`. On `done`,
`metrics` includes:
- MAE/R² per target, overall + per crop + per `data_source` (existing `ml/train.py` output).
- `DummyRegressor` baseline comparison per target, with a clear pass/fail flag if the new model
  fails to beat it on any target.
- The SCC-417 golden-record check (§10.3 of the original spec), pass/fail + delta.
- **New**: a diff against the metrics of whichever version is currently `is_current` in the
  registry (§5) — same metrics, side by side, so "is this actually better" is answered on-screen.

On completion, the artifact is written to local disk (as today) and uploaded to a new Supabase
Storage bucket (`ml-model-artifacts`) for durability — today, only versions someone remembered to
git-commit survive an instance rebuild; this makes every trained version recoverable.

## 5. Model Registry, Promotion, Rollback

New Supabase table `ml_model_versions`: `version` (PK, matches the artifact directory name),
`trained_at`, `row_count`, `metrics` (jsonb), `promoted_at` (nullable), `promoted_by` (nullable),
`is_current` (bool). Replaces today's implicit "whichever folder sorts latest" convention with an
explicit, auditable record — also gives the demo a natural "here's our model version history" view
for free.

`main.py`'s `load_artifacts()` is refactored from "runs once at import" to a function callable at
runtime, guarded by a lock so an in-flight `/predict` request always completes against a
consistent model. Boot-time loading reads the registry's `is_current` row (falling back to
local-latest only if the registry is unreachable), so a restart never silently reverts to the
wrong version.

**`POST /model/promote {version}`** — authenticated (reuse the existing `X-API-Key` /
`require_api_key` mechanism `/predict` already has), acquires the lock, calls `load_artifacts()`
for the target version, updates `is_current` in the registry on success. If the version's metrics
regressed against the previously-current version (per §4's diff), the suri UI requires an explicit
"promote anyway?" confirmation before sending the request — the backend doesn't hard-block it
(the admin is the actual gate, per the earlier discussion), but a regression can't be promoted by
an accidental click.

**Rollback** is the same endpoint, called with a previous version — no separate mechanism needed.

## 6. suri Admin UI

New section in suri's existing admin dashboard (`features/admin/components/ml-training/`),
admin/superadmin RBAC, following the existing pattern of thin-client admin pages that proxy to
`suri-ml-backend` via `ML_API_URL`/`ML_API_KEY` (same as `app/api/ml-fertilizer/route.ts` already
does for `/predict`). Three views:
1. **Upload** — file picker → validation report table (accepted rows with computed fields shown,
   rejected rows with reasons) → "Add to training data" button → commit.
2. **Retrain** — "Retrain" button → polls job status (visible "retraining…" state scoped to this
   page only, not a farmer-facing maintenance banner — `/predict` is unaffected and keeps serving
   real users throughout) → metrics table with the vs.-current diff → "Promote" / "Discard".
3. **History** — `ml_model_versions` listing: version, trained_at, row_count, metrics summary,
   promoted_at/by, with "Rollback to this version" on any past entry.

## 7. Error Handling & Safety

- Validation errors are always specific and row-level (never a generic "upload failed").
- Training-job failure (bad data slipped through, OOM, etc.) → `status: "failed"` with the error
  surfaced in the UI; the currently-live model is unaffected since training never touches it.
- Supabase Storage backup failure → logged as a warning, doesn't fail the retrain (same
  graceful-degrade pattern already used for the `/predict` audit log) — version stays usable
  locally, just not yet durable off-instance until the next successful backup.
- Instance reboot mid-training → job is simply lost, no resume logic (low value for a
  single-admin internal tool, explicitly not built). The promoted version reloads normally from
  the registry.
- Promote/rollback race: the lock (§5) guarantees no `/predict` request ever observes a
  half-swapped state, and `is_current` only updates after the in-memory swap succeeds.

## 8. Testing

- Unit tests for the upload validator: one fixture per rejection reason (unknown crop, out-of-range
  pH, missing OM/P/K, duplicate lab code), plus confirmation that accepted rows' derived fields
  exactly match calling `ml/agronomy.py`/`ml/remarks.py`/`ml/recipes.py` directly.
- Integration test: upload → commit → retrain → promote → rollback end-to-end against a small
  fixture dataset, asserting registry state and the actually-loaded in-memory model at each step.
- Concurrency test: fire `/predict` while a promote call is in flight; assert every response is
  fully served by either the old or the new model, never a mixed/inconsistent state.
- **More-data-for-an-existing-crop regression test**: confirm that uploading additional rows for
  an already-weak crop (e.g. Tomato, currently 20 real rows per the README's known limitations) and
  retraining measurably changes that crop's per-crop metrics — the concrete proof this pipeline
  does what it's for. Separate from §2.1's transcription tests, which prove a newly-added crop's
  guide values are correct, not that the retrain pipeline improves with more data.

## 9. Deliverables

```
suri-ml-backend/
  data/nutrient_recommendation_guide.py  # §2.1 — expanded from 9 to 39 crops, transcribed from the PDF
  main.py                          # §2.1 crop enum expansion; §5 swappable load_artifacts(), new endpoints
  data/training_upload.py         # §3 — parse/validate template, reuse derivation functions
  ml/retrain_job.py                # §4 — background job wrapper around ml/train.py
  supabase/migrations/<date>_ml_model_versions.sql   # §5
  tests/test_nutrient_recommendation_guide.py  # §2.1 — golden-value fixtures for each newly transcribed row
  tests/test_training_upload.py    # §8
  tests/test_retrain_pipeline.py   # §8

suri/
  features/admin/components/ml-training/
    MLTrainingManagement.tsx       # §6 — Upload / Retrain / History views
  app/api/admin/ml-training/       # thin proxy routes to suri-ml-backend, mirroring
                                    # app/api/ml-fertilizer/route.ts's existing pattern
```

## 10. Explicitly Out of Scope

- Adding support for any crop **not** listed in §2.1 (i.e. anything from the PDF's remaining ~22
  rows, or any crop outside the PDF entirely). Still reachable later with zero pipeline changes,
  per §1's forward-compatibility note — just not transcribed as part of this project.
- A Crop Reference CMS — an admin-facing UI to *author or edit* guide-table entries through the
  browser, as opposed to a developer transcribing official rates from the PDF (§2.1's method).
  Related, deferred separately.
- Farmer-facing maintenance mode / pausing `/predict` during retraining — decided against; the
  hot-swap architecture makes it unnecessary (§4, §7).
- Resuming an interrupted training job after an instance reboot.
- Any change to the bespoke `data/parse_real_soil_database.py` workbook-import path — it keeps
  working independently for whenever the next full DA export arrives.

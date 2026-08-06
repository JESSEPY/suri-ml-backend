# Real Soil Data Retraining — Final Whole-Branch Review Findings

**Status: NOT production-ready. Paused here — resume by fixing the Critical items below, then
re-running the final whole-branch review.**

## Context

All 12 tasks of `docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md` are implemented,
individually reviewed, and merged to `master` (see `.superpowers/sdd/2026-08-05-real-soil-data-retraining/progress.md`
for the full per-task ledger). The model has been retrained end-to-end on the real 3,076-row
dataset and verified serving correctly for the specific cases exercised in Task 12's smoke test
(see `docs/superpowers/specs/2026-08-06-real-soil-data-retraining-results.md`).

The final whole-branch review (dispatched on the most capable available model, per
`superpowers:subagent-driven-development`'s process, after all 12 tasks passed individually) found
three Critical and five Important issues that per-task reviews structurally could not catch,
because they only appear when looking at cross-task integration and the whole diff at once. This
doc is the durable record of those findings — the review itself was ephemeral (a subagent
transcript), this is what survives.

**Do not treat this as done.** The individual-task work was real and solid — every previously-caught
bug (int-cast, boundary off-by-one, negative-clip, CSV NA-sniffing, crop-text matching) is
genuinely fixed at HEAD, resolvers are consistently used everywhere, K-unit conversion is
consistent everywhere, and 213 tests pass. But the findings below reach real users with wrong,
sometimes dangerous, output, and none of them is caught by the existing suite.

## Critical (must fix before this is usable)

### 1. `ml/recipes.py` was never updated for `target_unit` — tree-crop formulations are wrong unit AND wrong scale

`ml/recipes.py` was not in the plan's "Modified files" list — a plan gap, not just an
implementation gap. It hard-codes `f"{int(round(kg))} kg/ha ({grade})"` and its signature
still says `plan_applications(n: int, p2o5: int, k2o: int)` "for the given kg/ha targets".

Confirmed live: a `/predict` call for Coconut (a `kg_tree` crop) returned:
```
target_n=0.947 target_p2o5=1.166 target_k2o=2.111  target_unit="kg_tree"
app1_formulation = "214 kg/ha (14-14-14) + 71 kg/ha (21-0-0)"
```
That's a ~200x over-application instruction, in the wrong unit, in the exact string a farmer
would read and act on. Also: of the 183 `kg_tree` rows in `suri_real_training_data.csv`, 79 have
`app1_formulation == "None"` (every term rounded to 0) and the other 104 carry a mislabeled
`"X kg/ha"` string that actually means "X kg per tree" — so the classifier heads were trained on
corrupted labels for every tree crop, not just serving them wrong at inference time.

**Fix direction:** `_format_terms`/`plan_applications` need the unit passed in, with rounding that
respects sub-1 amounts for `kg_tree` crops. Given the classifier heads were trained on already-bad
`kg_tree` labels, this likely needs the training CSV regenerated and the model retrained after the
formatting fix, not just a serving-layer patch. Consider whether `app*_formulation` should even be
model-predicted for `kg_tree` crops, given how few and how noisy those labels are (see the
architecture note at the bottom of this doc).

### 2. Zero training coverage for non-default `cultivation_method`/`age_band` values — API returns confident, badly wrong numbers

The real corpus only has `Lowland Wet`/`None` for `cultivation_method` and only the bearing age
bands (`10yrs & up`, `Over 7yrs`, `9yrs & up`) for `age_band` — every other value the API accepts
(`Upland`, `Lowland Dry`, `Below 1yr`, `1-3yrs`, etc.) has exactly zero training rows. The model
still returns a confident prediction for them. Measured against the deterministic guide table the
model is supposed to approximate (om=1.5, p=5.0, k=0.1):

| Request | Model | Guide | Error |
|---|---|---|---|
| Rice, `Lowland Dry` | N=56.7 | N=120 | 2.1x under |
| Rice, `Upland` | N=56.7 | N=45 | 1.3x over |
| Coconut, `Below 1yr` | 1.106 / 1.418 / 2.252 | 0.06 / 0.05 / 0.15 | **~18x over** |
| Mango, `1-3yrs` | 0.763 / 1.032 / 1.090 | 0.10 / 0.10 / 0.09 | **~8x over** |

The model returns *identical* values for `Upland` and `Lowland Dry` — it learned nothing from
those one-hot columns because no training row ever set them. Recommending 1.1 kg N on a coconut
seedling that needs 0.06 kg is real agronomic harm, on an input this whole task added specifically
to *improve* accuracy.

**Fix direction:** either reject uncovered `(crop, cultivation_method)`/`(crop, age_band)`
combinations with a 422 until real training data exists for them, or — since
`nutrient_requirements()` is already in-process and exact — serve the deterministic guide value
directly for any combination with zero training coverage, instead of the model's extrapolation.
Either way this needs to be an enforced, documented boundary, not silent.

### 3. The Supabase migration corrupts pre-existing rows and destroys the ability to detect it

`supabase/migrations/20260806_align_ml_training_schema.sql` is structurally safe (backfills before
`SET NOT NULL`, `target_unit` CASE matches `TARGET_UNIT` exactly) but not semantically safe:

- It renames `potassium_ppm` → `potassium_cmol_kg` **without converting the values**. Any
  pre-existing row keeps its old ppm number (e.g. 138) in a column now declared cmol/kg.
  Downstream, `k_cmol_to_ppm(138)` = 53,958 ppm — off by 391x, straight into the K-band lookup.
- It drops `is_synthetic` and its index, but never deletes the old synthetic rows the table was
  originally loaded with (per the repo's own README, ~15,000 synthetic rows). After the migration
  those rows survive, indistinguishable from real ones, with old-formula targets and a mis-united
  K column. `ml/train.py`'s no-`--csv` path is a plain `SELECT *` from this table — a Supabase-sourced
  retrain (the documented default) would train on that mixture.

**Fix direction:** add `DELETE FROM public.ml_soil_training_data WHERE is_synthetic;` (or a
documented truncate + re-ingest) **before** the `DROP COLUMN` step, and either convert the K
column's existing values or explicitly state that any surviving pre-migration row is unsupported
and must be deleted, not migrated.

## Important (should fix)

### 4. `README.md` documents an API that no longer exists

Never touched by this branch (also a plan gap — README.md isn't mentioned in the design spec or
plan). It still tells integrators to run `python -m data.generate_synthetic_data` (deleted, will
`ModuleNotFoundError`), send `potassium_ppm` (now a 422), expect `target_n_kg_ha` in the response,
and lists a crop set (`Cabbage, Corn, Rice, Sorghum, Sugarcane, Tomato`) where four of six are no
longer valid. No mention anywhere of `variety`/`cultivation_method`/`age_band`/`target_unit`/
`application_schedule`. This is the one integrator-facing doc in the repo and it actively
contradicts the design spec.

### 5. `resolve_variety()` doesn't gate by crop, unlike its two siblings — real feature/label desync

`data/nutrient_recommendation_guide.py`'s `resolve_variety()` is `return variety or "None"` —
`crop` is accepted and ignored. `resolve_cultivation_method`/`resolve_age_band` both correctly gate
on crop. This contradicts the plan's own Global Constraint ("every other crop's variety is always
`None`"). Measured: `POST /predict {crop: "Cassava", variety: "Hybrid"}` → N=69.1 vs.
`{crop: "Cassava"}` (no variety) → N=50.6 — 37% apart, on an axis the guide table ignores entirely
for Cassava, and a combination the parser never produces (so the model is extrapolating on an
unseen one-hot combination). Note: the plan prescribed the current implementation verbatim, so
this is a plan defect as much as an implementation one. Fix:
`return variety if variety in VARIETIES_FOR[crop] else "None"`.

### 6. The promised Coffee golden-record test was never written

The plan's Global Constraint says the retired SCC-417 fixture is "replaced, not just deleted" by
golden checks against two pptx worked examples — Coffee and Rice-Inbred. Only the Rice-Inbred test
exists. `GUIDE_ROWS["Coffee|Over 15yrs"]` is carried in production code solely to support the
missing test, is unreachable via `nutrient_requirements()` (Coffee isn't in `CROPS`), and is
asserted by nothing. Its transcribed values (0.165/0.10/0.30 kg_tree) also don't obviously
reconcile with the design spec's stated Coffee figures (90/60/40) — worth resolving either way.
Either write the test or delete the dead row and amend the Global Constraint to say why.

### 7. The test suite writes to the production Supabase audit-log table on every run

`tests/test_api.py` never monkeypatches `log_prediction` (except in the one test specifically
about logging failures) — every `/predict` call in the suite inserts a real row into
`ml_fertilizer_predictions` in the live Supabase project (confirmed: `HTTP/2 201 Created` observed
against the real project per test run). Roughly 25 rows per `pytest` run, permanently polluting the
table that exists specifically for retraining/traceability audit. `TestLoggingIsNonFatal` already
shows the correct monkeypatch pattern — it just isn't applied at the fixture level for the rest of
the file.

### 8. Stale docstrings still claim provenance to the retired SCC-417 record

`ml/remarks.py`'s module docstring still claims the OM/K bands are "calibrated to that single real
example" (SCC-417) and that the Olsen-P/pH classes "independently reproduce the real SCC-417
report" — the record the design spec traced to a document *outside* this dataset and retired
specifically because of that. Same issue in `ml/recipes.py`'s docstring, `tests/test_recipes.py`'s
and `tests/test_remarks.py`'s `class TestGoldenRecordSCC417` naming/docstrings, and two other
"synthetic" references (`data/ingest_to_supabase.py:1`, `ml/model.py:5`) left over from before the
generator was retired. Not wrong code, but false justification left attached to real constants.

## Minor (nice to have, non-blocking)

- Migration has no `BEGIN`/`COMMIT`, no `IF NOT EXISTS`/`IF EXISTS` guards — a partial failure
  leaves the table half-migrated with no clean re-run path. `target_unit` gets `SET NOT NULL` but
  no `SET DEFAULT`, inconsistent with its three sibling columns.
- `plan_applications(n: int, p2o5: int, k2o: int)` type hints are stale — every caller now passes
  floats.
- `normalize_lab_code()` maps any digit-free code to `{year}-SCC0`, silently colliding all such
  rows onto one primary key. Not triggered by the current real file (0 duplicate PKs across all
  3,076 rows, verified), but latent on a future re-parse if the source data ever has a code with no
  digits.
- `ml/artifacts/2026-08-04-020841/` (an old-schema preprocessor expecting `potassium_ppm`) is still
  on disk. Harmless today (`latest_version()` sorts lexically and the newer version wins), but a
  footgun if anyone ever pins an old version.
- `tests/fixtures.py` only ever generates default axis values, so no training-path test exercises a
  non-default `variety`/`cultivation_method`/`age_band` — consistent with the real corpus, but it
  means finding #2 above is structurally invisible to the test suite.

## Architecture note worth a deliberate decision, not a quick fix

Targets are now a *pure deterministic function* of `(crop, variety, cultivation_method, age_band,
om, p, k)` via `nutrient_requirements()`. Training an ensemble to approximate that lookup table
strictly adds error with no upside for any input combination the training data actually covers
well — overall MAE is 3.4-3.7 kg/ha, Tomato is 39.2 kg/ha, and even a *covered* combination
(Coconut bearing) predicts 0.947 against the guide's exact 0.5. Every deviation in finding #2's
table above is literally the model disagreeing with a table it could just read directly. This
branch made that mismatch sharper by replacing a fuzzy invented formula with an exact one.

Worth a deliberate decision on whether the ML layer should remain in the target-prediction path at
all, or whether it should be reserved for what it's actually suited for here — e.g. imputing a
result when the requester's soil-test inputs are incomplete/noisy (the `stk`/`iot` missingness case
this whole pipeline was already built around), while a complete, in-coverage request gets served
the exact deterministic value. This doesn't block fixing findings 1-8 above, but it's worth
resolving before investing more effort in model-quality tuning for a target the system can already
compute exactly.

## How to resume

1. Fix Critical #1-3, ideally in that order (#1 and #2 are both user-facing dosage errors; #3
   blocks safely re-running Task 12's Supabase ingestion).
2. Fix Important #4-8.
3. Regenerate the training CSV and retrain (Critical #1's fix likely requires this, since the
   `app*_formulation` classifier labels were trained on corrupted data).
4. Re-run the full test suite and the Task 12 smoke test, this time explicitly covering a
   non-default `cultivation_method`/`age_band` request (finding #2's exact gap) and a `kg_tree`
   crop's `app*_formulation` (finding #1's exact gap) — the existing smoke test missed both.
5. Re-run the final whole-branch review before considering this production-ready.
6. Decide the architecture question above — doesn't block 1-5, but affects how much further
   model-quality investment is worth making afterward.

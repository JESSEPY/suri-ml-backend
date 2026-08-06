# Real Soil Data Retraining — Final Whole-Branch Review Findings

**Status: All 3 Critical + 5 Important findings fixed and independently re-verified** (2026-08-06)
— see `docs/superpowers/plans/2026-08-06-final-review-fixes.md` for the fix plan (9 tasks, all
complete) and the verification notes below. Two things remain outside this fix's scope:

1. **The Supabase migration is written but not applied** (`supabase/migrations/20260806_align_ml_training_schema.sql`)
   — applying it, then running `python -m data.ingest_to_supabase`, is a deliberate, separate
   action against shared production infrastructure, not something done automatically as part of
   this fix.
2. **A new, out-of-scope discovery**: `lab-results-synthesizer.py` at the repo root is dead,
   disconnected reference material from the project's very first commit (predates this session
   entirely) — never imported by any real code path, but still generates fake data against the
   long-retired schema (`potassium_ppm`, the old 6-crop list, `SCC-26-{i}`-style lab codes — quite
   possibly the actual origin of the `SCC-26-417-417` golden-record lab code, given the numbering
   scheme). Flagged rather than deleted, since it predates this work and was explicitly labeled
   "reference material" in the initial commit — worth an explicit decision, not a unilateral one.

## Verification (2026-08-06)

Re-checked each finding directly against the current code and a live server, not just re-read the
diffs:

- **Critical #1** (`ml/recipes.py` unit bug): `POST /predict` for Coconut/bearing now returns
  `target_unit: "kg_tree"` with `app1_formulation` containing `"kg/tree"` and no `"kg/ha"` —
  confirmed live.
- **Critical #2** (zero training coverage): `POST /predict` for `{crop: "Rice", cultivation_method:
  "Upland", om: 1.5, p: 5.0, k: 0.1}` returns `target_n: 45.0`, `source: "table-lookup"` — matches
  `nutrient_requirements("Rice", 1.5, 5.0, 0.1, cultivation_method="Upland")` exactly (previously
  this combination — near-zero training coverage — got a model guess far from this value).
- **Critical #3** (migration): rewritten to `DELETE FROM public.ml_soil_training_data` before any
  rename, so no pre-existing row can carry a stale ppm-scale K value under the new
  `potassium_cmol_kg` name. `ml_fertilizer_predictions` (a separate, JSONB-based audit table) is
  correctly left untouched — an earlier draft of this fix mistakenly deleted from it too, caught in
  self-review before commit.
- **Important #4** (README): full rewrite, matches the current request/response shape verified
  live above.
- **Important #5** (`resolve_variety` gating): `resolve_variety("Cassava", "Hybrid")` now returns
  `"None"`, matching its two siblings.
- **Important #6** (Coffee golden test): resolved honestly — the unreachable, unverifiable dead row
  was removed and the original plan's Global Constraint amended to say why, rather than forcing an
  unsourced-density check.
- **Important #7** (tests writing to production Supabase): `tests/test_api.py` now has an autouse
  fixture no-opping `log_prediction` for the whole file; `TestLoggingIsNonFatal`'s one test that
  specifically needs it to raise correctly overrides the autouse fixture (verified by tracing
  pytest's fixture-then-test-body execution order, not just trusting the report).
- **Important #8** (stale docstrings): all corrected — `ml/remarks.py`, `ml/recipes.py`,
  `data/ingest_to_supabase.py`, `ml/model.py`, and (caught during this final re-verification pass,
  after the fix plan's Task 7 brief omitted it despite the original findings doc naming it)
  `tests/test_remarks.py`'s `TestGoldenRecordSCC417` class, renamed
  `TestKnownReferenceValues` with corrected provenance framing.

One additional gap found and fixed during this fix cycle, not in the original 8 findings: a
complete request with an unrecognized `age_band` (e.g. a typo) was raising an uncaught `KeyError`
from `nutrient_requirements()` inside `/predict` — an unhandled 500, unlike every other
invalid-input case in this API, which gets a clean 422. Fixed with a try/except and a regression
test (`main.py`, commit `fb2561d`).

Full suite: 228/228 passing throughout.

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

**Fix direction:** given the architecture decision below (route complete requests to the exact
table), `app1_formulation`/`app2_formulation` should become a deterministic function of
`(n, p2o5, k2o, target_unit)` — converting a known nutrient amount into a bag-recipe string is
arithmetic (exactly what the pptx's own worked examples do: `RR × product-%N` etc.), not something
that needs a trained classifier at all, for either serving path. This removes the unit bug at the
root (the formula gets `target_unit` as an explicit input) and the training-label noise problem
(79 degenerate `"None"` labels) simultaneously, rather than just making the existing classifier
unit-aware and re-training it on the same corrupted labels.

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

**Fix direction:** resolved by the architecture decision below — this stops being a
coverage-tracking problem once complete-data requests are routed to the exact table instead of the
model, since the table has no notion of "uncovered."

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

## Architecture decision: route complete requests to the exact table, reserve the model for incomplete ones

**Resolved** (discussed and agreed after the review — this directly subsumes the fix for Critical
#2 and reframes Critical #1; see "How to resume" below for how it folds into the fix plan).

Targets are now a *pure deterministic function* of `(crop, variety, cultivation_method, age_band,
om, p, k)` via `nutrient_requirements()`. Training an ensemble to approximate that lookup table
strictly adds error with no upside for any input combination the training data actually covers
well — overall MAE is 3.4-3.7 kg/ha, Tomato is 39.2 kg/ha, and even a *covered* combination
(Coconut bearing) predicts 0.947 against the guide's exact 0.5. Every deviation in finding #2's
table above is literally the model disagreeing with a table it could just read directly. This
branch made that mismatch sharper by replacing a fuzzy invented formula with an exact one.

**Why the ensemble exists at all, and why it still earns its place:** the model isn't redundant —
it's mis-scoped. `nutrient_requirements()` needs numeric `organic_matter_pct` (OM) plus P and K to
run at all. IoT probes and Rapid Soil Test Kits (`data_source` `iot`/`stk`) can read N/P/K/pH but
**cannot measure OM** — that needs a real lab combustion/wet-oxidation test. So any IoT/STK
request, and any partial `lab` submission (the source spreadsheet itself has plenty of real rows
where one field was RSTK-tested even though the rest came from the lab — the exact pattern
`data/parse_real_soil_database.py`'s sentinel handling was built around), structurally cannot run
the deterministic lookup. That's the model's real, non-redundant job: producing a reasonable
estimate when the exact inputs the table needs aren't all present.

**Decision:** route on actual field completeness, not on the declared `data_source` string —
`organic_matter_pct`, `phosphorus_ppm`, `potassium_cmol_kg` all present and numeric →
`main.py` calls `nutrient_requirements()` directly (exact, zero model-approximation error, no
training-coverage dependency, works even for crop/variety/cultivation_method/age_band
combinations the corpus has zero rows for, since it's a computation, not a learned pattern);
anything missing → fall through to the trained ensemble exactly as today. This is deliberately the
same completeness rule `data/parse_real_soil_database.py`'s `build_training_row()` already uses to
decide whether a row is trainable at all (`om is None or p is None or k is None: return None`) — a
`lab`-labeled row that's actually incomplete already gets excluded from training for this reason;
serving should treat it the same way it's already treated at training time.

**Why this is the right cut, not a patch:** it structurally eliminates Critical #2 (an uncovered
`cultivation_method`/`age_band` combination with complete OM/P/K now gets the exact guide answer,
not a model guess) rather than requiring a 422/coverage-tracking band-aid, and it removes Tomato's
17-row sample-size problem entirely for any complete-data request, since the table doesn't care how
few rows trained the ensemble. It doesn't touch or deprecate the model — it puts it back in the one
role it was actually designed for.

**Low-cost to add:** `main.py`'s `PredictionResponse` already has a `source: str = "ml-ensemble"`
field, currently hardcoded — it's already scaffolded to say which path answered a request; it just
needs to actually vary (`"table-lookup"` vs `"ml-ensemble"`).

## How to resume

Being executed via `docs/superpowers/plans/2026-08-06-final-review-fixes.md`. Summary:

1. Add the completeness router to `main.py` (complete OM/P/K → `nutrient_requirements()` directly;
   otherwise → the existing model path) — this is the fix for Critical #2 and reframes Critical #1.
2. Replace `ml/recipes.py`'s trained classifier heads with a deterministic
   `(n, p2o5, k2o, target_unit)` → formulation formula — fixes Critical #1 at the root.
3. Fix Critical #3 (the Supabase migration) — add the synthetic-row delete step and resolve the K
   unit conversion for any pre-existing row.
4. Fix Important #4-8 (README, `resolve_variety` crop-gating, the missing Coffee golden test, the
   test suite writing to the production audit log, stale SCC-417 docstrings).
5. Regenerate the training CSV and retrain if needed once #2 changes what the classifier heads are
   trained on.
6. Re-run the full test suite and an expanded smoke test — explicitly covering a non-default
   `cultivation_method`/`age_band` request and a `kg_tree` crop's `app*_formulation`, both of which
   the original Task 12 smoke test missed.
7. Re-run the final whole-branch review before considering this production-ready.

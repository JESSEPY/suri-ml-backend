# Real Soil Data Retraining — Results

Companion to `docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md` and
`docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md`. That spec/plan describe what
was built and why; this doc records what actually happened when it ran against the real data —
kept separate so the design record stays a design record, and this one gets updated on future
retrains without rewriting history.

## Status

All 12 plan tasks are implemented, reviewed, and merged to `master`. The model has been
retrained end-to-end on the real Region 7 National Soil Database and verified serving correctly
through `main.py`'s `/predict` endpoint.

**Not yet done:** the shared Supabase `ml_soil_training_data` table still has the pre-redesign
schema (missing `variety`/`cultivation_method`/`age_band`/`target_unit`, old `_kg_ha`-suffixed
target columns, old `potassium_ppm` name). Migration written at
`supabase/migrations/20260806_align_ml_training_schema.sql`, not yet applied. Once applied,
`python -m data.ingest_to_supabase` needs to run to actually populate it — the currently-served
model was trained directly from the local parsed CSV, not from Supabase, so this is independent
of model quality (see "Retraining without re-parsing" below).

## Data volume

Parsing `Region 7-National Soil Database_2022-2025.xlsx` (all 4 usable sheets) produces
**3,076 real training rows** (`data_source="lab"` only — see the design spec's "Data source
scope" section for why `stk`/`iot` get zero dedicated rows):

| Crop | Rows |
|---|---|
| Rice | 1,695 |
| Corn | 1,014 |
| Cassava | 91 |
| Peanut | 73 |
| Mango | 59 |
| Banana | 53 |
| Cacao | 54 |
| Tomato | 20 |
| Coconut | 17 |

Variety: 2,387 `None` / 308 OPV / 226 Hybrid / 155 Inbred. Cultivation method: 1,381 `None` /
1,695 Lowland Wet (the source spreadsheet's `Crops` text rarely states a method explicitly, so
most Rice rows fall back to the default — expected, not a parsing gap). Age band: 2,946 `None` /
59 Mango-bearing / 54 Cacao-bearing / 17 Coconut-bearing.

## Model quality (version `2026-08-06-011035`, trained on all 3,076 rows)

Held-out test set, overall:

| Target | MAE | R² | Mean-baseline MAE |
|---|---|---|---|
| target_n | 3.410 | 0.930 | 17.455 |
| target_p2o5 | 2.828 | 0.960 | 19.012 |
| target_k2o | 3.733 | 0.923 | 18.616 |

All three targets comfortably beat a mean-baseline dummy regressor, on real (not synthetic) data.

By crop (MAE; **note the two crop groups are on different scales and not directly comparable** —
Coconut/Cacao/Mango/Banana are `kg_tree` (~0.003–2 range), everything else is `kg_ha` (~20–300
range); this is exactly the split `LogTargetRegressor` (`ml/model.py`) exists to let one shared
regressor handle without the loss being dominated by the larger-scale rows):

| Crop | Unit | N MAE | P₂O₅ MAE | K₂O MAE |
|---|---|---|---|---|
| Banana | kg_tree | 0.446 | 0.416 | 0.444 |
| Cacao | kg_tree | 0.171 | 0.161 | 0.150 |
| Coconut | kg_tree | 0.534 | 0.526 | 0.342 |
| Mango | kg_tree | 0.295 | 0.203 | 0.355 |
| Rice | kg_ha | 2.974 | 2.425 | 3.215 |
| Corn | kg_ha | 4.274 | 3.831 | 4.674 |
| Cassava | kg_ha | 7.717 | 4.891 | 8.912 |
| Peanut | kg_ha | 1.107 | 2.289 | 1.977 |
| **Tomato** | kg_ha | **39.208** | 6.836 | **37.956** |

**Known weak spot: Tomato.** N and K₂O MAE are far worse than every other crop, both in absolute
terms and relative to its own ~20-150 kg/ha range. With only 20 rows total, its held-out test
fold is a handful of samples — this reads as a small-sample-size artifact, not a code defect
(the fix that unblocked this training run was unrelated — see "Bugs found during verification"
below — and Tomato was never touched by it). Worth watching on the next retrain as more Region 7
Tomato samples accumulate; not something to chase further with the current data volume.

Coconut (17 rows, the smallest crop) has the highest `kg_tree`-scale MAE of its group, consistent
with the same small-sample pattern at a less extreme scale.

## Bugs found during verification (both fixed, reviewed, committed)

Running the pipeline end-to-end against real data (not the synthetic/fixture data every unit
test up to this point had used) surfaced two real bugs neither the task-by-task reviews nor the
unit tests caught, because both needed a genuine CSV round-trip to manifest:

1. **`ml/train.py`'s `_load_frame()`** (commit `4dfed4d`) — plain `pd.read_csv()` silently
   converts the literal string `"None"` (a deliberate, valid category for
   `variety`/`cultivation_method`/`age_band`, not missing data) into `NaN` via pandas' default
   NA-sniffing. Corrupted 2,387 of 3,076 real rows and made training fail outright
   (`OneHotEncoder(handle_unknown="error")` correctly rejecting the resulting `NaN` category).
   Fixed with `keep_default_na=False, na_values=[""]`.
2. **`data/ingest_to_supabase.py`'s `ingest()`** (commit `3f4d66b`) — the identical bug at the
   Supabase-write path; would have silently written real `NULL`s into the database in place of
   the `"None"` sentinel. Same fix.

Both are covered by new regression tests asserting the sentinel string survives the round-trip
while genuinely-missing numeric cells still become `NaN`/`None` correctly.

## Known limitations carried over from the design spec (not re-litigated here)

- P-band classification assumes the **Olsen** method — the source spreadsheet never states which
  method Region 7 labs use; this is a documented assumption, isolated to one constant in
  `data/nutrient_recommendation_guide.py` (`P_METHOD`).
- The transcribed Nutrient Recommendation Guide table has one confirmed, unresolved discrepancy:
  the Rice-Inbred golden-record check (`tests/test_nutrient_recommendation_guide.py`) matches the
  pptx worked example's N and P but not K — see that test's docstring for the investigation
  already done.
- Targets are still formula/table-derived (from real soil chemistry), not learned from observed
  fertilizer-application outcomes — see the design spec's "Non-goals" section for why that's out
  of scope for this pass.

## Retraining without re-parsing

If only `data/nutrient_recommendation_guide.py`'s baseline values change (not the source
spreadsheet), the full sequence is: re-run `data/parse_real_soil_database.py` → re-run
`data/ingest_to_supabase.py` (safe, upserts by `(laboratory_code, data_source)`) → re-run
`python -m ml.train --csv suri_real_training_data.csv` → restart the service (it auto-loads the
latest `ml/artifacts/` version). No other code changes needed for a baseline-value-only update.

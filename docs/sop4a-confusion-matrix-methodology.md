# SOP 4.a — Confusion Matrix / Comparative Accuracy Rate

**This document exists to satisfy Chapter 3 of the capstone manuscript** ("Advancing
Agricultural Studies: Predictive Web-Based Modeling of Soil Capability and Regulatory
Constraints," SWU-005) — specifically SOP/RQ4.a, *"Predictive Accuracy of crop suitability
recommendations."* It is a thesis-defense reference, not application documentation.

The manuscript commits to this exact method:

> "the predictive accuracy of the Machine Learning model is evaluated by calculating the
> **Comparative Accuracy Rate** using a **Confusion Matrix** tested against **historical
> validation datasets**." — Chapter 3, p.53

> Table 7 (p.54), RQ4 row: Analysis Tool = Confusion Matrix (Comparative Accuracy Rate) —
> Excel Data Analysis Toolpak / Python SciPy.

> Table 17 (p.85), Performance Metrics Evaluation: *"a Confusion Matrix assessing precision
> through F1-scores and Root Mean Square Error (RMSE)."*

## Why a labeling rule was needed at all

The trained model (`ml/train.py`) is a **regressor** — it predicts continuous N/P₂O₅/K₂O
kg/ha fertilizer targets, not a category. A Confusion Matrix needs categorical
predicted-vs-actual pairs, so `ml/sop4a_confusion_matrix_eval.py` has to define that
category on both sides of the comparison.

### First attempt (superseded) and what it got wrong

The first version of this evaluation used a per-crop **median split** — "Low need" (≤
median) vs "High need" (> median) — computed from the train split. It produced a genuinely
alarming result: **Corn's `target_k2o` scored F1 = 0.000 despite 70.5% accuracy**, which read
as the model collapsing to a majority-class guess and never detecting a real high-need case.

That result did not survive cross-referencing against **`Nutrient Recommendation
Guide.pdf`** — the source BSWM/DA table that `data/nutrient_recommendation_guide.py` was
transcribed from. The guide's Corn K band is:

| Extractable K (ppm) | 0–35 | 36–55 | 56–250 | 251–300 | >300 |
|---|---|---|---|---|---|
| K₂O recommendation (kg/ha) | 60 | 60 | 60 | 20 | 0 |

The guide recommends a **flat 60 kg/ha across almost the entire real-world Extractable-K
range** (0–250 ppm covers nearly every real soil sample) — it only drops at soils that are
already K-saturated, which is rare. Confirmed directly against the real training data: across
all 1,044 real Corn rows, `target_k2o` takes **exactly three values** — `{0, 20, 60}` — with
60 sitting at the 25th, 50th, *and* 75th percentile simultaneously. A median split at 60
therefore left **zero true "High need" (>60) examples anywhere in the data** for the model to
be scored against — F1 was mathematically forced toward 0 regardless of how good the model
actually was. That's an artifact of imposing an arbitrary binary cut on a bounded,
ceiling-heavy distribution — not a model deficiency.

### The fix: use the guide's own tiers as the classes

Checking further, this pattern isn't unique to Corn K₂O — **every** crop-target's real value
in the dataset is drawn from a small, fixed set of published kg/ha levels (Cassava N: `{0, 30,
40, 50, 60}`; Rice K₂O: `{0, 20, 30, 45, 60}`, etc.), because that's exactly how the guide
itself is structured: a handful of discrete bands per nutrient, not a continuum.

So the corrected rule doesn't invent a threshold at all:
- **Actual label** = the real target value (it already sits exactly on one of the crop's
  guide tiers, since it was assigned by the guide/derivation logic in the first place).
- **Predicted label** = the model's continuous output, **snapped to its nearest guide tier**
  for that crop.
- **Confusion Matrix** = multi-class, over the crop's own known tiers.

This is a stronger claim for defense than a median split: the categories are the literal
published DA/BSWM standard (verifiable against the attached PDF), not a statistical
convenience.

## Data source

`suri_real_training_data.csv` — the cleaned output of `data/parse_real_soil_database.py`,
which parses the raw Region 7 National Soil Database export and derives the fertility
remarks and N/P₂O₅/K₂O targets via the same guide table. Every row is `data_source = "lab"`
(real DA lab data) — this is the "historical validation dataset" the manuscript refers to on
p.53. **Not** `suri_synthetic_training_data_region7.csv` (a separate, differently-schemed
synthetic/STK file) — that file was not used here.

Split sizes reproduced by the script (`train.py`'s exact grouped split, seed=42):
**train 2,366 / val 507 / test 508** rows.

## Results — per crop, per target (guide-tier confusion matrix)

| Crop | Target | Guide tiers (kg/ha) | Accuracy Rate | F1 (macro) | n |
|---|---|---|---|---|---|
| Cassava | target_n | 0, 30, 40, 50, 60 | 0.769 | 0.803 | 13 |
| Cassava | target_p2o5 | 0, 20, 30, 40, 60 | 0.846 | 0.667 | 13 |
| Cassava | target_k2o | 0, 30, 45, 60, 90 | 0.615 | 0.453 | 13 |
| Corn | target_n | 0, 40, 45, 50, 60, 90, 120 | 0.763 | 0.605 | 156 |
| Corn | target_p2o5 | 0, 30, 40, 60 | 0.910 | 0.846 | 156 |
| **Corn** | **target_k2o** | **0, 20, 60** | **0.994** | **0.983** | 156 |
| Rice | target_n | 20, 35, 40, 60, 90 | 0.940 | 0.927 | 252 |
| Rice | target_p2o5 | 0, 20, 30, 40, 45, 60 | 0.940 | 0.791 | 252 |
| Rice | target_k2o | 0, 20, 30, 45, 60 | 0.933 | 0.818 | 252 |

**Corn's `target_k2o` flips from the false "F1 = 0.000" reading to 0.994 accuracy / 0.983
macro-F1** once evaluated as the 3-class problem it actually is. Full per-crop breakdown
(all crops with ≥5 test rows) is written to
`ml/artifacts/sop4a_confusion_matrix/<timestamp>.json` on every run, alongside a pooled
per-target matrix across all crops (see caveat below on that pooled view).

## Limitations — report these at defense, don't let a panelist find them first

1. **Cassava's `target_k2o` is the real remaining weak spot**: 61.5% accuracy, 0.453 macro-F1
   — noticeably worse than every other crop-target shown here. With only n=13 test rows this
   is a small, noisy sample, so don't over-claim either way — state it as "the model's weakest
   nutrient-crop pairing in this evaluation, on a small test set," and tie it back to
   Cassava's overall small training footprint (101 real rows total vs. Corn's 1,044 and
   Rice's 1,688) rather than implying it's a fundamental flaw.
2. **The pooled, all-crops matrix per target** (in the saved JSON's `per_target` key, as
   opposed to `per_crop_target`) mixes crops that use **different units** — field-crop rows
   are kg/ha, while perennial/tree-crop rows (Cacao, Coconut, Mango, Banana, etc.) are
   kg/tree, so a "60" from a field crop and a "0.6" from a tree crop are not the same
   magnitude of anything. That pooled matrix is a rough global summary at best — the
   per-crop-target table above is the methodologically sound one to cite for SOP 4.a, and is
   what should go in the manuscript's Chapter 4 results.
3. **This is not a substitute for RMSE/MAE.** The guide-tier confusion matrix answers "did the
   system recommend the right published tier," which is the actual SOP 4.a question
   ("predictive accuracy of crop suitability recommendations"). RMSE (also reported per
   target above and by `train.py`'s own MAE/R² output) still matters separately as a measure
   of how far off a wrong prediction was, not just whether it was wrong.

## Reproducing this evaluation

```
cd suri-ml-backend
python -m ml.sop4a_confusion_matrix_eval
```

Reads `suri_real_training_data.csv`, reproduces the train/val/test split, trains the same
ensemble the production model uses, predicts on the held-out test split, and writes a JSON
result to `ml/artifacts/sop4a_confusion_matrix/`. No network access or Supabase credentials
required — this runs entirely off the local CSV and the local copy of the Nutrient
Recommendation Guide's tiers already encoded in `data/nutrient_recommendation_guide.py`.

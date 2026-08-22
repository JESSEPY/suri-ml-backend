"""SOP 4.a evaluation: Confusion Matrix / Comparative Accuracy Rate for the
fertilizer-target model, tested against the historical (real, lab-sourced)
validation dataset -- per Chapter 3 of the capstone manuscript (p.53, Table 7
p.54, Table 17 p.85): "the predictive accuracy of the Machine Learning model
is evaluated by calculating the Comparative Accuracy Rate using a Confusion
Matrix tested against historical validation datasets."

Why the labels are "guide tiers", not a statistical split: the trained model
(ml/train.py) is a regressor -- it predicts continuous N/P2O5/K2O kg/ha
targets, not a category. A Confusion Matrix needs categorical predicted/actual
pairs. An earlier version of this script manufactured that category with a
per-crop median split ("Low need" / "High need"), which produced a misleading
result: Corn's target_k2o showed F1 = 0.000 despite 70% accuracy, looking like
a model failure.

It wasn't. Cross-referencing `Nutrient Recommendation Guide.pdf` (the source
BSWM/DA table `data/nutrient_recommendation_guide.py` was transcribed from)
explains why: every target_n/p2o5/k2o value in the real data is not a free
continuous number -- it's assigned from a small, fixed set of published kg/ha
tiers per crop (see GuideRow's om_bands/p_bands/k_bands). Corn's K band table
is `((35,60), (55,60), (250,60), (300,20), (inf,0))` -- i.e. the guide
recommends a flat 60 kg/ha of K2O across almost the entire real-world
Extractable-K range (0-250 ppm), only dropping at soils that are already
K-saturated. Confirmed directly against the real CSV: Corn's target_k2o takes
exactly three values across all 1,044 real rows -- {0.0, 20.0, 60.0} -- with
60.0 at the 25th/50th/75th percentile alike. A median split at 60 leaves zero
true "High need" (>60) examples to score against, which is why F1 collapsed
to 0 -- there's no minority class for the model to have detected in the first
place. That's an artifact of imposing a binary cut on a bounded, ceiling-heavy
distribution, not a model deficiency.

The fix: don't invent a threshold at all. Use the crop's own real, observed
tiers as the class vocabulary (these ARE the guide's discrete recommendation
levels), snap the model's continuous prediction to its nearest tier, and
build a genuine multi-class Confusion Matrix from that. This is a stronger
claim for defense than the median split: the categories are literally the
published DA/BSWM standard, not a statistical convenience.

Uses the exact same split (train.py's RANDOM_SEED, grouped_split) so the
train/val/test partition here matches what a real training run produces --
this is not a fresh random split.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error

from ml.model import EnsembleFertilizerRegressor, LogTargetRegressor
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS, build_preprocessor
from ml.train import RANDOM_SEED, cv_weights, grouped_split

REAL_DATA_CSV = "suri_real_training_data.csv"
OUTPUT_DIR = "ml/artifacts/sop4a_confusion_matrix"


def _load_real_frame(csv_path: str = REAL_DATA_CSV) -> pd.DataFrame:
    # Same NA handling as ml.train._load_frame -- "None" is a real category
    # in variety/cultivation_method/age_band, not a missing value.
    return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])


def _guide_tiers(frame: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """The published kg/ha recommendation levels actually observed per
    (crop, target), sourced from the full real dataset -- these are a fixed
    external vocabulary (the BSWM/DA guide), not something to estimate from
    a train-only split the way a statistical threshold would be."""
    tiers: dict[tuple[str, str], list[float]] = {}
    for crop, group in frame.groupby("crop"):
        for target in TARGET_COLUMNS:
            tiers[(crop, target)] = sorted(group[target].unique().tolist())
    return tiers


def _snap_to_nearest_tier(value: float, tiers: list[float]) -> float:
    return min(tiers, key=lambda tier: abs(tier - value))


def _labels_for(df: pd.DataFrame, values: np.ndarray, tiers: dict[tuple[str, str], list[float]], snap: bool) -> dict[str, np.ndarray]:
    """values: (n_rows, n_targets) array aligned with df's row order.
    snap=True rounds continuous predictions to the nearest known tier;
    snap=False is used for the actual values, which already sit exactly on a
    tier (they were assigned by the guide), so no rounding is needed there."""
    # String labels, not raw floats: sklearn's type_of_target infers a float
    # array with many distinct values as "continuous" (it has no way to know
    # these floats are actually a small per-crop categorical vocabulary), so
    # confusion_matrix refuses it outright. Formatting as text sidesteps that
    # entirely -- "60" and "60.0" would otherwise silently become different
    # classes, hence the g-format to normalize e.g. 60.0 -> "60".
    labels: dict[str, np.ndarray] = {}
    crops = df["crop"].to_numpy()
    for index, target in enumerate(TARGET_COLUMNS):
        target_labels = np.empty(len(df), dtype=object)
        for row, crop in enumerate(crops):
            crop_tiers = tiers.get((crop, target))
            raw = float(values[row, index])
            tier_value = _snap_to_nearest_tier(raw, crop_tiers) if snap else raw
            target_labels[row] = f"{tier_value:g}"
        labels[target] = target_labels
    return labels


def evaluate_sop4a(csv_path: str = REAL_DATA_CSV, seed: int = RANDOM_SEED) -> dict:
    frame = _load_real_frame(csv_path)
    train_df, val_df, test_df = grouped_split(frame, seed=seed)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    weights = cv_weights(X_train, y_train, train_df["laboratory_code"].to_numpy())
    regressor = LogTargetRegressor(EnsembleFertilizerRegressor(weights=weights)).fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    # Tiers are drawn from the full frame (train+val+test): they're a fixed
    # published vocabulary, not a statistic that could leak test information.
    tiers = _guide_tiers(frame)
    actual_labels = _labels_for(test_df, y_test, tiers, snap=False)
    predicted_labels = _labels_for(test_df, predicted, tiers, snap=True)

    results: dict = {"per_target": {}, "per_crop_target": {}}

    for index, target in enumerate(TARGET_COLUMNS):
        y_true_cls = actual_labels[target]
        y_pred_cls = predicted_labels[target]
        class_labels = sorted({*y_true_cls, *y_pred_cls}, key=float)

        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=class_labels)
        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1_macro = f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        rmse = float(np.sqrt(mean_squared_error(y_test[:, index], predicted[:, index])))

        results["per_target"][target] = {
            "confusion_matrix": {"labels": class_labels, "matrix": cm.tolist()},
            "accuracy_rate": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "rmse": rmse,
            "n": len(y_true_cls),
        }

    # Per-crop-target breakdown, mirroring the MAE/R2 table's shape, for
    # crops with enough test rows to be meaningful (n >= 5).
    for crop in sorted(test_df["crop"].unique()):
        crop_mask = test_df["crop"].to_numpy() == crop
        if crop_mask.sum() < 5:
            continue
        results["per_crop_target"][crop] = {}
        for index, target in enumerate(TARGET_COLUMNS):
            y_true_cls = actual_labels[target][crop_mask]
            y_pred_cls = predicted_labels[target][crop_mask]
            acc = accuracy_score(y_true_cls, y_pred_cls)
            f1_macro = f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
            results["per_crop_target"][crop][target] = {
                "accuracy_rate": float(acc),
                "f1_macro": float(f1_macro),
                "n": int(crop_mask.sum()),
                "guide_tiers_kg_ha": tiers.get((crop, target)),
            }

    results["metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_csv": csv_path,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "label_rule": (
            "Actual = the real target value (already one of the crop's published "
            "guide tiers). Predicted = the model's continuous output snapped to "
            "its nearest guide tier. Multi-class confusion matrix over those "
            "tiers, per (crop, target)."
        ),
    }
    return results


def _print_summary(results: dict) -> None:
    print("\n=== SOP 4.a Confusion Matrix / Comparative Accuracy Rate (guide-tier labels) ===")
    meta = results["metadata"]
    print(f"Source: {meta['source_csv']}  (seed={meta['seed']})")
    print(f"Split sizes: {meta['split_sizes']}")
    for target, stats in results["per_target"].items():
        labels = stats["confusion_matrix"]["labels"]
        matrix = stats["confusion_matrix"]["matrix"]
        print(f"\n-- {target} (n={stats['n']}) --")
        print(f"  Tiers (kg/ha): {labels}")
        print(f"  Confusion Matrix:")
        for row in matrix:
            print(f"    {row}")
        print(f"  Accuracy Rate: {stats['accuracy_rate']:.3f}")
        print(f"  F1 (macro):    {stats['f1_macro']:.3f}")
        print(f"  F1 (weighted): {stats['f1_weighted']:.3f}")
        print(f"  RMSE:          {stats['rmse']:.3f}")


if __name__ == "__main__":
    results = evaluate_sop4a()
    _print_summary(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{results['metadata']['generated_at'][:19].replace(':', '')}.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved: {out_path}")

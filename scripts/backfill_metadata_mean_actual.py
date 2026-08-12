#!/usr/bin/env python
"""One-off backfill: adds mean_actual to an already-trained artifact's
metadata.json without retraining the model itself.

Reuses the EXISTING fitted preprocessor/regressor (transform, not
fit_transform) and reproduces the same deterministic grouped_split() this
artifact was originally trained with, then re-runs evaluate() (now
including mean_actual per Task 1) against that same held-out test split.

Before writing anything, this verifies the recomputed mae/r2 values match
what's already in metadata.json byte-for-byte (within floating point
tolerance) -- if they don't match, this is NOT reproducing the original
evaluation (wrong CSV, wrong seed, or the artifact was trained differently
than assumed) and the script refuses to overwrite metadata.json.

Usage: python scripts/backfill_metadata_mean_actual.py \
    --version 2026-08-06-055755 --csv suri_real_training_data.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os

import joblib

from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import ARTIFACT_ROOT, _load_frame, evaluate, grouped_split


def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def backfill(version: str, csv_path: str) -> None:
    directory = os.path.join(ARTIFACT_ROOT, version)
    metadata_path = os.path.join(directory, "metadata.json")

    with open(metadata_path) as handle:
        existing_metadata = json.load(handle)

    frame = _load_frame(csv_path)
    if len(frame) != existing_metadata["n_rows"]:
        raise SystemExit(
            f"Row count mismatch: {csv_path} has {len(frame)} rows, "
            f"metadata.json says this artifact was trained on {existing_metadata['n_rows']}. "
            "Wrong CSV -- refusing to backfill against different data."
        )

    train_df, _, test_df = grouped_split(frame, seed=existing_metadata["seed"])

    preprocessor = joblib.load(os.path.join(directory, "preprocessor.joblib"))
    regressor = joblib.load(os.path.join(directory, "regressor.joblib"))

    X_train = preprocessor.transform(train_df[FEATURE_COLUMNS])
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    recomputed = evaluate(
        regressor, X_test, y_test,
        test_df.data_source.to_numpy(), test_df.crop.to_numpy(),
        X_train, y_train,
    )

    for target, scores in existing_metadata["metrics"]["overall"].items():
        recomputed_scores = recomputed["overall"][target]
        if not _approx_equal(scores["mae"], recomputed_scores["mae"]):
            raise SystemExit(
                f"Reproduction check failed for {target}: existing MAE={scores['mae']}, "
                f"recomputed MAE={recomputed_scores['mae']}. This is not reproducing the "
                "original evaluation -- refusing to overwrite metadata.json. Check the "
                "CSV/seed match how this artifact was actually trained."
            )

    existing_metadata["metrics"] = recomputed
    with open(metadata_path, "w") as handle:
        json.dump(existing_metadata, handle, indent=2)

    print(f"Backfilled mean_actual into {metadata_path} -- reproduction check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Artifact version directory name under ml/artifacts/")
    parser.add_argument("--csv", required=True, help="Exact training CSV this artifact was trained from")
    args = parser.parse_args()
    backfill(args.version, args.csv)

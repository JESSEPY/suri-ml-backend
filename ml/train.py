"""Training, evaluation, and artifact writing (spec section 10).

Every split and cross-validation fold is grouped on laboratory_code — each
physical sample must stay entirely within one split, since a random row
split would otherwise leak information across train/test.

The regressor is wrapped in LogTargetRegressor (ml/model.py) because
targets now span both kg/ha (~20-300, field crops) and kg/tree (~0.003-2,
Coconut/Cacao/Mango/Banana) scales in the same three columns — fitting
directly on that mixture would let the loss be dominated by the
larger-scale rows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ml.model import EnsembleFertilizerRegressor, LogTargetRegressor, build_krr, build_xgb_regressor
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS, build_preprocessor

RANDOM_SEED = 42
ARTIFACT_ROOT = "ml/artifacts"
GROUP_COLUMN = "laboratory_code"


def grouped_split(frame: pd.DataFrame, seed: int = RANDOM_SEED):
    groups = frame[GROUP_COLUMN].to_numpy()

    holdout = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(holdout.split(frame, groups=groups))
    train_df = frame.iloc[train_idx]
    rest_df = frame.iloc[rest_idx]

    halve = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(halve.split(rest_df, groups=rest_df[GROUP_COLUMN].to_numpy()))
    return train_df, rest_df.iloc[val_idx], rest_df.iloc[test_idx]


def cv_weights(X, y, groups, n_splits: int = 5) -> list[tuple[float, float]]:
    """Set each target's (xgb, krr) voting weights from grouped cross-validated
    R2, so each nutrient favours whichever model actually performs better.

    Scored in log1p space, self-contained to this function — consistent
    with how the final model is actually fit (see LogTargetRegressor) —
    without requiring the caller to pre-transform y."""
    splitter = GroupKFold(n_splits=n_splits)
    y_log = np.log1p(y)
    weights: list[tuple[float, float]] = []

    for target in range(y_log.shape[1]):
        scores = {"xgb": [], "krr": []}
        for train_idx, test_idx in splitter.split(X, y_log[:, target], groups):
            for name, builder in (("xgb", build_xgb_regressor), ("krr", build_krr)):
                model = builder().fit(X[train_idx], y_log[train_idx, target])
                scores[name].append(r2_score(y_log[test_idx, target], model.predict(X[test_idx])))

        xgb_score = max(float(np.mean(scores["xgb"])), 0.0)
        krr_score = max(float(np.mean(scores["krr"])), 0.0)
        total = xgb_score + krr_score
        weights.append((0.5, 0.5) if total == 0 else (xgb_score / total, krr_score / total))

    return weights


def evaluate(model, X, y, sources, crops, X_train, y_train) -> dict:
    predicted = model.predict(X)

    overall = {}
    for index, name in enumerate(TARGET_COLUMNS):
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train[:, index])
        overall[name] = {
            "mae": float(mean_absolute_error(y[:, index], predicted[:, index])),
            "r2": float(r2_score(y[:, index], predicted[:, index])),
            "dummy_mae": float(mean_absolute_error(y[:, index], dummy.predict(X))),
        }

    def _grouped_metrics(labels: np.ndarray) -> dict:
        result = {}
        for label in np.unique(labels):
            mask = labels == label
            result[str(label)] = {
                name: {
                    "mae": float(mean_absolute_error(y[mask, index], predicted[mask, index])),
                    "r2": float(r2_score(y[mask, index], predicted[mask, index])),
                }
                for index, name in enumerate(TARGET_COLUMNS)
            }
        return result

    return {
        "overall": overall,
        "by_source": _grouped_metrics(sources),
        "by_crop": _grouped_metrics(crops),
    }


def train(frame: pd.DataFrame, output_dir: str | None = None, seed: int = RANDOM_SEED) -> dict:
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    output_dir = output_dir or os.path.join(ARTIFACT_ROOT, version)
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = grouped_split(frame, seed=seed)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    weights = cv_weights(X_train, y_train, train_df[GROUP_COLUMN].to_numpy())
    regressor = LogTargetRegressor(EnsembleFertilizerRegressor(weights=weights)).fit(X_train, y_train)

    metrics = evaluate(
        regressor, X_test, y_test,
        test_df.data_source.to_numpy(), test_df.crop.to_numpy(),
        X_train, y_train,
    )

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_rows": len(frame),
        "n_records": int(frame[GROUP_COLUMN].nunique()),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "voting_weights": [{"xgb": xgb, "krr": krr} for xgb, krr in weights],
        "metrics": metrics,
    }

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(regressor, os.path.join(output_dir, "regressor.joblib"))
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def _load_frame(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        # keep_default_na=False + explicit na_values=[""]: several of this
        # schema's nominal columns (variety/cultivation_method/age_band) use
        # the literal string "None" as a real, valid category (see
        # ml/preprocessing.py's VARIETIES/CULTIVATION_METHODS/AGE_BANDS) --
        # pandas' default NA-sniffing would otherwise silently read that
        # text as NaN, corrupting the category and tripping the encoder's
        # deliberate handle_unknown="error" on load. An actually-missing
        # numeric cell (e.g. an ungraded organic_matter_pct) is written by
        # csv.DictWriter as a genuinely empty cell, which na_values=[""]
        # still correctly treats as NaN.
        return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    from data.ingest_to_supabase import TABLE, get_client
    rows = get_client().table(TABLE).select("*").execute().data
    return pd.DataFrame(rows)


def _print_summary(metadata: dict) -> None:
    print(f"\nModel version: {metadata['version']}")
    print(f"Records: {metadata['n_records']}  Rows: {metadata['n_rows']}")
    print("\nRegression (held-out test):")
    for name, scores in metadata["metrics"]["overall"].items():
        print(f"  {name:22s} MAE {scores['mae']:7.3f}  R2 {scores['r2']:6.3f}"
              f"  (mean baseline MAE {scores['dummy_mae']:7.3f})")
    print("\nMAE by data source:")
    for source, targets in metadata["metrics"]["by_source"].items():
        errors = "  ".join(f"{name}={scores['mae']:6.3f}"
                           for name, scores in targets.items())
        print(f"  {source:4s}  {errors}")
    print("\nMAE by crop:")
    for crop, targets in metadata["metrics"]["by_crop"].items():
        errors = "  ".join(f"{name}={scores['mae']:6.3f}"
                           for name, scores in targets.items())
        print(f"  {crop:10s}  {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SURI fertilizer ensemble.")
    parser.add_argument("--csv", help="Train from a local CSV instead of Supabase.")
    args = parser.parse_args()

    result = train(_load_frame(args.csv))
    _print_summary(result)

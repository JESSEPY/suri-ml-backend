"""Ensemble model definitions (spec section 8).

XGBoost captures the non-linear pH/phosphorus interaction and categorical splits;
Kernel Ridge Regression contributes an L2-regularised smooth fit that resists
overfitting the synthetic set. sklearn's VotingRegressor does not support
multi-output, so EnsembleFertilizerRegressor holds one per target and exposes a
single multi-output fit/predict.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBClassifier, XGBRegressor

RANDOM_SEED = 42
N_TARGETS = 3


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_krr() -> KernelRidge:
    return KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1)


def build_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


class EnsembleFertilizerRegressor(BaseEstimator, RegressorMixin):
    """Multi-output wrapper holding one VotingRegressor per nutrient target.

    weights: optional per-target (xgb_weight, krr_weight) pairs. Task 8 sets
    these from grouped cross-validated R2, so each nutrient uses whichever model
    actually performs better for it rather than a single global guess.
    """

    def __init__(self, weights: list[tuple[float, float]] | None = None):
        self.weights = weights

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != N_TARGETS:
            raise ValueError(
                f"Expected y of shape (n_samples, {N_TARGETS}), got {y.shape}"
            )

        self.models_ = []
        for target in range(N_TARGETS):
            pair = self.weights[target] if self.weights else None
            voter = VotingRegressor(
                estimators=[("xgb", build_xgb_regressor()), ("krr", build_krr())],
                weights=list(pair) if pair else None,
            )
            voter.fit(X, y[:, target])
            self.models_.append(voter)
        return self

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models_])

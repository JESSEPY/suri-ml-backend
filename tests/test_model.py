import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from ml.model import (
    EnsembleFertilizerRegressor, build_classifier,
    build_xgb_regressor, build_krr,
)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def dataset():
    """A learnable non-linear relationship, standing in for the real one."""
    X = RNG.uniform(-1, 1, size=(400, 6))
    y = np.column_stack([
        50 + 20 * X[:, 0] - 10 * X[:, 1] ** 2,
        40 + 15 * X[:, 2] + 8 * X[:, 3],
        60 - 25 * X[:, 4] + 5 * X[:, 5],
    ])
    return X, y


class TestEnsembleRegressor:
    def test_predict_returns_one_column_per_target(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor().fit(X, y)
        assert model.predict(X).shape == (X.shape[0], 3)

    def test_beats_a_mean_baseline_on_every_target(self, dataset):
        """The guard against the original draft's failure mode: with randomly
        assigned targets the ensemble and the dummy score identically, and that
        equality is the detectable symptom."""
        X, y = dataset
        split = 300
        model = EnsembleFertilizerRegressor().fit(X[:split], y[:split])
        predicted = model.predict(X[split:])

        for target in range(3):
            dummy = DummyRegressor(strategy="mean").fit(X[:split], y[:split, target])
            ensemble_error = mean_absolute_error(y[split:, target], predicted[:, target])
            dummy_error = mean_absolute_error(y[split:, target], dummy.predict(X[split:]))
            assert ensemble_error < dummy_error

    def test_accepts_explicit_per_target_weights(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor(
            weights=[(0.8, 0.2), (0.5, 0.5), (0.3, 0.7)]
        ).fit(X, y)
        assert model.predict(X).shape == (X.shape[0], 3)

    def test_weights_actually_change_predictions(self, dataset):
        X, y = dataset
        xgb_heavy = EnsembleFertilizerRegressor(
            weights=[(1.0, 0.0)] * 3).fit(X, y).predict(X)
        krr_heavy = EnsembleFertilizerRegressor(
            weights=[(0.0, 1.0)] * 3).fit(X, y).predict(X)
        assert not np.allclose(xgb_heavy, krr_heavy)

    def test_holds_one_voting_regressor_per_target(self, dataset):
        X, y = dataset
        model = EnsembleFertilizerRegressor().fit(X, y)
        assert len(model.models_) == 3

    def test_rejects_wrong_target_shape(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            EnsembleFertilizerRegressor().fit(X, y[:, 0])


class TestLogTargetRegressor:
    def test_predictions_are_in_the_original_scale_not_log_scale(self, dataset):
        """The wrapped model is fit on log1p(y) internally, but predict()
        must return real, expm1'd values — a caller with no knowledge of
        the wrapper must see the same kind of numbers either way."""
        X, y = dataset
        from ml.model import LogTargetRegressor
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        predicted = wrapped.predict(X)
        assert predicted.shape == y.shape
        # y in this fixture ranges roughly 15-85; log1p of that range is
        # roughly 2.7-4.5 — a prediction still in log-space would be wildly
        # out of range, not just imprecise.
        assert predicted.min() > 5.0

    def test_handles_a_target_close_to_zero(self):
        """The smallest real targets in this system are ~0.003 (kg/tree,
        young Cacao) -- log1p/expm1 must round-trip near-zero values
        without producing negatives or NaNs."""
        from ml.model import LogTargetRegressor
        X = RNG.uniform(-1, 1, size=(60, 3))
        y = np.column_stack([
            np.full(60, 0.003), np.full(60, 0.01), np.full(60, 0.002),
        ])
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        predicted = wrapped.predict(X)
        assert np.all(predicted >= 0)
        assert not np.any(np.isnan(predicted))

    def test_exposes_the_wrapped_models_attribute(self, dataset):
        X, y = dataset
        from ml.model import LogTargetRegressor
        wrapped = LogTargetRegressor(EnsembleFertilizerRegressor()).fit(X, y)
        assert len(wrapped.models_) == 3


class TestComponentBuilders:
    def test_xgb_regressor_is_seeded(self):
        assert build_xgb_regressor().random_state == 42

    def test_krr_uses_an_rbf_kernel(self):
        assert build_krr().kernel == "rbf"

    def test_classifier_fits_string_labels(self):
        X = RNG.uniform(-1, 1, size=(60, 4))
        labels = np.array(["recipe-a", "recipe-b", "recipe-c"] * 20)
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(labels)
        model = build_classifier().fit(X, encoder.transform(labels))
        assert len(encoder.inverse_transform(model.predict(X))) == 60

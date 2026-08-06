import json
import numpy as np
import pytest

from tests.fixtures import sample_training_frame
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import grouped_split, cv_weights, train


@pytest.fixture(scope="module")
def frame():
    return sample_training_frame(num_records=200, seed=42)


class TestGroupedSplit:
    def test_no_laboratory_code_spans_two_splits(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        train_codes = set(train_df.laboratory_code)
        val_codes = set(val_df.laboratory_code)
        test_codes = set(test_df.laboratory_code)
        assert not (train_codes & val_codes)
        assert not (train_codes & test_codes)
        assert not (val_codes & test_codes)

    def test_splits_cover_every_row_exactly_once(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        assert len(train_df) + len(val_df) + len(test_df) == len(frame)

    def test_split_is_roughly_seventy_fifteen_fifteen(self, frame):
        train_df, val_df, test_df = grouped_split(frame, seed=42)
        assert 0.60 < len(train_df) / len(frame) < 0.80
        assert 0.08 < len(test_df) / len(frame) < 0.22

    def test_split_is_deterministic(self, frame):
        first = grouped_split(frame, seed=42)[0].laboratory_code.tolist()
        second = grouped_split(frame, seed=42)[0].laboratory_code.tolist()
        assert first == second

    def test_each_code_appears_exactly_once(self, frame):
        """Unlike the retired synthetic generator (which produced 3 rows per
        code, one per data_source), real training data is one row per
        laboratory_code — only "lab" has real training rows now."""
        train_df, _, _ = grouped_split(frame, seed=42)
        counts = train_df.groupby("laboratory_code").size()
        assert (counts == 1).all()


class TestCvWeights:
    def test_returns_one_weight_pair_per_target(self, frame):
        from ml.preprocessing import build_preprocessor
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(frame[FEATURE_COLUMNS])
        y = frame[TARGET_COLUMNS].to_numpy()
        weights = cv_weights(X, y, frame.laboratory_code.to_numpy())
        assert len(weights) == 3
        for xgb_weight, krr_weight in weights:
            assert xgb_weight >= 0 and krr_weight >= 0
            assert xgb_weight + krr_weight == pytest.approx(1.0)


class TestTrain:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        output = tmp_path_factory.mktemp("artifacts")
        return train(sample_training_frame(num_records=200, seed=42), output_dir=str(output)), output

    def test_writes_all_artifacts(self, result):
        _, output = result
        for name in ["preprocessor.joblib", "regressor.joblib", "clf_app1.joblib",
                     "clf_app2.joblib", "label_encoders.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 200
        assert "trained_at" in written
        assert "version" in written

    def test_reports_metrics_per_data_source(self, result):
        metadata, _ = result
        assert set(metadata["metrics"]["by_source"].keys()) == {"lab"}

    def test_reports_metrics_per_crop(self, result):
        """New this pass: with targets now spanning kg_ha and kg_tree
        scales in the same columns, aggregate metrics alone could hide a
        regression specific to the tree crops — this breakdown is how a
        retrain would actually catch that."""
        metadata, _ = result
        from data.nutrient_recommendation_guide import CROPS
        assert set(metadata["metrics"]["by_crop"].keys()) <= set(CROPS)
        assert len(metadata["metrics"]["by_crop"]) > 0

    def test_beats_the_mean_baseline_on_every_target(self, result):
        metadata, _ = result
        for target, scores in metadata["metrics"]["overall"].items():
            assert scores["mae"] < scores["dummy_mae"], target

    def test_no_golden_test_key(self, result):
        """Retired along with Sorghum — see the design spec's 'Golden test —
        retired' section."""
        metadata, _ = result
        assert "golden_test" not in metadata

    def test_predictions_are_non_negative(self, result):
        """A sanity floor independent of any specific golden record — every
        real target in this system (kg/ha or kg/tree) is >= 0; a
        LogTargetRegressor bug that let a negative slip through the expm1
        inverse would be a real, silent correctness break."""
        metadata, output = result
        import joblib
        import numpy as np
        regressor = joblib.load(output / "regressor.joblib")
        preprocessor = joblib.load(output / "preprocessor.joblib")
        frame = sample_training_frame(num_records=20, seed=99)
        X = preprocessor.transform(frame[FEATURE_COLUMNS])
        predicted = regressor.predict(X)
        assert np.all(predicted >= 0)

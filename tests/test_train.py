import json
import numpy as np
import pytest

from data.generate_synthetic_data import generate
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import grouped_split, cv_weights, train


@pytest.fixture(scope="module")
def frame():
    return generate(num_records=200, seed=42)


class TestGroupedSplit:
    def test_no_laboratory_code_spans_two_splits(self, frame):
        """Each record contributes three rows sharing one code. A random row
        split would put a record's lab view in train and its iot view in test —
        the model would be scored on data it had effectively already seen."""
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

    def test_all_three_views_of_a_code_land_together(self, frame):
        train_df, _, _ = grouped_split(frame, seed=42)
        counts = train_df.groupby("laboratory_code").size()
        assert (counts == 3).all()


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
        return train(generate(num_records=200, seed=42), output_dir=str(output)), output

    def test_writes_all_artifacts(self, result):
        _, output = result
        for name in ["preprocessor.joblib", "regressor.joblib", "clf_app1.joblib",
                     "clf_app2.joblib", "label_encoders.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 600
        assert "trained_at" in written
        assert "version" in written

    def test_reports_metrics_per_data_source(self, result):
        """IoT has strictly less information than lab and should score worse. If
        it does not, the missingness handling is suspect."""
        metadata, _ = result
        for source in ["lab", "stk", "iot"]:
            assert source in metadata["metrics"]["by_source"]

    def test_beats_the_mean_baseline_on_every_target(self, result):
        metadata, _ = result
        for target, scores in metadata["metrics"]["overall"].items():
            assert scores["mae"] < scores["dummy_mae"], target

    def test_golden_record_scc417_is_within_tolerance(self, result):
        """The only real labeled example available."""
        metadata, _ = result
        golden = metadata["golden_test"]
        assert abs(golden["predicted_n"] - 50) <= 15
        assert abs(golden["predicted_p2o5"] - 40) <= 15
        assert abs(golden["predicted_k2o"] - 60) <= 15
        assert golden["passed"] is True

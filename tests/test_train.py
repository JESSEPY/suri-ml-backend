import json
import numpy as np
import pandas as pd
import pytest

from tests.fixtures import sample_training_frame
from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train import grouped_split, cv_weights, train, _load_frame


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
        for name in ["preprocessor.joblib", "regressor.joblib", "metadata.json"]:
            assert (output / name).exists()

    def test_no_longer_writes_classifier_artifacts(self, result):
        """The classifier heads were removed -- app1/app2_formulation is now
        computed directly by ml.recipes.plan_applications(), not predicted.
        This test exists so a future change can't silently reintroduce the
        classifier-training path without a deliberate decision to do so."""
        _, output = result
        for name in ["clf_app1.joblib", "clf_app2.joblib", "label_encoders.joblib"]:
            assert not (output / name).exists()

    def test_metadata_records_provenance(self, result):
        metadata, output = result
        written = json.loads((output / "metadata.json").read_text())
        assert written["seed"] == 42
        assert written["n_rows"] == 200
        assert "trained_at" in written
        assert "version" in written

    def test_metadata_has_no_classification_keys(self, result):
        metadata, _ = result
        assert "classification_metrics" not in metadata
        assert "recipe_vocabulary" not in metadata

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

    def test_reports_mean_actual_per_crop_target(self, result):
        """Needed to normalize MAE into a relative error (MAE / mean_actual)
        for the confidence feature — see
        docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md."""
        metadata, _ = result
        for crop_metrics in metadata["metrics"]["by_crop"].values():
            for target_metrics in crop_metrics.values():
                assert "mean_actual" in target_metrics
                assert isinstance(target_metrics["mean_actual"], float)

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


class TestLoadFrameCsvNoneSentinel:
    """Regression test: data/parse_real_soil_database.py writes the literal
    4-character string "None" into variety/cultivation_method/age_band as a
    deliberate sentinel category (see ml/preprocessing.py's
    VARIETIES/CULTIVATION_METHODS/AGE_BANDS). Plain pd.read_csv() silently
    sniffs that text as a missing value and turns it into NaN, which then
    trips OneHotEncoder(handle_unknown="error") downstream. _load_frame()
    must round-trip the literal string, while still treating a genuinely
    blank numeric cell as missing.
    """

    @pytest.fixture(scope="class")
    def csv_path(self, tmp_path_factory):
        frame = sample_training_frame(num_records=5, seed=7)
        frame = frame.copy()
        frame.loc[0, "variety"] = "None"
        frame.loc[0, "cultivation_method"] = "None"
        frame.loc[0, "age_band"] = "None"
        # A genuinely missing optional numeric cell -- e.g. an ungraded
        # organic_matter_pct -- must still round-trip as NaN.
        frame.loc[1, "organic_matter_pct"] = np.nan

        path = tmp_path_factory.mktemp("data") / "sentinel.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_literal_none_string_survives_as_string(self, csv_path):
        loaded = _load_frame(csv_path)
        assert loaded.loc[0, "variety"] == "None"
        assert not pd.isna(loaded.loc[0, "variety"])
        assert loaded.loc[0, "cultivation_method"] == "None"
        assert not pd.isna(loaded.loc[0, "cultivation_method"])
        assert loaded.loc[0, "age_band"] == "None"
        assert not pd.isna(loaded.loc[0, "age_band"])

    def test_genuinely_blank_numeric_cell_still_reads_as_nan(self, csv_path):
        loaded = _load_frame(csv_path)
        assert pd.isna(loaded.loc[1, "organic_matter_pct"])

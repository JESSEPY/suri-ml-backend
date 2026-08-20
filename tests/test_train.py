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


class TestTrainWithNoData:
    """A full reset (every row cleared from ml_soil_training_data, ready to
    retrain from scratch) or simply clicking "Train first model" before
    uploading anything both reach train() with zero rows. Before this,
    grouped_split() raised a bare KeyError('laboratory_code') -- the empty
    frame _load_frame() returns has no columns at all, so indexing by
    GROUP_COLUMN fails -- which the admin UI's error-message heuristic
    then misread as an upload-format problem ("Training data has a problem
    with 'laboratory_code'"), not "there's no data yet."""

    def test_raises_a_clear_error_instead_of_a_bare_keyerror(self):
        # Matches _load_frame()'s real empty-table shape (pd.DataFrame([]) on
        # zero Supabase rows has no columns at all, not just no rows) rather
        # than a same-schema-but-empty frame, which wouldn't have reproduced
        # the original KeyError('laboratory_code') this guards against.
        empty = pd.DataFrame([])
        with pytest.raises(ValueError, match="No training data"):
            train(empty)


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

    def test_metrics_are_strict_json_serializable(self, result):
        """r2_score is undefined (NaN) for a crop/source whose held-out
        split has fewer than 2 samples or zero target variance -- this
        200-row fixture spreads across 81 crops, so several crops hit this
        every run. A raw NaN survives Python's own permissive json.dumps
        (it just prints the invalid token `NaN`) but breaks the moment
        registry.record_version() tries to upsert it to Supabase, which
        rejects it with "Out of range float values are not JSON compliant:
        nan". allow_nan=False here reproduces that failure locally."""
        metadata, _ = result
        json.dumps(metadata["metrics"], allow_nan=False)

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

    def test_reports_sample_count_per_crop_target(self, result):
        """Needed so a persistent per-crop confidence view (main.py's
        GET /model/confidence) can show how many rows a crop's tier is
        actually based on -- a high R2 on 5 rows and a high R2 on 500 rows
        must not render identically."""
        metadata, _ = result
        for crop, crop_metrics in metadata["metrics"]["by_crop"].items():
            for target_metrics in crop_metrics.values():
                assert "n" in target_metrics
                assert isinstance(target_metrics["n"], int)
                assert target_metrics["n"] > 0

    def test_reports_train_val_test_split_sizes_per_crop(self, result):
        """metadata["split_sizes"] is overall-only -- the admin Confidence
        view needs a per-crop breakdown too, so an admin can see a crop's
        real total row count (train+val+test) next to the much smaller
        held-out test count its tier is actually built on."""
        metadata, _ = result
        coverage = metadata["metrics"]["crop_coverage"]
        # crop_coverage covers every crop with ANY real data (train, val, or
        # test); by_crop only covers crops that happened to land in the test
        # split, which can be a strict subset -- a crop with rows only in
        # train/val (no test rows at all) has no tier, but still has real
        # data worth showing coverage for.
        assert set(coverage.keys()) >= set(metadata["metrics"]["by_crop"].keys())
        for crop, counts in coverage.items():
            assert set(counts.keys()) == {"train", "val", "test", "total"}
            for key in ("train", "val", "test", "total"):
                assert isinstance(counts[key], int)
            assert counts["total"] == counts["train"] + counts["val"] + counts["test"]
            # Where a crop has a tier at all, by_crop's own "n" (evaluate()'s
            # test-set count) must agree with crop_coverage's "test" -- same
            # number, two different views.
            if crop in metadata["metrics"]["by_crop"]:
                any_target = next(iter(metadata["metrics"]["by_crop"][crop].values()))
                assert counts["test"] == any_target["n"]

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


class TestFetchAllRowsPaginates:
    """PostgREST caps one response at 1000 rows; the real corpus is larger.
    A single .select("*") would silently return a truncated corpus and train
    on it without any error surfacing."""

    def test_keeps_requesting_until_a_short_page_arrives(self):
        from ml.train import fetch_all_rows

        pages = [
            [{"laboratory_code": f"A{i}"} for i in range(1000)],
            [{"laboratory_code": f"B{i}"} for i in range(1000)],
            [{"laboratory_code": f"C{i}"} for i in range(76)],
        ]
        requested_ranges = []

        class FakeQuery:
            def __init__(self, page):
                self._page = page

            def range(self, start, end):
                requested_ranges.append((start, end))
                return self

            def execute(self):
                return type("R", (), {"data": self._page})()

        class FakeTable:
            def __init__(self):
                self._calls = 0

            def select(self, _columns):
                page = pages[self._calls]
                self._calls += 1
                return FakeQuery(page)

        class FakeClient:
            def __init__(self):
                self._table = FakeTable()

            def table(self, _name):
                return self._table

        rows = fetch_all_rows(FakeClient(), "ml_soil_training_data")

        assert len(rows) == 2076
        assert requested_ranges == [(0, 999), (1000, 1999), (2000, 2999)]

    def test_single_short_page_stops_immediately(self):
        from ml.train import fetch_all_rows

        class FakeQuery:
            def range(self, start, end):
                return self

            def execute(self):
                return type("R", (), {"data": [{"laboratory_code": "A"}]})()

        class FakeClient:
            def table(self, _name):
                return type("T", (), {"select": lambda _self, _c: FakeQuery()})()

        assert len(fetch_all_rows(FakeClient(), "t")) == 1

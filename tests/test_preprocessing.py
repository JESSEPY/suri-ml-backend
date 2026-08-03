import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from data.generate_synthetic_data import generate
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS,
)


@pytest.fixture(scope="module")
def frame():
    return generate(num_records=100, seed=42)


@pytest.fixture(scope="module")
def fitted(frame):
    preprocessor = build_preprocessor()
    preprocessor.fit(frame[FEATURE_COLUMNS])
    return preprocessor


class TestColumnSets:
    def test_feature_columns_is_the_union_of_the_three_groups(self):
        assert set(FEATURE_COLUMNS) == set(CONTINUOUS) | set(ORDERED) | set(NOMINAL)

    def test_farm_area_is_not_a_feature(self):
        """Nutrient targets are kg/ha rates, identical regardless of farm size.
        Including area would invite a spurious area-to-dosage correlation."""
        assert "farm_area_sqm" not in FEATURE_COLUMNS

    def test_leakage_columns_are_not_features(self):
        for column in TARGET_COLUMNS + ["app1_formulation", "app2_formulation",
                                        "laboratory_code"]:
            assert column not in FEATURE_COLUMNS

    def test_deferred_metadata_is_not_a_feature(self):
        """Synthetic dates and sites carry no signal — including them would
        train on pure noise."""
        assert "date_of_sampling" not in FEATURE_COLUMNS
        assert "sampling_site" not in FEATURE_COLUMNS


class TestTransform:
    def test_output_is_two_dimensional_and_numeric(self, fitted, frame):
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert out.ndim == 2
        assert out.shape[0] == len(frame)

    def test_output_width_matches_feature_names(self, fitted, frame):
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert out.shape[1] == len(fitted.get_feature_names_out())

    def test_no_nan_survives_transform(self, fitted, frame):
        import numpy as np
        out = fitted.transform(frame[FEATURE_COLUMNS])
        assert not np.isnan(out).any()

    def test_transform_before_fit_raises(self, frame):
        with pytest.raises(NotFittedError):
            build_preprocessor().transform(frame[FEATURE_COLUMNS])


class TestMissingnessIndicators:
    def test_indicator_exists_for_organic_matter(self, fitted):
        """The IoT view has no organic matter at all — the model must be able
        to see that it was not measured, rather than trust the median."""
        names = list(fitted.get_feature_names_out())
        assert any("organic_matter_pct" in name and "missing" in name.lower()
                   for name in names)

    def test_indicator_exists_for_sensor_columns(self, fitted):
        names = list(fitted.get_feature_names_out())
        assert any("temp" in name and "missing" in name.lower() for name in names)


class TestEncoders:
    def test_unknown_remark_encodes_without_error(self, fitted, frame):
        row = frame[FEATURE_COLUMNS].head(1).copy()
        row.loc[:, "om_remark"] = "UNKNOWN"
        fitted.transform(row)

    def test_unknown_crop_raises(self, fitted, frame):
        """A silent all-zeros crop vector would yield a confident-looking
        prediction computed from no crop information at all."""
        row = frame[FEATURE_COLUMNS].head(1).copy()
        row.loc[:, "crop"] = "Durian"
        with pytest.raises(ValueError):
            fitted.transform(row)

    def test_ordinal_encoding_preserves_band_order(self, fitted):
        rows = pd.DataFrame([
            {**_base_row(), "k_remark": "DEFICIENT"},
            {**_base_row(), "k_remark": "SUFFICIENT"},
            {**_base_row(), "k_remark": "SUFFICIENT+"},
        ])[FEATURE_COLUMNS]
        out = fitted.transform(rows)
        index = list(fitted.get_feature_names_out()).index("ord__k_remark")
        assert out[0][index] < out[1][index] < out[2][index]


def _base_row() -> dict:
    return {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_ppm": 120.0,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": "SLIGHTLY ACIDIC", "om_remark": "MODERATELY LOW",
        "p_remark": "MEDIUM", "k_remark": "SUFFICIENT",
        "crop": "Corn", "data_source": "lab",
    }

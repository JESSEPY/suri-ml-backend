import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tests.fixtures import sample_training_frame
from ml.preprocessing import (
    build_preprocessor, FEATURE_COLUMNS, CONTINUOUS, ORDERED, NOMINAL,
    TARGET_COLUMNS, VARIETIES, CULTIVATION_METHODS, AGE_BANDS,
)


@pytest.fixture(scope="module")
def frame():
    return sample_training_frame(num_records=100, seed=42)


@pytest.fixture(scope="module")
def fitted(frame):
    return build_preprocessor().fit(frame[FEATURE_COLUMNS])


class TestSchema:
    def test_target_columns_are_unit_agnostic_names(self):
        assert TARGET_COLUMNS == ["target_n", "target_p2o5", "target_k2o"]

    def test_potassium_column_is_cmol_kg(self):
        assert "potassium_cmol_kg" in CONTINUOUS
        assert "potassium_ppm" not in CONTINUOUS

    def test_nominal_includes_all_four_categorical_axes(self):
        assert NOMINAL == ["crop", "variety", "cultivation_method", "age_band", "data_source"]


class TestPreprocessor:
    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            build_preprocessor().transform(pd.DataFrame())

    def test_fit_transform_produces_a_dense_2d_array(self, frame):
        result = build_preprocessor().fit_transform(frame[FEATURE_COLUMNS])
        assert result.ndim == 2
        assert result.shape[0] == len(frame)

    def test_missing_continuous_value_does_not_raise(self, frame):
        degraded = frame.copy()
        degraded.loc[0, "phosphorus_ppm"] = None
        build_preprocessor().fit_transform(degraded[FEATURE_COLUMNS])


def _base_row() -> dict:
    return {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_cmol_kg": 0.35,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": "SLIGHTLY ACIDIC", "om_remark": "MODERATELY LOW",
        "p_remark": "MEDIUM", "k_remark": "SUFFICIENT",
        "crop": "Corn", "variety": "None", "cultivation_method": "None",
        "age_band": "None", "data_source": "lab",
    }


class TestVarietyEncoding:
    def test_variety_is_a_feature_column(self):
        assert "variety" in FEATURE_COLUMNS
        assert "variety" in NOMINAL

    def test_none_variety_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Cassava", "variety": "None"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_known_variety_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Rice", "variety": "Hybrid"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_unknown_variety_raises(self, fitted):
        row = pd.DataFrame([{**_base_row(), "variety": "Nonexistent"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)


class TestCultivationMethodEncoding:
    def test_cultivation_method_is_a_feature_column(self):
        assert "cultivation_method" in FEATURE_COLUMNS

    def test_known_cultivation_method_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Rice", "cultivation_method": "Upland"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_unknown_cultivation_method_raises(self, fitted):
        row = pd.DataFrame([{**_base_row(), "cultivation_method": "Nonexistent"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)


class TestAgeBandEncoding:
    def test_age_band_is_a_feature_column(self):
        assert "age_band" in FEATURE_COLUMNS

    def test_known_age_band_is_accepted(self, fitted):
        row = pd.DataFrame([{**_base_row(), "crop": "Coconut", "age_band": "9yrs & up"}])[FEATURE_COLUMNS]
        fitted.transform(row)

    def test_bearing_alias_is_never_a_valid_category(self, fitted):
        """'bearing' must be resolved to a specific band before it ever
        reaches this layer — see data.nutrient_recommendation_guide.
        resolve_age_band. If the literal word leaks in here, that's a bug
        upstream, and this encoder must reject it like any other unknown
        category rather than silently accept it."""
        row = pd.DataFrame([{**_base_row(), "crop": "Coconut", "age_band": "bearing"}])[FEATURE_COLUMNS]
        with pytest.raises(ValueError):
            fitted.transform(row)

"""Feature preprocessing (spec section 7).

Continuous features are median-imputed with an explicit missingness indicator,
so the model learns "this was not measured" rather than silently trusting an
imputed guess — essential because each data source is missing different things.

Remarks are ordinal-encoded rather than one-hot because the bands are genuinely
ordered (LOW < MEDIUM < HIGH); one-hot would discard that ordering, which matters
for the Kernel Ridge half of the ensemble.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data.nutrient_recommendation_guide import CROPS, RICE_CULTIVATION_METHODS, BEARING_AGE_BAND, GUIDE_ROWS
from ml.remarks import K_ORDER, OM_ORDER, P_ORDER, PH_ORDER

CONTINUOUS = [
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
]

ORDERED = ["ph_remark", "om_remark", "p_remark", "k_remark"]

NOMINAL = ["crop", "variety", "cultivation_method", "age_band", "data_source"]

FEATURE_COLUMNS = CONTINUOUS + ORDERED + NOMINAL

TARGET_COLUMNS = ["target_n", "target_p2o5", "target_k2o"]

DATA_SOURCES = ["iot", "lab", "stk"]

# "None" (the string, not Python None) covers every crop/row that doesn't
# track this axis. See data/nutrient_recommendation_guide.py for how each
# is resolved from a request before it ever reaches this layer.
VARIETIES = ["None", "Hybrid", "Inbred", "OPV"]
CULTIVATION_METHODS = ["None", *RICE_CULTIVATION_METHODS]
AGE_BANDS = ["None"] + sorted({
    key.split("|", 1)[1] for key in GUIDE_ROWS if key.split("|", 1)[0] in BEARING_AGE_BAND
})

_ORDERED_CATEGORIES = [PH_ORDER, OM_ORDER, P_ORDER, K_ORDER]


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    ordered = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", OrdinalEncoder(
            categories=_ORDERED_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # handle_unknown="error", deliberately: an unrecognised crop, variety,
    # cultivation_method, or age_band must fail loudly rather than produce
    # an all-zeros vector and a confident-looking prediction built on no
    # real information. A farmer acting on that could over- or
    # under-fertilize a real field.
    nominal = OneHotEncoder(
        categories=[CROPS, VARIETIES, CULTIVATION_METHODS, AGE_BANDS, DATA_SOURCES],
        handle_unknown="error",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("cont", continuous, CONTINUOUS),
        ("ord", ordered, ORDERED),
        ("nom", nominal, NOMINAL),
    ])

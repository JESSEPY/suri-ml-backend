"""Feature preprocessing (spec section 7).

Continuous features are median-imputed with an explicit missingness indicator,
so the model learns "this was not measured" rather than silently trusting an
imputed guess — essential because each data source is missing different things.

Remarks are ordinal-encoded rather than one-hot because the bands are genuinely
ordered (LOW < MEDIUM < HIGH); one-hot would discard that ordering, which matters
for the Kernel Ridge half of the ensemble.

CANONICAL_CROPS/VARIETIES_BY_CROP/etc. are no longer read once at this
module's import time — see build_categories() below. See
docs/superpowers/specs/2026-09-01-crop-nutrient-reference-cms-design.md.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Module-qualified, not `from data.nutrient_recommendation_guide import X` --
# the latter copies a reference at import time, so a later reload-guide
# reassignment of guide.CANONICAL_CROPS etc. would be invisible here. `guide.`
# attribute access always reads the CURRENT value.
from data import nutrient_recommendation_guide as guide
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

_ORDERED_CATEGORIES = [PH_ORDER, OM_ORDER, P_ORDER, K_ORDER]


def build_categories() -> dict[str, list[str]]:
    """Re-derives the nominal encoder's category vocabulary from the guide
    module's CURRENT state (guide.CANONICAL_CROPS, guide.VARIETIES_BY_CROP,
    guide.CULTIVATION_METHODS_BY_CROP, guide.GUIDE_ROWS, guide.BEARING_AGE_BAND)
    every time it's called, rather than computing it once at this module's
    own import time. Called fresh inside build_preprocessor() (itself called
    fresh per training job), so a crop added via /admin/reload-guide is
    picked up by the very next retrain with no further code changes needed.

    "None" (the string, not Python None) covers every crop/row that doesn't
    track a given axis. See data/nutrient_recommendation_guide.py for how
    each is resolved from a request before it ever reaches this layer.
    """
    varieties = ["None"] + sorted({
        variety for varieties in guide.VARIETIES_BY_CROP.values() for variety in varieties
    })
    cultivation_methods = ["None"] + sorted({
        method for methods in guide.CULTIVATION_METHODS_BY_CROP.values() for method in methods
    })
    age_bands = ["None"] + sorted({
        key.split("|", 1)[1] for key in guide.GUIDE_ROWS
        if "|" in key and key.split("|", 1)[0] in guide.BEARING_AGE_BAND
    })
    return {
        "crops": list(guide.CANONICAL_CROPS),
        "varieties": varieties,
        "cultivation_methods": cultivation_methods,
        "age_bands": age_bands,
    }


def build_preprocessor() -> ColumnTransformer:
    categories = build_categories()

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
        categories=[
            categories["crops"], categories["varieties"],
            categories["cultivation_methods"], categories["age_bands"], DATA_SOURCES,
        ],
        handle_unknown="error",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("cont", continuous, CONTINUOUS),
        ("ord", ordered, ORDERED),
        ("nom", nominal, NOMINAL),
    ])

"""Test-only DataFrame builder in the training-data schema.

NOT used by any production path (ml/train.py, data/ingest_to_supabase.py,
data/parse_real_soil_database.py never import this). Exists purely so tests
of generic pipeline mechanics — grouping, splitting, chunking, JSON
serialization — have a fast, hermetic multi-row frame to exercise, without
depending on the real xlsx file or asserting anything about real-world
accuracy.
"""

from __future__ import annotations

import random

import pandas as pd

from data.nutrient_recommendation_guide import (
    CROPS, target_unit_for_crop, resolve_crop, resolve_variety, resolve_cultivation_method, resolve_age_band,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark
from data.nutrient_recommendation_guide import k_cmol_to_ppm

COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "cultivation_method", "age_band", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


def sample_training_frame(num_records: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    for index in range(num_records):
        crop_name = rng.choice(CROPS)
        crop = resolve_crop(crop_name)
        ph = round(rng.uniform(4.5, 8.5), 2)
        om = round(rng.uniform(0.5, 5.0), 2)
        phosphorus = round(rng.uniform(5.0, 40.0), 2)
        potassium = round(rng.uniform(0.1, 1.0), 3)

        variety = resolve_variety(crop, None)
        cultivation_method = resolve_cultivation_method(crop, None)
        age_band = resolve_age_band(crop, None)

        n, p2o5, k2o = nutrient_requirements(crop, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o, target_unit_for_crop(crop))

        rows.append({
            "laboratory_code": f"TEST-{index}",
            "data_source": "lab",
            "sampling_site": None,
            "date_of_sampling": None,
            "crop": crop,
            "variety": variety,
            "cultivation_method": cultivation_method,
            "age_band": age_band,
            "farm_area_sqm": round(rng.uniform(100, 10000), 2),
            "ph": ph,
            "organic_matter_pct": om,
            "total_nitrogen_pct": round(total_nitrogen_from_om(om), 4),
            "phosphorus_ppm": phosphorus,
            "potassium_cmol_kg": potassium,
            "temp": None, "moist": None, "ec": None, "sal": None,
            "ph_remark": derive_ph_remark(ph),
            "om_remark": derive_om_remark(om),
            "p_remark": derive_p_remark(phosphorus),
            "k_remark": derive_k_remark(k_cmol_to_ppm(potassium)),
            "target_n": n, "target_p2o5": p2o5, "target_k2o": k2o,
            "target_unit": target_unit_for_crop(crop),
            "app1_formulation": app1, "app2_formulation": app2,
        })

    return pd.DataFrame(rows, columns=COLUMNS)

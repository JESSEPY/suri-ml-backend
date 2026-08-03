"""Synthetic training data generator (spec section 6).

Produces a lab-grade record, then derives three training views from it — one per
data source, each carrying that source's real inference-time missingness pattern.
Training on the actual missingness the model will face is what makes it usable by
all three SURI flows.

Run as a script to write suri_synthetic_training_data_region7.csv.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from data.crop_baselines import CROPS
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import (
    derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark,
)

RANDOM_SEED = 42
DEFAULT_NUM_RECORDS = 5000
OUTPUT_CSV = "suri_synthetic_training_data_region7.csv"

LOCATIONS = [
    "Jampang, Argao, Cebu", "Tulic, Argao, Cebu",
    "Mantalongon, Dalaguete, Cebu", "Obo, Dalaguete, Cebu",
    "Campesa, Barili, Cebu", "Guadalupe, Carcar, Cebu",
    "Lamacan, Sibonga, Cebu", "San Nicholas, Cebu City, Cebu",
    "Sudlon II, Cebu City, Cebu", "Guba, Cebu City, Cebu",
    "Cansomoroy, Balamban, Cebu", "Luray, Toledo, Cebu",
    "Gairan, Bogo City, Cebu", "Buenos Aires, Carmen, Bohol",
    "Nueva Fuerza, Carmen, Bohol", "Biabas, Ubay, Bohol",
    "San Pascual, Ubay, Bohol", "Pula, Canlaon, Negros Oriental",
    "Linothangan, Canlaon, Negros Oriental",
]

COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_ppm",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
    "app1_formulation", "app2_formulation",
    "is_synthetic",
]

# STK bands map to representative ranges in LAB units. The kit's own `value`
# fields in suri/shared/constants/stk-options.ts are an internal app scale, not
# lab units — STK P reads 30/70/110 where real Olsen P spans 5-40 ppm.
_STK_PH_INDICATORS = [4.5, 5.5, 6.8, 7.8]
_STK_OM_BANDS = [(2.0, (0.5, 2.0)), (3.5, (2.0, 3.5)), (float("inf"), (3.5, 6.0))]
_STK_P_BANDS = [(10.0, (3.0, 10.0)), (20.0, (10.0, 20.0)), (float("inf"), (20.0, 40.0))]
_STK_K_BANDS = [(80.0, (30.0, 80.0)), (150.0, (80.0, 150.0)), (float("inf"), (150.0, 250.0))]

_IOT_SENSOR_NOISE = 0.10


def _coarsen(rng: random.Random, value: float, bands) -> float:
    """Place a lab value in its STK band, then resample within that band — the
    coarsening a kit user's reading actually represents."""
    for upper, (low, high) in bands:
        if value < upper:
            return round(rng.uniform(low, high), 2)
    return round(value, 2)


def _nearest(value: float, options: list[float]) -> float:
    return min(options, key=lambda option: abs(option - value))


def _row(base: dict, source: str, **overrides) -> dict:
    row = {**base, "data_source": source, **overrides}
    row["ph_remark"] = derive_ph_remark(row["ph"])
    row["om_remark"] = derive_om_remark(row["organic_matter_pct"])
    row["p_remark"] = derive_p_remark(row["phosphorus_ppm"])
    row["k_remark"] = derive_k_remark(row["potassium_ppm"])
    return row


def generate(num_records: int = DEFAULT_NUM_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    start_date = datetime(2025, 1, 1)
    rows: list[dict] = []

    for index in range(num_records):
        ph = round(rng.uniform(4.5, 8.5), 2)
        om = round(rng.uniform(0.5, 5.0), 2)
        phosphorus = round(rng.uniform(5.0, 40.0), 2)
        potassium = round(rng.uniform(50.0, 250.0), 2)
        crop = rng.choice(CROPS)

        n, p2o5, k2o = nutrient_requirements(crop, ph, om, phosphorus, potassium)
        app1, app2 = plan_applications(n, p2o5, k2o)

        base = {
            "laboratory_code": f"SCC-26-{400 + index}",
            "sampling_site": rng.choice(LOCATIONS),
            "date_of_sampling": (
                start_date + timedelta(days=rng.randint(0, 500))
            ).strftime("%Y-%m-%d"),
            "crop": crop,
            "farm_area_sqm": round(rng.uniform(100, 10000), 2),
            "temp": None, "moist": None, "ec": None, "sal": None,
            "target_n_kg_ha": n,
            "target_p2o5_kg_ha": p2o5,
            "target_k2o_kg_ha": k2o,
            "app1_formulation": app1,
            "app2_formulation": app2,
            "is_synthetic": True,
        }

        # Lab view — the full LS-0040 panel.
        rows.append(_row(
            base, "lab",
            ph=ph, organic_matter_pct=om,
            total_nitrogen_pct=round(total_nitrogen_from_om(om), 4),
            phosphorus_ppm=phosphorus, potassium_ppm=potassium,
        ))

        # STK view — coarsened to kit granularity. OM comes from the nitrogen
        # colour band, which the kit documents as an organic-matter proxy.
        stk_om = _coarsen(rng, om, _STK_OM_BANDS)
        rows.append(_row(
            base, "stk",
            ph=_nearest(ph, _STK_PH_INDICATORS),
            organic_matter_pct=stk_om,
            total_nitrogen_pct=round(total_nitrogen_from_om(stk_om), 4),
            phosphorus_ppm=_coarsen(rng, phosphorus, _STK_P_BANDS),
            potassium_ppm=_coarsen(rng, potassium, _STK_K_BANDS),
        ))

        # IoT view — no organic-matter equivalent exists on a probe.
        def noisy(value: float) -> float:
            return round(value * rng.uniform(1 - _IOT_SENSOR_NOISE, 1 + _IOT_SENSOR_NOISE), 2)

        rows.append(_row(
            base, "iot",
            ph=round(ph + rng.uniform(-0.15, 0.15), 2),
            organic_matter_pct=None, total_nitrogen_pct=None,
            phosphorus_ppm=noisy(phosphorus), potassium_ppm=noisy(potassium),
            temp=round(rng.uniform(24.0, 34.0), 2),
            moist=round(rng.uniform(20.0, 70.0), 2),
            ec=round(rng.uniform(100.0, 1500.0), 2),
            sal=round(rng.uniform(50.0, 800.0), 2),
        ))

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    frame = generate()
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(frame)} rows ({len(frame) // 3} records x 3 views) to {OUTPUT_CSV}")

"""SURI fertilizer prediction service (spec section 11).

Run: uvicorn main:app --port 8001

Port 8001, not 8000: suri/features/flows/iot/IoTFlow.tsx already hard-codes 8000
for a different, unrelated crop-capability contract.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from data.mode_of_application import application_schedule_for_crop
from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, nutrient_requirements, resolve_age_band,
    resolve_cultivation_method, resolve_variety, target_unit_for_crop,
)
from ml.agronomy import total_nitrogen_from_om
from ml.preprocessing import FEATURE_COLUMNS
from ml.recipes import plan_applications
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)
from ml.train import ARTIFACT_ROOT

logger = logging.getLogger("suri-ml")
logging.basicConfig(level=logging.INFO)

SQM_PER_HECTARE = 10_000

app = FastAPI(title="SURI Fertilizer ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SoilTestRequest(BaseModel):
    crop: str
    variety: str | None = None
    cultivation_method: str | None = None
    age_band: str | None = None
    data_source: Literal["lab", "stk", "iot"]
    ph: float = Field(ge=0, le=14)
    organic_matter_pct: float | None = Field(default=None, ge=0, le=100)
    total_nitrogen_pct: float | None = Field(default=None, ge=0)
    phosphorus_ppm: float | None = Field(default=None, ge=0)
    potassium_cmol_kg: float | None = Field(default=None, ge=0)
    temp: float | None = None
    moist: float | None = Field(default=None, ge=0, le=100)
    ec: float | None = Field(default=None, ge=0)
    sal: float | None = Field(default=None, ge=0)
    farm_area_sqm: float | None = Field(default=None, gt=0)
    laboratory_code: str | None = None

    @field_validator("crop")
    @classmethod
    def crop_must_be_known(cls, value: str) -> str:
        if value not in CROPS:
            raise ValueError(f"Unknown crop '{value}'. Valid crops: {', '.join(CROPS)}")
        return value

    def is_complete(self) -> bool:
        """A request nutrient_requirements() can answer exactly -- the same
        completeness rule data/parse_real_soil_database.py's
        build_training_row() uses to decide whether a row is trainable."""
        return (
            self.organic_matter_pct is not None
            and self.phosphorus_ppm is not None
            and self.potassium_cmol_kg is not None
        )


class FarmTotals(BaseModel):
    area_ha: float
    n_kg: float
    p2o5_kg: float
    k2o_kg: float


class PredictionResponse(BaseModel):
    target_n: float
    target_p2o5: float
    target_k2o: float
    target_unit: str
    app1_formulation: str
    app2_formulation: str
    application_schedule: str | None
    derived_remarks: dict[str, str]
    farm_totals: FarmTotals | None
    model_version: str
    source: str


def latest_version() -> str:
    if not os.path.isdir(ARTIFACT_ROOT):
        raise RuntimeError(
            f"No artifact directory at {ARTIFACT_ROOT}. Run: python -m ml.train --csv <path>"
        )
    versions = sorted(entry for entry in os.listdir(ARTIFACT_ROOT)
                      if os.path.isdir(os.path.join(ARTIFACT_ROOT, entry)))
    if not versions:
        raise RuntimeError(f"{ARTIFACT_ROOT} is empty. Run: python -m ml.train --csv <path>")
    return versions[-1]


def load_artifacts() -> dict:
    """Load at import time so a missing or corrupt model fails at boot rather
    than surfacing as a 500 on every request, or silently serving a stale model."""
    version = latest_version()
    directory = os.path.join(ARTIFACT_ROOT, version)
    return {
        "version": version,
        "preprocessor": joblib.load(os.path.join(directory, "preprocessor.joblib")),
        "regressor": joblib.load(os.path.join(directory, "regressor.joblib")),
    }


ARTIFACTS = load_artifacts()


def log_prediction(request: SoilTestRequest, response: PredictionResponse) -> None:
    from data.ingest_to_supabase import get_client
    get_client().table("ml_fertilizer_predictions").insert({
        "laboratory_code": request.laboratory_code,
        "data_source": request.data_source,
        "request_payload": request.model_dump(),
        "response_payload": response.model_dump(),
        "model_version": response.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def _build_feature_row(request: SoilTestRequest) -> pd.DataFrame:
    organic_matter = request.organic_matter_pct
    total_nitrogen = request.total_nitrogen_pct
    if total_nitrogen is None and organic_matter is not None:
        total_nitrogen = total_nitrogen_from_om(organic_matter)

    k_ppm = k_cmol_to_ppm(request.potassium_cmol_kg) if request.potassium_cmol_kg is not None else None

    row = {
        "ph": request.ph,
        "organic_matter_pct": organic_matter,
        "total_nitrogen_pct": total_nitrogen,
        "phosphorus_ppm": request.phosphorus_ppm,
        "potassium_cmol_kg": request.potassium_cmol_kg,
        "temp": request.temp, "moist": request.moist,
        "ec": request.ec, "sal": request.sal,
        "ph_remark": derive_ph_remark(request.ph),
        "om_remark": derive_om_remark(organic_matter),
        "p_remark": derive_p_remark(request.phosphorus_ppm),
        "k_remark": derive_k_remark(k_ppm),
        "crop": request.crop,
        "variety": resolve_variety(request.crop, request.variety),
        "cultivation_method": resolve_cultivation_method(request.crop, request.cultivation_method),
        "age_band": resolve_age_band(request.crop, request.age_band),
        "data_source": request.data_source,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.get("/health")
def health() -> dict:
    supabase_reachable = True
    try:
        from data.ingest_to_supabase import get_client
        get_client()
    except Exception:
        supabase_reachable = False

    return {
        "status": "ok",
        "model_version": ARTIFACTS["version"],
        "artifacts_loaded": True,
        "supabase_reachable": supabase_reachable,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SoilTestRequest) -> PredictionResponse:
    features = _build_feature_row(request)

    if request.is_complete():
        # Exact: the guide table already has the right answer for any
        # (crop, variety, cultivation_method, age_band) combination,
        # regardless of whether the model was ever trained on it. A
        # crop-inapplicable variety/cultivation_method value degrades
        # gracefully inside the resolvers (falls back to "None" or a
        # default), but an age_band string that isn't one of that crop's
        # real guide bands (e.g. a typo) has no fallback and raises KeyError
        # -- caught here and turned into the same clean 422 every other
        # invalid-input case in this API already gets, not an unhandled 500.
        try:
            n, p2o5, k2o = nutrient_requirements(
                request.crop, request.organic_matter_pct, request.phosphorus_ppm,
                request.potassium_cmol_kg, variety=request.variety,
                cultivation_method=request.cultivation_method, age_band=request.age_band,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=422,
                detail=f"Unrecognized age_band {request.age_band!r} for crop {request.crop!r}: {error}",
            )
        source = "table-lookup"
    else:
        try:
            X = ARTIFACTS["preprocessor"].transform(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        n, p2o5, k2o = ARTIFACTS["regressor"].predict(X)[0]
        n, p2o5, k2o = float(n), float(p2o5), float(k2o)
        source = "ml-ensemble"

    n, p2o5, k2o = round(n, 3), round(p2o5, 3), round(k2o, 3)
    target_unit = target_unit_for_crop(request.crop)
    app1, app2 = plan_applications(n, p2o5, k2o, target_unit)

    farm_totals = None
    if request.farm_area_sqm and target_unit == "kg_ha":
        hectares = request.farm_area_sqm / SQM_PER_HECTARE
        farm_totals = FarmTotals(
            area_ha=round(hectares, 4),
            n_kg=round(n * hectares, 3),
            p2o5_kg=round(p2o5 * hectares, 3),
            k2o_kg=round(k2o * hectares, 3),
        )

    response = PredictionResponse(
        target_n=n,
        target_p2o5=p2o5,
        target_k2o=k2o,
        target_unit=target_unit,
        app1_formulation=app1,
        app2_formulation=app2,
        application_schedule=application_schedule_for_crop(request.crop),
        derived_remarks={
            "ph_remark": features.at[0, "ph_remark"],
            "om_remark": features.at[0, "om_remark"],
            "p_remark": features.at[0, "p_remark"],
            "k_remark": features.at[0, "k_remark"],
        },
        farm_totals=farm_totals,
        model_version=ARTIFACTS["version"],
        source=source,
    )

    # An audit-log outage must never deny a farmer their recommendation.
    try:
        log_prediction(request, response)
    except Exception as error:
        logger.warning("Prediction logging failed (returning result anyway): %s", error)

    return response

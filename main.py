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

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from data.mode_of_application import application_schedule_for_crop
from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, max_total_need, nutrient_requirements, resolve_age_band,
    resolve_cultivation_method, resolve_variety, target_unit_for_crop,
)
from ml.agronomy import total_nitrogen_from_om
from ml.confidence import (
    DISAGREEMENT_DOWNGRADE_THRESHOLD, TIER_PCT_CEILING,
    confidence_from_metrics, downgrade_tier,
)
from ml.preprocessing import FEATURE_COLUMNS
from ml.recipes import plan_applications
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)
from ml import model_state, registry
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

# Unset in local dev / CI (no ML_API_KEY in the environment) means /predict
# stays open, same as before this check existed -- tests and localhost
# usage need no header. Set it in production so the endpoint isn't public
# once it's reachable from outside the instance.
ML_API_KEY = os.environ.get("ML_API_KEY")
if not ML_API_KEY:
    logger.warning("ML_API_KEY is not set -- /predict is unauthenticated.")


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    if ML_API_KEY and x_api_key != ML_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class SoilTestRequest(BaseModel):
    crop: str
    variety: str | None = None
    cultivation_method: str | None = None
    age_band: str | None = None
    data_source: Literal["lab", "stk", "iot"]
    ph: float = Field(ge=0, le=14)
    organic_matter_pct: float | None = Field(default=None, ge=0, le=100)
    total_nitrogen_pct: float | None = Field(default=None, ge=0, le=10)
    phosphorus_ppm: float | None = Field(default=None, ge=0, le=500)
    potassium_cmol_kg: float | None = Field(default=None, ge=0, le=6)
    temp: float | None = Field(default=None, ge=-10, le=60)
    moist: float | None = Field(default=None, ge=0, le=100)
    ec: float | None = Field(default=None, ge=0, le=20)
    sal: float | None = Field(default=None, ge=0, le=20)
    farm_area_sqm: float | None = Field(default=None, gt=0, le=2_000_000)
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


class Confidence(BaseModel):
    tier: Literal["high", "fair", "limited"]
    pct: int


class PredictionResponse(BaseModel):
    target_n: float
    target_p2o5: float
    target_k2o: float
    max_total_need: float
    target_unit: str
    app1_formulation: str
    app2_formulation: str
    application_schedule: str | None
    derived_remarks: dict[str, str]
    farm_totals: FarmTotals | None
    model_version: str
    source: str
    confidence: Confidence | None


def boot_version() -> str:
    """The registry is the authority on which version is live; the newest
    directory on disk is the fallback so a Supabase outage can't stop the
    service from booting. A registry version that isn't on disk (e.g. this
    instance was rebuilt) also falls back rather than crashing."""
    on_disk = model_state.available_versions(ARTIFACT_ROOT)
    if not on_disk:
        raise RuntimeError(
            f"No artifact directory at {ARTIFACT_ROOT}. Run: python -m ml.train --csv <path>"
        )

    promoted = registry.current_version()
    if promoted and promoted in on_disk:
        return promoted
    if promoted:
        logger.warning(
            "Registry says %s is current but it is not on disk; falling back to %s",
            promoted, on_disk[-1],
        )
    return on_disk[-1]


model_state.set_bundle(model_state.load_bundle(boot_version(), ARTIFACT_ROOT))


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


def _log_prediction_safe(request: SoilTestRequest, response: PredictionResponse) -> None:
    # An audit-log outage must never deny a farmer their recommendation --
    # scheduled as a background task (runs after the response is already
    # sent) so it no longer blocks the request on Supabase's network
    # latency, measured at ~3.3s per call vs. ~0.23s for the actual model
    # inference. See docs/superpowers/specs/2026-08-11-report-generation-
    # speed-design.md Finding 6 in the sibling `suri` repo.
    try:
        log_prediction(request, response)
    except Exception as error:
        logger.warning("Prediction logging failed (returning result anyway): %s", error)


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


_TARGET_KEYS = ("target_n", "target_p2o5", "target_k2o")


def _compute_confidence(bundle: dict, crop: str, X) -> Confidence | None:
    """Tier 1: this crop's historical held-out accuracy for the worst
    (lowest-confidence) of the three targets, since a report shows all
    three together and the weakest one is what should drive trust. Tier 2:
    if the two sub-models disagree a lot on THIS specific input, downgrade
    one level regardless of the crop's historical track record -- see
    docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md."""
    crop_metrics = bundle["metadata"].get("metrics", {}).get("by_crop", {}).get(crop)
    if not crop_metrics:
        return None

    tiers_and_pcts = [
        confidence_from_metrics(crop_metrics[key]["mae"], crop_metrics[key]["mean_actual"])
        for key in _TARGET_KEYS
        if key in crop_metrics
    ]
    if not tiers_and_pcts:
        return None

    tier_rank = {"high": 2, "fair": 1, "limited": 0}
    worst_tier, worst_pct = min(tiers_and_pcts, key=lambda pair: tier_rank[pair[0]])

    max_disagreement = 0.0
    for voter in bundle["regressor"].models_:
        xgb_log_pred = voter.named_estimators_["xgb"].predict(X)[0]
        krr_log_pred = voter.named_estimators_["krr"].predict(X)[0]
        blended_log_pred = voter.predict(X)[0]
        disagreement = abs(xgb_log_pred - krr_log_pred) / max(abs(blended_log_pred), 1e-6)
        max_disagreement = max(max_disagreement, disagreement)

    if max_disagreement > DISAGREEMENT_DOWNGRADE_THRESHOLD:
        worst_tier = downgrade_tier(worst_tier)
        worst_pct = min(worst_pct, TIER_PCT_CEILING[worst_tier])

    return Confidence(tier=worst_tier, pct=worst_pct)


@app.get("/health")
def health() -> dict:
    supabase_reachable = True
    try:
        from data.ingest_to_supabase import get_client
        get_client()
    except Exception:
        supabase_reachable = False

    bundle = model_state.snapshot()
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "artifacts_loaded": True,
        "supabase_reachable": supabase_reachable,
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(request: SoilTestRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    # One snapshot for the whole request: promotion rebinds the module's
    # bundle reference, and reading it more than once could otherwise pair an
    # old regressor's numbers with a new version string in the same response.
    bundle = model_state.snapshot()
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
        confidence = None
    else:
        try:
            X = bundle["preprocessor"].transform(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        n, p2o5, k2o = bundle["regressor"].predict(X)[0]
        n, p2o5, k2o = float(n), float(p2o5), float(k2o)
        source = "ml-ensemble"
        confidence = _compute_confidence(bundle, request.crop, X)

    n, p2o5, k2o = round(n, 3), round(p2o5, 3), round(k2o, 3)
    # A fixed property of the crop config, not the soil reading -- computed
    # the same way regardless of which branch above ran. In the
    # is_complete() branch a bad age_band would already have raised via
    # nutrient_requirements() above; this second try/except exists for the
    # ml-ensemble branch, which never calls nutrient_requirements() and so
    # never gets that check for free.
    try:
        crop_max_need = max_total_need(
            request.crop, variety=request.variety,
            cultivation_method=request.cultivation_method, age_band=request.age_band,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Unrecognized age_band {request.age_band!r} for crop {request.crop!r}: {error}",
        )
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
        max_total_need=crop_max_need,
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
        model_version=bundle["version"],
        source=source,
        confidence=confidence,
    )

    background_tasks.add_task(_log_prediction_safe, request, response)

    return response

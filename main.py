"""SURI fertilizer prediction service (spec section 11).

Run: uvicorn main:app --port 8001

Port 8001, not 8000: suri/features/flows/iot/IoTFlow.tsx already hard-codes 8000
for a different, unrelated crop-capability contract.
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from data.ingest_to_supabase import TABLE as TRAINING_TABLE, ingest_records
from data.mode_of_application import application_schedule_for_crop
from data import nutrient_recommendation_guide as guide
from data.nutrient_recommendation_guide import (
    k_cmol_to_ppm, max_total_need, nutrient_requirements, resolve_age_band,
    resolve_crop, resolve_cultivation_method, resolve_variety, target_unit_for_crop,
)
from data.training_upload import (
    TEMPLATE_COLUMNS, UploadFormatError, build_training_row, parse_upload, validate_rows,
)
from ml.agronomy import total_nitrogen_from_om
from ml.confidence import (
    DISAGREEMENT_DOWNGRADE_THRESHOLD, TIER_PCT_CEILING,
    apply_sample_size_floor, confidence_from_metrics, downgrade_tier,
)
from ml.preprocessing import FEATURE_COLUMNS
from ml.recipes import plan_applications
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)
from ml import artifacts_store, model_state, registry, retrain_job
from ml.train import ARTIFACT_ROOT

logger = logging.getLogger("suri-ml")
logging.basicConfig(level=logging.INFO)

SQM_PER_HECTARE = 10_000

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Deliberately a server-startup step rather than an import-time one: only
    # a process that is actually about to serve traffic should claim a version
    # in the shared registry. Importing main.py (tests, a REPL, a one-off
    # script) must never rewrite which model production considers live.
    _reconcile_boot_registry()
    _reload_guide_at_startup()
    yield


def _reload_guide_at_startup() -> None:
    """Loads crops/crop_variants/crop_nutrient_targets from Supabase into
    guide.GUIDE_ROWS/CROPS/etc. at real process startup -- this module no
    longer loads that data at import time (see
    data.nutrient_recommendation_guide.reload_guide's docstring for why).
    Never allowed to block startup: same posture as
    _reconcile_boot_registry -- a Supabase outage (or, for a `with
    TestClient(...)` context in tests, dummy/unreachable credentials)
    degrades to whatever guide state is already loaded (e.g. a test
    fixture's tests/conftest.py bootstrap) rather than crashing the
    process."""
    try:
        guide.reload_guide()
    except Exception as error:
        logger.warning("Could not load crop reference data at startup: %s", error)


app = FastAPI(title="SURI Fertilizer ML Service", version="1.0.0", lifespan=lifespan)
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
        if value not in guide.CROPS:
            raise ValueError(f"Unknown crop '{value}'. Valid crops: {', '.join(guide.CROPS)}")
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


class CommitRequest(BaseModel):
    rows: list[dict]


class PromoteRequest(BaseModel):
    version: str
    promoted_by: str | None = None


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


def boot_version() -> str | None:
    """The registry is the authority on which version is live; the newest
    directory on disk is the fallback so a Supabase outage can't stop the
    service from booting. A registry version that isn't on disk (e.g. this
    instance was rebuilt) also falls back rather than crashing.

    None means genuinely nothing exists yet -- a brand-new instance, or
    every version deliberately cleared to retrain from scratch. The service
    still starts in that case (see the module-level call site below); only
    /predict actually needs a model loaded, and everything an admin needs to
    produce the first one (upload data, retrain, promote) doesn't."""
    on_disk = model_state.available_versions(ARTIFACT_ROOT)
    if not on_disk:
        return None

    promoted = registry.current_version()
    if promoted and promoted in on_disk:
        return promoted
    if promoted:
        logger.warning(
            "Registry says %s is current but it is not on disk; falling back to %s",
            promoted, on_disk[-1],
        )
    return on_disk[-1]


_boot_version = boot_version()
if _boot_version is not None:
    model_state.set_bundle(model_state.load_bundle(_boot_version, ARTIFACT_ROOT))
else:
    logger.warning(
        "No trained model artifacts found -- starting with no model loaded. "
        "/predict will refuse requests (503) until the first retrain completes."
    )


def _reconcile_boot_registry() -> None:
    """Record whichever version this process actually booted, if the registry
    doesn't already say so.

    Until the retrain pipeline existed, an artifact directory could only reach
    disk by being git-committed, so "newest on disk" and "the model an admin
    approved" were the same thing and the fallback needed no record. Now a
    retrain writes directories nobody approved. If the registry has no current
    row -- it is brand new, or an outage happened during a promotion -- every
    restart would otherwise repeat the same blind newest-on-disk guess, so a
    reviewed-and-rejected model would silently go live on the next reboot.
    Writing down what was booted turns that one-time guess into an explicit
    row that all later boots honour.

    Never allowed to block startup: a registry outage degrades to the previous
    behaviour rather than taking the service down, same posture as the rest of
    this module.
    """
    try:
        bundle = model_state.snapshot()
        version = bundle["version"]
        if registry.current_version() == version:
            return

        metadata = bundle["metadata"]
        registry.record_version(
            version=version,
            trained_at=metadata["trained_at"],
            row_count=metadata["n_rows"],
            metrics=metadata.get("metrics", {}),
        )
        registry.set_current(version=version, promoted_by="boot-fallback")
        logger.info("Recorded booted model version %s as current.", version)
    except Exception as error:
        logger.warning("Could not record the booted model version: %s", error)


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


def _existing_laboratory_codes() -> set[str]:
    """Every laboratory_code already in the corpus, so an upload can flag
    duplicates before writing anything. Read in pages for the same reason
    ml/train.py's read is paged -- PostgREST caps a response at 1000 rows."""
    from data.ingest_to_supabase import get_client
    from ml.train import SUPABASE_PAGE_SIZE

    client = get_client()
    codes: set[str] = set()
    start = 0
    while True:
        page = client.table(TRAINING_TABLE).select("laboratory_code").range(
            start, start + SUPABASE_PAGE_SIZE - 1
        ).execute().data or []
        codes.update(row["laboratory_code"] for row in page)
        if len(page) < SUPABASE_PAGE_SIZE:
            return codes
        start += SUPABASE_PAGE_SIZE


def _commit_rows(rows: list[dict]) -> int:
    return ingest_records(rows)


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
        # Canonical name only: the encoder's vocabulary is CANONICAL_CROPS, so
        # an alias would trip handle_unknown="error".
        "crop": resolve_crop(request.crop),
        "variety": resolve_variety(request.crop, request.variety),
        "cultivation_method": resolve_cultivation_method(request.crop, request.cultivation_method),
        "age_band": resolve_age_band(request.crop, request.age_band),
        "data_source": request.data_source,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


_TARGET_KEYS = ("target_n", "target_p2o5", "target_k2o")


def trained_crops(bundle: dict) -> set[str]:
    """Crops the loaded model actually has held-out metrics for. Metrics are
    computed per crop over the test split, so presence here means the model
    both trained on and was evaluated for that crop."""
    return set(bundle["metadata"].get("metrics", {}).get("by_crop", {}))


def _historical_confidence(bundle: dict, crop: str) -> Confidence | None:
    """Tier 1: this crop's historical held-out accuracy for the worst
    (lowest-confidence) of the three targets, since a report shows all three
    together and the weakest one is what should drive trust. Tier 3: a
    target's tier is floored to Limited when its held-out sample count is
    too small to trust the error at all, regardless of how good it looks --
    see ml.confidence's module docstring for the Lettuce/Garlic cases that
    motivated this. No Tier 2 (ensemble-disagreement) check here -- that
    needs a specific prediction's X, so this is what GET /model/confidence
    uses to show a crop's confidence independent of any single request. See
    docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md."""
    crop_metrics = bundle["metadata"].get("metrics", {}).get("by_crop", {}).get(crop)
    if not crop_metrics:
        return None

    tiers_and_pcts = [
        apply_sample_size_floor(
            *confidence_from_metrics(crop_metrics[key]["mae"], crop_metrics[key]["mean_actual"]),
            crop_metrics[key].get("n"),
        )
        for key in _TARGET_KEYS
        if key in crop_metrics
    ]
    if not tiers_and_pcts:
        return None

    tier_rank = {"high": 2, "fair": 1, "limited": 0}
    worst_tier, worst_pct = min(tiers_and_pcts, key=lambda pair: tier_rank[pair[0]])
    return Confidence(tier=worst_tier, pct=worst_pct)


def _compute_confidence(bundle: dict, crop: str, X) -> Confidence | None:
    """Tier 1 (see _historical_confidence), then Tier 2: if the two
    sub-models disagree a lot on THIS specific input, downgrade one level
    regardless of the crop's historical track record -- see
    docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md."""
    historical = _historical_confidence(bundle, crop)
    if historical is None:
        return None
    worst_tier, worst_pct = historical.tier, historical.pct

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

    try:
        model_version = model_state.snapshot()["version"]
        artifacts_loaded = True
    except RuntimeError:
        model_version = None
        artifacts_loaded = False

    return {
        "status": "ok",
        "model_version": model_version,
        "artifacts_loaded": artifacts_loaded,
        "supabase_reachable": supabase_reachable,
        # Promotion hot-swaps the model in *this* process's memory, so the
        # service is only correct with a single uvicorn worker. Reporting the
        # PID makes a violation observable from outside: repeated /health
        # calls that come back with different pids mean more than one worker
        # is answering, and promotions are reaching only one of them.
        "pid": os.getpid(),
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(request: SoilTestRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    # One snapshot for the whole request: promotion rebinds the module's
    # bundle reference, and reading it more than once could otherwise pair an
    # old regressor's numbers with a new version string in the same response.
    try:
        bundle = model_state.snapshot()
    except RuntimeError:
        # Applies even to the table-lookup path below, which doesn't touch
        # the regressor at all -- model_version has nowhere honest to come
        # from for the response either way, and simplicity here beats
        # partially serving one path before any model exists.
        raise HTTPException(
            status_code=503,
            detail="No model has been trained yet — complete a first retrain before requesting predictions.",
        )
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
        canonical = resolve_crop(request.crop)
        known = trained_crops(bundle)
        if known and canonical not in known:
            # The guide table can still answer this crop exactly -- it just
            # needs the full soil panel. Say so, rather than returning a
            # number derived from no crop-specific training signal.
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No training data exists yet for {request.crop!r}, so an estimate from "
                    f"an incomplete soil reading is not available. Supply "
                    f"organic_matter_pct, phosphorus_ppm, and potassium_cmol_kg for an exact "
                    f"result from the official guide, or upload lab data for this crop and "
                    f"retrain."
                ),
            )
        try:
            X = bundle["preprocessor"].transform(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        n, p2o5, k2o = bundle["regressor"].predict(X)[0]
        n, p2o5, k2o = float(n), float(p2o5), float(k2o)
        source = "ml-ensemble"
        confidence = _compute_confidence(bundle, resolve_crop(request.crop), X)

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


@app.post("/training-data/validate", dependencies=[Depends(require_api_key)])
async def validate_training_upload(file: UploadFile = File(...)) -> dict:
    """Dry run: parses, validates, and shows what every accepted row would
    contribute. Writes nothing."""
    content = await file.read()
    try:
        raw_rows = parse_upload(content, file.filename or "upload.csv")
    except UploadFormatError as error:
        raise HTTPException(status_code=422, detail=str(error))

    # This handler has to be async (UploadFile.read() is a coroutine), which
    # means everything else in it runs *on the event loop* unless pushed off
    # it: _existing_laboratory_codes() makes several sequential blocking
    # Supabase calls and validate_rows() runs full agronomy derivation for
    # every row of the file. Left inline, a large upload would stall /predict
    # dispatch for the whole duration -- and "/predict is never paused" is a
    # hard constraint of this pipeline. (/training-data/commit is a plain def,
    # so FastAPI already threadpools it.)
    codes = await run_in_threadpool(_existing_laboratory_codes)
    result = await run_in_threadpool(validate_rows, raw_rows, codes)
    return {
        "template_columns": TEMPLATE_COLUMNS,
        "accepted": result.accepted,
        "rejected": result.rejected,
    }


@app.post("/training-data/commit", dependencies=[Depends(require_api_key)])
def commit_training_upload(request: CommitRequest) -> dict:
    """Re-validates from the raw rows rather than trusting anything derived
    client-side, then upserts only what passes."""
    if not request.rows:
        raise HTTPException(status_code=422, detail="No rows supplied.")

    result = validate_rows(request.rows, _existing_laboratory_codes())
    committed = _commit_rows([build_training_row(entry["raw"]) for entry in result.accepted])
    return {"committed": committed, "rejected": result.rejected}


@app.post("/retrain", dependencies=[Depends(require_api_key)])
def start_retrain() -> dict:
    try:
        return {"job_id": retrain_job.start_job()}
    except retrain_job.RetrainInProgress as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/retrain/{job_id}", dependencies=[Depends(require_api_key)])
def retrain_status(job_id: str) -> dict:
    job = retrain_job.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No retrain job {job_id!r}.")
    return job


@app.get("/model/versions", dependencies=[Depends(require_api_key)])
def model_versions() -> dict:
    try:
        current = model_state.snapshot()["version"]
    except RuntimeError:
        current = None
    return {
        "current": current,
        "versions": registry.list_versions(),
        "on_disk": model_state.available_versions(ARTIFACT_ROOT),
    }


@app.get("/model/confidence", dependencies=[Depends(require_api_key)])
def model_confidence() -> dict:
    """Always-available per-crop confidence for the live model -- unlike
    /predict's confidence field, this isn't tied to a specific prediction or
    to a retrain job that just finished, so a dashboard can show it at rest.
    No model loaded yet (a fresh instance, or every version deliberately
    cleared to retrain from scratch) is a real, expected state here, not an
    error -- reported as an empty by_crop rather than a 500."""
    try:
        bundle = model_state.snapshot()
    except RuntimeError:
        return {"version": None, "by_crop": {}}
    by_crop_metrics = bundle["metadata"].get("metrics", {}).get("by_crop", {})
    coverage_metrics = bundle["metadata"].get("metrics", {}).get("crop_coverage", {})

    by_crop = {}
    for crop in sorted(trained_crops(bundle)):
        confidence = _historical_confidence(bundle, crop)
        if confidence is None:
            continue
        by_crop[crop] = {
            "tier": confidence.tier,
            "pct": confidence.pct,
            "targets": {
                key: by_crop_metrics[crop][key]
                for key in _TARGET_KEYS
                if key in by_crop_metrics[crop]
            },
            # None for an artifact trained before crop_coverage existed --
            # same nullable-until-retrained pattern as by_crop's own "n".
            "coverage": coverage_metrics.get(crop),
        }

    return {"version": bundle["version"], "by_crop": by_crop}


@app.post("/model/promote", dependencies=[Depends(require_api_key)])
def promote_model(request: PromoteRequest) -> dict:
    # Checked before the string is ever joined onto ARTIFACT_ROOT, handed to
    # shutil.rmtree (inside download_version) or unpickled by joblib.load.
    # The endpoint is already behind require_api_key and the dashboard's
    # session check, so this is defence in depth rather than a live hole.
    if not model_state.is_version_name(request.version):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid version {request.version!r}; expected the YYYY-MM-DD-HHMMSS form.",
        )

    try:
        previous = model_state.snapshot()["version"]
    except RuntimeError:
        previous = None  # first-ever promotion -- nothing was live before this

    if request.version not in model_state.available_versions(ARTIFACT_ROOT):
        # Not on this instance -- try the off-instance backup before giving up,
        # which is what makes rollback survive an EC2 rebuild.
        if not artifacts_store.download_version(request.version, ARTIFACT_ROOT):
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {request.version!r} is not on disk and could not be restored.",
            )

    try:
        bundle = model_state.load_bundle(request.version, ARTIFACT_ROOT)
        model_state.verify_bundle(bundle)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not load {request.version!r}: {error}")

    # Swap first, then record: if the registry write fails, the service is
    # already serving the version the admin approved, and the next boot's
    # fallback picks a valid version rather than a phantom one.
    model_state.set_bundle(bundle)
    registry_recorded = True
    try:
        # Upsert before set_current, always. set_current is an
        # UPDATE ... WHERE version = $1: for a version nothing ever
        # registered -- the original bootstrap artifact pinned via .gitignore
        # is exactly that -- it matches zero rows, and PostgREST answers 200
        # with an empty body. The promotion would then live only in memory,
        # and the next restart's fallback would silently undo it, which is
        # precisely the roll-back-to-the-known-good-model case this pipeline
        # exists for. record_version() only writes the columns named here, so
        # re-upserting an already-registered version leaves promoted_at and
        # promoted_by (i.e. the promotion history) untouched.
        metadata = bundle["metadata"]
        registry.record_version(
            version=request.version,
            trained_at=metadata["trained_at"],
            row_count=metadata["n_rows"],
            metrics=metadata.get("metrics", {}),
        )
        registry.set_current(request.version, promoted_by=request.promoted_by)
    except Exception as error:
        # The in-memory swap already happened and is real, so this is a 200 --
        # but a silent log line would leave the admin and /model/versions with
        # no idea the promotion won't survive a restart. Say so in the body.
        registry_recorded = False
        logger.warning("Promoted %s in memory but could not record it: %s",
                       request.version, error)

    return {
        "current": request.version,
        "previous": previous,
        "registry_recorded": registry_recorded,
    }


@app.post("/admin/reload-guide", dependencies=[Depends(require_api_key)])
def reload_guide_endpoint() -> dict:
    """Re-reads crops/crop_variants/crop_nutrient_targets from Supabase and
    reassigns guide.GUIDE_ROWS/CROPS/etc. in place -- called by suri's
    crop-reference CMS after every create/bulk-upload so a newly added crop
    is immediately recognized by /predict and training-data upload, without
    a process restart. Single uvicorn worker (deploy/suri-ml.service), so
    there is no cross-worker inconsistency window."""
    try:
        guide.reload_guide()
    except Exception:
        logger.exception("Failed to reload guide data")
        raise HTTPException(status_code=502, detail="Reload failed") from None

    return {
        "success": True,
        "crop_count": len(guide.CROPS),
        "canonical_crop_count": len(guide.CANONICAL_CROPS),
    }


@app.delete("/model/versions", dependencies=[Depends(require_api_key)])
def delete_all_model_versions() -> dict:
    """Full reset: every version, everywhere -- registry, Storage, local
    disk, and the in-memory bundle /predict is currently serving. After
    this, /predict refuses requests (503) until the next retrain, the same
    boot-with-nothing-loaded state main.py already handles gracefully.

    Must be registered before DELETE /model/{version} below -- FastAPI
    matches routes in definition order, and "versions" is a syntactically
    valid value for that route's {version} path parameter. Defined after it,
    this would never be reached; every bulk-delete request would instead hit
    delete_model_version(version="versions")."""
    deleted_count = len(model_state.available_versions(ARTIFACT_ROOT))

    if os.path.isdir(ARTIFACT_ROOT):
        for entry in os.listdir(ARTIFACT_ROOT):
            shutil.rmtree(os.path.join(ARTIFACT_ROOT, entry), ignore_errors=True)

    artifacts_store.delete_all_versions()
    registry.delete_all()
    model_state.clear_bundle()

    return {"deleted_versions": deleted_count}


@app.delete("/model/{version}", dependencies=[Depends(require_api_key)])
def delete_model_version(version: str) -> dict:
    """Deletes one version everywhere it lives: the registry row, its
    Storage backup, and its local-disk directory. Never the live version --
    that would leave /predict serving a bundle whose files just vanished out
    from under it, functional only until the next restart notices."""
    if not model_state.is_version_name(version):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid version {version!r}; expected the YYYY-MM-DD-HHMMSS form.",
        )

    try:
        if model_state.snapshot()["version"] == version:
            raise HTTPException(
                status_code=422,
                detail=f"{version!r} is the live version and cannot be deleted. "
                        "Promote a different version first.",
            )
    except RuntimeError:
        pass  # no live model at all -- nothing this delete could break

    exists_on_disk = version in model_state.available_versions(ARTIFACT_ROOT)
    exists_in_registry = registry.get_version(version) is not None
    if not exists_on_disk and not exists_in_registry:
        raise HTTPException(status_code=404, detail=f"Version {version!r} does not exist.")

    registry.delete_version(version)
    artifacts_store.delete_version(version)
    shutil.rmtree(os.path.join(ARTIFACT_ROOT, version), ignore_errors=True)

    return {"deleted": version}


def _delete_all_training_data() -> int:
    from data.ingest_to_supabase import delete_all_records
    return delete_all_records()


def _list_training_data_page(page: int, page_size: int) -> tuple[list[dict], int]:
    from data.ingest_to_supabase import list_records_paginated
    return list_records_paginated(page, page_size)


@app.delete("/training-data", dependencies=[Depends(require_api_key)])
def delete_all_training_data() -> dict:
    """Clears ml_soil_training_data entirely. Does not touch any trained
    model -- an already-trained version keeps working exactly as before,
    this only affects what the *next* retrain would learn from."""
    return {"deleted": _delete_all_training_data()}


@app.get("/training-data", dependencies=[Depends(require_api_key)])
def list_training_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> dict:
    rows, total = _list_training_data_page(page, page_size)
    return {"rows": rows, "total": total, "page": page, "page_size": page_size}

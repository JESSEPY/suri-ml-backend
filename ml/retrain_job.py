"""Runs a retrain off the request thread and exposes its progress by job id.

Training never touches the bundle /predict is serving: it reads rows from
Supabase and writes a brand-new artifact directory. The live model changes
only when an admin promotes the result, which is a separate endpoint.

Job state lives in a module-level dict, which is correct precisely because
the service runs a single uvicorn worker (see the deploy unit file). The
indirection seams (_train, _record_version, ...) exist so the lifecycle can
be tested without training a real model or reaching Supabase.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("suri-ml")

_JOBS: dict[str, dict] = {}
_ACTIVE: str | None = None
_lock = threading.Lock()

TARGETS = ("target_n", "target_p2o5", "target_k2o")


class RetrainInProgress(RuntimeError):
    """A retrain is already running. Two concurrent runs on a 1 GiB instance
    would compete for memory and produce two artifacts racing to be newest,
    for no benefit."""


def _load_training_frame():
    from ml.train import _load_frame
    return _load_frame(None)


def _train(frame) -> dict:
    from ml.train import train
    return train(frame)


def _record_version(**kwargs) -> None:
    from ml import registry
    registry.record_version(**kwargs)


def _upload_version(version: str) -> bool:
    from ml import artifacts_store
    from ml.train import ARTIFACT_ROOT
    return artifacts_store.upload_version(version, ARTIFACT_ROOT)


def _current_metrics() -> dict | None:
    from ml import model_state
    try:
        return model_state.snapshot()["metadata"].get("metrics")
    except Exception:
        return None


def build_comparison(new_metrics: dict, current_metrics: dict | None) -> dict:
    """Everything the review screen needs to answer "is this better?":
    per-target MAE against the mean baseline, and against the live model."""
    new_overall = (new_metrics or {}).get("overall", {})
    current_overall = (current_metrics or {}).get("overall", {})

    targets: dict[str, dict] = {}
    beats_baseline = True
    regressed = False

    for name, scores in new_overall.items():
        mae = scores.get("mae")
        dummy_mae = scores.get("dummy_mae")
        target_beats = mae is not None and dummy_mae is not None and mae < dummy_mae
        beats_baseline = beats_baseline and target_beats

        current_mae = current_overall.get(name, {}).get("mae")
        delta = None if current_mae is None else round(mae - current_mae, 4)
        improved = None if delta is None else delta < 0
        if improved is False:
            regressed = True

        targets[name] = {
            "mae": mae,
            "r2": scores.get("r2"),
            "dummy_mae": dummy_mae,
            "beats_baseline": target_beats,
            "current_mae": current_mae,
            "mae_delta": delta,
            "improved": improved,
        }

    return {
        "targets": targets,
        "beats_baseline": beats_baseline,
        "regressed": regressed,
        "compared_against_live_model": bool(current_overall),
    }


def _run(job_id: str) -> None:
    global _ACTIVE
    try:
        current_metrics = _current_metrics()
        frame = _load_training_frame()
        metadata = _train(frame)

        _record_version(
            version=metadata["version"],
            trained_at=metadata["trained_at"],
            row_count=metadata["n_rows"],
            metrics=metadata["metrics"],
        )
        backed_up = _upload_version(metadata["version"])

        with _lock:
            _JOBS[job_id].update({
                "status": "done",
                "version": metadata["version"],
                "metrics": metadata["metrics"],
                "comparison": build_comparison(metadata["metrics"], current_metrics),
                "backed_up": backed_up,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
    except Exception as error:
        logger.exception("Retrain job %s failed", job_id)
        with _lock:
            _JOBS[job_id].update({
                "status": "failed",
                "error": str(error),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
    finally:
        with _lock:
            _ACTIVE = None


def start_job() -> str:
    global _ACTIVE
    with _lock:
        if _ACTIVE is not None:
            raise RetrainInProgress(f"Retrain {_ACTIVE} is already running.")
        job_id = uuid.uuid4().hex
        _ACTIVE = job_id
        _JOBS[job_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    threading.Thread(target=_run, args=(job_id,), daemon=True, name=f"retrain-{job_id}").start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def active_job_id() -> str | None:
    with _lock:
        return _ACTIVE

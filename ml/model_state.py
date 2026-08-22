"""Holds the artifact bundle /predict is currently serving, swappable at runtime.

The bundle dict is treated as immutable: promoting rebinds the module-level
reference to a brand-new dict rather than mutating the live one, and readers
take exactly one snapshot per request. That combination is what makes a
promotion invisible to an in-flight request instead of visible halfway
through it.
"""

from __future__ import annotations

import json
import os
import re
import threading

import joblib
import pandas as pd

from ml.artifacts_store import ARTIFACT_FILES

_lock = threading.Lock()
_current: dict | None = None

# ml/train.py::train() stamps every artifact directory with
# datetime.strftime("%Y-%m-%d-%H%M%S"). Nothing else is allowed to invent a
# version string, so anything on disk that doesn't match this shape is not an
# artifact directory -- a staging leftover (`<version>.partial`), a scratch
# folder, whatever -- and must never be a promotion or boot-fallback target.
# \Z, not $: in Python `$` also matches just before a trailing newline, so
# "2026-08-06-055755\n" would sneak through and reach the filesystem.
VERSION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}\Z")


def is_version_name(name: str) -> bool:
    """True only for the artifact-directory naming convention. Callers use it
    to refuse an attacker- or typo-supplied version before it reaches
    os.path.join, shutil.rmtree, or joblib.load (unpickling)."""
    return bool(VERSION_PATTERN.match(name or ""))


def load_bundle(version: str, artifact_root: str) -> dict:
    directory = os.path.join(artifact_root, version)
    with open(os.path.join(directory, "metadata.json")) as handle:
        metadata = json.load(handle)
    return {
        "version": version,
        "preprocessor": joblib.load(os.path.join(directory, "preprocessor.joblib")),
        "regressor": joblib.load(os.path.join(directory, "regressor.joblib")),
        "metadata": metadata,
    }


def set_bundle(bundle: dict) -> None:
    global _current
    with _lock:
        _current = bundle


def clear_bundle() -> None:
    """Explicitly unloads the live model -- used when every version has just
    been deleted (DELETE /model/versions), so /predict correctly starts
    refusing requests (503) immediately rather than continuing to serve
    whatever was already in memory until the next restart happens to notice
    the files are gone."""
    global _current
    with _lock:
        _current = None


def snapshot() -> dict:
    with _lock:
        if _current is None:
            raise RuntimeError("No model artifacts loaded")
        return _current


def _is_complete_version_dir(artifact_root: str, entry: str) -> bool:
    directory = os.path.join(artifact_root, entry)
    if not os.path.isdir(directory):
        return False
    return all(os.path.isfile(os.path.join(directory, name)) for name in ARTIFACT_FILES)


def available_versions(artifact_root: str) -> list[str]:
    """Only *complete*, correctly-named artifact directories, sorted ascending.

    ml/train.py::train() creates its output directory as its first action and
    writes metadata.json as its last, so a run killed in between (deploy, OOM,
    reboot) leaves an empty directory behind. A bare "every subdirectory
    counts" rule would let main.py::boot_version()'s newest-on-disk fallback
    pick that carcass, and load_bundle() would then raise FileNotFoundError at
    import time -- uvicorn exits, systemd's Restart=always retries, crash loop.
    Requiring all three artifact files (and the version naming convention,
    which also excludes a `<version>.partial` staging directory left by a
    SIGKILLed restore) makes an incomplete directory simply invisible."""
    if not os.path.isdir(artifact_root):
        return []
    return sorted(
        entry for entry in os.listdir(artifact_root)
        if is_version_name(entry) and _is_complete_version_dir(artifact_root, entry)
    )


def _probe_frame(bundle: dict) -> pd.DataFrame:
    """One syntactically valid row, used only to prove a candidate bundle can
    actually transform input before it is allowed to serve traffic.

    The crop comes from the bundle's OWN by_crop metrics, not from CROPS[0].
    CROPS[0] is whatever sorts first in the current code (today "Alugbati"),
    which for any artifact predating a crop-list expansion is a category its
    encoder was never fit against -- so verify_bundle() would reject a
    perfectly serviceable older artifact and break the rollback path the
    promotion registry exists to provide. It is not the probe's job to test
    coverage of the newest crop; main.py::predict() already refuses crops
    absent from a bundle's by_crop metrics with a 422, so a crop that IS in
    those metrics is exactly the kind of row this artifact will really see.

    Sorted rather than "first key" so the probe is reproducible regardless of
    the order train.py happened to write metadata.json in. CROPS[0] remains
    the fallback for an artifact carrying no metrics at all -- the same
    "degrade gracefully when metadata is empty" stance as main.py's
    `known and canonical not in known` guard."""
    from data.nutrient_recommendation_guide import (
        CROPS, resolve_age_band, resolve_cultivation_method, resolve_variety,
    )
    from ml.preprocessing import FEATURE_COLUMNS
    from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

    by_crop = bundle.get("metadata", {}).get("metrics", {}).get("by_crop", {})
    crop = next(iter(sorted(by_crop)), None) or CROPS[0]
    row = {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_cmol_kg": 0.3,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": derive_ph_remark(6.5),
        "om_remark": derive_om_remark(2.5),
        "p_remark": derive_p_remark(15.0),
        "k_remark": derive_k_remark(117.3),
        "crop": crop,
        "variety": resolve_variety(crop, None),
        "cultivation_method": resolve_cultivation_method(crop, None),
        "age_band": resolve_age_band(crop, None),
        "data_source": "lab",
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def verify_bundle(bundle: dict) -> None:
    """Promotion gate. An artifact trained against a different feature vocabulary
    (e.g. before the supported-crop list changed) loads fine from joblib but
    raises the moment it sees a real row -- which would turn every /predict
    into a 422 if it were promoted. Catch that here, before the swap."""
    try:
        transformed = bundle["preprocessor"].transform(_probe_frame(bundle))
        bundle["regressor"].predict(transformed)
    except Exception as error:
        raise ValueError(
            f"Artifact {bundle.get('version')!r} cannot serve requests: {error}"
        ) from error

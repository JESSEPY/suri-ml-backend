"""Copy trained artifact directories to a private Supabase Storage bucket.

Before this existed, the only artifacts that survived an EC2 rebuild were the
ones someone remembered to git-commit (see .gitignore's single pinned
exception). Every trained version now gets an off-instance copy, so rollback
targets stay recoverable.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger("suri-ml")

BUCKET = "ml-model-artifacts"
ARTIFACT_FILES = ("preprocessor.joblib", "regressor.joblib", "metadata.json")


def _client(client):
    if client is not None:
        return client
    from data.ingest_to_supabase import get_client
    return get_client()


def upload_version(version: str, artifact_root: str, client=None) -> bool:
    """Returns False (never raises) when the backup fails -- the caller's
    artifact is already on local disk and fully usable without it."""
    directory = os.path.join(artifact_root, version)
    try:
        bucket = _client(client).storage.from_(BUCKET)
        for name in ARTIFACT_FILES:
            with open(os.path.join(directory, name), "rb") as handle:
                bucket.upload(
                    f"{version}/{name}",
                    handle.read(),
                    file_options={"upsert": "true", "content-type": "application/octet-stream"},
                )
        return True
    except Exception as error:
        logger.warning("Artifact backup for %s failed: %s", version, error)
        return False


def download_version(version: str, artifact_root: str, client=None) -> bool:
    """Restores files to disk. Downloads into a staging directory first and only
    creates/replaces the real version directory on full success, so a failed
    restore (missing version, network error, partial download, ...) never
    leaves behind an empty or incomplete directory. That matters because
    boot_version()'s "newest on disk" fallback (ml/model_state.py,
    main.py::boot_version) would otherwise pick a poisoned directory as the
    live version and crash the next boot looking for a metadata.json that was
    never written."""
    directory = os.path.join(artifact_root, version)
    staging = directory + ".partial"
    try:
        bucket = _client(client).storage.from_(BUCKET)
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        for name in ARTIFACT_FILES:
            payload = bucket.download(f"{version}/{name}")
            with open(os.path.join(staging, name), "wb") as handle:
                handle.write(payload)
        os.makedirs(artifact_root, exist_ok=True)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.rename(staging, directory)
        return True
    except Exception as error:
        logger.warning("Artifact restore for %s failed: %s", version, error)
        shutil.rmtree(staging, ignore_errors=True)
        return False


def delete_version(version: str, client=None) -> bool:
    """Returns False (never raises) on failure -- same posture as
    upload_version, so a Storage outage doesn't block the caller's own
    local-disk and registry cleanup for the same version."""
    try:
        bucket = _client(client).storage.from_(BUCKET)
        bucket.remove([f"{version}/{name}" for name in ARTIFACT_FILES])
        return True
    except Exception as error:
        logger.warning("Artifact deletion for %s failed: %s", version, error)
        return False


def delete_all_versions(client=None) -> bool:
    """Lists every version folder, then every file inside each, before
    removing them all in one call -- Storage has no recursive "delete this
    prefix" operation, only delete-by-exact-key."""
    try:
        bucket = _client(client).storage.from_(BUCKET)
        paths = [
            f"{folder['name']}/{file['name']}"
            for folder in bucket.list()
            for file in bucket.list(folder["name"])
        ]
        if paths:
            bucket.remove(paths)
        return True
    except Exception as error:
        logger.warning("Bulk artifact deletion failed: %s", error)
        return False

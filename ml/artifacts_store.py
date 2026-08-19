"""Copy trained artifact directories to a private Supabase Storage bucket.

Before this existed, the only artifacts that survived an EC2 rebuild were the
ones someone remembered to git-commit (see .gitignore's single pinned
exception). Every trained version now gets an off-instance copy, so rollback
targets stay recoverable.
"""

from __future__ import annotations

import logging
import os

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
    directory = os.path.join(artifact_root, version)
    try:
        bucket = _client(client).storage.from_(BUCKET)
        os.makedirs(directory, exist_ok=True)
        for name in ARTIFACT_FILES:
            payload = bucket.download(f"{version}/{name}")
            with open(os.path.join(directory, name), "wb") as handle:
                handle.write(payload)
        return True
    except Exception as error:
        logger.warning("Artifact restore for %s failed: %s", version, error)
        return False

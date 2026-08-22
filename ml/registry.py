"""Supabase-backed record of every trained model version and which one is live.

main.py previously inferred the live model from directory-name sort order.
That works only while "newest on disk" and "the one an admin approved" are the
same thing -- which stops being true the moment rollback exists. This table is
the authority; the on-disk directories are just storage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("suri-ml")

TABLE = "ml_model_versions"


def _client(client):
    if client is not None:
        return client
    from data.ingest_to_supabase import get_client
    return get_client()


def record_version(
    version: str, trained_at: str, row_count: int, metrics: dict, client=None
) -> None:
    """Register a freshly trained version. Never marks it current -- promotion
    is a separate, explicitly-approved action."""
    _client(client).table(TABLE).upsert({
        "version": version,
        "trained_at": trained_at,
        "row_count": row_count,
        "metrics": metrics,
        "is_current": False,
    }, on_conflict="version").execute()


def list_versions(client=None) -> list[dict]:
    try:
        return _client(client).table(TABLE).select("*").order(
            "trained_at", desc=True
        ).execute().data or []
    except Exception as error:
        logger.warning("Could not list model versions: %s", error)
        return []


def get_version(version: str, client=None) -> dict | None:
    try:
        rows = _client(client).table(TABLE).select("*").eq(
            "version", version
        ).limit(1).execute().data or []
    except Exception as error:
        logger.warning("Could not read model version %s: %s", version, error)
        return None
    return rows[0] if rows else None


def current_version(client=None) -> str | None:
    """None means "the registry could not answer" -- unreachable Supabase, an
    empty table, or nothing promoted yet. Callers fall back to the newest
    artifact on disk, so a registry outage degrades to the old behaviour
    instead of refusing to boot."""
    try:
        rows = _client(client).table(TABLE).select("version").eq(
            "is_current", True
        ).limit(1).execute().data or []
    except Exception as error:
        logger.warning("Could not read the current model version: %s", error)
        return None
    return rows[0]["version"] if rows else None


def set_current(version: str, promoted_by: str | None = None, client=None) -> None:
    """Clear-then-set, in that order: a partial unique index allows at most one
    is_current row, so setting before clearing would violate it. The brief
    window where no row is current is safe -- boot-time fallback covers it."""
    handle = _client(client)
    handle.table(TABLE).update({"is_current": False}).eq("is_current", True).execute()
    handle.table(TABLE).update({
        "is_current": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_by": promoted_by,
    }).eq("version", version).execute()


def delete_version(version: str, client=None) -> None:
    """Caller's job to refuse deleting whichever version is currently live --
    this just removes the row."""
    _client(client).table(TABLE).delete().eq("version", version).execute()


def delete_all(client=None) -> None:
    """delete() with no filter is refused by Supabase -- neq on a column no
    real version ever equals is the standard "delete everything" idiom."""
    _client(client).table(TABLE).delete().neq("version", "").execute()

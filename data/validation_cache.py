"""Short-lived server-side cache for validated-but-not-yet-committed upload
batches.

POST /training-data/validate can produce a response too large to round-trip
through the browser for a large file: with a few thousand accepted rows,
each carrying its full raw+derived data, the JSON response can run into the
multi-megabyte range -- and the Vercel proxy in front of this service caps
both request AND response bodies at 4.5MB. Storing the full ValidationResult
here, keyed by a token, means /validate only ever has to send the browser a
bounded preview plus counts, and /commit only ever has to receive a token
back -- the full row data never crosses the network twice.

In-memory, per-process, same pattern as ml/retrain_job.py's _JOBS -- safe
because main.py runs as exactly one uvicorn worker, deliberately (see
deploy/suri-ml.service's comment on why: the admin retraining pipeline
hot-swaps a model in this process's own memory, so more than one worker
would leave some of them serving a stale model with no way to tell which
answered a given request -- the same single-process assumption this cache
relies on).
"""

from __future__ import annotations

import time
import uuid

from data.training_upload import ValidationResult

# Long enough for an admin to review a validation report before committing;
# short enough that an abandoned upload (validated, tab closed, never
# committed) doesn't sit in memory indefinitely.
_BATCH_TTL_SECONDS = 15 * 60

_BATCHES: dict[str, tuple[float, ValidationResult]] = {}


def store(result: ValidationResult) -> str:
    """Caches a validation result and returns its token."""
    _evict_expired()
    token = uuid.uuid4().hex
    _BATCHES[token] = (time.monotonic() + _BATCH_TTL_SECONDS, result)
    return token


def pop(token: str) -> ValidationResult | None:
    """Returns and removes the cached result, or None if the token is
    unknown or expired. One-shot by design: a batch can be committed at
    most once, so re-using a token (a double-click, a retried request)
    must fail loudly instead of silently recommitting or double-counting
    rows."""
    _evict_expired()
    entry = _BATCHES.pop(token, None)
    if entry is None:
        return None
    expires_at, result = entry
    if time.monotonic() > expires_at:
        return None
    return result


def _evict_expired() -> None:
    now = time.monotonic()
    expired = [token for token, (expires_at, _) in _BATCHES.items() if now > expires_at]
    for token in expired:
        del _BATCHES[token]

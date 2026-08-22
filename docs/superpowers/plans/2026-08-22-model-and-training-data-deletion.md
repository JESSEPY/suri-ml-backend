# Model and Training Data Deletion — Completed

**Status:** Implemented and merged into `new-ml-retrain-feature` (commit `db19eac`). This doc records what was built, not a forward plan — written after the fact so the admin-side delete/reset flows have a durable reference before this branch merges to `main`.

**Goal:** Let an admin clear demo state cleanly — delete one non-live model version, wipe all model history, or wipe all training data — as a prerequisite for the frontend's type-to-confirm delete UI (`suri` repo, `MLTrainingManagement.tsx`).

## What changed

**`ml/registry.py`**
- `delete_version(version)` — removes one row from the version registry.
- `delete_all()` — removes every row (`neq("version", "")`, the standard Supabase "no real row equals this" idiom for an unconditional delete, since `.delete()` with no filter is refused).

**`ml/artifacts_store.py`**
- `delete_version(version)` — removes one version's files from Supabase Storage. Never raises; returns `False` on failure, same posture as `upload_version`, so a Storage outage doesn't block the caller's local-disk/registry cleanup for the same version.
- `delete_all_versions()` — lists every version folder, then every file inside each, then removes them all in one call (Storage has no recursive delete-by-prefix, only delete-by-exact-key).

**`ml/model_state.py`**
- `clear_bundle()` — explicitly unloads the in-memory live model. Used after every version has been deleted, so `/predict` starts refusing requests (503) immediately instead of continuing to serve a bundle whose files just vanished on disk.

**`data/ingest_to_supabase.py`**
- `delete_all_records()` — clears `ml_soil_training_data` entirely (same `neq` idiom as `registry.delete_all()`). Does not touch any trained model.
- `list_records_paginated(page, page_size)` — 1-indexed page over `.range()`, returns `(rows, total_count)` in one round trip.

**`main.py`** — four new endpoints:
- `DELETE /model/versions` — full reset: every version everywhere (registry, Storage, local disk, in-memory bundle). Must be registered *before* `DELETE /model/{version}` — FastAPI matches routes in definition order, and `"versions"` is a syntactically valid value for that route's `{version}` path parameter; defined after it, every bulk-delete request would instead hit `delete_model_version(version="versions")`.
- `DELETE /model/{version}` — deletes one version everywhere. Refuses (422) to delete the currently-live version, and refuses (422) a malformed version string before touching anything.
- `DELETE /training-data` — clears all training data.
- `GET /training-data?page=&page_size=` — paginated browse, backing the frontend's training-data preview table.

## Verification

676 backend tests pass (added `TestDeleteVersion`, `TestDeleteAll`, `TestDeleteAllVersions`, `TestDeleteModelVersion`, `TestDeleteAllModelVersions`, `TestDeleteAllTrainingData`, `TestListTrainingData` across `test_registry.py`, `test_artifacts_store.py`, `test_model_state.py`, `test_ingest_to_supabase.py`, `test_pipeline_api.py`). Re-verified 2026-08-22 against the current branch tip (includes the unrelated `79df0f0` SOP 4.a confusion-matrix commit and the pending `db542a3` numeric-bounds merge from `origin/main`) — full suite still green, and `git merge-tree` against `origin/main` reports no conflicts despite both touching `main.py`.

## Consumed by

`suri` repo's `MLTrainingManagement.tsx` (admin ML dashboard) — see that repo's `docs/superpowers/plans/2026-08-22-ml-admin-delete-flows-and-trained-crop-filtering.md`.

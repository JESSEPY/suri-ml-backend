"""The upload/retrain/promote HTTP surface.

Supabase is never reached: every seam is monkeypatched. These tests assert
routing, status codes, and payload shape."""

import io

import pytest
from fastapi.testclient import TestClient

import main as service
from data.training_upload import TEMPLATE_COLUMNS
from ml import model_state, retrain_job


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


@pytest.fixture(autouse=True)
def _no_supabase(monkeypatch):
    monkeypatch.setattr(service, "_existing_laboratory_codes", lambda: set())
    monkeypatch.setattr(service, "_commit_rows", lambda rows: len(rows))


def upload_csv(rows):
    lines = [",".join(TEMPLATE_COLUMNS)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in TEMPLATE_COLUMNS))
    return ("\n".join(lines) + "\n").encode()


VALID = {
    "laboratory_code": "2026-API1", "crop": "Corn", "ph": "6.4",
    "organic_matter_pct": "2.8", "phosphorus_ppm": "18.0", "potassium_cmol_kg": "0.35",
}

# /model/promote now rejects anything that isn't ml/train.py's
# "%Y-%m-%d-%H%M%S" stamp, so stand-in versions have to look like real ones.
CANDIDATE = "2026-08-18-101112"
BROKEN = "2026-08-18-101113"


class TestValidateEndpoint:
    def test_accepts_a_good_file_and_writes_nothing(self, client):
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv([VALID])), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["accepted"]) == 1
        assert body["rejected"] == []
        assert body["accepted"][0]["derived"]["target_unit"] == "kg_ha"

    def test_reports_bad_rows_without_failing_the_request(self, client):
        rows = [VALID, {**VALID, "laboratory_code": "2026-API2", "crop": "Papaya"}]
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv(rows)), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["accepted"]) == 1
        assert "Papaya" in body["rejected"][0]["reason"]

    def test_unsupported_file_type_is_a_422(self, client):
        response = client.post(
            "/training-data/validate",
            files={"file": ("notes.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert response.status_code == 422

    def test_the_blocking_work_never_runs_on_the_event_loop(self, client, monkeypatch):
        """The handler must be async to await UploadFile.read(), which means
        anything after that read runs *on the event loop* unless pushed off
        it. _existing_laboratory_codes() makes several sequential blocking
        Supabase calls and validate_rows() runs full agronomy derivation per
        row; inline, a large upload stalls /predict dispatch for the whole
        duration -- and "/predict is never paused" is a hard constraint of
        this pipeline.

        asyncio.get_running_loop() succeeds only when called from the thread
        currently running the loop, so it is a direct test of the property
        rather than a proxy for it.
        """
        import asyncio

        original_codes = service._existing_laboratory_codes
        original_validate = service.validate_rows
        on_loop = []

        def note(name):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return  # a worker thread: exactly right
            on_loop.append(name)

        def codes():
            note("_existing_laboratory_codes")
            return original_codes()

        def validate(raw_rows, existing_codes):
            note("validate_rows")
            return original_validate(raw_rows, existing_codes)

        monkeypatch.setattr(service, "_existing_laboratory_codes", codes)
        monkeypatch.setattr(service, "validate_rows", validate)

        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv([VALID])), "text/csv")},
        )

        assert response.status_code == 200
        assert len(response.json()["accepted"]) == 1  # the seams really ran
        assert on_loop == []


class TestCommitEndpoint:
    def test_revalidates_and_commits_only_good_rows(self, client):
        response = client.post("/training-data/commit", json={
            "rows": [VALID, {**VALID, "laboratory_code": "2026-API3", "crop": "Nope"}],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["committed"] == 1
        assert len(body["rejected"]) == 1

    def test_empty_payload_is_a_422(self, client):
        assert client.post("/training-data/commit", json={"rows": []}).status_code == 422


class TestRetrainEndpoints:
    def test_starts_a_job_and_reports_its_status(self, client, monkeypatch):
        monkeypatch.setattr(retrain_job, "_JOBS", {})
        monkeypatch.setattr(retrain_job, "_ACTIVE", None)
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "v-test", "trained_at": "t", "n_rows": 5,
            "metrics": {"overall": {"target_n": {"mae": 1.0, "r2": 0.9, "dummy_mae": 9.0}}},
        })
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job_id = client.post("/retrain").json()["job_id"]

        import time
        for _ in range(500):
            body = client.get(f"/retrain/{job_id}").json()
            if body["status"] in {"done", "failed"}:
                break
            time.sleep(0.01)

        assert body["status"] == "done"
        assert body["version"] == "v-test"

    def test_unknown_job_is_a_404(self, client):
        assert client.get("/retrain/does-not-exist").status_code == 404

    def test_a_second_concurrent_retrain_is_a_409(self, client, monkeypatch):
        monkeypatch.setattr(retrain_job, "_ACTIVE", "already-running")
        assert client.post("/retrain").status_code == 409


class TestModelVersionEndpoints:
    def test_lists_versions_and_the_live_one(self, client, monkeypatch):
        monkeypatch.setattr(service.registry, "list_versions",
                            lambda: [{"version": "v1", "is_current": True}])
        body = client.get("/model/versions").json()
        assert body["current"] == model_state.snapshot()["version"]
        assert body["versions"][0]["version"] == "v1"

    def test_promoting_an_unknown_version_is_a_404(self, client):
        response = client.post("/model/promote", json={"version": "2099-01-01-000000"})
        assert response.status_code == 404

    def test_promoting_swaps_the_live_bundle(self, client, monkeypatch):
        live = model_state.snapshot()
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], CANDIDATE])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})
        monkeypatch.setattr(service.model_state, "verify_bundle", lambda bundle: None)
        monkeypatch.setattr(service.registry, "record_version", lambda **kwargs: None)
        monkeypatch.setattr(service.registry, "set_current", lambda *args, **kwargs: None)

        try:
            body = client.post("/model/promote", json={"version": CANDIDATE}).json()
            assert body["current"] == CANDIDATE
            assert body["registry_recorded"] is True
            assert model_state.snapshot()["version"] == CANDIDATE
        finally:
            model_state.set_bundle(live)

    def test_an_unservable_artifact_is_refused_and_the_live_model_kept(self, client, monkeypatch):
        live = model_state.snapshot()
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], BROKEN])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})

        def refuse(_bundle):
            raise ValueError("cannot serve requests: categories changed")

        monkeypatch.setattr(service.model_state, "verify_bundle", refuse)

        response = client.post("/model/promote", json={"version": BROKEN})
        assert response.status_code == 422
        assert model_state.snapshot()["version"] == live["version"]

    @pytest.mark.parametrize("version", [
        "../../etc/passwd", "latest", "", "2026-08-06-055755/../../evil",
        "2026-8-6-055755", "2026-08-06-055755.partial",
    ])
    def test_a_malformed_version_is_refused_before_touching_the_disk(
        self, client, monkeypatch, version
    ):
        """request.version reaches os.path.join, shutil.rmtree (via
        download_version) and joblib.load (unpickling). Reject anything that
        isn't ml/train.py's stamp before any of that runs -- these seams blow
        up if they are ever reached."""
        def explode(*_args, **_kwargs):
            raise AssertionError("a malformed version must not reach the filesystem")

        monkeypatch.setattr(service.model_state, "available_versions", explode)
        monkeypatch.setattr(service.artifacts_store, "download_version", explode)
        monkeypatch.setattr(service.model_state, "load_bundle", explode)

        assert client.post("/model/promote", json={"version": version}).status_code == 422


class TestModelConfidenceEndpoint:
    """Persistent, always-available per-crop confidence -- unlike /predict's
    confidence field, this doesn't need a specific request's inputs, so it
    reads the live bundle's held-out metrics only (no ensemble-disagreement
    Tier 2 check, which needs a real X)."""

    def test_reports_a_tier_and_pct_per_trained_crop(self, client):
        live = model_state.snapshot()
        trained = set(live["metadata"].get("metrics", {}).get("by_crop", {}))

        body = client.get("/model/confidence").json()

        assert body["version"] == live["version"]
        assert set(body["by_crop"].keys()) == trained
        for crop_entry in body["by_crop"].values():
            assert crop_entry["tier"] in {"high", "fair", "limited"}
            assert 5 <= crop_entry["pct"] <= 100

    def test_includes_the_raw_per_target_metrics_for_the_detail_view(self, client):
        live = model_state.snapshot()
        trained = set(live["metadata"].get("metrics", {}).get("by_crop", {}))
        any_crop = next(iter(trained))

        body = client.get("/model/confidence").json()

        targets = body["by_crop"][any_crop]["targets"]
        assert set(targets.keys()) == {"target_n", "target_p2o5", "target_k2o"}
        for target_metrics in targets.values():
            assert "mae" in target_metrics
            assert "r2" in target_metrics
            assert "mean_actual" in target_metrics


class TestPromoteRegistersBeforePromoting:
    """set_current() is an UPDATE ... WHERE version = $1. Promoting a version
    the registry never recorded -- the bootstrap artifact pinned via
    .gitignore is exactly that -- matched zero rows, PostgREST answered 200,
    and the promotion survived only in memory until the next restart quietly
    reverted it. That is the rollback case the pipeline exists for."""

    def _promotable(self, monkeypatch, live):
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], CANDIDATE])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})
        monkeypatch.setattr(service.model_state, "verify_bundle", lambda bundle: None)

    def test_the_version_is_upserted_before_set_current(self, client, monkeypatch):
        live = model_state.snapshot()
        self._promotable(monkeypatch, live)

        calls = []
        monkeypatch.setattr(service.registry, "record_version",
                            lambda **kwargs: calls.append(("record_version", kwargs)))
        monkeypatch.setattr(service.registry, "set_current",
                            lambda version, **kwargs: calls.append(("set_current", version)))

        try:
            client.post("/model/promote",
                        json={"version": CANDIDATE, "promoted_by": "admin@suri"})
        finally:
            model_state.set_bundle(live)

        assert [name for name, _ in calls] == ["record_version", "set_current"]
        recorded = calls[0][1]
        assert recorded["version"] == CANDIDATE
        # Sourced from the bundle that was just loaded and verified, not
        # re-read from disk.
        assert recorded["trained_at"] == live["metadata"]["trained_at"]
        assert recorded["row_count"] == live["metadata"]["n_rows"]
        assert recorded["metrics"] == live["metadata"]["metrics"]
        assert calls[1][1] == CANDIDATE

    def test_a_registry_outage_still_promotes_but_says_so(self, client, monkeypatch):
        """The in-memory swap already happened and is real, so this stays a
        200 -- but the admin has to be able to see the promotion won't
        survive a restart."""
        live = model_state.snapshot()
        self._promotable(monkeypatch, live)

        def unreachable(**_kwargs):
            raise RuntimeError("supabase unreachable")

        monkeypatch.setattr(service.registry, "record_version", unreachable)
        monkeypatch.setattr(service.registry, "set_current", unreachable)

        try:
            response = client.post("/model/promote", json={"version": CANDIDATE})
            assert response.status_code == 200
            assert response.json()["registry_recorded"] is False
            assert model_state.snapshot()["version"] == CANDIDATE
        finally:
            model_state.set_bundle(live)


class TestBootRegistryReconciliation:
    """A boot that fell back to newest-on-disk used to leave no trace, so
    every restart made the same blind guess -- and a retrained-then-rejected
    model would go live on the next reboot with nobody promoting it."""

    def test_records_and_promotes_what_was_actually_booted(self, monkeypatch):
        booted = model_state.snapshot()
        recorded, promoted = {}, {}

        monkeypatch.setattr(service.registry, "current_version", lambda: None)
        monkeypatch.setattr(service.registry, "record_version",
                            lambda **kwargs: recorded.update(kwargs))
        monkeypatch.setattr(service.registry, "set_current",
                            lambda **kwargs: promoted.update(kwargs))

        service._reconcile_boot_registry()

        assert recorded["version"] == booted["version"]
        assert recorded["trained_at"] == booted["metadata"]["trained_at"]
        assert recorded["row_count"] == booted["metadata"]["n_rows"]
        assert promoted == {"version": booted["version"], "promoted_by": "boot-fallback"}

    def test_writes_nothing_when_the_registry_already_agrees(self, monkeypatch):
        booted = model_state.snapshot()
        monkeypatch.setattr(service.registry, "current_version", lambda: booted["version"])

        def explode(**_kwargs):
            raise AssertionError("the registry already names this version; leave it alone")

        monkeypatch.setattr(service.registry, "record_version", explode)
        monkeypatch.setattr(service.registry, "set_current", explode)

        service._reconcile_boot_registry()

    def test_a_registry_outage_never_blocks_startup(self, monkeypatch):
        def unreachable(*_args, **_kwargs):
            raise RuntimeError("supabase unreachable")

        monkeypatch.setattr(service.registry, "current_version", unreachable)
        service._reconcile_boot_registry()  # must not raise

    def test_it_runs_on_server_startup_not_on_import(self, monkeypatch):
        """Importing main.py (a test run, a REPL, a script) must never rewrite
        which version production considers live -- only a process about to
        serve traffic may claim one."""
        calls = []
        monkeypatch.setattr(service, "_reconcile_boot_registry", lambda: calls.append(1))

        with TestClient(service.app):
            pass

        assert calls == [1]

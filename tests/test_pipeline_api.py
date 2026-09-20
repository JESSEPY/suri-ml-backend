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
        assert body["accepted_count"] == 1
        assert len(body["accepted_preview"]) == 1
        assert body["rejected"] == []
        assert body["accepted_preview"][0]["derived"]["target_unit"] == "kg_ha"

    def test_reports_bad_rows_without_failing_the_request(self, client):
        rows = [VALID, {**VALID, "laboratory_code": "2026-API2", "crop": "Papaya"}]
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv(rows)), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["accepted_count"] == 1
        assert "Papaya" in body["rejected"][0]["reason"]

    def test_returns_a_usable_token(self, client):
        """The token is what /commit takes instead of the full row list --
        the whole point of this response shape is that the browser never
        has to send the accepted rows back."""
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv([VALID])), "text/csv")},
        )
        token = response.json()["token"]
        assert isinstance(token, str) and len(token) > 0

    def test_accepted_preview_is_capped_even_when_more_rows_are_accepted(self, client):
        """A large file must not blow the response back up to full size --
        the preview is bounded regardless of how many rows were accepted."""
        rows = [{**VALID, "laboratory_code": f"2026-API{i}"} for i in range(300)]
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv(rows)), "text/csv")},
        )
        body = response.json()
        assert body["accepted_count"] == 300
        assert len(body["accepted_preview"]) < 300

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
        assert response.json()["accepted_count"] == 1  # the seams really ran
        assert on_loop == []


class TestCommitEndpoint:
    def _validate(self, client, rows):
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv(rows)), "text/csv")},
        )
        return response.json()["token"]

    def test_commits_only_what_validate_already_accepted(self, client):
        """Filtering bad rows out now happens once, at /validate -- /commit
        just commits whatever that same request already accepted, by
        token. A row invalid enough to reject never makes it into the
        cached batch at all, so there's nothing left for /commit to
        re-reject here."""
        token = self._validate(client, [VALID, {**VALID, "laboratory_code": "2026-API3", "crop": "Nope"}])
        response = client.post("/training-data/commit", json={"token": token})
        assert response.status_code == 200
        body = response.json()
        assert body["committed"] == 1
        assert len(body["rejected"]) == 1  # the "Nope" row, carried over from validate

    def test_unknown_token_is_a_404(self, client):
        response = client.post("/training-data/commit", json={"token": "does-not-exist"})
        assert response.status_code == 404

    def test_token_is_one_shot(self, client):
        """Committing the same token twice (double-click, retried request)
        must not silently recommit or double-count rows."""
        token = self._validate(client, [VALID])
        first = client.post("/training-data/commit", json={"token": token})
        assert first.status_code == 200
        second = client.post("/training-data/commit", json={"token": token})
        assert second.status_code == 404

    def test_no_accepted_rows_is_a_422(self, client):
        token = self._validate(client, [{**VALID, "crop": "Nope"}])
        response = client.post("/training-data/commit", json={"token": token})
        assert response.status_code == 422

    def test_a_code_committed_elsewhere_since_validate_is_rejected_not_silently_written(
        self, client, monkeypatch
    ):
        """The original commit endpoint re-validated from raw rows partly to
        catch a laboratory_code that became a duplicate in the gap between
        validate and commit (a second upload finishing first). The token
        design still needs this: it just checks against a fresh read of
        existing codes at commit time instead of re-deriving everything."""
        token = self._validate(client, [VALID])
        monkeypatch.setattr(service, "_existing_laboratory_codes", lambda: {VALID["laboratory_code"]})
        response = client.post("/training-data/commit", json={"token": token})
        assert response.status_code == 422
        assert response.json()["detail"] == "No rows to commit."


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

    def test_includes_coverage_when_the_artifact_has_it(self, client):
        """coverage is None for an artifact trained before crop_coverage
        existed (same nullable-until-retrained pattern as by_crop's "n") --
        every crop entry still carries the key so the frontend has one
        consistent shape to check, retrained or not."""
        body = client.get("/model/confidence").json()
        for crop_entry in body["by_crop"].values():
            assert "coverage" in crop_entry
            coverage = crop_entry["coverage"]
            if coverage is not None:
                assert set(coverage.keys()) == {"train", "val", "test", "total"}
                assert coverage["total"] == coverage["train"] + coverage["val"] + coverage["test"]

    def test_a_crop_with_very_few_held_out_rows_is_forced_to_limited(self):
        """Reproduces the real case that motivated this: Lettuce showed
        "High confidence * 89%" on 3 held-out rows. A flattering error from
        1-3 rows proves nothing about generalization -- calling
        _historical_confidence directly (rather than through the endpoint)
        so this doesn't depend on which crops the live artifact happens to
        have thin data for right now."""
        bundle = {"metadata": {"metrics": {"by_crop": {"Lettuce": {
            "target_n": {"mae": 5.96, "mean_actual": 56.67, "n": 3},
            "target_p2o5": {"mae": 2.07, "mean_actual": 26.67, "n": 3},
            "target_k2o": {"mae": 0.99, "mean_actual": 30.0, "n": 3},
        }}}}}
        confidence = service._historical_confidence(bundle, "Lettuce")
        assert confidence.tier == "limited"

    def test_a_crop_with_plenty_of_held_out_rows_is_not_forced(self):
        bundle = {"metadata": {"metrics": {"by_crop": {"Rice": {
            "target_n": {"mae": 3.41, "mean_actual": 57.47, "n": 239},
        }}}}}
        confidence = service._historical_confidence(bundle, "Rice")
        assert confidence.tier == "high"


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


def _raise_no_model_loaded(*_args, **_kwargs):
    raise RuntimeError("No model artifacts loaded")


class TestBootWithoutArtifacts:
    """A full registry/artifact reset (every version deleted, ready to
    retrain from a freshly re-uploaded training set) used to crash main.py
    at import time -- boot_version() raised, which meant the whole admin UI
    (upload, retrain, version history) was unusable for the very first
    bootstrap. Exactly backwards for a project whose point is "no CLI
    needed": an admin should be able to do the first retrain from the UI,
    which needs the server to be *up* with no model loaded yet."""

    def test_boot_version_returns_none_with_no_artifacts_on_disk(self, monkeypatch):
        monkeypatch.setattr(service.model_state, "available_versions", lambda root: [])
        assert service.boot_version() is None


class TestEndpointsWithoutAModelLoaded:
    """model_state.snapshot() raises RuntimeError whenever nothing has been
    loaded -- true both for "never booted one" (see TestBootWithoutArtifacts)
    and, exercised more easily here, "was loaded, then unset". Every reader
    must degrade gracefully rather than surface that as an unhandled 500."""

    def test_health_reports_no_model_instead_of_crashing(self, client, monkeypatch):
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] is None
        assert body["artifacts_loaded"] is False

    def test_predict_returns_a_clean_503_not_a_500(self, client, monkeypatch):
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        response = client.post("/predict", json={"crop": "Corn", "data_source": "stk", "ph": 6.5})
        assert response.status_code == 503
        assert "retrain" in response.json()["detail"].lower()

    def test_predict_serves_the_table_lookup_path_without_a_model(self, client, monkeypatch):
        """A complete request is answered exactly by the official guide table
        and never touches the ML ensemble, so it must not depend on one being
        trained. model_version reports "table-lookup" rather than a fake
        version (the audit log column is NOT NULL). An incomplete reading
        still needs the ensemble and keeps the 503 above."""
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        response = client.post("/predict", json={
            "crop": "Corn", "data_source": "lab", "ph": 6.5,
            "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0, "potassium_cmol_kg": 0.35,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "table-lookup"
        assert body["model_version"] == "table-lookup"
        assert body["confidence"] is None

    def test_model_versions_reports_no_current_instead_of_crashing(self, client, monkeypatch):
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        response = client.get("/model/versions")
        assert response.status_code == 200
        assert response.json()["current"] is None

    def test_model_confidence_returns_an_empty_response_instead_of_crashing(self, client, monkeypatch):
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        response = client.get("/model/confidence")
        assert response.status_code == 200
        assert response.json() == {"version": None, "by_crop": {}}

    def test_promote_succeeds_as_a_first_promotion_with_no_previous_version(
        self, client, monkeypatch
    ):
        live = model_state.snapshot()  # captured before monkeypatching it away
        monkeypatch.setattr(service.model_state, "snapshot", _raise_no_model_loaded)
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"]])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: live)
        monkeypatch.setattr(service.model_state, "verify_bundle", lambda bundle: None)
        monkeypatch.setattr(service.registry, "record_version", lambda **kwargs: None)
        monkeypatch.setattr(service.registry, "set_current", lambda *args, **kwargs: None)

        try:
            body = client.post("/model/promote", json={"version": live["version"]}).json()
            assert body["previous"] is None
            assert body["current"] == live["version"]
        finally:
            model_state.set_bundle(live)


class TestDeleteModelVersion:
    def test_refuses_to_delete_the_live_version(self, client):
        live = model_state.snapshot()["version"]
        response = client.delete(f"/model/{live}")
        assert response.status_code == 422
        assert "live" in response.json()["detail"].lower()

    @pytest.mark.parametrize("version", [
        "latest", "2026-08-06-055755.partial", "2026-8-6-55755",
    ])
    def test_a_malformed_version_is_refused_before_touching_anything(
        self, client, monkeypatch, version
    ):
        # A value containing "/" (e.g. "../../etc/passwd") never reaches this
        # handler at all -- {version} is a single path segment by default,
        # so FastAPI 404s at the routing layer before any of our code runs.
        # These are same-segment strings that pass routing but must still
        # fail is_version_name()'s format check.
        def explode(*_args, **_kwargs):
            raise AssertionError("a malformed version must not reach deletion")

        monkeypatch.setattr(service.registry, "delete_version", explode)
        monkeypatch.setattr(service.artifacts_store, "delete_version", explode)

        response = client.delete(f"/model/{version}")
        assert response.status_code == 422

    def test_deletes_a_non_live_version_everywhere(self, client, monkeypatch, tmp_path):
        target = "2026-01-01-000000"
        (tmp_path / target).mkdir()
        (tmp_path / target / "metadata.json").write_text("{}")
        monkeypatch.setattr(service, "ARTIFACT_ROOT", str(tmp_path))
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [target])

        registry_calls, storage_calls = [], []
        monkeypatch.setattr(service.registry, "delete_version", registry_calls.append)
        monkeypatch.setattr(service.artifacts_store, "delete_version",
                            lambda version, **_kw: storage_calls.append(version) or True)

        response = client.delete(f"/model/{target}")

        assert response.status_code == 200
        assert registry_calls == [target]
        assert storage_calls == [target]
        assert not (tmp_path / target).exists()

    def test_404_when_the_version_exists_nowhere(self, client, monkeypatch):
        monkeypatch.setattr(service.model_state, "available_versions", lambda root: [])
        monkeypatch.setattr(service.registry, "get_version", lambda version: None)
        response = client.delete("/model/2099-01-01-000000")
        assert response.status_code == 404


class TestDeleteAllModelVersions:
    def test_clears_registry_storage_disk_and_the_live_bundle(self, client, monkeypatch, tmp_path):
        live = model_state.snapshot()  # captured so it can be restored after
        (tmp_path / "2026-01-01-000000").mkdir()
        (tmp_path / "2026-01-02-000000").mkdir()
        monkeypatch.setattr(service, "ARTIFACT_ROOT", str(tmp_path))

        registry_called = []
        monkeypatch.setattr(service.registry, "delete_all", lambda: registry_called.append(1))
        monkeypatch.setattr(service.artifacts_store, "delete_all_versions", lambda **_kw: True)

        try:
            response = client.delete("/model/versions")
            assert response.status_code == 200
            assert registry_called == [1]
            assert list(tmp_path.iterdir()) == []
            with pytest.raises(RuntimeError, match="No model artifacts loaded"):
                model_state.snapshot()
        finally:
            model_state.set_bundle(live)


class TestDeleteAllTrainingData:
    def test_deletes_everything_and_reports_the_count(self, client, monkeypatch):
        monkeypatch.setattr(service, "_delete_all_training_data", lambda: 3362)
        response = client.delete("/training-data")
        assert response.status_code == 200
        assert response.json() == {"deleted": 3362}


class TestListTrainingData:
    def test_returns_the_requested_page(self, client, monkeypatch):
        rows = [{"laboratory_code": "A"}]
        monkeypatch.setattr(
            service, "_list_training_data_page", lambda page, page_size: (rows, 57)
        )
        response = client.get("/training-data?page=2&page_size=25")
        assert response.status_code == 200
        body = response.json()
        assert body == {"rows": rows, "total": 57, "page": 2, "page_size": 25}

    def test_defaults_to_page_1_of_25(self, client, monkeypatch):
        captured = {}

        def fake_list(page, page_size):
            captured["args"] = (page, page_size)
            return [], 0

        monkeypatch.setattr(service, "_list_training_data_page", fake_list)
        client.get("/training-data")
        assert captured["args"] == (1, 25)

    def test_page_size_is_capped(self, client, monkeypatch):
        captured = {}

        def fake_list(page, page_size):
            captured["args"] = (page, page_size)
            return [], 0

        monkeypatch.setattr(service, "_list_training_data_page", fake_list)
        response = client.get("/training-data?page_size=10000")
        assert response.status_code == 422


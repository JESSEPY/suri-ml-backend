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
                            lambda root: [live["version"], "candidate"])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})
        monkeypatch.setattr(service.model_state, "verify_bundle", lambda bundle: None)
        monkeypatch.setattr(service.registry, "set_current", lambda **kwargs: None)

        try:
            body = client.post("/model/promote", json={"version": "candidate"}).json()
            assert body["current"] == "candidate"
            assert model_state.snapshot()["version"] == "candidate"
        finally:
            model_state.set_bundle(live)

    def test_an_unservable_artifact_is_refused_and_the_live_model_kept(self, client, monkeypatch):
        live = model_state.snapshot()
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], "broken"])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})

        def refuse(_bundle):
            raise ValueError("cannot serve requests: categories changed")

        monkeypatch.setattr(service.model_state, "verify_bundle", refuse)

        response = client.post("/model/promote", json={"version": "broken"})
        assert response.status_code == 422
        assert model_state.snapshot()["version"] == live["version"]

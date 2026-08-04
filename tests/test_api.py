import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


GOLDEN_PAYLOAD = {
    "crop": "Sorghum", "data_source": "lab", "ph": 8.17,
    "organic_matter_pct": 3.21, "phosphorus_ppm": 18.96,
    "potassium_ppm": 138.0, "farm_area_sqm": 210,
    "laboratory_code": "SCC-26-417-417",
}


class TestHealth:
    def test_reports_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_loaded"] is True
        assert body["model_version"]


class TestPredict:
    def test_golden_record_returns_expected_shape(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        for field in ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
                      "app1_formulation", "app2_formulation", "derived_remarks",
                      "model_version", "source"]:
            assert field in body
        assert body["source"] == "ml-ensemble"

    def test_golden_record_is_agronomically_sane(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert abs(body["target_n_kg_ha"] - 50) <= 15
        assert abs(body["target_p2o5_kg_ha"] - 40) <= 15
        assert abs(body["target_k2o_kg_ha"] - 60) <= 15

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert body["derived_remarks"] == {
            "ph_remark": "MODERATELY ALKALINE", "om_remark": "MODERATELY LOW",
            "p_remark": "HIGH", "k_remark": "SUFFICIENT+",
        }

    def test_farm_totals_scale_the_per_hectare_rate(self, client):
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        totals = body["farm_totals"]
        assert totals["area_ha"] == pytest.approx(0.021)
        assert totals["n_kg"] == pytest.approx(body["target_n_kg_ha"] * 0.021, rel=1e-3)

    def test_farm_totals_absent_when_area_not_given(self, client):
        payload = {key: value for key, value in GOLDEN_PAYLOAD.items()
                   if key != "farm_area_sqm"}
        assert client.post("/predict", json=payload).json()["farm_totals"] is None

    def test_minimal_payload_is_accepted(self, client):
        """Only crop, data_source and ph are required — everything else is
        imputed and flagged."""
        response = client.post("/predict", json={
            "crop": "Corn", "data_source": "iot", "ph": 6.4,
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("source", ["lab", "stk", "iot"])
    def test_every_data_source_is_servable(self, client, source):
        payload = {**GOLDEN_PAYLOAD, "data_source": source}
        assert client.post("/predict", json=payload).status_code == 200

    def test_total_nitrogen_derived_from_organic_matter(self, client):
        """The caller need not send it — the lab itself calculates it."""
        body = client.post("/predict", json=GOLDEN_PAYLOAD).json()
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"


class TestValidation:
    def test_ph_above_fourteen_is_rejected(self, client):
        assert client.post("/predict", json={**GOLDEN_PAYLOAD, "ph": 15.0}).status_code == 422

    def test_negative_ppm_is_rejected(self, client):
        payload = {**GOLDEN_PAYLOAD, "phosphorus_ppm": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_data_source_is_rejected(self, client):
        payload = {**GOLDEN_PAYLOAD, "data_source": "satellite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict", json={"crop": "Corn"}).status_code == 422

    def test_unknown_crop_is_rejected_with_the_valid_list(self, client):
        """A silent fallback would return a confident number computed from no
        crop information at all."""
        response = client.post("/predict", json={**GOLDEN_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Sorghum" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a recommendation."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=GOLDEN_PAYLOAD).status_code == 200

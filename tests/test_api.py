import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


SAMPLE_PAYLOAD = {
    "crop": "Corn", "data_source": "lab", "ph": 6.4,
    "organic_matter_pct": 2.8, "phosphorus_ppm": 18.0,
    "potassium_cmol_kg": 0.35, "farm_area_sqm": 5000,
    "laboratory_code": "TEST-001",
}


class TestHealth:
    def test_reports_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_loaded"] is True
        assert body["model_version"]


class TestPredict:
    def test_returns_expected_shape(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        for field in ["target_n", "target_p2o5", "target_k2o", "target_unit",
                      "app1_formulation", "app2_formulation", "application_schedule",
                      "derived_remarks", "model_version", "source"]:
            assert field in body
        assert body["source"] == "ml-ensemble"
        assert body["target_unit"] == "kg_ha"

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["ph_remark"] == "SLIGHTLY ACIDIC"
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"
        assert body["derived_remarks"]["p_remark"] == "HIGH"
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_k_remark_uses_the_converted_ppm_value_not_the_raw_cmol_kg(self, client):
        """0.35 cmol/kg -> 136.85 ppm, which is SUFFICIENT+ on the (unchanged,
        ppm-scale) K bands. If a caller forgot the x391 conversion and fed
        0.35 straight into derive_k_remark, that would read as DEFICIENT
        (well under the 60 ppm floor) — this test exists to catch exactly
        that regression."""
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_application_schedule_present_for_a_covered_crop(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["application_schedule"] is not None
        assert len(body["application_schedule"]) > 0

    def test_farm_totals_scale_the_per_hectare_rate_for_kg_ha_crops(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        totals = body["farm_totals"]
        assert totals["area_ha"] == pytest.approx(0.5)
        assert totals["n_kg"] == pytest.approx(body["target_n"] * 0.5, rel=1e-3)

    def test_farm_totals_absent_when_area_not_given(self, client):
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "farm_area_sqm"}
        assert client.post("/predict", json=payload).json()["farm_totals"] is None

    def test_farm_totals_absent_for_kg_tree_crops(self, client):
        """A per-tree rate can't be scaled by hectares alone — this API
        doesn't collect a tree count, so totalling isn't attempted."""
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert body["farm_totals"] is None

    def test_minimal_payload_is_accepted(self, client):
        """Only crop, data_source and ph are required — everything else is
        imputed and flagged."""
        response = client.post("/predict", json={
            "crop": "Corn", "data_source": "iot", "ph": 6.4,
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("source", ["lab", "stk", "iot"])
    def test_every_data_source_is_servable(self, client, source):
        payload = {**SAMPLE_PAYLOAD, "data_source": source}
        assert client.post("/predict", json=payload).status_code == 200

    def test_variety_is_accepted_and_optional(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "variety": "Hybrid"}
        assert client.post("/predict", json=payload).status_code == 200
        without_variety = {**SAMPLE_PAYLOAD, "crop": "Rice"}
        assert client.post("/predict", json=without_variety).status_code == 200

    def test_cultivation_method_is_accepted_and_optional(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "cultivation_method": "Upland"}
        assert client.post("/predict", json=payload).status_code == 200
        without = {**SAMPLE_PAYLOAD, "crop": "Rice"}
        assert client.post("/predict", json=without).status_code == 200

    def test_age_band_bearing_alias_is_accepted(self, client):
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_age_band_omitted_behaves_like_bearing(self, client):
        with_bearing = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        without = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        response_a = client.post("/predict", json=with_bearing).json()
        response_b = client.post("/predict", json=without).json()
        assert response_a["target_n"] == pytest.approx(response_b["target_n"])


class TestValidation:
    def test_ph_above_fourteen_is_rejected(self, client):
        assert client.post("/predict", json={**SAMPLE_PAYLOAD, "ph": 15.0}).status_code == 422

    def test_negative_ppm_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "phosphorus_ppm": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_data_source_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "data_source": "satellite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict", json={"crop": "Corn"}).status_code == 422

    def test_unknown_crop_is_rejected_with_the_valid_list(self, client):
        """A silent fallback would return a confident number computed from no
        crop information at all."""
        response = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Corn" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a recommendation."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200

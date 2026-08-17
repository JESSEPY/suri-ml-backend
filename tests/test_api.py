import pytest
from fastapi.testclient import TestClient

import main as service


@pytest.fixture(autouse=True)
def _no_real_supabase_writes(monkeypatch):
    """Every /predict call in this file would otherwise insert a real row
    into the live ml_fertilizer_predictions table in production Supabase --
    confirmed happening (HTTP 201) before this fixture existed. Autouse so
    every test in this file is covered without each one remembering to opt
    in. TestLoggingIsNonFatal overrides this per-test (see below) since it
    specifically needs log_prediction to raise, not silently succeed."""
    monkeypatch.setattr(service, "log_prediction", lambda *args, **kwargs: None)


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
        for field in ["target_n", "target_p2o5", "target_k2o", "max_total_need", "target_unit",
                      "app1_formulation", "app2_formulation", "application_schedule",
                      "derived_remarks", "model_version", "source"]:
            assert field in body
        assert body["target_unit"] == "kg_ha"

    def test_remarks_are_derived_from_the_numerics(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["derived_remarks"]["ph_remark"] == "SLIGHTLY ACIDIC"
        assert body["derived_remarks"]["om_remark"] == "MODERATELY LOW"
        assert body["derived_remarks"]["p_remark"] == "HIGH"
        assert body["derived_remarks"]["k_remark"] == "SUFFICIENT+"

    def test_k_remark_uses_the_converted_ppm_value_not_the_raw_cmol_kg(self, client):
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
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert body["farm_totals"] is None

    def test_minimal_payload_is_accepted(self, client):
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


class TestCompletenessRouter:
    """The fix for Critical #1/#2: a request with complete OM/P/K is
    answered by the exact guide table, not the model -- regardless of
    whether the specific crop/variety/cultivation_method/age_band
    combination has any training coverage."""

    def test_complete_request_reports_table_lookup_source(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["source"] == "table-lookup"

    def test_incomplete_request_reports_ml_ensemble_source(self, client):
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "phosphorus_ppm"}
        body = client.post("/predict", json=payload).json()
        assert body["source"] == "ml-ensemble"

    def test_table_lookup_has_no_confidence(self, client):
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["source"] == "table-lookup"
        assert body["confidence"] is None

    def test_ml_ensemble_reports_a_confidence_tier_and_pct(self, client):
        # Same pattern test_incomplete_request_reports_ml_ensemble_source
        # already uses: drop phosphorus_ppm entirely so is_complete() is
        # False and the request routes to ml-ensemble.
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "phosphorus_ppm"}
        body = client.post("/predict", json=payload).json()
        assert body["source"] == "ml-ensemble"
        assert body["confidence"]["tier"] in {"high", "fair", "limited"}
        assert 5 <= body["confidence"]["pct"] <= 100

    def test_complete_request_matches_nutrient_requirements_exactly(self, client):
        """The whole point of the router: for a complete request, /predict's
        answer must be bit-for-bit what the deterministic table says, not a
        model approximation of it."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        expected_n, expected_p, expected_k = nutrient_requirements(
            "Corn", om=2.8, p_ppm=18.0, k_cmol_kg=0.35,
        )
        assert body["target_n"] == pytest.approx(expected_n)
        assert body["target_p2o5"] == pytest.approx(expected_p)
        assert body["target_k2o"] == pytest.approx(expected_k)

    def test_uncovered_cultivation_method_is_still_exact(self, client):
        """Before this fix: Rice + Upland (near-zero real training coverage)
        returned a model guess far from the guide's actual answer. Now it
        must match the table exactly, the same as any other combination."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        payload = {**SAMPLE_PAYLOAD, "crop": "Rice", "cultivation_method": "Upland",
                   "organic_matter_pct": 1.5, "phosphorus_ppm": 5.0, "potassium_cmol_kg": 0.1}
        body = client.post("/predict", json=payload).json()
        expected_n, _, _ = nutrient_requirements(
            "Rice", om=1.5, p_ppm=5.0, k_cmol_kg=0.1, cultivation_method="Upland",
        )
        assert body["target_n"] == pytest.approx(expected_n)

    def test_uncovered_age_band_is_still_exact(self, client):
        """Before this fix: Coconut + Below 1yr (zero real training
        coverage) returned a model guess ~18x the guide's actual answer."""
        from data.nutrient_recommendation_guide import nutrient_requirements
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "Below 1yr",
                   "organic_matter_pct": 1.5, "phosphorus_ppm": 5.0, "potassium_cmol_kg": 0.1}
        body = client.post("/predict", json=payload).json()
        expected_n, _, _ = nutrient_requirements(
            "Coconut", om=1.5, p_ppm=5.0, k_cmol_kg=0.1, age_band="Below 1yr",
        )
        assert body["target_n"] == pytest.approx(expected_n)

    def test_kg_tree_formulation_uses_the_correct_unit_and_scale(self, client):
        """Before this fix: a Coconut request's app1_formulation was a
        classifier-predicted string like '214 kg/ha (14-14-14) + 71 kg/ha
        (21-0-0)' -- kg/ha, at field-crop scale, for a kg_tree crop. Now the
        formulation is computed directly from the same exact target the
        table just produced."""
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "bearing"}
        body = client.post("/predict", json=payload).json()
        assert body["target_unit"] == "kg_tree"
        assert "kg/tree" in body["app1_formulation"]
        assert "kg/ha" not in body["app1_formulation"]

    def test_max_total_need_matches_the_guide_function(self, client):
        from data.nutrient_recommendation_guide import max_total_need
        body = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert body["max_total_need"] == pytest.approx(max_total_need("Corn"))

    def test_max_total_need_present_and_consistent_on_the_ml_ensemble_branch(self, client):
        """A fixed crop-config property, not soil-dependent -- must be the
        same value whether the request routed to table-lookup or
        ml-ensemble."""
        from data.nutrient_recommendation_guide import max_total_need
        payload = {key: value for key, value in SAMPLE_PAYLOAD.items()
                   if key != "phosphorus_ppm"}
        body = client.post("/predict", json=payload).json()
        assert body["source"] == "ml-ensemble"
        assert body["max_total_need"] == pytest.approx(max_total_need("Corn"))

    def test_unrecognized_age_band_is_rejected_not_a_500(self, client):
        """Unlike variety/cultivation_method (which degrade gracefully to a
        default inside the resolvers), an age_band string that isn't one of
        the crop's real guide bands has no fallback -- nutrient_requirements()
        raises KeyError, which must surface as a clean 422, not an unhandled
        server error."""
        payload = {**SAMPLE_PAYLOAD, "crop": "Coconut", "age_band": "nonsense"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestValidation:
    def test_ph_above_fourteen_is_rejected(self, client):
        assert client.post("/predict", json={**SAMPLE_PAYLOAD, "ph": 15.0}).status_code == 422

    def test_negative_ppm_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "phosphorus_ppm": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_total_nitrogen_above_ten_percent_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "total_nitrogen_pct": 10.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_phosphorus_above_five_hundred_ppm_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "phosphorus_ppm": 500.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_potassium_above_six_cmol_kg_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "potassium_cmol_kg": 6.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_temp_below_negative_ten_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "temp": -10.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_temp_above_sixty_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "temp": 60.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_ec_above_twenty_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "ec": 20.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_sal_above_twenty_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "sal": 20.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_farm_area_above_two_hundred_hectares_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "farm_area_sqm": 2_000_001}
        assert client.post("/predict", json=payload).status_code == 422

    def test_unknown_data_source_is_rejected(self, client):
        payload = {**SAMPLE_PAYLOAD, "data_source": "satellite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        assert client.post("/predict", json={"crop": "Corn"}).status_code == 422

    def test_unknown_crop_is_rejected_with_the_valid_list(self, client):
        response = client.post("/predict", json={**SAMPLE_PAYLOAD, "crop": "Durian"})
        assert response.status_code == 422
        assert "Corn" in response.text


class TestLoggingIsNonFatal:
    def test_prediction_succeeds_when_supabase_is_down(self, client, monkeypatch):
        """An audit-log outage is not a reason to deny a farmer a
        recommendation. This overrides the module's autouse no-op fixture
        with a raising one, specifically to prove /predict survives it."""
        def explode(*args, **kwargs):
            raise RuntimeError("supabase unreachable")
        monkeypatch.setattr(service, "log_prediction", explode)
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200

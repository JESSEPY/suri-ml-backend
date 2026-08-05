from data.nutrient_recommendation_guide import CROPS
from data.mode_of_application import application_schedule_for_crop


class TestApplicationScheduleForCrop:
    def test_known_crop_returns_text(self):
        text = application_schedule_for_crop("Rice")
        assert isinstance(text, str) and len(text) > 0

    def test_unknown_crop_returns_none(self):
        assert application_schedule_for_crop("Durian") is None

    def test_every_supported_crop_has_an_entry(self):
        """All 9 crops this system serves are covered by MODE OF
        APPLICATION.pdf under their own name or a listed alias."""
        for crop in CROPS:
            assert application_schedule_for_crop(crop) is not None, crop

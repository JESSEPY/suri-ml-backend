from data.nutrient_recommendation_guide import CROPS
from data.mode_of_application import application_schedule_for_crop


class TestApplicationScheduleForCrop:
    def test_known_crop_returns_text(self):
        text = application_schedule_for_crop("Rice")
        assert isinstance(text, str) and len(text) > 0

    def test_unknown_crop_returns_none(self):
        assert application_schedule_for_crop("Durian") is None

    def test_covered_crops_have_an_entry(self):
        """These crops are confirmed covered by MODE OF APPLICATION.pdf,
        transcribed under their own name or a listed alias. CROPS itself is
        not a reliable proxy for "has application-schedule text" -- most of
        the ongoing crop expansion has no entry in this document at all,
        and that's expected, not a bug (see test below)."""
        covered = [
            "Rice", "Corn", "Cassava", "Peanut", "Mango", "Cacao", "Banana",
            "Tomato", "Coconut",
            "Ampalaya", "Asparagus", "Cabbage", "Cauliflower", "Celery",
            "Chayote", "Cucumber", "Eggplant", "Lettuce", "Radish",
            "Squash", "Strawberry",
        ]
        for crop in covered:
            assert application_schedule_for_crop(crop) is not None, crop

    def test_an_uncovered_crop_returns_none_gracefully(self):
        """Alugbati has no entry anywhere in MODE OF APPLICATION.pdf -- the
        system must degrade to no schedule text rather than fabricate one."""
        assert application_schedule_for_crop("Alugbati") is None

import pytest
from ml.recipes import plan_applications, NO_APPLICATION


class TestGoldenRecordSCC417:
    """Working backward from the real report's own numbers must reproduce its
    exact application strings."""

    def test_reproduces_real_report_exactly(self):
        app1, app2 = plan_applications(50, 40, 60)
        expected = "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)"
        assert app1 == expected
        assert app2 == expected


class TestFormatting:
    def test_terms_are_joined_with_plus(self):
        app1, _ = plan_applications(90, 60, 60)
        assert " + " in app1

    def test_each_term_has_kg_ha_and_grade(self):
        app1, _ = plan_applications(90, 60, 60)
        for term in app1.split(" + "):
            assert " kg/ha (" in term
            assert term.endswith(")")

    def test_zero_quantity_terms_are_omitted(self):
        """When the Complete base already supplies all the N and K needed,
        no Ammosul or MOP term should appear."""
        app1, _ = plan_applications(20, 150, 20)
        assert "21-0-0" not in app1
        assert "0-0-60" not in app1
        assert "14-14-14" in app1


class TestSingleApplication:
    def test_low_requirement_gives_no_second_application(self):
        _, app2 = plan_applications(20, 20, 20)
        assert app2 == NO_APPLICATION

    def test_high_requirement_gives_a_second_application(self):
        _, app2 = plan_applications(120, 90, 90)
        assert app2 != NO_APPLICATION


class TestVocabularyStability:
    def test_same_input_gives_same_output(self):
        assert plan_applications(90, 60, 60) == plan_applications(90, 60, 60)

    def test_quantities_are_whole_numbers(self):
        app1, _ = plan_applications(93, 61, 57)
        for term in app1.split(" + "):
            quantity = term.split(" kg/ha")[0]
            assert quantity.isdigit()

    def test_vocabulary_is_finite_over_the_target_space(self):
        """Rounding to whole kg/ha keeps the classifier's label space finite."""
        seen = set()
        for n in range(20, 201, 10):
            for p in range(20, 151, 10):
                for k in range(20, 201, 10):
                    seen.update(plan_applications(n, p, k))
        assert len(seen) < 5000
        assert all(isinstance(label, str) and label for label in seen)

import pytest
from ml.recipes import plan_applications, NO_APPLICATION


class TestKgHaFormatting:
    """Working backward from real DA-report-style numbers must reproduce the
    same kind of application strings this system has always produced for
    field crops — the kg_ha formatting itself hasn't changed, only that it's
    now explicit about which unit it's formatting for."""

    def test_reproduces_expected_kg_ha_formulation(self):
        app1, app2 = plan_applications(50, 40, 60, "kg_ha")
        expected = "143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)"
        assert app1 == expected
        assert app2 == expected

    def test_terms_are_joined_with_plus(self):
        app1, _ = plan_applications(90, 60, 60, "kg_ha")
        assert " + " in app1

    def test_each_term_has_kg_ha_and_grade(self):
        app1, _ = plan_applications(90, 60, 60, "kg_ha")
        for term in app1.split(" + "):
            assert " kg/ha (" in term
            assert term.endswith(")")

    def test_zero_quantity_terms_are_omitted(self):
        app1, _ = plan_applications(20, 150, 20, "kg_ha")
        assert "21-0-0" not in app1
        assert "0-0-60" not in app1
        assert "14-14-14" in app1

    def test_low_requirement_gives_no_second_application(self):
        _, app2 = plan_applications(20, 20, 20, "kg_ha")
        assert app2 == NO_APPLICATION

    def test_high_requirement_gives_a_second_application(self):
        _, app2 = plan_applications(120, 90, 90, "kg_ha")
        assert app2 != NO_APPLICATION

    def test_quantities_are_whole_numbers(self):
        app1, _ = plan_applications(93, 61, 57, "kg_ha")
        for term in app1.split(" + "):
            quantity = term.split(" kg/ha")[0]
            assert quantity.isdigit()


class TestKgTreeFormatting:
    """The regression this class exists to catch: kg_tree amounts are
    legitimately sub-1 and were previously formatted as '0 kg/ha (...)' and
    dropped entirely (rounding to int(round(kg)) == 0), or — worse, at
    serving time via the old trained classifier — shown with the wrong unit
    label at a wildly wrong scale."""

    def test_sub_one_amount_is_not_dropped(self):
        app1, _ = plan_applications(0.947, 1.166, 2.111, "kg_tree")
        assert app1 != NO_APPLICATION
        assert "14-14-14" in app1

    def test_uses_kg_tree_label_not_kg_ha(self):
        app1, _ = plan_applications(0.947, 1.166, 2.111, "kg_tree")
        assert "kg/tree" in app1
        assert "kg/ha" not in app1

    def test_kg_tree_always_gets_a_second_application(self):
        """Every kg_tree crop's Mode of Application guidance describes at
        least two splits per year regardless of amount — the kg_ha
        single-application thresholds (calibrated to a ~20-300 scale) would
        otherwise always trivially apply at kg_tree's ~0.003-2 scale and
        wrongly collapse every kg_tree recommendation to one application."""
        _, app2 = plan_applications(0.06, 0.05, 0.15, "kg_tree")
        assert app2 != NO_APPLICATION

    def test_tiny_amount_is_not_rounded_to_zero_and_dropped(self):
        """0.003 kg/tree (the smallest real target in this system, young
        Cacao) must still appear as a non-zero term, not vanish the way
        int(round(0.003)) == 0 would have dropped it."""
        app1, _ = plan_applications(0.003, 0.002, 0.001, "kg_tree")
        assert app1 != NO_APPLICATION


class TestVocabularyStability:
    def test_same_input_gives_same_output(self):
        assert plan_applications(90, 60, 60, "kg_ha") == plan_applications(90, 60, 60, "kg_ha")

    def test_vocabulary_is_finite_over_the_kg_ha_target_space(self):
        seen = set()
        for n in range(20, 201, 10):
            for p in range(20, 151, 10):
                for k in range(20, 201, 10):
                    seen.update(plan_applications(n, p, k, "kg_ha"))
        assert len(seen) < 5000
        assert all(isinstance(label, str) and label for label in seen)

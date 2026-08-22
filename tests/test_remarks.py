import pytest
from ml.remarks import (
    derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark,
    UNKNOWN, PH_ORDER, OM_ORDER, P_ORDER, K_ORDER,
)


class TestKnownReferenceValues:
    """Fixed reference points for the band functions, at the same numeric
    values a real DA Region 7 report (laboratory_code SCC-26-417-417) once
    supplied. pH and P are independently confirmed, so those two assertions
    still reflect a plausible real reading. The OM/K boundaries were, at the
    time these two assertions were written, calibrated against that same
    now-untraceable record -- see the OM/K provenance note in remarks.py's
    module docstring for how the boundaries were since replaced with DA's
    Table 3 (Nutrient Recommendation Guide) bins, and why "moderately low"/
    "sufficient+" are no longer necessarily what these two exact values
    would resolve to."""

    def test_ph_8_17_is_moderately_alkaline(self):
        assert derive_ph_remark(8.17) == "MODERATELY ALKALINE"

    def test_om_3_21_is_low(self):
        assert derive_om_remark(3.21) == "LOW"

    def test_p_18_96_is_high(self):
        assert derive_p_remark(18.96) == "HIGH"

    def test_k_138_is_sufficient(self):
        assert derive_k_remark(138.00) == "SUFFICIENT"


class TestPhBands:
    @pytest.mark.parametrize("value,expected", [
        (4.0, "STRONGLY ACIDIC"),
        (4.5, "MODERATELY ACIDIC"),
        (5.5, "MODERATELY ACIDIC"),
        (5.6, "SLIGHTLY ACIDIC"),
        (6.5, "SLIGHTLY ACIDIC"),
        (6.6, "NEUTRAL"),
        (7.3, "NEUTRAL"),
        (7.4, "SLIGHTLY ALKALINE"),
        (7.8, "SLIGHTLY ALKALINE"),
        (7.9, "MODERATELY ALKALINE"),
        (8.4, "MODERATELY ALKALINE"),
        (8.5, "STRONGLY ALKALINE"),
        (9.9, "STRONGLY ALKALINE"),
    ])
    def test_band_boundaries(self, value, expected):
        assert derive_ph_remark(value) == expected


class TestOtherBands:
    # OM boundaries are the DA Nutrient Recommendation Guide's real Table 3
    # bins (0-2.0, 2.1-3.5, 3.6-4.5, 4.6-5.5, >5.6) -- see remarks.py's
    # module docstring for provenance. 2.1/3.6/4.6/5.6 are each the printed
    # start of the *next* band, used as this function's exclusive upper
    # bound so a value like 2.0 correctly stays in "VERY LOW".
    @pytest.mark.parametrize("value,expected", [
        (1.0, "VERY LOW"), (2.1, "LOW"), (3.6, "MODERATELY LOW"),
        (4.6, "MODERATE"), (5.6, "HIGH"), (7.0, "HIGH"),
    ])
    def test_om_boundaries(self, value, expected):
        assert derive_om_remark(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (2.0, "VERY LOW"), (5.0, "LOW"), (10.0, "MEDIUM"),
        (17.0, "HIGH"), (40.0, "HIGH"),
    ])
    def test_p_boundaries(self, value, expected):
        assert derive_p_remark(value) == expected

    # Extractable K boundaries are the DA Nutrient Recommendation Guide's
    # real Table 3 bins (0-35, 36-55, 56-250, 251-300, >301), same
    # next-band-start-as-exclusive-upper-bound convention as OM above.
    # DA's table attaches fertilizer rates to each bin, not label words --
    # DEFICIENT/LOW/SUFFICIENT/SUFFICIENT+/EXCESSIVE are this project's own
    # naming for those five rate tiers, not DA terminology.
    @pytest.mark.parametrize("value,expected", [
        (20.0, "DEFICIENT"), (36.0, "LOW"), (56.0, "SUFFICIENT"),
        (251.0, "SUFFICIENT+"), (301.0, "EXCESSIVE"), (400.0, "EXCESSIVE"),
    ])
    def test_k_boundaries(self, value, expected):
        assert derive_k_remark(value) == expected


class TestMissingValues:
    def test_none_returns_unknown(self):
        assert derive_ph_remark(None) == UNKNOWN
        assert derive_om_remark(None) == UNKNOWN
        assert derive_p_remark(None) == UNKNOWN
        assert derive_k_remark(None) == UNKNOWN


class TestOrderLists:
    def test_orders_exclude_unknown(self):
        for order in (PH_ORDER, OM_ORDER, P_ORDER, K_ORDER):
            assert UNKNOWN not in order

    def test_orders_have_expected_lengths(self):
        assert len(PH_ORDER) == 7
        assert len(OM_ORDER) == 5
        assert len(P_ORDER) == 4
        assert len(K_ORDER) == 5

    def test_every_derived_remark_is_in_its_order_list(self):
        assert derive_ph_remark(8.17) in PH_ORDER
        assert derive_om_remark(3.21) in OM_ORDER
        assert derive_p_remark(18.96) in P_ORDER
        assert derive_k_remark(138.0) in K_ORDER

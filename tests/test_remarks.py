import pytest
from ml.remarks import (
    derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark,
    UNKNOWN, PH_ORDER, OM_ORDER, P_ORDER, K_ORDER,
)


class TestGoldenRecordSCC417:
    """The real DA Region 7 report SCC-26-417-417 is the only real labeled
    example available. Every derived remark must match it exactly."""

    def test_ph_8_17_is_moderately_alkaline(self):
        assert derive_ph_remark(8.17) == "MODERATELY ALKALINE"

    def test_om_3_21_is_moderately_low(self):
        assert derive_om_remark(3.21) == "MODERATELY LOW"

    def test_p_18_96_is_high(self):
        assert derive_p_remark(18.96) == "HIGH"

    def test_k_138_is_sufficient_plus(self):
        assert derive_k_remark(138.00) == "SUFFICIENT+"


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
    @pytest.mark.parametrize("value,expected", [
        (0.5, "VERY LOW"), (1.0, "LOW"), (2.0, "MODERATELY LOW"),
        (3.4, "MODERATE"), (5.0, "HIGH"), (7.0, "HIGH"),
    ])
    def test_om_boundaries(self, value, expected):
        assert derive_om_remark(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (2.0, "VERY LOW"), (5.0, "LOW"), (10.0, "MEDIUM"),
        (17.0, "HIGH"), (40.0, "HIGH"),
    ])
    def test_p_boundaries(self, value, expected):
        assert derive_p_remark(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (30.0, "DEFICIENT"), (60.0, "SUFFICIENT"),
        (120.0, "SUFFICIENT+"), (250.0, "SUFFICIENT+"),
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
        assert len(K_ORDER) == 3

    def test_every_derived_remark_is_in_its_order_list(self):
        assert derive_ph_remark(8.17) in PH_ORDER
        assert derive_om_remark(3.21) in OM_ORDER
        assert derive_p_remark(18.96) in P_ORDER
        assert derive_k_remark(138.0) in K_ORDER

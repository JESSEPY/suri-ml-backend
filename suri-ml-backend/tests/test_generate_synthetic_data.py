import pytest
from data.generate_synthetic_data import generate, COLUMNS
from data.crop_baselines import CROPS


@pytest.fixture(scope="module")
def df():
    return generate(num_records=200, seed=42)


class TestShape:
    def test_emits_three_views_per_record(self, df):
        assert len(df) == 200 * 3

    def test_has_exactly_the_declared_columns(self, df):
        assert list(df.columns) == COLUMNS

    def test_every_lab_code_has_all_three_sources(self, df):
        per_code = df.groupby("laboratory_code")["data_source"].apply(set)
        assert all(sources == {"lab", "stk", "iot"} for sources in per_code)

    def test_lab_code_and_source_are_unique_together(self, df):
        assert not df.duplicated(subset=["laboratory_code", "data_source"]).any()


class TestReproducibility:
    def test_same_seed_gives_identical_output(self):
        assert generate(50, seed=7).equals(generate(50, seed=7))

    def test_different_seed_gives_different_output(self):
        assert not generate(50, seed=7).equals(generate(50, seed=8))


class TestTargetsAreSharedAcrossViews:
    def test_all_views_of_a_record_share_identical_targets(self, df):
        targets = ["target_n_kg_ha", "target_p2o5_kg_ha", "target_k2o_kg_ha",
                   "app1_formulation", "app2_formulation"]
        for _, group in df.groupby("laboratory_code"):
            for column in targets:
                assert group[column].nunique() == 1


class TestMissingnessPatternPerSource:
    def test_lab_view_has_the_full_panel(self, df):
        lab = df[df.data_source == "lab"]
        for column in ["ph", "organic_matter_pct", "total_nitrogen_pct",
                       "phosphorus_ppm", "potassium_ppm"]:
            assert lab[column].notna().all()

    def test_lab_and_stk_have_no_sensor_columns(self, df):
        subset = df[df.data_source.isin(["lab", "stk"])]
        for column in ["temp", "moist", "ec", "sal"]:
            assert subset[column].isna().all()

    def test_iot_view_is_missing_om_and_total_nitrogen(self, df):
        """No IoT probe measures organic matter — genuinely absent, not zero."""
        iot = df[df.data_source == "iot"]
        assert iot["organic_matter_pct"].isna().all()
        assert iot["total_nitrogen_pct"].isna().all()

    def test_iot_view_has_sensor_columns(self, df):
        iot = df[df.data_source == "iot"]
        for column in ["temp", "moist", "ec", "sal"]:
            assert iot[column].notna().all()

    def test_stk_view_has_derived_om(self, df):
        """STK's nitrogen colour reading is a proxy for OM banding, so OM and
        Total N are derived rather than missing."""
        stk = df[df.data_source == "stk"]
        assert stk["organic_matter_pct"].notna().all()
        assert stk["total_nitrogen_pct"].notna().all()


class TestValueRanges:
    def test_total_nitrogen_is_five_percent_of_om(self, df):
        present = df[df.organic_matter_pct.notna()]
        expected = present.organic_matter_pct * 0.05
        assert (present.total_nitrogen_pct - expected).abs().max() < 1e-9

    def test_stk_ph_uses_only_the_kit_indicator_values(self, df):
        stk = df[df.data_source == "stk"]
        assert set(stk.ph.unique()).issubset({4.5, 5.5, 6.8, 7.8})

    def test_targets_are_within_clamp_ranges(self, df):
        assert df.target_n_kg_ha.between(20, 200).all()
        assert df.target_p2o5_kg_ha.between(20, 150).all()
        assert df.target_k2o_kg_ha.between(20, 200).all()

    def test_crops_are_from_the_known_set(self, df):
        assert set(df.crop.unique()).issubset(set(CROPS))

    def test_remarks_are_populated_wherever_the_numeric_is(self, df):
        with_ph = df[df.ph.notna()]
        assert (with_ph.ph_remark != "UNKNOWN").all()

    def test_iot_om_remark_is_unknown(self, df):
        iot = df[df.data_source == "iot"]
        assert (iot.om_remark == "UNKNOWN").all()


class TestSignalExists:
    def test_targets_are_not_random_noise(self, df):
        """The original draft synthesizer assigned targets uniformly at random,
        so a model could learn nothing but the mean. Higher available P must
        actually reduce the P2O5 requirement."""
        lab = df[df.data_source == "lab"]
        low_p = lab[lab.phosphorus_ppm < 12].target_p2o5_kg_ha.mean()
        high_p = lab[lab.phosphorus_ppm > 28].target_p2o5_kg_ha.mean()
        assert high_p < low_p

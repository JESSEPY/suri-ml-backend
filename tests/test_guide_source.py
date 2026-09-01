from data.guide_source import GuideRow, GuideSource, build_guide_source


class FakeSupabaseResponse:
    def __init__(self, data):
        self.data = data


class FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def execute(self):
        return FakeSupabaseResponse(self._rows)


class FakeSupabaseClient:
    """Minimal stand-in for supabase.Client -- only .table(name).select().execute()
    is exercised by build_guide_source, so that's all this fakes."""

    def __init__(self, crops, crop_variants, crop_nutrient_targets):
        self._tables = {
            "crops": FakeTable(crops),
            "crop_variants": FakeTable(crop_variants),
            "crop_nutrient_targets": FakeTable(crop_nutrient_targets),
        }

    def table(self, name):
        return self._tables[name]


def _minimal_client() -> FakeSupabaseClient:
    crops = [
        {"id": "c-rice", "value": "Rice", "unit": "kg_ha", "alias_of_crop_id": None},
        {"id": "c-cassava", "value": "Cassava", "unit": "kg_ha", "alias_of_crop_id": None},
        {"id": "c-camote", "value": "Camote", "unit": "kg_ha", "alias_of_crop_id": "c-cassava"},
    ]
    crop_variants = [
        {"crop_id": "c-rice", "dimension": "cultivation_method", "option_value": "Upland", "is_default": False},
        {"crop_id": "c-rice", "dimension": "cultivation_method", "option_value": "Lowland Wet", "is_default": True},
    ]
    crop_nutrient_targets = [
        {
            "crop_id": "c-rice", "variant_key": "Upland",
            "om_bands": [[2.0, 45], [3.5, 40], [4.5, 35], [5.5, 30], [None, 0]],
            "p_bands": [[6.0, 60], [10.0, 40], [15.0, 30], [20.0, 20], [None, 0]],
            "k_bands": [[35.0, 60], [55.0, 45], [250.0, 30], [300.0, 0], [None, 0]],
        },
        {
            "crop_id": "c-rice", "variant_key": "Lowland Wet",
            "om_bands": [[2.0, 90], [3.5, 60], [4.5, 40], [5.5, 35], [None, 20]],
            "p_bands": [[6.0, 60], [10.0, 45], [15.0, 30], [30.0, 20], [None, 0]],
            "k_bands": [[35.0, 60], [55.0, 45], [250.0, 30], [300.0, 20], [None, 0]],
        },
        {
            "crop_id": "c-cassava", "variant_key": None,
            "om_bands": [[2.0, 60], [3.5, 50], [4.5, 40], [5.5, 30], [None, 0]],
            "p_bands": [[6.0, 60], [10.0, 40], [15.0, 30], [20.0, 20], [None, 0]],
            "k_bands": [[35.0, 90], [55.0, 60], [250.0, 45], [300.0, 30], [None, 0]],
        },
    ]
    return FakeSupabaseClient(crops, crop_variants, crop_nutrient_targets)


class TestBuildGuideSource:
    def test_produces_guide_rows_keyed_like_the_old_literal(self):
        source = build_guide_source(_minimal_client())
        assert set(source.guide_rows.keys()) == {"Rice|Upland", "Rice|Lowland Wet", "Cassava"}

    def test_terminal_null_band_becomes_infinity(self):
        source = build_guide_source(_minimal_client())
        row = source.guide_rows["Rice|Upland"]
        assert row.om_bands[-1] == (float("inf"), 0)

    def test_cultivation_methods_by_crop_derived_from_variants(self):
        source = build_guide_source(_minimal_client())
        assert source.cultivation_methods_by_crop == {"Rice": ("Lowland Wet", "Upland")}

    def test_default_cultivation_method_row_derived_from_is_default_flag(self):
        source = build_guide_source(_minimal_client())
        assert source.default_cultivation_method_row == {"Rice": "Lowland Wet"}

    def test_alias_resolved_by_value_not_id(self):
        source = build_guide_source(_minimal_client())
        assert source.crop_aliases == {"Camote": "Cassava"}

    def test_target_unit_read_directly_from_crops_table(self):
        source = build_guide_source(_minimal_client())
        assert source.target_unit == {"Rice": "kg_ha", "Cassava": "kg_ha"}

    def test_returns_a_guide_source_instance(self):
        assert isinstance(build_guide_source(_minimal_client()), GuideSource)
        assert isinstance(build_guide_source(_minimal_client()).guide_rows["Cassava"], GuideRow)

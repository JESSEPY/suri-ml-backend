"""reload_guide() itself (as opposed to build_guide_source(), covered by
tests/test_guide_source.py): specifically, that a bad Supabase read leaves
the previously-loaded in-memory guide untouched rather than wiping it out.

reload_guide() achieves this for free, not by catching anything -- it builds
the new GuideSource entirely (via build_guide_source(), which now validates
that every crop with cultivation_method options has a default, see
data/guide_source.py) before reassigning any module global, so a validation
failure never reaches the reassignment step. This test exists to pin that
ordering so it can't regress silently.
"""

import pytest

import data.nutrient_recommendation_guide as guide
from tests.guide_fixture import FakeSupabaseClient


def _client_with_undefaulted_cultivation_method() -> FakeSupabaseClient:
    crops = [{"id": "c-rice", "value": "Rice", "unit": "kg_ha", "alias_of_crop_id": None}]
    crop_variants = [
        {"crop_id": "c-rice", "dimension": "cultivation_method", "option_value": "Upland", "is_default": False},
    ]
    return FakeSupabaseClient(crops, crop_variants, [])


class TestReloadGuideFailureLeavesPreviousStateIntact:
    def test_a_bad_reload_raises(self):
        with pytest.raises(ValueError, match="No default cultivation_method"):
            guide.reload_guide(client=_client_with_undefaulted_cultivation_method())

    def test_a_bad_reload_does_not_wipe_guide_rows(self):
        guide_rows_before = guide.GUIDE_ROWS
        crops_before = guide.CROPS

        with pytest.raises(ValueError):
            guide.reload_guide(client=_client_with_undefaulted_cultivation_method())

        assert guide.GUIDE_ROWS is guide_rows_before
        assert guide.CROPS is crops_before
        assert guide.GUIDE_ROWS
        assert guide.CROPS

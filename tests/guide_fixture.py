"""Loads tests/fixtures/crop_reference_seed.csv into a fake Supabase client
shaped like data.guide_source.build_guide_source() expects, so
tests/conftest.py can populate data.nutrient_recommendation_guide's module
state (GUIDE_ROWS, CROPS, etc.) without a live Supabase connection.

The fixture CSV is the same historical ~85-crop dataset that used to be a
hardcoded Python literal in nutrient_recommendation_guide.py, generated once
via scripts/export_guide_to_csv.py before that literal was deleted (see
docs/superpowers/plans/2026-09-01-crop-nutrient-reference-cms.md Task 3/4).
Unlike data/crop_reference_seed.csv (gitignored, regenerable, used for the
real one-time production seed upload), this copy under tests/fixtures/ is
committed -- *.csv is gitignored repo-wide except tests/fixtures/*.csv.
"""

from __future__ import annotations

import csv
import os
import uuid

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "crop_reference_seed.csv")

OM_BAND_COUNT = 5
P_BAND_COUNT = 6
K_BAND_COUNT = 5


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeResponse(self._rows)


class FakeSupabaseClient:
    """Minimal stand-in for supabase.Client -- only .table(name).select().execute()
    is exercised by build_guide_source, so that's all this fakes."""

    def __init__(self, crops, crop_variants, crop_nutrient_targets):
        self._tables = {
            "crops": _FakeTable(crops),
            "crop_variants": _FakeTable(crop_variants),
            "crop_nutrient_targets": _FakeTable(crop_nutrient_targets),
        }

    def table(self, name):
        return self._tables[name]


def _parse_band_pairs(cells: list[str], count: int) -> list[list]:
    pairs = []
    for i in range(count):
        bound_raw = cells[i * 2].strip()
        rate_raw = cells[i * 2 + 1].strip()
        if not rate_raw:
            continue
        pairs.append([float(bound_raw) if bound_raw else None, float(rate_raw)])
    return pairs


def load_fixture_client() -> FakeSupabaseClient:
    with open(FIXTURE_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header[:6] == ["crop", "alias_of", "dimension", "variant_key", "is_default", "unit"], (
        f"tests/fixtures/crop_reference_seed.csv header drifted from the expected shape: {header[:6]}"
    )

    crop_id_by_value: dict[str, str] = {}
    crops: list[dict] = []
    crop_variants: list[dict] = []
    crop_nutrient_targets: list[dict] = []
    alias_rows: list[tuple[str, str]] = []

    om_end = 6 + OM_BAND_COUNT * 2
    p_end = om_end + P_BAND_COUNT * 2
    k_end = p_end + K_BAND_COUNT * 2

    for row in rows:
        if not row:
            continue
        crop, alias_of, dimension, variant_key, is_default_raw, unit = row[:6]
        band_cells = row[6:k_end]

        if alias_of:
            alias_rows.append((crop, alias_of))
            continue

        crop_id = crop_id_by_value.setdefault(crop, str(uuid.uuid4()))
        if unit and not any(c["id"] == crop_id for c in crops):
            crops.append({"id": crop_id, "value": crop, "unit": unit, "alias_of_crop_id": None})

        if dimension and variant_key:
            crop_variants.append({
                "crop_id": crop_id, "dimension": dimension, "option_value": variant_key,
                "is_default": is_default_raw.strip().lower() == "true",
            })

        om_bands = _parse_band_pairs(band_cells[0:om_end - 6], OM_BAND_COUNT)
        p_bands = _parse_band_pairs(band_cells[om_end - 6:p_end - 6], P_BAND_COUNT)
        k_bands = _parse_band_pairs(band_cells[p_end - 6:k_end - 6], K_BAND_COUNT)
        has_band_data = om_bands or p_bands or k_bands
        if not has_band_data:
            # Declaration-only row (e.g. Rice's variety axis: tracked as an
            # ML feature, but doesn't determine which GUIDE_ROWS row is used
            # -- that's cultivation_method for Rice). Registers a
            # crop_variants entry above; must NOT also become a
            # crop_nutrient_targets row, or build_guide_source() would
            # synthesize a bogus GUIDE_ROWS["Rice|Hybrid"] with empty bands.
            continue

        crop_nutrient_targets.append({
            "crop_id": crop_id,
            "variant_key": variant_key or None,
            "om_bands": om_bands, "p_bands": p_bands, "k_bands": k_bands,
        })

    for alias_crop, canonical in alias_rows:
        canonical_id = crop_id_by_value[canonical]
        crops.append({
            "id": str(uuid.uuid4()), "value": alias_crop, "unit": "kg_ha",
            "alias_of_crop_id": canonical_id,
        })

    return FakeSupabaseClient(crops, crop_variants, crop_nutrient_targets)

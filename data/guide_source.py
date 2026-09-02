"""Builds the guide data structures data/nutrient_recommendation_guide.py's
GUIDE_ROWS/CROP_ALIASES/VARIETIES_BY_CROP/etc. used to be hardcoded literals
for, from the crops/crop_variants/crop_nutrient_targets tables in Supabase
(schema: docs/superpowers/specs/2026-09-01-crop-nutrient-reference-cms-design.md).
suri's admin CMS is the source of truth for those tables; this module only
reads them.

Isolated from nutrient_recommendation_guide.py so it's testable against a
fake Supabase client (see tests/test_guide_source.py) without a real
network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuideRow:
    unit: str
    om_bands: tuple[tuple[float, float], ...]
    p_bands: tuple[tuple[float, float], ...]
    k_bands: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class GuideSource:
    guide_rows: dict[str, GuideRow] = field(default_factory=dict)
    crop_aliases: dict[str, str] = field(default_factory=dict)
    varieties_by_crop: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cultivation_methods_by_crop: dict[str, tuple[str, ...]] = field(default_factory=dict)
    default_variety_row: dict[str, str] = field(default_factory=dict)
    default_cultivation_method_row: dict[str, str] = field(default_factory=dict)
    bearing_age_band: dict[str, str] = field(default_factory=dict)
    target_unit: dict[str, str] = field(default_factory=dict)


def _bands_from_json(raw: list) -> tuple[tuple[float, float], ...]:
    """A JSON null upper bound means 'no further class beyond this one' --
    converts to Python's float("inf"), matching GuideRow's existing
    convention exactly (see the terminal band in every literal row today)."""
    return tuple(
        (float("inf") if upper is None else float(upper), float(rate))
        for upper, rate in raw
    )


def build_guide_source(client) -> GuideSource:
    crops_rows = client.table("crops").select("id, value, unit, alias_of_crop_id").execute().data
    variant_rows = (
        client.table("crop_variants")
        .select("crop_id, dimension, option_value, is_default")
        .execute()
        .data
    )
    target_rows = (
        client.table("crop_nutrient_targets")
        .select("crop_id, variant_key, om_bands, p_bands, k_bands")
        .execute()
        .data
    )

    crop_by_id = {row["id"]: row for row in crops_rows}
    value_by_id = {row["id"]: row["value"] for row in crops_rows}

    guide_rows: dict[str, GuideRow] = {}
    for row in target_rows:
        crop_value = value_by_id[row["crop_id"]]
        key = f"{crop_value}|{row['variant_key']}" if row["variant_key"] else crop_value
        if key in guide_rows:
            raise ValueError(f"Duplicate crop_nutrient_targets row for {key!r}")
        guide_rows[key] = GuideRow(
            unit=crop_by_id[row["crop_id"]]["unit"],
            om_bands=_bands_from_json(row["om_bands"]),
            p_bands=_bands_from_json(row["p_bands"]),
            k_bands=_bands_from_json(row["k_bands"]),
        )

    crop_aliases = {
        row["value"]: value_by_id[row["alias_of_crop_id"]]
        for row in crops_rows
        if row["alias_of_crop_id"] is not None
    }

    varieties_by_crop: dict[str, list[str]] = {}
    cultivation_methods_by_crop: dict[str, list[str]] = {}
    default_variety_row: dict[str, str] = {}
    default_cultivation_method_row: dict[str, str] = {}
    bearing_age_band: dict[str, str] = {}

    for variant in variant_rows:
        crop_value = value_by_id[variant["crop_id"]]
        dimension = variant["dimension"]
        option = variant["option_value"]

        if dimension == "variety":
            varieties_by_crop.setdefault(crop_value, []).append(option)
            if variant["is_default"]:
                default_variety_row[crop_value] = option
        elif dimension == "cultivation_method":
            cultivation_methods_by_crop.setdefault(crop_value, []).append(option)
            if variant["is_default"]:
                default_cultivation_method_row[crop_value] = option
        elif dimension == "age_band":
            if variant["is_default"]:
                bearing_age_band[crop_value] = option
        else:
            raise ValueError(f"Unknown crop_variants.dimension {dimension!r} for crop {crop_value!r}")

    # The DB schema's partial unique index enforces AT MOST one is_default=true
    # row per (crop_id, dimension), never AT LEAST one -- a crop can end up
    # with cultivation_method options and no default marked. Unlike variety
    # (nutrient_recommendation_guide.py's _guide_row_key only ever consults
    # DEFAULT_VARIETY_ROW for a crop already known to be in it, so a variety
    # dimension with no default is a legitimate, harmless shape -- Rice's
    # fixture data is exactly this: variety tracked as an ML feature, with
    # cultivation_method as the actual GUIDE_ROWS-keying dimension),
    # resolve_cultivation_method() looks up DEFAULT_CULTIVATION_METHOD_ROW[canonical]
    # unconditionally for any crop with cultivation_method options. An admin
    # data-entry gap here must fail the reload loudly (leaving the previous
    # good in-memory guide in place -- reload_guide() only reassigns module
    # state after this function returns successfully) rather than surface as
    # a KeyError mid-request later.
    missing_cultivation_method_defaults = sorted(
        set(cultivation_methods_by_crop) - set(default_cultivation_method_row)
    )
    if missing_cultivation_method_defaults:
        raise ValueError(
            f"No default cultivation_method marked for: {', '.join(missing_cultivation_method_defaults)}"
        )

    target_unit = {row["value"]: row["unit"] for row in crops_rows if row["alias_of_crop_id"] is None}

    return GuideSource(
        guide_rows=guide_rows,
        crop_aliases=crop_aliases,
        varieties_by_crop={k: tuple(sorted(v)) for k, v in varieties_by_crop.items()},
        cultivation_methods_by_crop={k: tuple(sorted(v)) for k, v in cultivation_methods_by_crop.items()},
        default_variety_row=default_variety_row,
        default_cultivation_method_row=default_cultivation_method_row,
        bearing_age_band=bearing_age_band,
        target_unit=target_unit,
    )

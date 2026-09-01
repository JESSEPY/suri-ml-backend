#!/usr/bin/env python3
"""One-off: dumps the current hardcoded guide data (GUIDE_ROWS, CROP_ALIASES,
VARIETIES_BY_CROP, CULTIVATION_METHODS_BY_CROP, DEFAULT_VARIETY_ROW,
DEFAULT_CULTIVATION_METHOD_ROW, BEARING_AGE_BAND, TARGET_UNIT) into the CSV
shape suri's bulk crop-reference upload expects. Must run BEFORE the
literals are deleted (see docs/superpowers/plans/2026-09-01-crop-nutrient-reference-cms.md
Task 4) -- this is the one-time migration path for today's ~85 crops.

Usage: python scripts/export_guide_to_csv.py > data/crop_reference_seed.csv
"""

import csv
import sys

from data.nutrient_recommendation_guide import (
    BEARING_AGE_BAND, CROP_ALIASES, CULTIVATION_METHODS_BY_CROP,
    DEFAULT_CULTIVATION_METHOD_ROW, DEFAULT_VARIETY_ROW, GUIDE_ROWS,
    TARGET_UNIT, VARIETIES_BY_CROP,
)

# Explicit (bound, rate) pairs, not rate-only columns against an assumed
# shared boundary sequence -- REVISED from the plan's original rates-only
# design after this script's first run against real data showed WHICH
# column collapses (two adjacent guide-table classes sharing one printed
# rate) varies per crop row. Rice|Upland's 5-band P axis uses bounds
# 6/10/15/20/inf (collapses the 20.1-30.0 class into the terminal one);
# Rice|Lowland Wet's 5-band P axis uses bounds 6/10/15/30/inf (collapses
# 15.1-20.0 into the 20.1-30.0 class instead) -- see
# nutrient_recommendation_guide.py's module docstring. A fixed shared
# boundary sequence cannot represent both correctly; explicit per-band
# bounds can, and need no reconstruction logic on the reading side.
HEADER = [
    "crop", "alias_of", "dimension", "variant_key", "is_default", "unit",
    "om_b1", "om_r1", "om_b2", "om_r2", "om_b3", "om_r3", "om_b4", "om_r4", "om_b5", "om_r5",
    "p_b1", "p_r1", "p_b2", "p_r2", "p_b3", "p_r3", "p_b4", "p_r4", "p_b5", "p_r5", "p_b6", "p_r6",
    "k_b1", "k_r1", "k_b2", "k_r2", "k_b3", "k_r3", "k_b4", "k_r4", "k_b5", "k_r5",
]


def _default_for(crop: str, dimension: str) -> str | None:
    if dimension == "variety":
        return DEFAULT_VARIETY_ROW.get(crop)
    if dimension == "cultivation_method":
        return DEFAULT_CULTIVATION_METHOD_ROW.get(crop)
    if dimension == "age_band":
        return BEARING_AGE_BAND.get(crop)
    return None


def _dimension_for(crop: str) -> str | None:
    if crop in CULTIVATION_METHODS_BY_CROP:
        return "cultivation_method"
    if crop in DEFAULT_VARIETY_ROW:
        return "variety"
    if crop in BEARING_AGE_BAND:
        return "age_band"
    return None


def _band_pairs(bands: tuple, count: int) -> list[str]:
    """Each band is (upper_bound, rate); the terminal band's upper_bound is
    always float("inf") and is written as an empty bound cell (infinity
    isn't valid CSV/JSON). A crop with fewer real bands than `count` leaves
    its TRAILING pairs both blank -- unlike a rates-only encoding, leaving
    trailing pairs blank never loses boundary information, since every
    populated pair carries its own real bound explicitly."""
    cells: list[str] = []
    for i in range(count):
        if i < len(bands):
            upper, rate = bands[i]
            cells.append("" if upper == float("inf") else str(upper))
            cells.append(str(rate))
        else:
            cells.append("")
            cells.append("")
    return cells


def main() -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(HEADER)

    for key, row in GUIDE_ROWS.items():
        if "|" in key:
            crop, variant_key = key.split("|", 1)
        else:
            crop, variant_key = key, ""

        dimension = _dimension_for(crop) if variant_key else ""
        is_default = ""
        if dimension:
            is_default = "true" if _default_for(crop, dimension) == variant_key else "false"

        writer.writerow([
            crop, "", dimension, variant_key, is_default, row.unit,
            *_band_pairs(row.om_bands, 5),
            *_band_pairs(row.p_bands, 6),
            *_band_pairs(row.k_bands, 5),
        ])

    # Declaration-only rows: a crop can track a variety axis for ML feature
    # purposes WITHOUT that axis determining which GUIDE_ROWS row is used --
    # Rice is the one case today (its row is picked by cultivation_method;
    # variety (Hybrid/Inbred) is tracked in VARIETIES_BY_CROP purely as an
    # additional one-hot feature, never referenced by _guide_row_key). Such
    # a crop's variety options would otherwise never be exported at all,
    # since the loop above only emits ONE dimension per crop (whichever one
    # _dimension_for() says determines its guide row). These rows carry no
    # band data and no unit -- they only register crop_variants entries.
    for crop, varieties in sorted(VARIETIES_BY_CROP.items()):
        if _dimension_for(crop) == "variety":
            continue  # already covered by the per-target-row loop above
        for variety in varieties:
            writer.writerow([crop, "", "variety", variety, "false", "", *([""] * 32)])

    for alias, canonical in sorted(CROP_ALIASES.items()):
        writer.writerow([alias, canonical, "", "", "", "", *([""] * 32)])


if __name__ == "__main__":
    main()

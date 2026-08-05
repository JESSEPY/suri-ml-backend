"""Soil-test to fertilizer requirement lookup.

nutrient_requirements() is a thin delegate to
data/nutrient_recommendation_guide.py's transcribed BSWM table — see that
module for the actual band-lookup mechanism and its known transcription
caveats. This file's only remaining independent logic is
total_nitrogen_from_om(), which is unrelated to fertilizer targets: it's
how the lab itself derives a Total Nitrogen percentage when a report only
has Organic Matter.
"""

from __future__ import annotations

from data.nutrient_recommendation_guide import nutrient_requirements as _guide_lookup

# Total Nitrogen is calculated by the lab as OM x 0.05, never measured directly.
OM_TO_TOTAL_N = 0.05


def total_nitrogen_from_om(om: float) -> float:
    return om * OM_TO_TOTAL_N


def nutrient_requirements(
    crop: str, om: float, p_ppm: float, k_cmol_kg: float,
    variety: str | None = None, cultivation_method: str | None = None,
    age_band: str | None = None,
) -> tuple[float, float, float]:
    """Return (n, p2o5, k2o) — kg/ha for field crops, kg/tree for
    Coconut/Cacao/Mango/Banana (see
    data.nutrient_recommendation_guide.target_unit_for_crop). Floats, not
    ints: several tree-crop rates are sub-1 kg/tree. Raises KeyError for an
    unknown crop."""
    return _guide_lookup(
        crop, om, p_ppm, k_cmol_kg,
        variety=variety, cultivation_method=cultivation_method, age_band=age_band,
    )

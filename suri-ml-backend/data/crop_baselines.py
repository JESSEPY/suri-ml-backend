"""Per-crop baseline nutrient requirements in kg/ha, before soil-test adjustment.

Sorghum is anchored exactly to the real DA report SCC-26-417-417 (50/40/60).
The other five are conventional Philippine baseline rates used as placeholder
scaffolding while awaiting the real DA historical export. They set the model's
central tendency per crop, so they need agronomist review before any production
use — which is why they are isolated here: correcting them is a single edit plus
a retrain, with no code changes anywhere else.
"""

from __future__ import annotations

# crop -> (base_n, base_p2o5, base_k2o) in kg/ha
CROP_BASELINES: dict[str, tuple[int, int, int]] = {
    "Sorghum": (50, 40, 60),
    "Corn": (90, 60, 60),
    "Rice": (90, 30, 30),
    "Cabbage": (120, 90, 90),
    "Tomato": (100, 80, 80),
    "Sugarcane": (120, 60, 120),
}

CROPS: list[str] = sorted(CROP_BASELINES.keys())

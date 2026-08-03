"""Soil-test to fertilizer requirement formulas (spec section 6.2).

Each nutrient requirement is its crop baseline adjusted by what the soil already
supplies: organic matter is the nitrogen reserve, and measured available P and
exchangeable K offset the P2O5 and K2O requirements respectively.
"""

from __future__ import annotations

from data.crop_baselines import CROP_BASELINES

# Total Nitrogen is calculated by the lab as OM x 0.05, never measured directly.
OM_TO_TOTAL_N = 0.05

N_MIN, N_MAX = 20, 200
P_MIN, P_MAX = 20, 150
K_MIN, K_MAX = 20, 200

_PH_OPTIMUM = 6.5
_PH_PENALTY_MAX = 0.30


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def total_nitrogen_from_om(om: float) -> float:
    return om * OM_TO_TOTAL_N


def ph_p_penalty(ph: float) -> float:
    """Phosphorus availability peaks near pH 6.5 and falls off in both
    directions — fixed by iron/aluminium in acid soils, by calcium in alkaline
    soils. Modeled as an inverted-U adding up to 30% to the P2O5 requirement.

    This non-monotonic interaction between two input features is the specific
    reason the architecture needs XGBoost rather than a purely linear model.
    """
    return 1.0 + _PH_PENALTY_MAX * min((ph - _PH_OPTIMUM) ** 2 / 4.0, 1.0)


def nutrient_requirements(
    crop: str, ph: float, om: float, p_ppm: float, k_ppm: float
) -> tuple[int, int, int]:
    """Return (n, p2o5, k2o) in whole kg/ha, matching how the real reports
    present them. Raises KeyError for an unknown crop."""
    base_n, base_p, base_k = CROP_BASELINES[crop]

    n = base_n * (1 - 0.35 * _clamp((om - 2.5) / 3.5, -1, 1))
    p = base_p * (1 - 0.40 * _clamp((p_ppm - 15) / 25, -1, 1)) * ph_p_penalty(ph)
    k = base_k * (1 - 0.40 * _clamp((k_ppm - 120) / 130, -1, 1))

    return (
        int(round(_clamp(n, N_MIN, N_MAX))),
        int(round(_clamp(p, P_MIN, P_MAX))),
        int(round(_clamp(k, K_MIN, K_MAX))),
    )

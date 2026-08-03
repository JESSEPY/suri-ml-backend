"""Derivation of DA RSL qualitative remarks from numeric soil values.

Band tables per spec section 4. Boundaries use [lower, upper) semantics:
a value is placed in the first band whose upper bound it falls strictly below.

Provenance: the pH classes are the standard USDA/BSWM soil reaction classes and
the Olsen-P classes are the standard Olsen interpretation; both independently
reproduce the real SCC-417 report without having been fitted to it. The OM and K
bands are calibrated to that single real example and should be confirmed against
DA RSL's published interpretation chart before production use. Changing any
boundary is a one-constant edit here plus a retrain.
"""

from __future__ import annotations

UNKNOWN = "UNKNOWN"

# (exclusive upper bound, label), ascending. The final entry catches the rest.
_PH_BANDS = [
    (4.5, "STRONGLY ACIDIC"),
    (5.6, "MODERATELY ACIDIC"),
    (6.6, "SLIGHTLY ACIDIC"),
    (7.4, "NEUTRAL"),
    (7.9, "SLIGHTLY ALKALINE"),
    (8.5, "MODERATELY ALKALINE"),
    (float("inf"), "STRONGLY ALKALINE"),
]

_OM_BANDS = [
    (1.0, "VERY LOW"),
    (2.0, "LOW"),
    (3.4, "MODERATELY LOW"),
    (5.0, "MODERATE"),
    (float("inf"), "HIGH"),
]

_P_BANDS = [
    (5.0, "VERY LOW"),
    (10.0, "LOW"),
    (17.0, "MEDIUM"),
    (float("inf"), "HIGH"),
]

_K_BANDS = [
    (60.0, "DEFICIENT"),
    (120.0, "SUFFICIENT"),
    (float("inf"), "SUFFICIENT+"),
]

# Ascending category order for Task 6's OrdinalEncoder. UNKNOWN is deliberately
# absent — it is encoded as -1 via handle_unknown="use_encoded_value".
PH_ORDER = [label for _, label in _PH_BANDS]
OM_ORDER = [label for _, label in _OM_BANDS]
P_ORDER = [label for _, label in _P_BANDS]
K_ORDER = [label for _, label in _K_BANDS]


def _band(value: float | None, bands: list[tuple[float, str]]) -> str:
    if value is None:
        return UNKNOWN
    for upper, label in bands:
        if value < upper:
            return label
    return bands[-1][1]


def derive_ph_remark(ph: float | None) -> str:
    return _band(ph, _PH_BANDS)


def derive_om_remark(om: float | None) -> str:
    return _band(om, _OM_BANDS)


def derive_p_remark(p_ppm: float | None) -> str:
    return _band(p_ppm, _P_BANDS)


def derive_k_remark(k_ppm: float | None) -> str:
    return _band(k_ppm, _K_BANDS)

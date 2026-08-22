"""Derivation of DA RSL qualitative remarks from numeric soil values.

Band tables per spec section 4. Boundaries use [lower, upper) semantics:
a value is placed in the first band whose upper bound it falls strictly below.

Provenance: the pH classes are the standard USDA/BSWM soil reaction classes
and the Olsen-P classes are the standard Olsen interpretation.

The OM and K bands are DA's real numeric bins from the official BSWM
Nutrient Recommendation Guide (see `Nutrient Recommendation Guide.md` in
this repo's root, Table 3: "% OM" and "Extractable K" column headers) --
confirmed as DA's actual operational numbers, not just a reference table,
against a real DA-RFO7 Regional Soils Laboratory Fertilizer Recommendation
Report (laboratory_code SCC 417 / Soil Chemical Test Report SCC-26-417-417,
Sorghum, N=50): Table 3's Corn/Sorghum/Native row gives N=50 for exactly
the 2nd of 5 OM bands (2.1-3.5%), matching that real report's output.
Table 3 attaches a fertilizer *rate* to each bin, not a word -- so while the
five OM bin *boundaries* below are DA's, and the five K bin *boundaries* are
DA's, the *label text* for each bin (VERY LOW/LOW/MODERATELY LOW/MODERATE/
HIGH for OM; DEFICIENT/LOW/SUFFICIENT/SUFFICIENT+/EXCESSIVE for K) is this
project's own naming, not something DA's table itself prints. Changing any
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

# DA Nutrient Recommendation Guide Table 3's real OM bins (0-2.0, 2.1-3.5,
# 3.6-4.5, 4.6-5.5, >5.6) -- each entry's upper bound is the printed start
# of the *next* bin, so e.g. 2.0 still resolves to "VERY LOW".
_OM_BANDS = [
    (2.1, "VERY LOW"),
    (3.6, "LOW"),
    (4.6, "MODERATELY LOW"),
    (5.6, "MODERATE"),
    (float("inf"), "HIGH"),
]

_P_BANDS = [
    (5.0, "VERY LOW"),
    (10.0, "LOW"),
    (17.0, "MEDIUM"),
    (float("inf"), "HIGH"),
]

# DA Nutrient Recommendation Guide Table 3's real Extractable K bins
# (0-35, 36-55, 56-250, 251-300, >301 ppm), same next-bin-start-as-
# exclusive-upper-bound convention as OM above.
_K_BANDS = [
    (36.0, "DEFICIENT"),
    (56.0, "LOW"),
    (251.0, "SUFFICIENT"),
    (301.0, "SUFFICIENT+"),
    (float("inf"), "EXCESSIVE"),
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

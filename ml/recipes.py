"""Fertilizer product catalog and application planner.

Deterministic — not a trained classifier. Converts a known nutrient target
into product-bag amounts by covering the P2O5 requirement with a Complete
(14-14-14) base, then topping up nitrogen with Ammosul and potassium with
Muriate of Potash, splitting the total across a basal application at
planting and a topdress one month later — the structure of a real DA
Fertilizer Recommendation Report. This file's own docstring previously
claimed provenance from a specific real report (SCC-26-417-417); that
record was traced to a document outside this system's real dataset and
retired (see docs/superpowers/specs/2026-08-05-real-soil-data-retraining-design.md's
"Golden test — retired" section) — the arithmetic here doesn't depend on
that record, so the claim is removed rather than corrected to a new one.
"""

from __future__ import annotations

NO_APPLICATION = "None"

# Standard Philippine fertilizer grades as (N%, P2O5%, K2O%).
COMPLETE = ("14-14-14", 0.14, 0.14, 0.14)
AMMOSUL = ("21-0-0", 0.21, 0.0, 0.0)
MOP = ("0-0-60", 0.0, 0.0, 0.60)

# Below these combined kg/ha requirements a single basal application is
# sufficient. Calibrated to the kg_ha scale only -- see plan_applications
# for why kg_tree always splits into two instead of using a scaled version
# of these.
_SINGLE_APPLICATION_MAX_N = 30
_SINGLE_APPLICATION_MAX_P = 25
_SINGLE_APPLICATION_MAX_K = 30

_NUM_APPLICATIONS = 2

_UNIT_LABEL = {"kg_ha": "kg/ha", "kg_tree": "kg/tree"}


def _format_terms(terms: list[tuple[str, float]], target_unit: str) -> str:
    """Render [(grade, kg)] as '143 kg/ha (14-14-14) + 24 kg/ha (21-0-0)' for
    kg_ha, or '0.947 kg/tree (14-14-14)' for kg_tree -- kg_ha amounts round
    to whole numbers (matching the real DA report format this mirrors);
    kg_tree amounts stay at 3-decimal precision (matching
    nutrient_requirements()'s own rounding) since real kg_tree amounts are
    legitimately sub-1 and a whole-number round would drop every one of
    them. Terms are dropped only when they round to exactly zero at their
    own unit's precision, not the other unit's."""
    label = _UNIT_LABEL[target_unit]
    parts = []
    for grade, kg in terms:
        if target_unit == "kg_ha":
            amount = int(round(kg))
        else:
            amount = round(kg, 3)
        if amount > 0:
            parts.append(f"{amount} {label} ({grade})")
    return " + ".join(parts) if parts else NO_APPLICATION


def plan_applications(n: float, p2o5: float, k2o: float, target_unit: str) -> tuple[str, str]:
    """Return (app1_formulation, app2_formulation) for the given nutrient
    targets. target_unit ("kg_ha" or "kg_tree", from
    data.nutrient_recommendation_guide.target_unit_for_crop) controls both
    the formatted unit label and the application-count logic."""
    complete_grade, complete_n, _, complete_k = COMPLETE

    kg_complete = p2o5 / COMPLETE[2]
    n_supplied = kg_complete * complete_n
    k_supplied = kg_complete * complete_k

    kg_ammosul = max(0.0, n - n_supplied) / AMMOSUL[1]
    kg_mop = max(0.0, k2o - k_supplied) / MOP[3]

    if target_unit == "kg_ha":
        single_application = (
            n <= _SINGLE_APPLICATION_MAX_N
            and p2o5 <= _SINGLE_APPLICATION_MAX_P
            and k2o <= _SINGLE_APPLICATION_MAX_K
        )
    else:
        # kg_tree crops (Coconut/Cacao/Mango/Banana): every crop's Mode of
        # Application guidance (data/mode_of_application.py) describes at
        # least two applications per year regardless of amount -- the
        # kg_ha thresholds above don't transfer to this scale (every real
        # kg_tree value is far under all three), so always split.
        single_application = False
    splits = 1 if single_application else _NUM_APPLICATIONS

    per_application = [
        (complete_grade, kg_complete / splits),
        (AMMOSUL[0], kg_ammosul / splits),
        (MOP[0], kg_mop / splits),
    ]
    formulation = _format_terms(per_application, target_unit)

    if single_application:
        return formulation, NO_APPLICATION
    return formulation, formulation

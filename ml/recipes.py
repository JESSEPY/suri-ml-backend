"""Fertilizer product catalog and application planner (spec section 6.3).

The planner covers the P2O5 requirement with a Complete (14-14-14) base, then
tops up nitrogen with Ammosul and potassium with Muriate of Potash, splitting
the total evenly across a basal application at planting and a topdress one month
later — the structure of the real DA Fertilizer Recommendation Report.

Validated against the real SCC-417 report: inputs (50, 40, 60) reproduce its
"143 kg/ha (14-14-14) + 24 kg/ha (21-0-0) + 17 kg/ha (0-0-60)" exactly.
"""

from __future__ import annotations

NO_APPLICATION = "None"

# Standard Philippine fertilizer grades as (N%, P2O5%, K2O%).
COMPLETE = ("14-14-14", 0.14, 0.14, 0.14)
AMMOSUL = ("21-0-0", 0.21, 0.0, 0.0)
MOP = ("0-0-60", 0.0, 0.0, 0.60)

# Below these combined requirements a single basal application is sufficient.
_SINGLE_APPLICATION_MAX_N = 30
_SINGLE_APPLICATION_MAX_P = 25
_SINGLE_APPLICATION_MAX_K = 30

_NUM_APPLICATIONS = 2


def _format_terms(terms: list[tuple[str, float]]) -> str:
    """Render [(grade, kg)] as '143 kg/ha (14-14-14) + 24 kg/ha (21-0-0)',
    dropping any term that rounds to zero."""
    parts = [
        f"{int(round(kg))} kg/ha ({grade})"
        for grade, kg in terms
        if int(round(kg)) > 0
    ]
    return " + ".join(parts) if parts else NO_APPLICATION


def plan_applications(n: int, p2o5: int, k2o: int) -> tuple[str, str]:
    """Return (app1_formulation, app2_formulation) for the given kg/ha targets."""
    complete_grade, complete_n, _, complete_k = COMPLETE

    kg_complete = p2o5 / COMPLETE[2]
    n_supplied = kg_complete * complete_n
    k_supplied = kg_complete * complete_k

    kg_ammosul = max(0.0, n - n_supplied) / AMMOSUL[1]
    kg_mop = max(0.0, k2o - k_supplied) / MOP[3]

    single_application = (
        n <= _SINGLE_APPLICATION_MAX_N
        and p2o5 <= _SINGLE_APPLICATION_MAX_P
        and k2o <= _SINGLE_APPLICATION_MAX_K
    )
    splits = 1 if single_application else _NUM_APPLICATIONS

    per_application = [
        (complete_grade, kg_complete / splits),
        (AMMOSUL[0], kg_ammosul / splits),
        (MOP[0], kg_mop / splits),
    ]
    formulation = _format_terms(per_application)

    if single_application:
        return formulation, NO_APPLICATION
    return formulation, formulation

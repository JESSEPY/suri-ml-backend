"""Confidence tier computation for ML-ensemble fertilizer predictions.

Normalized error (MAE / mean actual value) instead of raw R2 -- R2 is
mathematically unstable on the low-variance tree-crop subgroups (Banana,
Cacao show R2 in the -200 to -700 range from tiny test-set variance, not
because the model is uniquely broken there). See
docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md.

Thresholds (15%/40% tier bands, 50% disagreement downgrade trigger used by
main.py) are an explicit, isolated policy choice -- not a derived truth.
Change them here; nothing else needs to know why.
"""

from __future__ import annotations

from typing import Literal

Tier = Literal["high", "fair", "limited"]

HIGH_CONFIDENCE_MAX_ERROR = 0.15
FAIR_CONFIDENCE_MAX_ERROR = 0.40
DISAGREEMENT_DOWNGRADE_THRESHOLD = 0.50

# Displayed pct is reclamped to a tier's ceiling after a Tier-2 downgrade
# (main.py) so the number never contradicts the word next to it -- e.g. a
# downgrade from "high" to "fair" must not still show "92%".
TIER_PCT_CEILING: dict[Tier, int] = {"high": 100, "fair": 85, "limited": 60}

_DOWNGRADE: dict[Tier, Tier] = {"high": "fair", "fair": "limited", "limited": "limited"}


def confidence_from_metrics(mae: float, mean_actual: float) -> tuple[Tier, int]:
    """(tier, displayed percentage) from a crop/target's held-out MAE and
    mean actual value. mean_actual <= 0 is defensive (shouldn't happen for
    a real fertilizer target) and returns the most conservative answer
    rather than dividing by zero."""
    if mean_actual <= 0:
        return "limited", 5

    normalized_error = mae / mean_actual
    pct = round(max(5, min(100, 100 - normalized_error * 100)))

    if normalized_error <= HIGH_CONFIDENCE_MAX_ERROR:
        return "high", pct
    if normalized_error <= FAIR_CONFIDENCE_MAX_ERROR:
        return "fair", pct
    return "limited", pct


def downgrade_tier(tier: Tier) -> Tier:
    return _DOWNGRADE[tier]

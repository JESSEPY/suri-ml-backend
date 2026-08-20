"""Confidence tier computation for ML-ensemble fertilizer predictions.

Normalized error (MAE / mean actual value) instead of raw R2 -- R2 is
mathematically unstable on the low-variance tree-crop subgroups (Banana,
Cacao show R2 in the -200 to -700 range from tiny test-set variance, not
because the model is uniquely broken there). See
docs/superpowers/specs/2026-08-12-ml-prediction-confidence-design.md.

Thresholds (15%/40% tier bands, 50% disagreement downgrade trigger used by
main.py) are an explicit, isolated policy choice -- not a derived truth.
Change them here; nothing else needs to know why.

Tier 3 (apply_sample_size_floor, below): a crop's normalized error can look
flattering purely by chance when it's computed from only 1-3 held-out rows
-- a single test row proves nothing about generalization, good or bad. This
surfaced concretely after the crop-coverage expansion: Lettuce showed "High
confidence · 89%" on 3 held-out rows (with a negative R2 on that same
target), and Garlic/Ginger/Squash showed "Fair confidence" on exactly 1 row
each. The floor forces Limited below the row-count threshold regardless of
how good the error looks, the same way Tier 2 forces a downgrade regardless
of how good the historical error looks.
"""

from __future__ import annotations

from typing import Literal

Tier = Literal["high", "fair", "limited"]

HIGH_CONFIDENCE_MAX_ERROR = 0.15
FAIR_CONFIDENCE_MAX_ERROR = 0.40
DISAGREEMENT_DOWNGRADE_THRESHOLD = 0.50

# Below this many held-out rows, the normalized error isn't trustworthy
# evidence of anything -- see the module docstring's Tier 3 note.
LOW_SAMPLE_THRESHOLD = 10

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


def apply_sample_size_floor(tier: Tier, pct: int, n: int | None) -> tuple[Tier, int]:
    """Forces Limited when n is a known count below LOW_SAMPLE_THRESHOLD.
    n is None for artifacts trained before sample-count tracking existed --
    treated as "unknown", not "too few": forcing every crop in an old
    artifact to Limited would be a false downgrade, not an honest one."""
    if n is not None and n < LOW_SAMPLE_THRESHOLD:
        return "limited", min(pct, TIER_PCT_CEILING["limited"])
    return tier, pct

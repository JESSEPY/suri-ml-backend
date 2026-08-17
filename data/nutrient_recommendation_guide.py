"""Hand-transcribed BSWM Nutrient Recommendation Guide — the official
soil-test-band -> fertilizer-rate reference (Nutrient Recommendation
Guide.pdf), replacing the old data/crop_baselines.py baseline+formula
approach entirely.

Mechanism: for a crop (+ variety / cultivation_method / age_band where the
guide splits it), three INDEPENDENT band lookups -- %OM -> N rate,
Available P (ppm) -> P rate, Extractable K (ppm) -> K rate -- each an
ascending list of (upper_bound, rate) pairs using the same
[lower, upper) semantics as ml/remarks.py's bands. There is no pH axis in
the official guide; pH still gets its own qualitative remark elsewhere but
no longer numerically adjusts any rate here.

TRANSCRIPTION STATUS: manually cross-checked cell-by-cell against
suri/public/Nutrient Recommendation Guide.pdf (Task 1 of the implementation
plan, docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md).

The source table's "Available P" columns are dual-labeled: the same 6
physical columns carry both a Bray#1 range label (0-2.0, 2.1-6.0, 6.1-10.0,
10.1-15.0, 15.1-25.0, >25.1) and an Olsen range label (0-6.0, 6.1-10.0,
10.1-15.0, 15.1-20.0, 20.1-30.0, >30.1) underneath -- same rate values
either way, only the boundary each column represents differs by method.
This file uses the Olsen boundaries throughout (see P_METHOD below). Most
crop rows' 5th and 6th P columns (20.1-30.0 and >30.1) hold the same rate
as an adjacent column, so 4 finite bounds (6/10/15/20) plus a terminal
band suffice; the manual re-check found two crops where that collapse is
NOT valid and the 5th column needed its own band: Rice (Lowland Wet/Dry)
and Corn (Native/Hybrid) -- see the inline comments on those GUIDE_ROWS
entries. The same re-check found and fixed three more real transcription
errors: Rice|Upland's K band for 251-300 ppm (was 20, is actually 0),
Rice|Lowland Wet's OM bands for the 2.1-3.5/3.6-4.5/4.6-5.5 columns (were
75/60/45, are actually 60/40/35), Rice|Lowland Dry's OM band for >5.6 (was
20, is actually 45), and Corn|Native/Hybrid's K band for 251-300 ppm (was
0, is actually 20). Every other crop row (Cassava, Peanut, Tomato, Mango,
Cacao, Coconut, Banana, and the reference-only Coffee row) was checked
against the PDF and transcribed correctly as-is.

The TestGoldenRecordRiceInbred K-band discrepancy (pptx says K=40 for the
Rice-Inbred wet-season worked example; this table has no "40" anywhere in
Rice's K column under any cultivation method -- Upland/Wet/Dry all use the
same 60/45/30/{0 or 20}/0 rate set) was NOT resolved by this re-check: the
re-verified PDF values confirm 40 genuinely does not appear in the source
table, so the discrepancy is real and unexplained by transcription error,
not an artifact of a bad transcription. See that test's docstring.

A second planned golden-record check, against the pptx's Coffee worked
example (90/60/40, labeled kg/ha), was abandoned rather than forced: this
guide's Coffee row is in kg/tree (Coffee isn't a supported crop and its row
existed solely for this cross-check), and reconciling kg/tree against a
kg/ha-labeled worked example requires an assumed coffee planting density
this project never sourced. A check built on an assumed, unsourced density
wouldn't actually verify the transcription, so the dead row was removed
instead of kept around unused. See
docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md's amended
Global Constraint on this point.
"""

from __future__ import annotations

from dataclasses import dataclass

# cmol(+)/kg -> mg/kg (ppm) for potassium: equivalent weight 39.1 g/mol,
# charge 1 -> 1 cmol/kg = 10 mmol/kg x 39.1 mg/mmol = 391 mg/kg.
K_CMOL_TO_PPM = 391.0

# The source spreadsheet never states which method Region 7 labs use for
# Available P. Assumed Olsen: Region 7 (Central Visayas) soils are
# predominantly limestone-derived/calcareous, the textbook Olsen use-case.
# Isolated here as a single named constant so it's a one-line fix if wrong
# -- flip to "bray1" and swap which _P_BANDS values are used per row if a
# lab confirms otherwise.
P_METHOD = "olsen"


def k_cmol_to_ppm(k_cmol_kg: float) -> float:
    return k_cmol_kg * K_CMOL_TO_PPM


@dataclass(frozen=True)
class GuideRow:
    unit: str  # "kg_ha" or "kg_tree"
    om_bands: tuple[tuple[float, float], ...]
    p_bands: tuple[tuple[float, float], ...]
    k_bands: tuple[tuple[float, float], ...]


def _band_value(value: float, bands: tuple[tuple[float, float], ...]) -> float:
    # Bands are printed in the guide as e.g. "0-2.0" then "2.1-3.5" -- a 0.1
    # gap between the printed labels, not a continuous boundary. The stored
    # `upper` is the literal printed upper bound of each band (2.0, 3.5, ...),
    # so a value exactly AT that bound belongs to the band that prints it,
    # not the next one up -- hence <=, not <.
    for upper, rate in bands:
        if value <= upper:
            return rate
    return bands[-1][1]


CROPS: list[str] = [
    "Corn", "Rice", "Cassava", "Peanut", "Mango", "Cacao", "Banana", "Tomato", "Coconut",
]

TARGET_UNIT: dict[str, str] = {
    "Rice": "kg_ha", "Corn": "kg_ha", "Cassava": "kg_ha", "Peanut": "kg_ha", "Tomato": "kg_ha",
    "Mango": "kg_tree", "Cacao": "kg_tree", "Coconut": "kg_tree", "Banana": "kg_tree",
}

# crop -> its guide row-label for the mature/fruit-bearing stage, used when
# a request passes the "bearing" alias (or omits age_band entirely).
BEARING_AGE_BAND: dict[str, str] = {
    "Coconut": "9yrs & up", "Cacao": "Over 7yrs", "Mango": "10yrs & up",
}

RICE_CULTIVATION_METHODS = ("Upland", "Lowland Wet", "Lowland Dry")

# crop -> the variety strings its guide rows actually distinguish. Every
# other crop's variety is always "None" -- the guide has no per-variety
# rows for them, so accepting an arbitrary variety string would just be an
# unseen, meaningless one-hot combination at training/serving time.
VARIETIES_BY_CROP: dict[str, tuple[str, ...]] = {
    "Rice": ("Hybrid", "Inbred"),
    "Corn": ("OPV", "Hybrid"),
}

# Guide row key -> GuideRow. Keys: bare crop name for crops the guide
# doesn't further split (Cassava, Peanut, Tomato, Banana); "{crop}|{axis
# value}" for Rice (cultivation_method), Corn (Native/Hybrid), and the
# age-banded tree crops (Coconut/Cacao/Mango). A planned "Coffee|..." row,
# included solely for a pptx golden-record cross-check, was removed --
# Coffee was never in CROPS, and the cross-check itself was abandoned as
# unverifiable (see the module docstring above).
GUIDE_ROWS: dict[str, GuideRow] = {
    "Rice|Upland": GuideRow(
        "kg_ha",
        om_bands=((2.0, 45), (3.5, 40), (4.5, 35), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        # K re-verified: the source PDF's Upland row reads 60/45/30/0/0, not
        # .../20/0 -- the 251-300 column is "0" here (unlike the Lowland
        # Wet/Dry rows, which are "20").
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 0), (float("inf"), 0)),
    ),
    "Rice|Lowland Wet": GuideRow(
        "kg_ha",
        # OM/P re-verified against the source PDF (see Task 1 manual cross-check):
        # OM row reads 90/60/40/35/20, not 90/75/60/45/20; the P row's Olsen
        # columns read 60/45/30/20/20/0 -- the "20" rate spans the 15.1-20.0
        # AND 20.1-30.0 Olsen columns, so the correct upper bound is 30, not 20.
        om_bands=((2.0, 90), (3.5, 60), (4.5, 40), (5.5, 35), (float("inf"), 20)),
        p_bands=((6.0, 60), (10.0, 45), (15.0, 30), (30.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 20), (float("inf"), 0)),
    ),
    "Rice|Lowland Dry": GuideRow(
        "kg_ha",
        # OM last band re-verified as 45 (not 20) -- the source PDF's Dry row
        # reads 120/75/60/45/45, i.e. the >5.6 OM column repeats the 4.6-5.5
        # value rather than dropping to the Wet row's 20. P has the same
        # 15.1-20.0/20.1-30.0 both-"20" collapse as Lowland Wet above.
        om_bands=((2.0, 120), (3.5, 75), (4.5, 60), (5.5, 45), (float("inf"), 45)),
        p_bands=((6.0, 60), (10.0, 45), (15.0, 30), (30.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 45), (250.0, 30), (300.0, 20), (float("inf"), 0)),
    ),
    "Corn|Native": GuideRow(
        "kg_ha",
        # P/K re-verified: the source PDF's Native row reads P
        # 60/60/40/40/30/0 (Olsen) -- the 20.1-30.0 column is "30", distinct
        # from both its neighbors, so it needs its own band (was dropped
        # entirely in the original transcription). K reads 60/60/60/20/0,
        # not .../0/0 -- the 251-300 column is "20".
        om_bands=((2.0, 60), (3.5, 50), (4.5, 45), (5.5, 40), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 60), (15.0, 40), (20.0, 40), (30.0, 30), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 60), (250.0, 60), (300.0, 20), (float("inf"), 0)),
    ),
    "Corn|Hybrid": GuideRow(
        "kg_ha",
        # Same P/K re-verification as Corn|Native (Hybrid row: P
        # 60/60/40/40/30/0, K 60/60/60/20/0).
        om_bands=((2.0, 120), (3.5, 90), (4.5, 60), (5.5, 50), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 60), (15.0, 40), (20.0, 40), (30.0, 30), (float("inf"), 0)),
        k_bands=((35.0, 60), (55.0, 60), (250.0, 60), (300.0, 20), (float("inf"), 0)),
    ),
    "Cassava": GuideRow(
        "kg_ha",
        om_bands=((2.0, 60), (3.5, 50), (4.5, 40), (5.5, 30), (float("inf"), 0)),
        p_bands=((6.0, 60), (10.0, 40), (15.0, 30), (20.0, 20), (float("inf"), 0)),
        k_bands=((35.0, 90), (55.0, 60), (250.0, 45), (300.0, 30), (float("inf"), 0)),
    ),
    "Peanut": GuideRow(
        "kg_ha",
        om_bands=((2.0, 10), (3.5, 0), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 40), (10.0, 30), (15.0, 20), (20.0, 10), (float("inf"), 0)),
        k_bands=((35.0, 45), (55.0, 30), (250.0, 15), (300.0, 0), (float("inf"), 0)),
    ),
    "Tomato": GuideRow(
        "kg_ha",
        om_bands=((2.0, 90), (3.5, 80), (4.5, 70), (5.5, 60), (float("inf"), 0)),
        p_bands=((6.0, 120), (10.0, 90), (15.0, 60), (20.0, 30), (float("inf"), 0)),
        k_bands=((35.0, 150), (55.0, 120), (250.0, 90), (300.0, 60), (float("inf"), 0)),
    ),
    "Mango|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.01), (3.5, 0.01), (4.5, 0.01), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.01), (10.0, 0.01), (15.0, 0.008), (20.0, 0.005), (float("inf"), 0)),
        k_bands=((35.0, 0.15), (55.0, 0.009), (250.0, 0.003), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|1-3yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.05), (4.5, 0.02), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.10), (10.0, 0.05), (15.0, 0.03), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.15), (55.0, 0.09), (250.0, 0.06), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|4-6yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.15), (3.5, 0.10), (4.5, 0.05), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.03), (float("inf"), 0)),
        k_bands=((35.0, 0.30), (55.0, 0.15), (250.0, 0.09), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|7-10yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.15), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.10), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 0.45), (55.0, 0.30), (250.0, 0.15), (300.0, 0), (float("inf"), 0)),
    ),
    "Mango|10yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.15), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|NT": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.003), (3.5, 0.002), (4.5, 0), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.004), (10.0, 0.003), (15.0, 0.002), (20.0, 0.001), (float("inf"), 0)),
        k_bands=((35.0, 0.003), (55.0, 0.002), (250.0, 0.001), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|1yr": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.055), (3.5, 0.035), (4.5, 0.02), (5.5, 0.005), (float("inf"), 0)),
        p_bands=((6.0, 0.11), (10.0, 0.07), (15.0, 0.04), (20.0, 0.015), (float("inf"), 0)),
        k_bands=((35.0, 0.075), (55.0, 0.05), (250.0, 0.024), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|2yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.065), (3.5, 0.04), (4.5, 0.025), (5.5, 0.01), (float("inf"), 0)),
        p_bands=((6.0, 0.13), (10.0, 0.08), (15.0, 0.05), (20.0, 0.025), (float("inf"), 0)),
        k_bands=((35.0, 0.08), (55.0, 0.06), (250.0, 0.03), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|3yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.075), (3.5, 0.045), (4.5, 0.03), (5.5, 0.015), (float("inf"), 0)),
        p_bands=((6.0, 0.15), (10.0, 0.09), (15.0, 0.055), (20.0, 0.025), (float("inf"), 0)),
        k_bands=((35.0, 0.10), (55.0, 0.075), (250.0, 0.036), (300.0, 0), (float("inf"), 0)),
    ),
    "Cacao|Over 7yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.10), (3.5, 0.06), (4.5, 0.035), (5.5, 0.02), (float("inf"), 0)),
        p_bands=((6.0, 0.12), (10.0, 0.07), (15.0, 0.045), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.12), (55.0, 0.09), (250.0, 0.045), (300.0, 0), (float("inf"), 0)),
    ),
    "Coconut|Below 1yr": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.06), (3.5, 0.06), (4.5, 0.04), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.05), (10.0, 0.03), (15.0, 0.02), (20.0, 0.01), (float("inf"), 0)),
        k_bands=((35.0, 0.20), (55.0, 0.15), (250.0, 0.10), (300.0, 0), (float("inf"), 0)),
    ),
    "Coconut|1-5yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.30), (3.5, 0.20), (4.5, 0.10), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.20), (10.0, 0.10), (15.0, 0.05), (20.0, 0.02), (float("inf"), 0)),
        k_bands=((35.0, 0.60), (55.0, 0.45), (250.0, 0.30), (300.0, 0.10), (float("inf"), 0)),
    ),
    "Coconut|6-8yrs": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.40), (3.5, 0.30), (4.5, 0.20), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.30), (10.0, 0.20), (15.0, 0.10), (20.0, 0.05), (float("inf"), 0)),
        k_bands=((35.0, 1.20), (55.0, 0.90), (250.0, 0.60), (300.0, 0.30), (float("inf"), 0)),
    ),
    "Coconut|9yrs & up": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.50), (3.5, 0.40), (4.5, 0.30), (5.5, 0), (float("inf"), 0)),
        p_bands=((6.0, 0.40), (10.0, 0.30), (15.0, 0.20), (20.0, 0.10), (float("inf"), 0)),
        k_bands=((35.0, 1.80), (55.0, 1.50), (250.0, 1.20), (300.0, 0.60), (float("inf"), 0)),
    ),
    "Banana|Fruit": GuideRow(
        "kg_tree",
        om_bands=((2.0, 0.16), (3.5, 0.12), (4.5, 0.09), (5.5, 0.04), (float("inf"), 0)),
        p_bands=((6.0, 0.05), (10.0, 0.045), (15.0, 0.04), (20.0, 0.035), (float("inf"), 0)),
        k_bands=((35.0, 0.33), (55.0, 0.23), (250.0, 0.13), (300.0, 0.08), (float("inf"), 0)),
    ),
}


def max_total_need(
    crop: str, variety: str | None = None, cultivation_method: str | None = None,
    age_band: str | None = None,
) -> float:
    """Worst-case (poorest-possible-soil) total N+P2O5+K2O for this crop
    config -- a fixed property of its guide row, independent of any soil
    reading (unlike nutrient_requirements(), it takes no om/p/k args).
    Normalizes a soil-driven total need into a 0-1 fraction of the crop's
    own range, so crops with structurally different absolute scales (e.g.
    Peanut's near-zero appetite vs. Tomato's) can't be compared by raw
    kg/ha magnitude. Raises KeyError for the same inputs
    nutrient_requirements() would. See shared/utils/ml-crop-ranking.ts
    (selfRelativeNeedScore) in the sibling `suri` repo for the consumer."""
    if crop not in CROPS:
        raise KeyError(f"Unknown crop {crop!r}")
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
    row = GUIDE_ROWS[key]
    return round(
        max(rate for _, rate in row.om_bands)
        + max(rate for _, rate in row.p_bands)
        + max(rate for _, rate in row.k_bands),
        3,
    )


def target_unit_for_crop(crop: str) -> str:
    return TARGET_UNIT[crop]


def resolve_variety(crop: str, variety: str | None) -> str:
    if variety in VARIETIES_BY_CROP.get(crop, ()):
        return variety
    return "None"


def resolve_cultivation_method(crop: str, cultivation_method: str | None) -> str:
    if crop != "Rice":
        return "None"
    if cultivation_method in RICE_CULTIVATION_METHODS:
        return cultivation_method
    return "Lowland Wet"


def resolve_age_band(crop: str, age_band: str | None) -> str:
    if crop not in BEARING_AGE_BAND:
        return "None"
    if age_band in (None, "bearing"):
        return BEARING_AGE_BAND[crop]
    return age_band


def _guide_row_key(
    crop: str, variety: str | None, cultivation_method: str | None, age_band: str | None,
) -> str:
    if crop == "Rice":
        return f"Rice|{resolve_cultivation_method(crop, cultivation_method)}"
    if crop == "Corn":
        resolved_variety = resolve_variety(crop, variety)
        return f"Corn|{'Hybrid' if resolved_variety == 'Hybrid' else 'Native'}"
    if crop in BEARING_AGE_BAND:
        return f"{crop}|{resolve_age_band(crop, age_band)}"
    if crop == "Banana":
        return "Banana|Fruit"
    return crop


def nutrient_requirements(
    crop: str, om: float, p_ppm: float, k_cmol_kg: float,
    variety: str | None = None, cultivation_method: str | None = None,
    age_band: str | None = None,
) -> tuple[float, float, float]:
    """Direct BSWM guide lookup: three independent band lookups (OM% -> N,
    P ppm -> P, K ppm, converted from cmol/kg -> K). Raises KeyError for an
    unknown crop or an unresolvable guide row (e.g. an age_band string this
    crop's guide rows don't have).

    Returns floats, not ints: tree-crop rates (Cacao/Coconut/Mango/Banana)
    are legitimately sub-1 kg/tree -- e.g. every one of Banana|Fruit's bands
    is below 1 (0.16, 0.12, ..., 0.05, 0.045, ..., 0.33, 0.23, ...). Casting
    to int would round every one of those down to 0, silently zeroing out
    entire crops' worth of recommendations (and, once a later task's parser
    uses this function to build training labels, corrupting them too)."""
    if crop not in CROPS:
        raise KeyError(f"Unknown crop {crop!r}")
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
    row = GUIDE_ROWS[key]
    k_ppm = k_cmol_to_ppm(k_cmol_kg)
    return (
        round(float(_band_value(om, row.om_bands)), 3),
        round(float(_band_value(p_ppm, row.p_bands)), 3),
        round(float(_band_value(k_ppm, row.k_bands)), 3),
    )

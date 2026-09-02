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

TRANSCRIPTION STATUS: the 35 canonical crops in this file were NOT all
verified the same way. Three tiers, strongest first -- know which one a row
belongs to before trusting or "correcting" its numbers.

  Tier 1 -- manual cell-by-cell re-check against the PDF. The original nine
  crops: Rice, Corn, Cassava, Peanut, Tomato, Mango, Cacao, Coconut, Banana.
  Every cell was read off suri/public/Nutrient Recommendation Guide.pdf by
  hand (Task 1 of docs/superpowers/plans/2026-08-05-real-soil-data-retraining.md);
  that pass found and fixed the four transcription errors and the two
  non-collapsible P columns detailed further down this docstring.

  Tier 2 -- two independent transcriptions, agreeing. The 21 crops added in
  Phase 1 of the crop-coverage expansion: Alugbati, Ampalaya, Asparagus,
  Beans, Cabbage, Cauliflower, Celery, Chayote, Cucumber, Eggplant, Gabi,
  Garlic, Ginger, Lettuce, Onion, Peas, Potato, Radish, Squash, Strawberry,
  Sugarcane. Each cell was transcribed twice from the source -- once into
  Nutrient Recommendation Guide.md and once for the expansion plan -- and the
  two passes agree exactly, including the cells the first pass had flagged as
  uncertain. Short of a DA-issued digital copy this is the strongest
  verification available, so no separate per-cell PDF re-check was required.
  (The aliases registered alongside them, e.g. Sorghum sharing Corn's printed
  row, add no new numbers and inherit their canonical crop's tier.)

  Tier 3 -- transcribed, plus documented inference where the source is
  defective. The 5 tree/plantation crops added in Phase 2: Pineapple,
  Avocado, Citrus, Guava, Lanzones. Most cells are straight transcriptions,
  but a few could not be read off the page as printed and were reconstructed:
  Guava's third age row repeats the label "4-6 yrs" and is encoded here as
  "7 yrs & up", and four Citrus cells across the 5 / 6-10 / 11-15 yrs rows
  are missing their decimal point in the source (.03 -> .30, 15 -> .15,
  45 -> .45, 40 -> .40). Every such inference carries an inline comment at
  the value it affects. Papaya was deliberately NOT added: its 1-2 yrs K row
  is non-monotonic in the source and could not be confirmed, and guessing a
  potassium rate a farmer would actually apply was judged unacceptable.
  Also note that kg_ha vs kg_tree is itself an inference for all tree rows --
  the guide prints units in only two places.

Full detail on the Phase 1/2 work, including the source-anomaly table and the
Papaya decision: docs/superpowers/plans/2026-08-18-crop-coverage-expansion.md.

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

from data.guide_source import GuideRow, build_guide_source  # noqa: F401 (GuideRow re-exported for existing callers)

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


# Module-level state, reassigned wholesale by reload_guide() -- never
# mutated in place. Starts empty here; reload_guide() populates it for
# real at process startup (main.py's lifespan) and again on every
# POST /admin/reload-guide. Every consumer (this file's own
# resolve_*/nutrient_requirements functions, ml/preprocessing.py's
# build_categories(), main.py's and data/training_upload.py's crop
# validators) reads these as plain module attributes, so a reassignment
# here is visible everywhere immediately -- see the module-qualified-import
# fix in main.py/data/training_upload.py/ml/preprocessing.py, which is what
# makes that true.
GUIDE_ROWS: dict[str, GuideRow] = {}
CROP_ALIASES: dict[str, str] = {}
VARIETIES_BY_CROP: dict[str, tuple[str, ...]] = {}
CULTIVATION_METHODS_BY_CROP: dict[str, tuple[str, ...]] = {}
DEFAULT_VARIETY_ROW: dict[str, str] = {}
DEFAULT_CULTIVATION_METHOD_ROW: dict[str, str] = {}
BEARING_AGE_BAND: dict[str, str] = {}
TARGET_UNIT: dict[str, str] = {}
CANONICAL_CROPS: list[str] = []
CROPS: list[str] = []


def reload_guide(client=None) -> None:
    """Re-reads crops/crop_variants/crop_nutrient_targets from Supabase and
    reassigns every module-level global above in place. The single function
    both main.py's real startup (its FastAPI `lifespan`, alongside the
    existing _reconcile_boot_registry() step) and its POST /admin/reload-guide
    call -- one code path, two triggers, per the design spec's Decision 4.

    Deliberately NOT called at this module's own import time: that would
    make importing this module (which nearly every test does transitively)
    require a live, reachable Supabase project with the crops/crop_variants/
    crop_nutrient_targets tables already populated -- this repo's test
    suite is explicitly designed to run against dummy Supabase credentials
    with no network access (see tests/conftest.py's docstring). Production
    startup calls this explicitly; tests populate guide state via
    tests/conftest.py's pytest_configure, injecting a fake `client` built
    from the committed tests/fixtures/crop_reference_seed.csv instead of a
    real network call.

    `client` defaults to the real Supabase client (data.ingest_to_supabase.
    get_client()) when not supplied -- production callers never pass one.

    get_client is imported here, not at module level: data.ingest_to_supabase
    transitively imports ml.agronomy, which imports nutrient_requirements
    FROM THIS MODULE -- a top-level import here would be circular (this
    module wouldn't have finished defining nutrient_requirements yet). Since
    reload_guide() is never called during this module's own import anymore,
    deferring the import to call time breaks the cycle."""
    from data.ingest_to_supabase import get_client

    global GUIDE_ROWS, CROP_ALIASES, VARIETIES_BY_CROP, CULTIVATION_METHODS_BY_CROP
    global DEFAULT_VARIETY_ROW, DEFAULT_CULTIVATION_METHOD_ROW, BEARING_AGE_BAND
    global TARGET_UNIT, CANONICAL_CROPS, CROPS

    source = build_guide_source(client if client is not None else get_client())

    GUIDE_ROWS = source.guide_rows
    CROP_ALIASES = source.crop_aliases
    VARIETIES_BY_CROP = source.varieties_by_crop
    CULTIVATION_METHODS_BY_CROP = source.cultivation_methods_by_crop
    DEFAULT_VARIETY_ROW = source.default_variety_row
    DEFAULT_CULTIVATION_METHOD_ROW = source.default_cultivation_method_row
    BEARING_AGE_BAND = source.bearing_age_band
    TARGET_UNIT = source.target_unit
    CANONICAL_CROPS = sorted({key.split("|", 1)[0] for key in GUIDE_ROWS})
    CROPS = CANONICAL_CROPS + sorted(CROP_ALIASES)


def resolve_crop(crop: str) -> str:
    if crop in CROP_ALIASES:
        return CROP_ALIASES[crop]
    if crop in CANONICAL_CROPS:
        return crop
    raise KeyError(f"Unknown crop {crop!r}")


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
    crop = resolve_crop(crop)
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
    row = GUIDE_ROWS[key]
    return round(
        max(rate for _, rate in row.om_bands)
        + max(rate for _, rate in row.p_bands)
        + max(rate for _, rate in row.k_bands),
        3,
    )


def target_unit_for_crop(crop: str) -> str:
    return TARGET_UNIT[resolve_crop(crop)]


def resolve_variety(crop: str, variety: str | None) -> str:
    if variety in VARIETIES_BY_CROP.get(resolve_crop(crop), ()):
        return variety
    return "None"


def resolve_cultivation_method(crop: str, cultivation_method: str | None) -> str:
    canonical = resolve_crop(crop)
    options = CULTIVATION_METHODS_BY_CROP.get(canonical)
    if options is None:
        return "None"
    if cultivation_method in options:
        return cultivation_method
    return DEFAULT_CULTIVATION_METHOD_ROW[canonical]


def resolve_age_band(crop: str, age_band: str | None) -> str:
    canonical = resolve_crop(crop)
    if canonical not in BEARING_AGE_BAND:
        return "None"
    if age_band in (None, "bearing"):
        return BEARING_AGE_BAND[canonical]
    return age_band


def _guide_row_key(
    crop: str, variety: str | None, cultivation_method: str | None, age_band: str | None,
) -> str:
    if crop in CULTIVATION_METHODS_BY_CROP:
        return f"{crop}|{resolve_cultivation_method(crop, cultivation_method)}"
    if crop in DEFAULT_VARIETY_ROW:
        resolved = resolve_variety(crop, variety)
        label = resolved if f"{crop}|{resolved}" in GUIDE_ROWS else DEFAULT_VARIETY_ROW[crop]
        return f"{crop}|{label}"
    if crop in BEARING_AGE_BAND:
        return f"{crop}|{resolve_age_band(crop, age_band)}"
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
    crop = resolve_crop(crop)
    key = _guide_row_key(crop, variety, cultivation_method, age_band)
    row = GUIDE_ROWS[key]
    k_ppm = k_cmol_to_ppm(k_cmol_kg)
    return (
        round(float(_band_value(om, row.om_bands)), 3),
        round(float(_band_value(p_ppm, row.p_bands)), 3),
        round(float(_band_value(k_ppm, row.k_bands)), 3),
    )

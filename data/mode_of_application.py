"""Hand-transcribed advisory application-timing/method text, from
MODE OF APPLICATION (1).pdf. Purely informational — no training or model
impact. Not routed through suri's RAG pipeline (suri/docs/11-rag-fertilizer
-pipeline.md), which is being retired in favor of deterministic backend
logic like this (suri/docs/12-ml-backend-integration-gaps.md, Gap 5).
"""

from __future__ import annotations

APPLICATION_SCHEDULE: dict[str, str] = {
    "Rice": (
        "Basal: apply half of the recommended N plus all P and K 10-14 days "
        "after transplanting (DAT). Topdress a quarter of the N at active "
        "tillering (23-27 DAT). Topdress the remaining quarter of N, and any "
        "remaining K, at panicle initiation (33-37 DAT)."
    ),
    "Corn": (
        "Wet season: apply half of the N and K plus all P in a band 5cm to "
        "the side and 5cm below the seed at planting; sidedress the "
        "remainder when plants are about 4 weeks old (knee-high), followed "
        "by hilling up and irrigation. Dry season: apply all fertilizer at "
        "planting, or up to 3 weeks after planting."
    ),
    "Cassava": (
        "Apply the P and K plus half of the N at planting. Sidedress the "
        "remaining N 2-3 months after planting, during hilling up."
    ),
    "Peanut": (
        "Apply the fertilizer in the furrow and cover with 2-3 inches of "
        "soil before sowing the seeds. Inoculating the seeds with "
        "nitrogen-fixing bacteria improves yield and quality."
    ),
    "Tomato": (
        "Apply half of the N 3 inches from the base of the plant and 3 "
        "inches deeper than the root system, immediately before or at "
        "planting. Sidedress the remaining N when plants start to bloom or "
        "small fruits appear. Avoid heavy N application — it delays "
        "ripening and produces soft fruit that splits easily."
    ),
    "Mango": (
        "Bearing trees: apply half of the recommended fertilizer at the "
        "start of the rainy season in 6-8 holes around the tree; apply the "
        "remaining half before the end of the rainy season, in 6-8 holes "
        "around the tree."
    ),
    "Cacao": (
        "Established crop: mix the recommended fertilizer and apply in 2 "
        "equal doses — the first at the start of the rainy season, the "
        "second before the end of the rainy season — in 5-6 holes bored 5cm "
        "deep around the base of the plant."
    ),
    "Coconut": (
        "Bearing trees: apply the recommended fertilizer annually, in at "
        "least 10 holes bored 8-16 inches deep at a 1-meter radius from the "
        "base of the tree."
    ),
    "Banana": (
        "Apply the recommended fertilizer in four equal splits: at "
        "planting, 3 months after transplanting, 6 months after "
        "transplanting, and at 9 months or just before/at initial "
        "flowering. Apply around the planting hole 30-60cm from the "
        "pseudostem, in a ring or band; on slopes, fork the fertilizer in "
        "on the uphill side of the plant."
    ),
    "Ampalaya": (
        "Mix half of the recommended fertilizer with the soil at planting "
        "time. Sidedress with the remaining fertilizer when the first set "
        "of fruits begin to form."
    ),
    "Asparagus": (
        "Broadcast the fertilizer before planting and work it into a depth "
        "of about 7.5cm. During the cutting season, scatter nitrogen "
        "fertilizer along the rows at intervals of about 14 days."
    ),
    "Cabbage": (
        "Apply the fertilizer including half the nitrogen as a double band "
        "10-12.5cm on both sides of the row, 10cm deep, at planting. "
        "Sidedress with the remaining nitrogen fertilizer 10-14 days after "
        "planting; part of this may also be applied 2 weeks before harvest."
    ),
    "Cauliflower": (
        # Source lists this jointly as "Brocolli/Cauliflower" -- same text
        # applies to both; Broccoli is not yet a canonical crop here.
        "Apply the fertilizer including half of the nitrogen in a double "
        "band 10-12.5cm on each side of the row, 10cm deep, at planting. "
        "Topdress with the remaining nitrogen fertilizer 10-14 days after "
        "planting."
    ),
    "Celery": (
        # Source lists this jointly as "Celery/Turnip(Singkamas)" and gives
        # a range, not a single fraction -- transcribed as printed.
        "One-half to two-thirds of the recommended fertilizer is broadcast "
        "and plowed into the soil during land preparation. Sidedress with "
        "the remaining fertilizer during the vegetative period."
    ),
    "Chayote": (
        "Mix the fertilizer including two-thirds of the nitrogen with the "
        "soil at planting time. Sidedress with the remaining nitrogen "
        "fertilizer when fruits begin to form."
    ),
    "Cucumber": (
        "Broadcast or apply the fertilizer including two-thirds of the "
        "nitrogen in a band 3-4 inches to the side of the row, 2 inches "
        "deeper than the seed, before sowing. Topdress with the remaining "
        "nitrogen fertilizer when the first fruits are half formed."
    ),
    "Eggplant": (
        "Apply all P and K fertilizer including half of the nitrogen into "
        "the furrows at transplanting time. Sidedress with a quarter of "
        "the nitrogen fertilizer about a month after transplanting. A "
        "second sidedressing with the remaining quarter of nitrogen is "
        "made immediately after the 2nd picking of marketable fruits."
    ),
    "Lettuce": (
        # Source lists this jointly as "Leafy vegetable (Lettuce/Mustard/
        # Pechay)" -- Mustard and Pechay are not yet canonical crops here.
        "Sensitive to heavy application of nitrogen. Apply one-third of "
        "the nitrogen fertilizer with the potash and phosphate at "
        "planting. Sidedress the remaining two-thirds of nitrogen in two "
        "equal doses, at 14 days and at 30 days after planting."
    ),
    "Radish": (
        "Broadcast the fertilizer including half the nitrogen and work it "
        "into the soil 2-3 inches deep at seedling time. Topdress with "
        "the remaining nitrogen fertilizer after seedlings have taken "
        "root."
    ),
    "Squash": (
        "Apply the fertilizer including half of the nitrogen at planting "
        "time, mixing thoroughly with the soil. Sidedress with the "
        "remaining nitrogen when fruits begin to form."
    ),
    "Strawberry": (
        # Source has two entries mentioning strawberry: "Strawberry
        # (similar to coffee)" (a perennial-tree schedule) and a specific
        # "Muskmelon/Strawberry" entry with field-crop timing. This
        # GUIDE_ROWS entry treats Strawberry as kg_ha (an annual field
        # crop), so the Muskmelon/Strawberry entry is the applicable one --
        # used here, not the coffee cross-reference.
        "Apply the fertilizer including half of the nitrogen in the form "
        "of a broad band beneath the seed row at the preparation of the "
        "seedbed. Topdress with the remaining nitrogen fertilizer when "
        "the first fruits are half grown. This may be repeated once more "
        "or two weeks before harvest."
    ),
}


def application_schedule_for_crop(crop: str) -> str | None:
    from data.nutrient_recommendation_guide import resolve_crop

    try:
        canonical = resolve_crop(crop)
        return APPLICATION_SCHEDULE.get(canonical)
    except KeyError:
        return None

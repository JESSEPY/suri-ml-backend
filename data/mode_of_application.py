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
}


def application_schedule_for_crop(crop: str) -> str | None:
    return APPLICATION_SCHEDULE.get(crop)

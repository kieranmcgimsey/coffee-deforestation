"""Deforestation attribution: classify what replaced ALL lost forest.

What: For every Hansen loss pixel (not just coffee-linked ones), determines
the current land cover using FDP coffee probability + ESA WorldCover.
Why: Shows the full deforestation picture — what fraction of forest loss
became coffee vs other crops vs built-up vs degraded land vs regrowth.
Assumes: Hansen GFC, FDP, and WorldCover are available via GEE.
Produces: Per-class pixel fractions and per-year breakdowns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import ee

    from coffee_deforestation.config import AOIConfig, PipelineConfig


def classify_all_loss_replacement(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Classify replacement land cover for ALL Hansen loss pixels.

    Logic:
    1. Get all Hansen loss pixels (loss == 1)
    2. For each loss pixel, determine current land cover:
       - If FDP coffee probability > threshold → class 0 (coffee)
       - Else remap WorldCover to simplified classes
    3. Returns ee.Image with 'replacement_class' band:
       0=coffee, 1=other_crops, 2=built/industrial,
       3=bare/degraded, 4=water, 5=forest_regrowth
    """
    import ee

    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen, get_worldcover
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]

    hansen = get_hansen(aoi)  # type: ignore[arg-type]
    fdp = get_fdp_coffee(aoi)  # type: ignore[arg-type]
    worldcover = get_worldcover(aoi)  # type: ignore[arg-type]

    # All forest loss pixels
    loss_mask = hansen.select("loss").eq(1)

    # Coffee probability threshold
    fdp_threshold = getattr(
        getattr(pipeline_config, "change_detection", None),
        "fdp_coffee_threshold", 0.5
    )
    coffee_mask = fdp.select("coffee_prob").gt(fdp_threshold)

    # Remap WorldCover to simplified classes
    # WorldCover codes: 10=tree, 20=shrub, 30=grass, 40=cropland,
    # 50=built, 60=bare, 70=snow, 80=water, 90=herbaceous_wetland, 95=mangrove, 100=moss
    wc_class = worldcover.select("worldcover")

    # Create replacement class image
    # Default: remap WorldCover on loss pixels
    replacement = (
        wc_class
        .remap(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            [5,  5,  3,  1,  2,  3,  3,  4,  5,  5,  3],  # regrowth, regrowth, bare, crops, built, bare, bare, water, regrowth, regrowth, bare
        )
        .rename("replacement_class")
    )

    # Override with coffee where FDP says coffee
    replacement = replacement.where(coffee_mask, 0)

    # Mask to loss pixels only
    replacement = replacement.updateMask(loss_mask).toInt8()

    return replacement


def compute_attribution(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> dict:
    """Compute deforestation attribution fractions via reduceRegion.

    Returns dict with total_loss_ha and per-class percentages.
    """
    import ee

    from coffee_deforestation.data.ancillary import get_hansen
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]
    replacement = classify_all_loss_replacement(aoi, pipeline_config)

    # Count pixels per replacement class
    hist_result = replacement.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=30,
        maxPixels=10_000_000,
        bestEffort=True,
    ).getInfo()

    raw_hist = hist_result.get("replacement_class", {}) or {}
    total_px = sum(raw_hist.values()) or 1

    # Map class codes to names
    class_names = {
        "0": "coffee",
        "1": "other_crops",
        "2": "built_industrial",
        "3": "bare_degraded",
        "4": "water",
        "5": "regrowth",
    }

    attribution: dict[str, float] = {}
    for code, name in class_names.items():
        count = raw_hist.get(code, 0)
        attribution[f"{name}_pct"] = round(count / total_px * 100, 1)

    # Compute total loss area (all loss pixels, not just classified)
    hansen = get_hansen(aoi)  # type: ignore[arg-type]
    loss_area = hansen.select("loss").eq(1).multiply(ee.Image.pixelArea()).divide(10000)
    total_loss = loss_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=30,
        maxPixels=10_000_000,
        bestEffort=True,
    ).getInfo()

    attribution["total_loss_ha"] = round(float(total_loss.get("loss", 0) or 0), 1)

    logger.info(
        f"Deforestation attribution for {aoi.id}: "  # type: ignore[union-attr]
        f"coffee={attribution.get('coffee_pct', 0):.1f}%, "
        f"other_crops={attribution.get('other_crops_pct', 0):.1f}%, "
        f"built={attribution.get('built_industrial_pct', 0):.1f}%, "
        f"total={attribution.get('total_loss_ha', 0):.0f} ha"
    )

    return attribution


def compute_attribution_by_year(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    years: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Compute attribution breakdown per Hansen loss year.

    Returns dict mapping year → {coffee_pct, other_crops_pct, ...}
    """
    import ee

    from coffee_deforestation.data.ancillary import get_hansen
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    if years is None:
        years = list(range(2005, 2024))

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]
    replacement = classify_all_loss_replacement(aoi, pipeline_config)

    hansen = get_hansen(aoi)  # type: ignore[arg-type]
    loss_year = hansen.select("lossyear")

    class_names = {
        "0": "coffee",
        "1": "other_crops",
        "2": "built_industrial",
        "3": "bare_degraded",
        "4": "water",
        "5": "regrowth",
    }

    yearly: dict[int, dict[str, float]] = {}

    for year in years:
        # Hansen loss year is encoded as year - 2000 (e.g., 2015 → 15)
        year_code = year - 2000
        year_mask = loss_year.eq(year_code)
        year_replacement = replacement.updateMask(year_mask)

        try:
            hist_result = year_replacement.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=region,
                scale=30,
                maxPixels=10_000_000,
                bestEffort=True,
            ).getInfo()

            raw_hist = hist_result.get("replacement_class", {}) or {}
            total_px = sum(raw_hist.values()) or 1

            year_attr: dict[str, float] = {}
            for code, name in class_names.items():
                count = raw_hist.get(code, 0)
                year_attr[f"{name}_pct"] = round(count / total_px * 100, 1)
            year_attr["total_pixels"] = float(total_px)

            if total_px > 0:
                yearly[year] = year_attr
        except Exception as e:
            logger.warning(f"Attribution for year {year} failed: {e}")

    logger.info(f"Attribution by year: {len(yearly)} years computed")
    return yearly

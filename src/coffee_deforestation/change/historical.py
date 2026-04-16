"""Historical look-back: Hansen 2000–2024 per-pixel forest trajectory.

What: For every pixel currently classified as coffee, determines when it stopped
being forest by walking back through Hansen annual loss layers.
Why: Quantifies "coffee on former forest" — the key metric for deforestation-linked
coffee supply chain analysis.
Assumes: Hansen GFC data is available. ML coffee classification exists.
Produces: A raster of loss_year_before_coffee: 0 = never was forest, 2000 = still forest,
2001–2023 = year of forest loss.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def compute_historical_trajectory(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    coffee_mask: ee.Image | None = None,
) -> ee.Image:
    """Compute per-pixel historical forest trajectory.

    For each pixel, determines:
    - was_forest_2000: whether treecover2000 > threshold
    - loss_year: year of Hansen loss (0 if no loss)
    - loss_year_before_coffee: loss_year masked to coffee pixels only

    If coffee_mask is None, uses FDP threshold as proxy.
    """
    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    hansen = get_hansen(aoi)
    threshold = pipeline_config.change_detection.hansen_treecover_2000_threshold

    logger.info(f"Computing historical trajectory for {aoi.id}")

    # Was this pixel forest in 2000?
    was_forest_2000 = hansen.select("treecover2000").gt(threshold).rename("was_forest_2000")

    # Year of loss (0 = no loss, 1-23 = 2001-2023)
    loss_year = hansen.select("lossyear").rename("loss_year")

    # Coffee mask: use provided ML classification or fall back to FDP
    if coffee_mask is None:
        fdp = get_fdp_coffee(aoi)
        coffee_mask = fdp.select("coffee_prob").gt(
            pipeline_config.change_detection.fdp_coffee_threshold
        )

    # "Coffee on former forest": pixels that are currently coffee AND were forest AND lost
    coffee_on_former_forest = (
        coffee_mask
        .And(was_forest_2000)
        .And(loss_year.gt(0))
        .rename("coffee_on_former_forest")
    )

    # Loss year only where coffee replaced forest
    loss_year_before_coffee = loss_year.updateMask(coffee_on_former_forest).rename(
        "loss_year_before_coffee"
    )

    result = (
        was_forest_2000
        .addBands(loss_year)
        .addBands(coffee_on_former_forest)
        .addBands(loss_year_before_coffee)
    )

    logger.info(f"Historical trajectory computed for {aoi.id}")
    return result


def compute_historical_stats(
    trajectory: ee.Image,
    aoi: AOIConfig,
) -> dict:
    """Compute summary statistics from the historical trajectory.

    Returns: dict with fractions and counts for the key metrics.
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)

    stats = trajectory.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.sum(), sharedInputs=False
        ),
        geometry=region,
        scale=30,
        maxPixels=10_000_000,
        bestEffort=True,
    ).getInfo()

    logger.info(f"Historical stats for {aoi.id}: {stats}")
    return stats

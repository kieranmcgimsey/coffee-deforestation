"""Contextual features: elevation, slope, distance-to-forest-edge, distance-to-road.

What: Computes terrain and proximity features from SRTM, Hansen, and road data.
Why: Coffee cultivation has specific elevation/slope preferences (600-1800m, moderate slopes).
Distance to forest edge and roads indicate expansion pressure.
Assumes: GEE is initialized. Ancillary data modules are available.
Produces: ee.Image with contextual feature bands.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def compute_contextual_features(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Compute all contextual features for an AOI.

    Returns: ee.Image with bands: elevation, slope, distance_to_forest_edge, distance_to_road
    """
    from coffee_deforestation.data.ancillary import get_hansen, get_roads, get_srtm

    logger.debug(f"Computing contextual features for {aoi.id}")

    # Elevation and slope from SRTM
    srtm = get_srtm(aoi)
    elevation = srtm.select("elevation")
    slope = srtm.select("slope")

    # Distance to forest edge from Hansen
    hansen = get_hansen(aoi)
    forest_mask = hansen.select("treecover2000").gt(
        pipeline_config.change_detection.hansen_treecover_2000_threshold
    )
    # Forest edge = forest pixels adjacent to non-forest
    forest_edge = forest_mask.focal_min(1).neq(forest_mask.focal_max(1))
    distance_to_forest = forest_edge.Not().cumulativeCost(
        forest_edge, maxDistance=50000
    ).sqrt().rename("distance_to_forest_edge").toFloat()

    # Distance to road
    road_distance = get_roads(aoi)

    return elevation.addBands([slope, distance_to_forest, road_distance])

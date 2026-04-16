"""Replacement classification: what replaced lost forest?

What: For every Hansen loss pixel, classifies the post-loss land cover using
the trained ML model applied to the post-loss year feature stack.
Why: Distinguishes coffee expansion from other causes of deforestation (e.g.,
urbanization, other crops).
Assumes: Trained ML model and multi-year feature stacks exist.
Produces: A "replacement class" raster and per-polygon aggregation.
"""

from __future__ import annotations

import ee
import numpy as np
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def classify_replacement_gee(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    feature_stack: ee.Image,
    trained_classifier: ee.Classifier | None = None,
) -> ee.Image:
    """Classify post-loss land cover for Hansen loss pixels using GEE.

    If trained_classifier is None, uses FDP probability as a proxy
    (same as rule-based approach but framed as replacement classification).
    """
    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen

    hansen = get_hansen(aoi)
    loss_mask = hansen.select("loss").eq(1)

    logger.info(f"Classifying replacement land cover for {aoi.id}")

    if trained_classifier is not None:
        # Use GEE classifier on feature stack, masked to loss pixels
        classified = feature_stack.updateMask(loss_mask).classify(trained_classifier)
        return classified.rename("replacement_class")
    else:
        # Proxy: use FDP coffee probability + WorldCover for non-coffee
        from coffee_deforestation.data.ancillary import get_worldcover

        fdp = get_fdp_coffee(aoi)
        worldcover = get_worldcover(aoi)
        threshold = pipeline_config.change_detection.fdp_coffee_threshold

        # Coffee if FDP > threshold
        is_coffee = fdp.select("coffee_prob").gt(threshold)

        # Remap WorldCover for non-coffee pixels
        # 10=forest(1), 40=cropland(2), 50=built(3), 80=water(4)
        wc_class = worldcover.remap(
            [10, 20, 30, 40, 50, 60, 80, 90, 95],
            [1, 1, 2, 2, 3, 3, 4, 1, 1],
        )

        # Coffee overrides
        replacement = wc_class.where(is_coffee, 0).updateMask(loss_mask)
        return replacement.rename("replacement_class").toByte()


def aggregate_replacement_by_hotspot(
    replacement_raster: np.ndarray,
    hotspot_labels: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Aggregate replacement class distribution per hotspot polygon.

    replacement_raster: 2D array of class codes
    hotspot_labels: 2D array of hotspot IDs (0 = background)

    Returns: dict[hotspot_id] -> {class_name: fraction}
    """
    if class_names is None:
        class_names = ["coffee", "forest", "cropland", "built_bare", "water"]

    unique_hotspots = np.unique(hotspot_labels)
    unique_hotspots = unique_hotspots[unique_hotspots > 0]

    results = {}
    for hid in unique_hotspots:
        mask = hotspot_labels == hid
        classes_in_hotspot = replacement_raster[mask]
        total = len(classes_in_hotspot)
        if total == 0:
            continue

        distribution = {}
        for i, name in enumerate(class_names):
            count = np.sum(classes_in_hotspot == i)
            distribution[name] = round(float(count / total), 4)

        results[int(hid)] = distribution

    logger.info(f"Aggregated replacement classes for {len(results)} hotspots")
    return results

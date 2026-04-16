"""Feature stack assembly for per-pixel classification.

What: Combines spectral indices, SAR features, and contextual features into
a single multi-band image per AOI per year.
Why: The ML classifier needs a consistent feature vector across all pixels.
Assumes: S2 and S1 composites are already built. Contextual features are available.
Produces: ee.Image with ~20 feature bands ready for sampling or prediction.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig
from coffee_deforestation.features.contextual import compute_contextual_features
from coffee_deforestation.features.indices import compute_all_indices, compute_ndvi
from coffee_deforestation.features.sar_features import (
    compute_sar_features,
    compute_temporal_sar_features,
)


FEATURE_BAND_ORDER = [
    # Spectral indices (from S2)
    "ndvi", "evi", "ndwi", "nbr", "savi",
    # S2 key bands (reflectance)
    "B4", "B8", "B11", "B12",
    # SAR features (from S1)
    "vv_median", "vh_median", "vv_vh_ratio",
    # Contextual
    "elevation", "slope", "distance_to_forest_edge", "distance_to_road",
    # Temporal features (multi-year)
    "ndvi_delta",   # NDVI(last_year) - NDVI(first_year)
    "vv_stddev",    # inter-annual VV standard deviation
    "vh_stddev",    # inter-annual VH standard deviation
]


def build_feature_stack(
    s2_composite: ee.Image,
    s1_composite: ee.Image,
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    *,
    s2_composites_all: dict[int, ee.Image] | None = None,
    s1_composites_all: dict[int, ee.Image] | None = None,
) -> ee.Image:
    """Assemble the full feature stack from S2, S1, and contextual data.

    When multi-year composite dicts are provided, temporal features (NDVI
    change, SAR inter-annual variability) are computed from the full time
    series. Otherwise temporal bands are filled with zeros for backward
    compatibility.

    Returns an ee.Image with bands in FEATURE_BAND_ORDER (19 bands).
    """
    logger.info(f"Building feature stack for {aoi.id}")

    # Spectral indices from S2
    s2_with_indices = compute_all_indices(s2_composite)

    # SAR features from S1
    sar_features = compute_sar_features(s1_composite)

    # Contextual features (elevation, slope, distance to forest/road)
    contextual = compute_contextual_features(aoi, pipeline_config)

    # Combine single-year bands
    stack = (
        s2_with_indices
        .select(["ndvi", "evi", "ndwi", "nbr", "savi", "B4", "B8", "B11", "B12"])
        .addBands(sar_features)
        .addBands(contextual)
    )

    # --- Temporal features (multi-year) ---

    # NDVI delta: last year minus first year
    if s2_composites_all and len(s2_composites_all) >= 2:
        years_sorted = sorted(s2_composites_all.keys())
        first_ndvi = compute_ndvi(s2_composites_all[years_sorted[0]])
        last_ndvi = compute_ndvi(s2_composites_all[years_sorted[-1]])
        ndvi_delta = last_ndvi.subtract(first_ndvi).rename("ndvi_delta")
        logger.info(f"NDVI delta: {years_sorted[-1]} minus {years_sorted[0]}")
    else:
        ndvi_delta = ee.Image.constant(0).rename("ndvi_delta")

    # SAR temporal variability: inter-annual VV/VH standard deviation
    if s1_composites_all and len(s1_composites_all) >= 2:
        temporal_sar = compute_temporal_sar_features(s1_composites_all)
        logger.info(f"Temporal SAR features from {len(s1_composites_all)} years")
    else:
        temporal_sar = (
            ee.Image.constant(0).rename("vv_stddev")
            .addBands(ee.Image.constant(0).rename("vh_stddev"))
        )

    stack = stack.addBands(ndvi_delta).addBands(temporal_sar)

    # Ensure consistent band order
    stack = stack.select(FEATURE_BAND_ORDER)

    logger.info(f"Feature stack built for {aoi.id}: {len(FEATURE_BAND_ORDER)} bands")
    return stack.toFloat()


def get_feature_names() -> list[str]:
    """Return the ordered list of feature band names."""
    return list(FEATURE_BAND_ORDER)

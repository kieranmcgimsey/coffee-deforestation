"""Temporal analysis: before/after composites, NDVI change maps, per-year stats.

What: Downloads multi-year satellite composites and computes change metrics
to demonstrate real temporal analysis.
Why: Shows actual before/after landscape change rather than static snapshots.
Assumes: GEE is authenticated. S2/S1 composites can be built for target years.
Produces: numpy arrays for before/after RGB, NDVI change maps, per-year stats dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from coffee_deforestation.config import AOIConfig, PipelineConfig


def download_composite_rgb(
    aoi: AOIConfig,
    year: int,
    pipeline_config: PipelineConfig,
    scale: int = 300,
) -> np.ndarray | None:
    """Download S2 RGB composite for a specific year via sampleRectangle.

    Returns (H, W, 3) numpy array of RGB reflectance (0-1), or None on failure.
    """
    import ee

    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.data.sentinel2 import build_s2_composite

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]

    try:
        composite = build_s2_composite(aoi, year, pipeline_config)  # type: ignore[arg-type]
        rgb = composite.select(["B4", "B3", "B2"])

        sampled = (
            rgb
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=scale)  # type: ignore[union-attr]
            .sampleRectangle(region=region, defaultValue=0)
        )
        result = sampled.getInfo()

        properties = result.get("properties", {})
        r = np.array(properties.get("B4", []), dtype=np.float32)
        g = np.array(properties.get("B3", []), dtype=np.float32)
        b = np.array(properties.get("B2", []), dtype=np.float32)

        if r.size == 0:
            logger.warning(f"Empty RGB result for {aoi.id} year {year}")  # type: ignore[union-attr]
            return None

        rgb_array = np.dstack([r, g, b])
        logger.info(
            f"Downloaded RGB {year} for {aoi.id}: "  # type: ignore[union-attr]
            f"{rgb_array.shape[0]}x{rgb_array.shape[1]} @ {scale}m"
        )
        return rgb_array

    except Exception as e:
        logger.warning(f"RGB download failed for {aoi.id} year {year}: {e}")  # type: ignore[union-attr]
        return None


def compute_ndvi_change_map(
    aoi: AOIConfig,
    year_before: int,
    year_after: int,
    pipeline_config: PipelineConfig,
    scale: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute NDVI change between two years via sampleRectangle.

    Returns (ndvi_before, ndvi_after, delta) numpy arrays, or None on failure.
    Delta = after - before: negative = vegetation loss, positive = recovery.
    """
    import ee

    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.data.sentinel2 import build_s2_composite
    from coffee_deforestation.features.indices import compute_ndvi

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]

    try:
        s2_before = build_s2_composite(aoi, year_before, pipeline_config)  # type: ignore[arg-type]
        s2_after = build_s2_composite(aoi, year_after, pipeline_config)  # type: ignore[arg-type]

        ndvi_before = compute_ndvi(s2_before)
        ndvi_after = compute_ndvi(s2_after)

        # Download both via sampleRectangle
        combined = ndvi_before.rename("ndvi_before").addBands(
            ndvi_after.rename("ndvi_after")
        )
        sampled = (
            combined
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=scale)  # type: ignore[union-attr]
            .sampleRectangle(region=region, defaultValue=0)
        )
        result = sampled.getInfo()

        properties = result.get("properties", {})
        before = np.array(properties.get("ndvi_before", []), dtype=np.float32)
        after = np.array(properties.get("ndvi_after", []), dtype=np.float32)

        if before.size == 0:
            logger.warning(f"Empty NDVI result for {aoi.id}")  # type: ignore[union-attr]
            return None

        delta = after - before

        logger.info(
            f"NDVI change {year_before}→{year_after} for {aoi.id}: "  # type: ignore[union-attr]
            f"mean delta = {np.nanmean(delta):.4f}"
        )

        return before, after, delta

    except Exception as e:
        logger.warning(f"NDVI change failed for {aoi.id}: {e}")  # type: ignore[union-attr]
        return None


def compute_real_yearly_stats(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    years: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Compute per-year Hansen loss statistics via reduceRegion.

    For each loss year, returns:
    - total_loss_pixels: count of ALL forest loss pixels
    - coffee_loss_pixels: count of loss pixels where FDP coffee > threshold
    - total_loss_ha: area of all loss (approximate)
    - coffee_loss_ha: area of coffee-linked loss
    """
    import ee

    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    if years is None:
        years = list(range(2005, 2024))

    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]
    hansen = get_hansen(aoi)  # type: ignore[arg-type]
    fdp = get_fdp_coffee(aoi)  # type: ignore[arg-type]

    fdp_threshold = getattr(
        getattr(pipeline_config, "change_detection", None),
        "fdp_coffee_threshold", 0.5
    )
    coffee_mask = fdp.select("coffee_prob").gt(fdp_threshold)
    loss_year = hansen.select("lossyear")
    pixel_area_ha = ee.Image.pixelArea().divide(10000)

    yearly_stats: dict[int, dict[str, float]] = {}

    for year in years:
        year_code = year - 2000
        year_loss = loss_year.eq(year_code)

        try:
            # Total loss this year
            total_area = (
                year_loss.multiply(pixel_area_ha)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=30,
                    maxPixels=10_000_000,
                    bestEffort=True,
                )
                .getInfo()
            )

            # Coffee-linked loss this year
            coffee_loss = year_loss.And(coffee_mask)
            coffee_area = (
                coffee_loss.multiply(pixel_area_ha)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=30,
                    maxPixels=10_000_000,
                    bestEffort=True,
                )
                .getInfo()
            )

            total_ha = float(total_area.get("lossyear", 0) or 0)
            coffee_ha = float(coffee_area.get("lossyear", 0) or 0)

            if total_ha > 0:
                yearly_stats[year] = {
                    "total_loss_ha": round(total_ha, 1),
                    "coffee_loss_ha": round(coffee_ha, 1),
                    "coffee_fraction": round(coffee_ha / max(total_ha, 1e-9), 3),
                }
        except Exception as e:
            logger.warning(f"Yearly stats for {year} failed: {e}")

    logger.info(f"Real yearly stats: {len(yearly_stats)} years for {aoi.id}")  # type: ignore[union-attr]
    return yearly_stats

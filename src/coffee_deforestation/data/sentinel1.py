"""Sentinel-1 SAR composite building with speckle filtering.

What: Builds dry-season median composites from Sentinel-1 GRD (VV/VH) with
focal median speckle filter.
Why: SAR penetrates clouds — critical for tropical AOIs where optical is limited.
Assumes: GEE is initialized. AOI and year are valid.
Produces: An ee.Image with VV and VH bands (dB scale, speckle-filtered).
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def _get_date_range(aoi: AOIConfig, year: int) -> tuple[str, str]:
    """Get dry-season date range for an AOI and year."""
    ds = aoi.dry_season
    if ds.cross_year:
        start = f"{year - 1}-{ds.start_month:02d}-01"
        end = f"{year}-{ds.end_month:02d}-28"
    else:
        start = f"{year}-{ds.start_month:02d}-01"
        end = f"{year}-{ds.end_month:02d}-30"
    return start, end


def build_s1_composite(
    aoi: AOIConfig,
    year: int,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Build a speckle-filtered Sentinel-1 median composite for a dry-season window.

    Returns an ee.Image with bands: VV, VH (dB scale, focal-median filtered).
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    start_date, end_date = _get_date_range(aoi, year)
    region = aoi_to_geometry(aoi)
    s1_config = pipeline_config.s1_processing

    logger.info(f"Building S1 composite for {aoi.id} {year} ({start_date} to {end_date})")

    # Load S1 GRD collection — try configured orbit pass, fall back to any pass
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(s1_config.polarizations)
    )

    # Median composite
    composite = s1_col.median().clip(region)

    # Apply focal median speckle filter
    # GEE's S1 GRD is already in dB; focal_median smooths speckle noise
    radius = s1_config.speckle_filter_radius_m
    composite = composite.focal_median(radius=radius, units="meters")

    image_count = s1_col.size()

    logger.info(f"S1 composite built for {aoi.id} {year}")

    return composite.set({
        "aoi_id": aoi.id,
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "image_count": image_count,
    }).toFloat()

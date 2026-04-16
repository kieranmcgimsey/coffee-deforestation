"""Sentinel-2 composite building with s2cloudless cloud masking.

What: Builds dry-season median composites from Sentinel-2 L2A imagery with
cloud/shadow masking using the S2 cloud probability collection.
Why: Cloud-free optical composites are the foundation of spectral features.
Assumes: GEE is initialized. AOI and year are valid.
Produces: A cloud-masked S2 median composite (ee.Image) for the dry season window.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def _get_date_range(aoi: AOIConfig, year: int) -> tuple[str, str]:
    """Get dry-season date range for an AOI and year.

    For cross-year seasons (e.g., Dec-Mar), start date is Dec of previous year.
    """
    ds = aoi.dry_season
    if ds.cross_year:
        start = f"{year - 1}-{ds.start_month:02d}-01"
        end = f"{year}-{ds.end_month:02d}-28"
    else:
        start = f"{year}-{ds.start_month:02d}-01"
        end = f"{year}-{ds.end_month:02d}-30"
    return start, end


def build_s2_composite(
    aoi: AOIConfig,
    year: int,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Build a cloud-masked Sentinel-2 median composite for a dry-season window.

    Cloud masking uses s2cloudless probability thresholding with dilation.
    The median composite naturally suppresses residual cloud/shadow contamination.

    Returns an ee.Image with bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    (scaled to reflectance 0-1).
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    start_date, end_date = _get_date_range(aoi, year)
    region = aoi_to_geometry(aoi)
    cloud_config = pipeline_config.cloud_masking

    logger.info(f"Building S2 composite for {aoi.id} {year} ({start_date} to {end_date})")

    # Load S2 SR collection
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )

    # Load matching cloud probability collection
    cloud_prob_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    # Join S2 with cloud probability on system:index
    join = ee.Join.saveFirst("cloud_prob")
    filter_match = ee.Filter.equals(
        leftField="system:index", rightField="system:index"
    )
    joined = join.apply(s2_col, cloud_prob_col, filter_match)

    # Cloud masking parameters
    threshold = cloud_config.cloud_probability_threshold
    dilation_pixels = int(cloud_config.cloud_dilation_m / 10)

    # Apply cloud masking inside map — keep it minimal for GEE proxy compatibility
    def apply_mask(img):
        cloud_prob = ee.Image(img.get("cloud_prob"))
        cloud_mask = cloud_prob.gt(threshold)
        cloud_mask = cloud_mask.focalMax(dilation_pixels, "circle", "pixels")
        return ee.Image(img).updateMask(cloud_mask.Not())

    masked_col = ee.ImageCollection(joined.map(apply_mask))

    # Select relevant bands and compute median composite
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    composite = masked_col.select(bands).median().clip(region)

    # Scale to reflectance (0-1)
    composite = composite.divide(10000).toFloat()

    image_count = masked_col.size()

    logger.info(f"S2 composite built for {aoi.id} {year}")

    return composite.set({
        "aoi_id": aoi.id,
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "image_count": image_count,
    })

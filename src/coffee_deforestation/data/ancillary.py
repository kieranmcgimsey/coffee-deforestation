"""Ancillary data layers from GEE: Hansen GFC, FDP coffee, WorldCover, SRTM, roads.

What: Loads and preprocesses static/semi-static reference datasets for each AOI.
Why: These provide labels (FDP, WorldCover), baselines (Hansen forest), and
contextual features (elevation, slope, roads).
Assumes: GEE is initialized. AOI config is valid.
Produces: ee.Image layers clipped to the AOI.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig


def get_hansen(aoi: AOIConfig) -> ee.Image:
    """Load Hansen Global Forest Change dataset clipped to AOI.

    Returns an ee.Image with bands:
    - treecover2000: tree canopy cover in 2000 (%)
    - loss: binary forest loss (1 = loss occurred)
    - lossyear: year of loss (0 = no loss, 1-23 = 2001-2023)
    - gain: binary forest gain
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    logger.debug(f"Loaded Hansen GFC for {aoi.id}")
    return hansen.clip(region)


def get_hansen_loss_by_year(aoi: AOIConfig, year: int) -> ee.Image:
    """Get a binary mask of Hansen loss for a specific year.

    year should be 2001-2023 (Hansen lossyear values are 1-23).
    """
    hansen = get_hansen(aoi)
    loss_year_code = year - 2000
    return hansen.select("lossyear").eq(loss_year_code).rename("loss_in_year")


def get_hansen_cumulative_loss(aoi: AOIConfig, up_to_year: int) -> ee.Image:
    """Get cumulative Hansen loss from 2001 up to (inclusive) a given year."""
    hansen = get_hansen(aoi)
    loss_year_code = up_to_year - 2000
    return (
        hansen.select("lossyear")
        .gt(0)
        .And(hansen.select("lossyear").lte(loss_year_code))
        .rename("cumulative_loss")
    )


def get_fdp_coffee(aoi: AOIConfig, year: int = 2023) -> ee.Image:
    """Load Forest Data Partnership coffee probability layer.

    The FDP model_2025a collection contains two images: coffee_2020 and coffee_2023.
    Defaults to the 2023 layer (most recent). Band name is 'probability'.

    Returns an ee.Image with a single band 'coffee_prob' (0-1).
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)

    # model_2025a is an ImageCollection with coffee_2020 and coffee_2023
    asset_id = f"projects/forestdatapartnership/assets/coffee/model_2025a/coffee_{year}"
    fdp = ee.Image(asset_id)
    logger.debug(f"Loaded FDP coffee {year} for {aoi.id}")

    return fdp.select("probability").clip(region).rename("coffee_prob").toFloat()


def get_worldcover(aoi: AOIConfig) -> ee.Image:
    """Load ESA WorldCover 2021 (v200) land cover classification.

    Returns an ee.Image with a single band 'worldcover' containing class codes:
    10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland, 50=Built-up,
    60=Bare/sparse, 70=Snow/ice, 80=Water, 90=Herbaceous wetland, 95=Mangroves, 100=Moss/lichen
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)
    worldcover = ee.Image("ESA/WorldCover/v200/2021")

    logger.debug(f"Loaded WorldCover for {aoi.id}")
    return worldcover.clip(region).rename("worldcover")


def get_srtm(aoi: AOIConfig) -> ee.Image:
    """Load SRTM elevation and compute slope.

    Returns an ee.Image with bands: elevation (m), slope (degrees).
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)
    srtm = ee.Image("USGS/SRTMGL1_003").clip(region)

    elevation = srtm.select("elevation")
    slope = ee.Terrain.slope(srtm)

    logger.debug(f"Loaded SRTM for {aoi.id}")
    return elevation.addBands(slope).toFloat()


def get_roads(aoi: AOIConfig) -> ee.Image:
    """Compute distance-to-nearest-road raster from OpenStreetMap roads.

    Uses TIGER/2016/Roads (US) as a fallback reference; for global coverage,
    we rasterize OSM roads from a suitable collection. If no global road
    dataset is available, returns a constant image.
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)

    # Use WorldCover built-up class as a road proximity proxy.
    # Global OSM roads (projects/sat-io/open-datasets/OSM/roads_global) is not
    # available as a GEE asset. See DECISIONS.md for rationale.
    worldcover = get_worldcover(aoi)
    built_up = worldcover.eq(50)  # Built-up class
    distance = built_up.Not().cumulativeCost(
        built_up, maxDistance=50000
    ).sqrt().rename("distance_to_road")

    logger.debug(f"Computed distance-to-road for {aoi.id}")
    return distance.clip(region).toFloat()

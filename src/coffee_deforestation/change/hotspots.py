"""Hotspot polygonization, ranking, and enrichment.

What: Converts raster coffee-deforestation candidates into vector polygons,
filters by minimum area, ranks by size, and enriches with metadata.
Why: Polygons are the unit of analysis for reporting and spatial queries.
Assumes: Coffee deforestation raster exists. GEE or local rasterio processing.
Produces: GeoJSON file with ranked hotspot polygons.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import shape

from coffee_deforestation.config import AOIConfig, PipelineConfig


def polygonize_hotspots_gee(
    candidates: ee.Image,
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> list[dict]:
    """Polygonize candidate coffee-deforestation pixels using GEE.

    Converts connected candidate pixels into vector polygons,
    filters by minimum area, and returns as a list of GeoJSON features.
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)
    min_area_ha = pipeline_config.change_detection.min_hotspot_area_ha
    min_area_m2 = min_area_ha * 10000

    logger.info(f"Polygonizing hotspots for {aoi.id} (min area: {min_area_ha} ha)")

    # Reduce to vectors
    candidate_mask = candidates.select("coffee_deforestation")
    vectors = candidate_mask.selfMask().reduceToVectors(
        geometry=region,
        scale=30,  # Hansen resolution
        geometryType="polygon",
        eightConnected=True,
        labelProperty="label",
        maxPixels=1_000_000,
        bestEffort=True,
    )

    # Filter by area
    def add_area(feature):
        return feature.set("area_m2", feature.geometry().area(maxError=1))

    vectors = vectors.map(add_area).filter(ee.Filter.gt("area_m2", min_area_m2))

    # Limit to top 5000 by area (GEE has a 5000-element getInfo limit)
    vectors = vectors.sort("area_m2", False).limit(5000)

    # Get as GeoJSON
    try:
        geojson = vectors.getInfo()
        features = geojson.get("features", [])
        logger.info(f"Found {len(features)} hotspot polygons for {aoi.id}")
        return features
    except Exception as e:
        logger.error(f"Polygonization failed for {aoi.id}: {e}")
        return []


def enrich_hotspots(
    features: list[dict],
    aoi: AOIConfig,
    candidates: ee.Image,
) -> list[dict]:
    """Enrich hotspot features with metadata: area_ha, centroid, loss_year, rank."""
    from coffee_deforestation.data.ancillary import get_hansen

    enriched = []

    for i, feature in enumerate(features):
        geom = shape(feature["geometry"])
        props = feature.get("properties", {})

        # Compute area in hectares using UTM projection
        area_m2 = props.get("area_m2", geom.area)
        area_ha = area_m2 / 10000

        centroid = geom.centroid
        enriched_props = {
            "hotspot_id": f"{aoi.id}_h{i+1:03d}",
            "aoi_id": aoi.id,
            "area_ha": round(area_ha, 2),
            "centroid_lon": round(centroid.x, 6),
            "centroid_lat": round(centroid.y, 6),
            "rank": 0,  # Will be set after sorting
        }

        enriched.append({
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": enriched_props,
        })

    # Sort by area (largest first) and assign ranks
    enriched.sort(key=lambda f: f["properties"]["area_ha"], reverse=True)
    for i, feat in enumerate(enriched):
        feat["properties"]["rank"] = i + 1

    # Extract loss_year per hotspot using batch GEE reduceRegions
    try:
        hansen = get_hansen(aoi)
        loss_year_img = hansen.select("lossyear")

        # Build a FeatureCollection from our enriched features
        ee_features = []
        for feat in enriched:
            ee_geom = ee.Geometry(feat["geometry"])
            ee_feat = ee.Feature(ee_geom, {"idx": feat["properties"]["rank"]})
            ee_features.append(ee_feat)

        fc = ee.FeatureCollection(ee_features)

        # Batch reduceRegions — one GEE call for all polygons
        reduced = loss_year_img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mode(),
            scale=30,
        )

        # Extract results
        reduced_info = reduced.getInfo()
        loss_years_by_rank = {}
        for rf in reduced_info.get("features", []):
            rp = rf.get("properties", {})
            rank = rp.get("idx")
            mode_ly = rp.get("mode")
            if rank is not None and mode_ly is not None:
                loss_years_by_rank[rank] = int(mode_ly) + 2000

        # Apply to enriched features
        assigned = 0
        for feat in enriched:
            rank = feat["properties"]["rank"]
            ly = loss_years_by_rank.get(rank)
            feat["properties"]["loss_year"] = ly
            if ly:
                assigned += 1

        logger.info(f"Assigned loss_year to {assigned}/{len(enriched)} hotspots")

    except Exception as e:
        logger.warning(f"Loss year extraction failed: {e}. Hotspots will lack loss_year.")
        for feat in enriched:
            feat["properties"]["loss_year"] = None

    return enriched


def save_hotspots(
    features: list[dict],
    output_path: Path,
    aoi: AOIConfig,
) -> Path:
    """Save hotspot features as GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "aoi_id": aoi.id,
            "aoi_name": aoi.name,
            "count": len(features),
        },
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"Saved {len(features)} hotspots to {output_path}")

    # Export as GeoPackage for GIS interoperability
    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gpkg_path = output_path.with_suffix(".gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")
        logger.info(f"Saved GeoPackage: {gpkg_path}")
    except Exception as e:
        logger.warning(f"GeoPackage export failed: {e}")

    # Export as CSV for mobile / spreadsheet use
    try:
        import csv as csv_mod

        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv_mod.DictWriter(
                csvfile,
                fieldnames=["hotspot_id", "lat", "lon", "area_ha", "loss_year", "rank"],
            )
            writer.writeheader()
            for feat in features:
                props = feat.get("properties", {})
                writer.writerow({
                    "hotspot_id": props.get("hotspot_id", ""),
                    "lat": props.get("centroid_lat", ""),
                    "lon": props.get("centroid_lon", ""),
                    "area_ha": props.get("area_ha", ""),
                    "loss_year": props.get("loss_year", ""),
                    "rank": props.get("rank", ""),
                })
        logger.info(f"Saved CSV: {csv_path}")
    except Exception as e:
        logger.warning(f"CSV export failed: {e}")

    return output_path

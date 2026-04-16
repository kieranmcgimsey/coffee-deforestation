"""Raster-wide prediction from trained models.

What: Applies a trained classifier to a full feature stack to produce a
coffee probability raster and hard classification map.
Why: Per-pixel classification is the core output of the ML pipeline.
Assumes: Trained model and feature stack GeoTIFF exist.
Produces: Classification raster (class codes) and probability raster.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio

if TYPE_CHECKING:
    import ee

    from coffee_deforestation.config import AOIConfig
from loguru import logger


def predict_from_raster(
    model: Any,
    feature_stack_path: Path,
    output_dir: Path,
    aoi_id: str,
) -> tuple[Path, Path]:
    """Apply a trained model to a feature stack GeoTIFF.

    Reads the feature stack, reshapes to (n_pixels, n_features), predicts,
    and writes classification + probability rasters.

    Returns: (classification_path, probability_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_stack_path) as src:
        data = src.read()  # (bands, height, width)
        profile = src.profile.copy()
        nodata_mask = np.any(np.isnan(data), axis=0)  # (height, width)

        n_bands, height, width = data.shape
        # Reshape to (n_pixels, n_features)
        flat = data.reshape(n_bands, -1).T  # (n_pixels, n_features)

        # Mask out nodata pixels
        valid_mask = ~np.any(np.isnan(flat), axis=1)
        valid_pixels = flat[valid_mask]

    logger.info(
        f"Predicting {aoi_id}: {valid_pixels.shape[0]} valid pixels "
        f"out of {height * width} total"
    )

    # Predict
    predictions = np.full(height * width, -1, dtype=np.int16)
    probabilities = np.full(height * width, np.nan, dtype=np.float32)

    if len(valid_pixels) > 0:
        pred = model.predict(valid_pixels)  # type: ignore[union-attr]
        predictions[valid_mask] = pred.astype(np.int16)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(valid_pixels)  # type: ignore[union-attr]
            # Coffee probability is class 0
            coffee_prob = proba[:, 0] if proba.shape[1] > 0 else np.zeros(len(valid_pixels))
            probabilities[valid_mask] = coffee_prob.astype(np.float32)

    # Reshape back to raster
    class_raster = predictions.reshape(height, width)
    prob_raster = probabilities.reshape(height, width)

    # Apply nodata mask
    class_raster[nodata_mask] = -1
    prob_raster[nodata_mask] = np.nan

    # Write classification raster
    class_path = output_dir / f"classification_{aoi_id}.tif"
    class_profile = profile.copy()
    class_profile.update(count=1, dtype="int16", nodata=-1)
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(class_raster[np.newaxis, :, :])

    # Write probability raster
    prob_path = output_dir / f"coffee_prob_{aoi_id}.tif"
    prob_profile = profile.copy()
    prob_profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(prob_path, "w", **prob_profile) as dst:
        dst.write(prob_raster[np.newaxis, :, :])

    logger.info(f"Saved classification: {class_path}, probability: {prob_path}")
    return class_path, prob_path


def predict_from_gee(
    model: Any,
    feature_stack: ee.Image,
    aoi: AOIConfig,
    output_dir: Path,
    scale: int = 300,
) -> tuple[Path, Path] | None:
    """Download feature stack via sampleRectangle, predict locally, save as GeoTIFF.

    Uses sampleRectangle() at reduced resolution to stay within GEE's
    getInfo() pixel limits (262,144 pixels) without Drive export.

    Args:
        model: trained sklearn-compatible classifier
        feature_stack: ee.Image with bands matching FEATURE_BAND_ORDER
        aoi: AOIConfig with bbox and epsg_utm
        output_dir: directory for output GeoTIFFs
        scale: resolution in metres (default 300m to fit pixel limit)

    Returns (classification_path, probability_path) or None on failure.
    """
    import ee
    from rasterio.transform import from_bounds

    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.features.stack import FEATURE_BAND_ORDER

    output_dir.mkdir(parents=True, exist_ok=True)
    region = aoi_to_geometry(aoi)  # type: ignore[arg-type]

    # Reproject and sample at reduced resolution
    try:
        reprojected = feature_stack.reproject(  # type: ignore[union-attr]
            crs=f"EPSG:{aoi.epsg_utm}",  # type: ignore[union-attr]
            scale=scale,
        )
        sampled = reprojected.sampleRectangle(
            region=region,
            defaultValue=0,
        )
        result = sampled.getInfo()
    except Exception as e:
        logger.warning(
            f"sampleRectangle failed for {aoi.id}: {e}. "  # type: ignore[union-attr]
            f"AOI may be too large at {scale}m resolution."
        )
        return None

    properties = result.get("properties", {})

    # Extract arrays per band
    arrays = []
    for band_name in FEATURE_BAND_ORDER:
        band_data = properties.get(band_name, [])
        if not band_data:
            logger.warning(f"Missing band {band_name} in sampleRectangle result")
            return None
        arrays.append(np.array(band_data, dtype=np.float32))

    if not arrays or arrays[0].size == 0:
        logger.warning(f"Empty sampleRectangle result for {aoi.id}")  # type: ignore[union-attr]
        return None

    data = np.stack(arrays, axis=0)  # (bands, height, width)
    n_bands, height, width = data.shape
    logger.info(
        f"Downloaded feature stack for {aoi.id}: "  # type: ignore[union-attr]
        f"{height}x{width} pixels @ {scale}m ({n_bands} bands)"
    )

    # Reshape for prediction
    flat = data.reshape(n_bands, -1).T  # (n_pixels, n_features)
    valid_mask = ~np.any(np.isnan(flat), axis=1)

    predictions = np.full(height * width, -1, dtype=np.int16)
    probabilities = np.full(height * width, np.nan, dtype=np.float32)

    valid_pixels = flat[valid_mask]
    if len(valid_pixels) > 0:
        pred = model.predict(valid_pixels)  # type: ignore[union-attr]
        predictions[valid_mask] = pred.astype(np.int16)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(valid_pixels)  # type: ignore[union-attr]
            coffee_prob = proba[:, 0] if proba.shape[1] > 0 else np.zeros(len(valid_pixels))
            probabilities[valid_mask] = coffee_prob.astype(np.float32)

    class_raster = predictions.reshape(height, width)
    prob_raster = probabilities.reshape(height, width)

    # Build GeoTIFF with proper CRS and transform
    bbox = aoi.bbox  # type: ignore[union-attr]
    transform = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    aoi_id = aoi.id  # type: ignore[union-attr]
    class_path = output_dir / f"classification_{aoi_id}.tif"
    with rasterio.open(class_path, "w", count=1, dtype="int16", nodata=-1, **profile) as dst:
        dst.write(class_raster[np.newaxis, :, :])

    prob_path = output_dir / f"coffee_prob_{aoi_id}.tif"
    with rasterio.open(prob_path, "w", count=1, dtype="float32", nodata=float("nan"), **profile) as dst:
        dst.write(prob_raster[np.newaxis, :, :])

    logger.info(f"Saved classification ({height}x{width} @ ~{scale}m): {class_path}")
    return class_path, prob_path


def predict_array(
    model: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict on a numpy array. Returns (predictions, probabilities or None)."""
    predictions = model.predict(X)  # type: ignore[union-attr]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)  # type: ignore[union-attr]
    return predictions, probabilities

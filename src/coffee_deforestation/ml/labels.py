"""Label sampling from FDP coffee probability and ESA WorldCover.

What: Generates stratified, balanced training samples by combining FDP coffee
labels with WorldCover land cover classes. Samples pixels with known class
membership and extracts their feature stack values.
Why: The ML classifier needs labeled training data. FDP provides coffee labels;
WorldCover provides non-coffee classes.
Assumes: Feature stack rasters and label rasters are available (as GEE images or
local GeoTIFFs). GEE is initialized for GEE-based sampling.
Produces: A labeled numpy array (samples × features) with class labels, ready
for train/test split.
"""

from __future__ import annotations

from pathlib import Path

import ee
import numpy as np
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig

# Class code mapping
CLASS_MAP = {
    "coffee": 0,
    "forest": 1,
    "non_coffee_cropland": 2,
    "built_bare": 3,
    "water": 4,
}

# WorldCover codes → our class codes
WORLDCOVER_MAP = {
    10: 1,   # Tree cover → forest
    20: 1,   # Shrubland → forest (conservative)
    40: 2,   # Cropland → non-coffee cropland
    50: 3,   # Built-up → built/bare
    60: 3,   # Bare/sparse → built/bare
    80: 4,   # Water → water
    90: 1,   # Herbaceous wetland → forest (conservative)
    95: 1,   # Mangroves → forest
}


def create_label_image(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Create a combined label image from FDP and WorldCover.

    FDP coffee probability > threshold → class 0 (coffee).
    WorldCover classes remapped to our class codes for everything else.
    Coffee labels override WorldCover where they overlap.
    """
    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_worldcover

    fdp = get_fdp_coffee(aoi)
    worldcover = get_worldcover(aoi)
    threshold = pipeline_config.change_detection.fdp_coffee_threshold

    # Coffee mask from FDP
    coffee_mask = fdp.select("coffee_prob").gt(threshold)

    # Remap WorldCover to our class codes
    wc_from = list(WORLDCOVER_MAP.keys())
    wc_to = list(WORLDCOVER_MAP.values())
    wc_remapped = worldcover.remap(wc_from, wc_to).rename("label")

    # Coffee overrides WorldCover
    labels = wc_remapped.where(coffee_mask, CLASS_MAP["coffee"])

    return labels.toByte()


def sample_training_data_gee(
    feature_stack: ee.Image,
    label_image: ee.Image,
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    seed: int = 42,
) -> dict:
    """Sample balanced training data from GEE.

    Returns a dict ready for ee export or getInfo with columns for each
    feature band plus 'label'.
    """
    from coffee_deforestation.data.gee_client import aoi_to_geometry

    region = aoi_to_geometry(aoi)
    n_per_class = pipeline_config.ml.samples_per_class_per_aoi

    # Add label band to feature stack
    stacked = feature_stack.addBands(label_image)

    # Stratified sampling — scale=30 for large AOIs to stay within GEE memory
    samples = stacked.stratifiedSample(
        numPoints=n_per_class,
        classBand="label",
        region=region,
        scale=30,
        seed=seed,
        geometries=False,
    )

    return samples


def samples_to_numpy(
    samples_info: list[dict],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GEE sample dicts to numpy arrays.

    Returns: (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    X_list = []
    y_list = []

    for row in samples_info:
        features = [row.get(fname, np.nan) for fname in feature_names]
        label = row.get("label", -1)
        if label >= 0 and not any(v is None for v in features):
            X_list.append(features)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Converted {len(X)} samples with {X.shape[1]} features")
    for cls_name, cls_code in CLASS_MAP.items():
        count = np.sum(y == cls_code)
        logger.info(f"  {cls_name} (class {cls_code}): {count} samples")

    return X, y


def save_samples(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    aoi_id: str,
) -> tuple[Path, Path]:
    """Save training samples to numpy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / f"X_{aoi_id}.npy"
    y_path = output_dir / f"y_{aoi_id}.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    logger.info(f"Saved samples: {x_path} ({X.shape}), {y_path} ({y.shape})")
    return x_path, y_path


def load_samples(
    data_dir: Path,
    aoi_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load training samples from numpy files."""
    X = np.load(data_dir / f"X_{aoi_id}.npy")
    y = np.load(data_dir / f"y_{aoi_id}.npy")
    return X, y

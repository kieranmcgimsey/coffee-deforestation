"""AOI validation via cheap GEE queries.

What: Runs a fast sanity check for each AOI before committing to expensive exports.
Checks that the AOI has sufficient coffee pixels, forest cover, and loss history.
Why: Prevents wasting hours on GEE exports for AOIs that won't produce useful results.
Assumes: GEE is initialized.
Produces: A validation result dict per AOI with pass/fail status and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig
from coffee_deforestation.data.gee_client import aoi_to_geometry


@dataclass
class AOIValidationResult:
    aoi_id: str
    coffee_fraction: float
    forest_2000_fraction: float
    hansen_loss_pixels: int
    passed: bool
    messages: list[str]

    def summary_row(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.aoi_id:<15} | "
            f"Coffee: {self.coffee_fraction:6.2%} | "
            f"Forest-2000: {self.forest_2000_fraction:6.2%} | "
            f"Loss pixels: {self.hansen_loss_pixels:>8} | "
            f"[{status}]"
        )


def validate_aoi(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> AOIValidationResult:
    """Run validation checks on a single AOI.

    Checks:
    1. FDP coffee probability > 0.5 fraction >= min_coffee_fraction
    2. Hansen treecover2000 > 50% fraction >= min_forest_2000_fraction
    3. Hansen loss pixel count >= min_hansen_loss_pixels
    """
    region = aoi_to_geometry(aoi)
    val_config = pipeline_config.validation
    messages: list[str] = []

    logger.info(f"Validating AOI: {aoi.id} ({aoi.name})")

    # Compute area stats at coarse scale (100m) for speed
    scale = 100

    # 1. Coffee fraction from FDP (2023 layer)
    fdp = ee.Image("projects/forestdatapartnership/assets/coffee/model_2025a/coffee_2023")
    coffee_mask = fdp.select("probability").gt(0.5).rename("coffee")
    coffee_stats = coffee_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    ).getInfo()
    coffee_fraction = coffee_stats.get("coffee", 0) or 0

    # 2. Forest 2000 fraction from Hansen
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    forest_mask = hansen.select("treecover2000").gt(
        pipeline_config.change_detection.hansen_treecover_2000_threshold
    ).rename("forest")
    forest_stats = forest_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    ).getInfo()
    forest_fraction = forest_stats.get("forest", 0) or 0

    # 3. Hansen loss pixel count
    loss_mask = hansen.select("loss").rename("loss")
    loss_stats = loss_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    ).getInfo()
    loss_pixels = int(loss_stats.get("loss", 0) or 0)

    # Evaluate thresholds
    passed = True

    if coffee_fraction < val_config.min_coffee_fraction:
        messages.append(
            f"Coffee fraction {coffee_fraction:.2%} < {val_config.min_coffee_fraction:.2%}"
        )
        passed = False
    else:
        messages.append(f"Coffee fraction OK: {coffee_fraction:.2%}")

    if forest_fraction < val_config.min_forest_2000_fraction:
        messages.append(
            f"Forest-2000 fraction {forest_fraction:.2%} < {val_config.min_forest_2000_fraction:.2%}"
        )
        passed = False
    else:
        messages.append(f"Forest-2000 fraction OK: {forest_fraction:.2%}")

    if loss_pixels < val_config.min_hansen_loss_pixels:
        messages.append(
            f"Hansen loss pixels {loss_pixels} < {val_config.min_hansen_loss_pixels}"
        )
        passed = False
    else:
        messages.append(f"Hansen loss pixels OK: {loss_pixels}")

    status = "PASSED" if passed else "FAILED"
    logger.info(f"AOI {aoi.id} validation {status}")

    return AOIValidationResult(
        aoi_id=aoi.id,
        coffee_fraction=coffee_fraction,
        forest_2000_fraction=forest_fraction,
        hansen_loss_pixels=loss_pixels,
        passed=passed,
        messages=messages,
    )

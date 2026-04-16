"""Rule-based coffee-linked deforestation detection using Hansen + FDP overlay.

What: Identifies pixels where Hansen forest loss coincides with FDP coffee probability,
as a simple baseline before ML models are trained.
Why: Provides an interpretable, reproducible baseline for comparison with ML results.
Assumes: Hansen and FDP data are loaded. GEE is initialized.
Produces: ee.Image with candidate coffee-deforestation pixels per loss year.
"""

from __future__ import annotations

import ee
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig


def detect_coffee_deforestation_rule_based(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
) -> ee.Image:
    """Rule-based detection: Hansen loss ∧ FDP coffee prob > threshold.

    Returns an ee.Image with bands:
    - coffee_deforestation: binary mask (1 = candidate event)
    - loss_year: year of loss (2001-2023) where coffee deforestation detected, 0 elsewhere
    """
    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen

    logger.info(f"Running rule-based coffee deforestation detection for {aoi.id}")

    hansen = get_hansen(aoi)
    fdp = get_fdp_coffee(aoi)

    threshold = pipeline_config.change_detection.fdp_coffee_threshold

    # Hansen loss mask (any year)
    loss = hansen.select("loss").eq(1)
    loss_year = hansen.select("lossyear")

    # FDP coffee probability above threshold
    coffee = fdp.select("coffee_prob").gt(threshold)

    # Candidate: loss pixel that is now coffee
    candidate = loss.And(coffee).rename("coffee_deforestation")

    # Loss year only where candidate is true
    candidate_year = loss_year.updateMask(candidate).rename("loss_year")

    result = candidate.addBands(candidate_year)

    logger.info(f"Rule-based detection complete for {aoi.id}")
    return result


def detect_by_year_range(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    start_year: int = 2019,
    end_year: int = 2024,
) -> ee.Image:
    """Detect coffee-linked deforestation within a specific year range."""
    result = detect_coffee_deforestation_rule_based(aoi, pipeline_config)

    # Filter to year range (Hansen lossyear is coded as year - 2000)
    start_code = start_year - 2000
    end_code = end_year - 2000

    loss_year = result.select("loss_year")
    in_range = loss_year.gte(start_code).And(loss_year.lte(end_code))

    return result.updateMask(in_range)

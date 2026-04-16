"""Build the JSON summary consumed by the reporting pipeline.

What: Aggregates outputs from all pipeline stages into a single structured
JSON per AOI, validated against the pydantic schema.
Why: Single source of truth for all downstream consumers (reports, HTML, maps).
Assumes: All upstream stages have completed and produced valid outputs.
Produces: A validated AOISummary JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from coffee_deforestation.config import AOIConfig, PROJECT_ROOT
from coffee_deforestation.data.validate_aoi import AOIValidationResult
from coffee_deforestation.stats.schema import (
    AblationResult,
    AblationSummary,
    AOIMetadata,
    AOISummary,
    BBoxSummary,
    ChangeDetectionSummary,
    DataCoverageSummary,
    HistoricalSummary,
    HotspotSummary,
    ModelMetrics,
    ValidationSummary,
)


def compute_per_year_stats(
    s2_composites: dict,
    s1_composites: dict,
    aoi: AOIConfig,
    hotspot_features: list[dict] | None = None,
) -> dict:
    """Compute per-year NDVI and VV means via GEE reduceRegion.

    Uses ee.Reducer.mean() on each annual composite to extract real
    time-series statistics rather than synthetic approximations.

    Returns dict with ndvi_by_year, vv_mean_by_year, and optionally
    per-hotspot NDVI trajectories for the top 10 hotspots.
    """
    import ee

    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.features.indices import compute_ndvi

    region = aoi_to_geometry(aoi)
    ndvi_by_year: dict[int, float] = {}
    vv_by_year: dict[int, float] = {}

    for year in sorted(s2_composites.keys()):
        s2 = s2_composites[year]
        try:
            ndvi = compute_ndvi(s2)
            stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1_000_000,
                bestEffort=True,
            ).getInfo()
            ndvi_by_year[year] = round(float(stats.get("ndvi", 0) or 0), 4)
        except Exception as e:
            logger.warning(f"NDVI reduceRegion failed for {year}: {e}")

    for year in sorted(s1_composites.keys()):
        s1 = s1_composites[year]
        try:
            stats = s1.select("VV").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1_000_000,
                bestEffort=True,
            ).getInfo()
            vv_by_year[year] = round(float(stats.get("VV", 0) or 0), 2)
        except Exception as e:
            logger.warning(f"VV reduceRegion failed for {year}: {e}")

    # Per-hotspot NDVI trajectories (top 10 only)
    hotspot_trajectories: dict[str, dict[int, float]] = {}
    if hotspot_features:
        for feat in hotspot_features[:10]:
            hid = feat["properties"]["hotspot_id"]
            trajectory: dict[int, float] = {}
            try:
                h_geom = ee.Geometry(feat["geometry"])
                for year in sorted(s2_composites.keys()):
                    ndvi = compute_ndvi(s2_composites[year])
                    val = ndvi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=h_geom,
                        scale=30,
                        maxPixels=100_000,
                        bestEffort=True,
                    ).getInfo()
                    trajectory[year] = round(float(val.get("ndvi", 0) or 0), 4)
                hotspot_trajectories[hid] = trajectory
            except Exception as e:
                logger.warning(f"Hotspot trajectory failed for {hid}: {e}")

    logger.info(
        f"Per-year stats: {len(ndvi_by_year)} NDVI years, "
        f"{len(vv_by_year)} VV years, {len(hotspot_trajectories)} hotspot trajectories"
    )
    return {
        "ndvi_by_year": ndvi_by_year,
        "vv_mean_by_year": vv_by_year,
        "hotspot_ndvi_trajectories": hotspot_trajectories,
    }


def build_summary(
    aoi: AOIConfig,
    validation_result: AOIValidationResult,
    hotspot_features: list[dict],
    years_processed: list[int],
    figures: list[str] | None = None,
    maps: list[str] | None = None,
    model_metrics: ModelMetrics | None = None,
    ablation_results: dict | None = None,
    historical_stats: dict | None = None,
    per_year_stats: dict | None = None,
) -> AOISummary:
    """Build a complete AOI summary from pipeline outputs."""
    logger.info(f"Building summary for {aoi.id}")

    # Metadata
    metadata = AOIMetadata(
        aoi_id=aoi.id,
        name=aoi.name,
        country=aoi.country,
        coffee_type=aoi.coffee_type,
        role=aoi.role,
        bbox=BBoxSummary(
            west=aoi.bbox.west,
            south=aoi.bbox.south,
            east=aoi.bbox.east,
            north=aoi.bbox.north,
        ),
        epsg_utm=aoi.epsg_utm,
    )

    # Validation
    validation = ValidationSummary(
        coffee_fraction=validation_result.coffee_fraction,
        forest_2000_fraction=validation_result.forest_2000_fraction,
        hansen_loss_pixels=validation_result.hansen_loss_pixels,
        passed=validation_result.passed,
    )

    # Data coverage
    data_coverage = DataCoverageSummary(
        years_processed=years_processed,
        s2_composite_count=len(years_processed),
        s1_composite_count=len(years_processed),
    )

    # Change detection summary — aggregate by loss year
    areas = [f["properties"].get("area_ha", 0) for f in hotspot_features]
    total_area = sum(areas)

    hotspots_by_loss_year: dict[str, int] = {}
    area_ha_by_loss_year: dict[str, float] = {}
    for f in hotspot_features:
        ly = f["properties"].get("loss_year")
        area = f["properties"].get("area_ha", 0)
        if ly is not None:
            key = str(ly)
            hotspots_by_loss_year[key] = hotspots_by_loss_year.get(key, 0) + 1
            area_ha_by_loss_year[key] = round(
                area_ha_by_loss_year.get(key, 0) + area, 2
            )

    change_detection = ChangeDetectionSummary(
        method="rule_based_hansen_fdp",
        total_hotspots=len(hotspot_features),
        total_area_ha=round(total_area, 2),
        largest_hotspot_ha=round(max(areas), 2) if areas else 0.0,
        smallest_hotspot_ha=round(min(areas), 2) if areas else 0.0,
        hotspots_by_loss_year=hotspots_by_loss_year,
        area_ha_by_loss_year=area_ha_by_loss_year,
    )

    # Per-hotspot NDVI trajectories from per_year_stats
    hotspot_trajectories = (
        per_year_stats.get("hotspot_ndvi_trajectories", {}) if per_year_stats else {}
    )

    # Top hotspots (up to 10)
    top_hotspots = [
        HotspotSummary(
            hotspot_id=f["properties"]["hotspot_id"],
            area_ha=f["properties"]["area_ha"],
            centroid_lon=f["properties"]["centroid_lon"],
            centroid_lat=f["properties"]["centroid_lat"],
            rank=f["properties"]["rank"],
            loss_year=f["properties"].get("loss_year"),
            ndvi_trajectory=hotspot_trajectories.get(
                f["properties"]["hotspot_id"]
            ),
        )
        for f in hotspot_features[:10]
    ]

    # ML model metrics (Stage 2)
    ml_metrics = model_metrics or ModelMetrics()

    # Ablation summary (Stage 2)
    ablation = AblationSummary()
    if ablation_results:
        ablation = AblationSummary(
            s1_only=AblationResult(
                f1_coffee=ablation_results.get("s1_only", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s1_only", {}).get("accuracy", 0.0),
            ),
            s2_only=AblationResult(
                f1_coffee=ablation_results.get("s2_only", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s2_only", {}).get("accuracy", 0.0),
            ),
            s1_s2=AblationResult(
                f1_coffee=ablation_results.get("s1_s2", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s1_s2", {}).get("accuracy", 0.0),
            ),
        )

    # Historical summary (Stage 2) + per-year time series (Stage 4)
    historical = HistoricalSummary()
    if historical_stats:
        historical = HistoricalSummary(
            was_forest_2000_fraction=float(
                historical_stats.get("was_forest_2000_mean", 0.0) or 0.0
            ),
            coffee_on_former_forest_fraction=float(
                historical_stats.get("coffee_on_former_forest_mean", 0.0) or 0.0
            ),
            mean_loss_year_offset=historical_stats.get("loss_year_before_coffee_mean"),
            replacement_class_distribution=historical_stats.get(
                "replacement_class_distribution", {}
            ),
        )
    if per_year_stats:
        historical.ndvi_by_year = per_year_stats.get("ndvi_by_year", {})
        historical.vv_mean_by_year = per_year_stats.get("vv_mean_by_year", {})

    return AOISummary(
        metadata=metadata,
        validation=validation,
        data_coverage=data_coverage,
        change_detection=change_detection,
        top_hotspots=top_hotspots,
        model_metrics=ml_metrics,
        ablation=ablation,
        historical=historical,
        figures=figures or [],
        maps=maps or [],
    )


def save_summary(summary: AOISummary, output_dir: Path | None = None) -> Path:
    """Save an AOI summary to JSON, validated against the schema."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"summary_{summary.metadata.aoi_id}.json"

    with open(output_path, "w") as f:
        f.write(summary.model_dump_json(indent=2))

    logger.info(f"Saved summary to {output_path}")
    return output_path

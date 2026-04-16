"""Pydantic models for inter-stage JSON contracts.

What: Defines the schema for the per-AOI stats summary JSON consumed by
the reporting agents and HTML reports.
Why: Enforces type safety and completeness at every pipeline boundary.
Assumes: All upstream stages produce data matching these schemas.
Produces: Validated pydantic models that serialize to/from JSON.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class BBoxSummary(BaseModel):
    west: float
    south: float
    east: float
    north: float


class AOIMetadata(BaseModel):
    aoi_id: str
    name: str
    country: str
    coffee_type: str
    role: str
    bbox: BBoxSummary
    epsg_utm: int


class ValidationSummary(BaseModel):
    coffee_fraction: float
    forest_2000_fraction: float
    hansen_loss_pixels: int
    passed: bool


class DataCoverageSummary(BaseModel):
    """Coverage statistics for the data acquisition stage."""
    years_processed: list[int]
    s2_composite_count: int = 0
    s1_composite_count: int = 0


class ChangeDetectionSummary(BaseModel):
    """Summary of coffee-linked deforestation detection."""
    method: str = "rule_based_hansen_fdp"
    total_hotspots: int = 0
    total_area_ha: float = 0.0
    largest_hotspot_ha: float = 0.0
    smallest_hotspot_ha: float = 0.0
    hotspots_by_loss_year: dict[str, int] = Field(default_factory=dict)
    area_ha_by_loss_year: dict[str, float] = Field(default_factory=dict)


class HotspotSummary(BaseModel):
    """Summary of a single hotspot polygon."""
    hotspot_id: str
    area_ha: float
    centroid_lon: float
    centroid_lat: float
    rank: int
    loss_year: int | None = None
    ndvi_trajectory: dict[int, float] | None = None


class ModelMetrics(BaseModel):
    """ML model evaluation metrics (populated in Stage 2)."""
    model_type: str = "none"
    accuracy: float = 0.0
    f1_coffee: float = 0.0
    precision_coffee: float = 0.0
    recall_coffee: float = 0.0


class AblationResult(BaseModel):
    """Metrics for one ablation condition."""
    f1_coffee: float = 0.0
    accuracy: float = 0.0


class AblationSummary(BaseModel):
    """S1-only / S2-only / S1+S2 ablation results."""
    s1_only: AblationResult = Field(default_factory=AblationResult)
    s2_only: AblationResult = Field(default_factory=AblationResult)
    s1_s2: AblationResult = Field(default_factory=AblationResult)


class HistoricalSummary(BaseModel):
    """Per-AOI historical forest trajectory summary (Stage 2)."""
    was_forest_2000_fraction: float = 0.0
    coffee_on_former_forest_fraction: float = 0.0
    mean_loss_year_offset: float | None = None  # mean offset from 2000 (e.g. 12 = 2012)
    replacement_class_distribution: dict[str, float] = Field(default_factory=dict)
    ndvi_by_year: dict[int, float] = Field(default_factory=dict)
    vv_mean_by_year: dict[int, float] = Field(default_factory=dict)


class DeforestationAttribution(BaseModel):
    """Breakdown of ALL forest loss by replacement land cover (not just coffee)."""
    total_loss_ha: float = 0.0
    coffee_pct: float = 0.0
    other_crops_pct: float = 0.0
    built_industrial_pct: float = 0.0
    bare_degraded_pct: float = 0.0
    water_pct: float = 0.0
    regrowth_pct: float = 0.0
    by_year: dict[int, dict[str, float]] = Field(default_factory=dict)


class YearlyLossStats(BaseModel):
    """Per-year deforestation statistics from Hansen GFC."""
    total_loss_ha: float = 0.0
    coffee_loss_ha: float = 0.0
    coffee_fraction: float = 0.0


class AOISummary(BaseModel):
    """Complete per-AOI summary consumed by reporting agents."""
    metadata: AOIMetadata
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "0.1.0"
    validation: ValidationSummary
    data_coverage: DataCoverageSummary
    change_detection: ChangeDetectionSummary
    top_hotspots: list[HotspotSummary] = Field(default_factory=list)
    model_metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    ablation: AblationSummary = Field(default_factory=AblationSummary)
    historical: HistoricalSummary = Field(default_factory=HistoricalSummary)
    deforestation_attribution: DeforestationAttribution = Field(
        default_factory=DeforestationAttribution
    )
    yearly_loss: dict[int, YearlyLossStats] = Field(default_factory=dict)
    figures: list[str] = Field(default_factory=list)
    maps: list[str] = Field(default_factory=list)

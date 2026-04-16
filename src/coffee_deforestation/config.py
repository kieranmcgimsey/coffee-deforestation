"""Configuration management for coffee deforestation pipeline.

What: Loads and validates YAML config files (aois.yaml, pipeline.yaml) and environment
variables into typed pydantic models.
Why: Single source of truth for all configuration, with validation at load time.
Assumes: Config files exist at config/ relative to the project root.
Produces: Typed AOIConfig, PipelineConfig, and Settings objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _project_root() -> Path:
    """Walk up from this file to find the project root (where pyproject.toml lives)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _project_root()


# --- AOI models ---


class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    def to_list(self) -> list[float]:
        """Return [west, south, east, north] for GEE."""
        return [self.west, self.south, self.east, self.north]

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south


class DrySeason(BaseModel):
    start_month: int
    end_month: int
    cross_year: bool = False


class PatchConfig(BaseModel):
    """A sub-region patch within a larger AOI."""
    name: str
    bbox: BBox


class AOIConfig(BaseModel):
    id: str = ""
    name: str
    country: str
    coffee_type: str
    role: str
    bbox: BBox
    dry_season: DrySeason
    epsg_utm: int = 0  # Auto-computed from bbox if not specified
    region_bbox: BBox | None = None  # country-scale envelope for basemap context
    patches: list[PatchConfig] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute UTM zone from bbox center if not specified."""
        if self.epsg_utm == 0:
            self.epsg_utm = _utm_epsg_from_bbox(self.bbox)

    def get_effective_patches(self) -> list[PatchConfig]:
        """Return patches list; if empty, wrap the main bbox as a single patch."""
        if self.patches:
            return self.patches
        return [PatchConfig(name=self.name, bbox=self.bbox)]


def _utm_epsg_from_bbox(bbox: BBox) -> int:
    """Compute the UTM EPSG code from a bounding box center.

    Saves users from having to look up UTM zones manually.
    Formula: zone = floor((lon + 180) / 6) + 1
    EPSG = 32600 + zone (northern) or 32700 + zone (southern)
    """
    center_lon = (bbox.west + bbox.east) / 2
    center_lat = (bbox.south + bbox.north) / 2
    zone = int((center_lon + 180) / 6) + 1
    if center_lat >= 0:
        return 32600 + zone  # Northern hemisphere
    return 32700 + zone  # Southern hemisphere


# --- Pipeline models ---


class CloudMaskingConfig(BaseModel):
    cloud_probability_threshold: int = 40
    cloud_dilation_m: int = 50


class S1ProcessingConfig(BaseModel):
    speckle_filter_radius_m: int = 50
    polarizations: list[str] = Field(default_factory=lambda: ["VV", "VH"])
    orbit_pass: str = "DESCENDING"


class FeaturesConfig(BaseModel):
    spectral_indices: list[str] = Field(default_factory=list)
    sar_features: list[str] = Field(default_factory=list)
    contextual: list[str] = Field(default_factory=list)


class RFConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 5
    random_state: int = 42


class XGBConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42


class MLConfig(BaseModel):
    random_forest: RFConfig = Field(default_factory=RFConfig)
    xgboost: XGBConfig = Field(default_factory=XGBConfig)
    samples_per_class_per_aoi: int = 10000
    test_split: float = 0.2
    classes: list[str] = Field(default_factory=list)


class ChangeDetectionConfig(BaseModel):
    fdp_coffee_threshold: float = 0.5
    min_hotspot_area_ha: float = 0.5
    hansen_treecover_2000_threshold: int = 50


class ValidationConfig(BaseModel):
    min_coffee_fraction: float = 0.02
    min_forest_2000_fraction: float = 0.10
    min_hansen_loss_pixels: int = 100


class CacheConfig(BaseModel):
    base_dir: str = "outputs/cache"
    hash_length: int = 16


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "outputs/logs"


class TemporalConfig(BaseModel):
    years: list[int] = Field(default_factory=lambda: [2019, 2020, 2021, 2022, 2023, 2024])


class PipelineConfig(BaseModel):
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    cloud_masking: CloudMaskingConfig = Field(default_factory=CloudMaskingConfig)
    s1_processing: S1ProcessingConfig = Field(default_factory=S1ProcessingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    change_detection: ChangeDetectionConfig = Field(default_factory=ChangeDetectionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# --- Environment settings ---


class Settings(BaseSettings):
    gee_service_account_key_path: str = ""
    gee_project: str = ""
    google_drive_export_folder: str = "coffee_deforestation_exports"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    llm_dry_run: bool = True

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


# --- Loaders ---


def load_aois(path: Path | None = None) -> dict[str, AOIConfig]:
    """Load AOI configurations from YAML."""
    if path is None:
        path = PROJECT_ROOT / "config" / "aois.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    aois: dict[str, AOIConfig] = {}
    for aoi_id, data in raw["aois"].items():
        aois[aoi_id] = AOIConfig(id=aoi_id, **data)
    return aois


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from YAML."""
    if path is None:
        path = PROJECT_ROOT / "config" / "pipeline.yaml"
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return PipelineConfig(**raw)


def load_settings() -> Settings:
    """Load environment settings."""
    return Settings()

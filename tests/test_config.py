"""Tests for configuration loading and validation."""

from coffee_deforestation.config import (
    AOIConfig,
    PipelineConfig,
    load_aois,
    load_pipeline_config,
)


def test_load_aois():
    """Verify all 3 AOIs load correctly from YAML."""
    aois = load_aois()
    assert len(aois) == 3
    assert "lam_dong" in aois
    assert "huila" in aois
    assert "sul_de_minas" in aois


def test_aoi_fields():
    """Verify AOI fields are populated."""
    aois = load_aois()
    lam_dong = aois["lam_dong"]
    assert lam_dong.name == "Lâm Đồng"
    assert lam_dong.country == "Vietnam"
    assert lam_dong.bbox.west < lam_dong.bbox.east
    assert lam_dong.bbox.south < lam_dong.bbox.north
    assert lam_dong.dry_season.cross_year is True


def test_bbox_to_list():
    """Verify bbox serialization for GEE."""
    aois = load_aois()
    bbox_list = aois["lam_dong"].bbox.to_list()
    assert len(bbox_list) == 4
    assert bbox_list[0] < bbox_list[2]  # west < east


def test_pipeline_config():
    """Verify pipeline config loads with expected defaults."""
    config = load_pipeline_config()
    assert len(config.temporal.years) == 6
    assert config.cloud_masking.cloud_probability_threshold == 40
    assert config.ml.random_forest.n_estimators == 500
    assert config.change_detection.fdp_coffee_threshold == 0.5


def test_pipeline_config_features():
    """Verify feature configuration."""
    config = load_pipeline_config()
    assert "ndvi" in config.features.spectral_indices
    assert "vv_median" in config.features.sar_features
    assert "elevation" in config.features.contextual

"""Test fixtures: mock GEE, mock Anthropic, synthetic data."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coffee_deforestation.config import (
    AOIConfig,
    BBox,
    DrySeason,
    PipelineConfig,
    load_pipeline_config,
)


@pytest.fixture
def sample_aoi() -> AOIConfig:
    """A minimal AOI config for testing."""
    return AOIConfig(
        id="test_aoi",
        name="Test AOI",
        country="Testland",
        coffee_type="Testusta",
        role="Unit test",
        bbox=BBox(west=108.3, south=11.9, east=108.4, north=12.0),
        dry_season=DrySeason(start_month=12, end_month=3, cross_year=True),
        epsg_utm=32648,
    )


@pytest.fixture
def sahara_aoi() -> AOIConfig:
    """An AOI in the Sahara desert — should fail validation."""
    return AOIConfig(
        id="sahara",
        name="Sahara Desert",
        country="Algeria",
        coffee_type="None",
        role="Should fail validation",
        bbox=BBox(west=2.0, south=28.0, east=2.3, north=28.3),
        dry_season=DrySeason(start_month=6, end_month=9, cross_year=False),
        epsg_utm=32631,
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Load the real pipeline config for testing."""
    return load_pipeline_config()


@pytest.fixture
def mock_ee():
    """Mock the ee module for tests that don't need real GEE."""
    with patch("coffee_deforestation.data.gee_client.ee") as mock:
        mock.Geometry.Rectangle.return_value = MagicMock()
        mock.Image.return_value = MagicMock()
        mock.ImageCollection.return_value = MagicMock()
        yield mock


@pytest.fixture
def sample_hotspot_features() -> list[dict]:
    """Sample hotspot GeoJSON features for testing."""
    return [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[108.3, 11.95], [108.31, 11.95],
                                  [108.31, 11.96], [108.3, 11.96],
                                  [108.3, 11.95]]],
            },
            "properties": {
                "hotspot_id": "test_aoi_h001",
                "aoi_id": "test_aoi",
                "area_ha": 25.5,
                "centroid_lon": 108.305,
                "centroid_lat": 11.955,
                "rank": 1,
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[108.35, 11.92], [108.36, 11.92],
                                  [108.36, 11.93], [108.35, 11.93],
                                  [108.35, 11.92]]],
            },
            "properties": {
                "hotspot_id": "test_aoi_h002",
                "aoi_id": "test_aoi",
                "area_ha": 12.3,
                "centroid_lon": 108.355,
                "centroid_lat": 11.925,
                "rank": 2,
            },
        },
    ]


@pytest.fixture
def sample_raster_data() -> dict[str, np.ndarray]:
    """Synthetic raster data for visualization testing."""
    shape = (100, 100)
    rng = np.random.default_rng(42)
    return {
        "red": rng.uniform(0.02, 0.15, shape).astype(np.float32),
        "green": rng.uniform(0.03, 0.12, shape).astype(np.float32),
        "blue": rng.uniform(0.02, 0.08, shape).astype(np.float32),
        "ndvi": rng.uniform(-0.1, 0.9, shape).astype(np.float32),
        "vv": rng.uniform(-20, -5, shape).astype(np.float32),
        "coffee_prob": rng.uniform(0, 1, shape).astype(np.float32),
        "loss_year": rng.choice([0, 0, 0, 5, 10, 15, 20], shape).astype(np.int16),
    }


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test artifacts."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output

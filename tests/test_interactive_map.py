"""Tests for interactive folium map generation."""

import json
from pathlib import Path

from coffee_deforestation.config import AOIConfig
from coffee_deforestation.viz.interactive import create_aoi_map, save_map


def test_create_aoi_map(sample_aoi: AOIConfig):
    """Map creates without errors."""
    m = create_aoi_map(sample_aoi)
    assert m is not None


def test_create_map_with_hotspots(
    sample_aoi: AOIConfig,
    sample_hotspot_features: list[dict],
    tmp_path: Path,
):
    """Map creates with hotspot overlay."""
    geojson_path = tmp_path / "hotspots.geojson"
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": sample_hotspot_features}, f)

    m = create_aoi_map(sample_aoi, hotspot_geojson_path=geojson_path)
    assert m is not None


def test_save_map(sample_aoi: AOIConfig, tmp_path: Path):
    """Map saves as HTML."""
    m = create_aoi_map(sample_aoi)
    path = save_map(m, sample_aoi, output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".html"
    content = path.read_text()
    assert "folium" in content.lower() or "leaflet" in content.lower()

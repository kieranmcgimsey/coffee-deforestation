"""Tests for hotspot polygonization and enrichment."""

import json
from pathlib import Path

from coffee_deforestation.change.hotspots import enrich_hotspots, save_hotspots
from coffee_deforestation.config import AOIConfig


def test_enrich_hotspots(sample_aoi: AOIConfig, sample_hotspot_features: list[dict]):
    """Enriched hotspots have correct fields and ranking."""
    # Create raw features (without enrichment fields)
    raw = [
        {
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {"area_m2": f["properties"]["area_ha"] * 10000},
        }
        for f in sample_hotspot_features
    ]

    from unittest.mock import MagicMock
    candidates = MagicMock()

    enriched = enrich_hotspots(raw, sample_aoi, candidates)

    assert len(enriched) == 2
    # Should be sorted by area (largest first)
    assert enriched[0]["properties"]["area_ha"] >= enriched[1]["properties"]["area_ha"]
    # Rank should be assigned
    assert enriched[0]["properties"]["rank"] == 1
    assert enriched[1]["properties"]["rank"] == 2
    # IDs should be prefixed with AOI
    assert enriched[0]["properties"]["hotspot_id"].startswith("test_aoi_")


def test_save_hotspots(
    sample_aoi: AOIConfig,
    sample_hotspot_features: list[dict],
    tmp_path: Path,
):
    """Hotspots save as valid GeoJSON."""
    output_path = tmp_path / "hotspots.geojson"
    save_hotspots(sample_hotspot_features, output_path, sample_aoi)

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    assert data["properties"]["aoi_id"] == "test_aoi"

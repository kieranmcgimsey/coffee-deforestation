"""Tests for pydantic stats schema validation."""

import json

from coffee_deforestation.stats.schema import (
    AOIMetadata,
    AOISummary,
    BBoxSummary,
    ChangeDetectionSummary,
    DataCoverageSummary,
    HotspotSummary,
    ValidationSummary,
)


def _make_summary() -> AOISummary:
    """Create a minimal valid AOISummary."""
    return AOISummary(
        metadata=AOIMetadata(
            aoi_id="test",
            name="Test",
            country="Testland",
            coffee_type="Testusta",
            role="test",
            bbox=BBoxSummary(west=1, south=2, east=3, north=4),
            epsg_utm=32648,
        ),
        validation=ValidationSummary(
            coffee_fraction=0.05,
            forest_2000_fraction=0.30,
            hansen_loss_pixels=500,
            passed=True,
        ),
        data_coverage=DataCoverageSummary(
            years_processed=[2023, 2024],
            s2_composite_count=2,
            s1_composite_count=2,
        ),
        change_detection=ChangeDetectionSummary(
            total_hotspots=5,
            total_area_ha=150.0,
            largest_hotspot_ha=50.0,
            smallest_hotspot_ha=2.0,
        ),
        top_hotspots=[
            HotspotSummary(
                hotspot_id="test_h001",
                area_ha=50.0,
                centroid_lon=2.0,
                centroid_lat=3.0,
                rank=1,
            ),
        ],
    )


def test_summary_creation():
    """AOISummary creates successfully with valid data."""
    summary = _make_summary()
    assert summary.metadata.aoi_id == "test"
    assert summary.change_detection.total_hotspots == 5


def test_summary_serialization():
    """AOISummary serializes to and from JSON."""
    summary = _make_summary()
    json_str = summary.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["metadata"]["aoi_id"] == "test"

    # Round-trip
    restored = AOISummary.model_validate_json(json_str)
    assert restored.metadata.aoi_id == summary.metadata.aoi_id


def test_summary_defaults():
    """AOISummary fills defaults correctly."""
    summary = _make_summary()
    assert summary.model_metrics.model_type == "none"
    assert summary.figures == []
    assert summary.pipeline_version == "0.1.0"


def test_hotspot_summary():
    """HotspotSummary validates correctly."""
    hs = HotspotSummary(
        hotspot_id="test_h001",
        area_ha=25.5,
        centroid_lon=108.305,
        centroid_lat=11.955,
        rank=1,
    )
    assert hs.area_ha == 25.5


def test_change_detection_defaults():
    """ChangeDetectionSummary fills defaults."""
    cd = ChangeDetectionSummary()
    assert cd.method == "rule_based_hansen_fdp"
    assert cd.total_hotspots == 0
    assert cd.hotspots_by_loss_year == {}

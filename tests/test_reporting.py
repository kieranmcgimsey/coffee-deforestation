"""Tests for report generation and saving."""

from pathlib import Path

import pytest

from coffee_deforestation.reporting.llm_client import generate_report, save_report
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
    return AOISummary(
        metadata=AOIMetadata(
            aoi_id="test",
            name="Test Region",
            country="Testland",
            coffee_type="Testusta",
            role="test",
            bbox=BBoxSummary(west=1, south=2, east=3, north=4),
            epsg_utm=32648,
        ),
        validation=ValidationSummary(
            coffee_fraction=0.10,
            forest_2000_fraction=0.40,
            hansen_loss_pixels=2000,
            passed=True,
        ),
        data_coverage=DataCoverageSummary(
            years_processed=[2023, 2024],
        ),
        change_detection=ChangeDetectionSummary(
            total_hotspots=3,
            total_area_ha=100.0,
            largest_hotspot_ha=50.0,
            smallest_hotspot_ha=5.0,
        ),
        top_hotspots=[
            HotspotSummary(
                hotspot_id="test_h001", area_ha=50.0,
                centroid_lon=2.0, centroid_lat=3.0, rank=1,
            ),
        ],
    )


def test_generate_dry_run():
    """Dry run produces markdown report."""
    report = generate_report(_make_summary(), dry_run=True)
    assert isinstance(report, str)
    assert len(report) > 100
    assert "# Test Region" in report


def test_real_llm_raises_without_key():
    """Real LLM call raises ValueError when no API key is configured."""
    with pytest.raises((ValueError, NotImplementedError)):
        generate_report(_make_summary(), dry_run=False)


def test_save_report(tmp_path: Path):
    """Report saves to disk."""
    report = generate_report(_make_summary(), dry_run=True)
    path = save_report(report, "test", output_dir=tmp_path)
    assert path.exists()
    assert path.read_text() == report

"""Tests for LLM dry-run report generation."""

from coffee_deforestation.reporting.llm_client import generate_report
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
            aoi_id="lam_dong",
            name="Lâm Đồng",
            country="Vietnam",
            coffee_type="Robusta, highland",
            role="Primary showcase",
            bbox=BBoxSummary(west=108.2, south=11.8, east=108.5, north=12.1),
            epsg_utm=32648,
        ),
        validation=ValidationSummary(
            coffee_fraction=0.15,
            forest_2000_fraction=0.45,
            hansen_loss_pixels=5000,
            passed=True,
        ),
        data_coverage=DataCoverageSummary(
            years_processed=[2019, 2020, 2021, 2022, 2023, 2024],
            s2_composite_count=6,
            s1_composite_count=6,
        ),
        change_detection=ChangeDetectionSummary(
            total_hotspots=15,
            total_area_ha=350.0,
            largest_hotspot_ha=80.0,
            smallest_hotspot_ha=0.8,
        ),
        top_hotspots=[
            HotspotSummary(
                hotspot_id="lam_dong_h001",
                area_ha=80.0,
                centroid_lon=108.35,
                centroid_lat=11.95,
                rank=1,
            ),
        ],
    )


def test_dry_run_report_generated():
    """Dry-run report is generated from template."""
    summary = _make_summary()
    report = generate_report(summary, dry_run=True)
    assert "Lâm Đồng" in report
    assert "Vietnam" in report
    assert "15" in report  # hotspot count
    assert "350.0" in report  # total area


def test_dry_run_report_structure():
    """Dry-run report has expected sections."""
    summary = _make_summary()
    report = generate_report(summary, dry_run=True)
    report_lower = report.lower()
    assert "executive summary" in report_lower
    assert "area context" in report_lower
    assert "headline findings" in report_lower
    assert "hotspot" in report_lower
    assert "methodology" in report_lower


def test_dry_run_report_numbers():
    """Dry-run report includes actual numbers from summary."""
    summary = _make_summary()
    report = generate_report(summary, dry_run=True)
    assert "80.0" in report  # largest hotspot
    assert "15.0%" in report  # coffee fraction
    assert "45.0%" in report  # forest fraction

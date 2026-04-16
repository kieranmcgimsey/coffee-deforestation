"""Tests for stats summarization."""

from pathlib import Path

from coffee_deforestation.config import AOIConfig
from coffee_deforestation.data.validate_aoi import AOIValidationResult
from coffee_deforestation.stats.summarize import build_summary, save_summary


def test_build_summary(sample_aoi: AOIConfig, sample_hotspot_features: list[dict]):
    """Summary builds correctly from pipeline outputs."""
    validation = AOIValidationResult(
        aoi_id="test_aoi",
        coffee_fraction=0.08,
        forest_2000_fraction=0.35,
        hansen_loss_pixels=1000,
        passed=True,
        messages=["OK"],
    )
    summary = build_summary(
        sample_aoi,
        validation,
        sample_hotspot_features,
        years_processed=[2023, 2024],
    )
    assert summary.metadata.aoi_id == "test_aoi"
    assert summary.change_detection.total_hotspots == 2
    assert summary.change_detection.total_area_ha > 0
    assert len(summary.top_hotspots) == 2


def test_save_summary(sample_aoi: AOIConfig, sample_hotspot_features: list[dict], tmp_path: Path):
    """Summary saves as valid JSON."""
    validation = AOIValidationResult(
        aoi_id="test_aoi",
        coffee_fraction=0.08,
        forest_2000_fraction=0.35,
        hansen_loss_pixels=1000,
        passed=True,
        messages=["OK"],
    )
    summary = build_summary(sample_aoi, validation, sample_hotspot_features, [2023])
    path = save_summary(summary, output_dir=tmp_path)
    assert path.exists()

    import json
    with open(path) as f:
        data = json.load(f)
    assert data["metadata"]["aoi_id"] == "test_aoi"

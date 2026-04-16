"""Tests for Stage 4: multi-temporal analysis, ML prediction, temporal stats."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coffee_deforestation.config import AOIConfig
from coffee_deforestation.stats.schema import (
    AOIMetadata,
    AOISummary,
    BBoxSummary,
    ChangeDetectionSummary,
    DataCoverageSummary,
    HistoricalSummary,
    HotspotSummary,
    ValidationSummary,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaTemporalFields:
    """Verify new temporal fields serialize/deserialize correctly."""

    def test_change_detection_area_by_year(self):
        cd = ChangeDetectionSummary(
            total_hotspots=100,
            total_area_ha=500.0,
            area_ha_by_loss_year={"2019": 100.0, "2020": 200.0, "2023": 200.0},
        )
        dumped = cd.model_dump()
        assert dumped["area_ha_by_loss_year"]["2020"] == 200.0
        reloaded = ChangeDetectionSummary.model_validate(dumped)
        assert reloaded.area_ha_by_loss_year == cd.area_ha_by_loss_year

    def test_hotspot_summary_loss_year(self):
        hs = HotspotSummary(
            hotspot_id="h001", area_ha=10.0,
            centroid_lon=108.3, centroid_lat=11.9, rank=1,
            loss_year=2021,
            ndvi_trajectory={2019: 0.7, 2020: 0.65, 2021: 0.4, 2022: 0.45},
        )
        dumped = hs.model_dump()
        assert dumped["loss_year"] == 2021
        assert dumped["ndvi_trajectory"][2021] == 0.4
        reloaded = HotspotSummary.model_validate(dumped)
        assert reloaded.loss_year == 2021

    def test_historical_ndvi_by_year(self):
        h = HistoricalSummary(
            ndvi_by_year={2019: 0.65, 2020: 0.63},
            vv_mean_by_year={2019: -12.0, 2020: -11.5},
        )
        dumped = h.model_dump()
        assert dumped["ndvi_by_year"][2019] == 0.65
        reloaded = HistoricalSummary.model_validate(dumped)
        assert reloaded.vv_mean_by_year[2020] == -11.5

    def test_defaults_backward_compat(self):
        """New fields default to empty so old data still loads."""
        cd = ChangeDetectionSummary()
        assert cd.area_ha_by_loss_year == {}
        hs = HotspotSummary(
            hotspot_id="h001", area_ha=5.0,
            centroid_lon=0.0, centroid_lat=0.0, rank=1,
        )
        assert hs.loss_year is None
        assert hs.ndvi_trajectory is None
        h = HistoricalSummary()
        assert h.ndvi_by_year == {}


# ---------------------------------------------------------------------------
# Feature stack tests
# ---------------------------------------------------------------------------


class TestTemporalFeatureStack:
    """Verify temporal bands are wired into the feature stack."""

    def test_feature_band_order_has_temporal(self):
        from coffee_deforestation.features.stack import FEATURE_BAND_ORDER
        assert "ndvi_delta" in FEATURE_BAND_ORDER
        assert "vv_stddev" in FEATURE_BAND_ORDER
        assert "vh_stddev" in FEATURE_BAND_ORDER
        assert len(FEATURE_BAND_ORDER) == 19

    def test_temporal_sar_features_defined(self):
        from coffee_deforestation.features.sar_features import compute_temporal_sar_features
        assert callable(compute_temporal_sar_features)


# ---------------------------------------------------------------------------
# Area-by-loss-year aggregation
# ---------------------------------------------------------------------------


class TestAreaByLossYear:
    """Test area_ha_by_loss_year aggregation in build_summary."""

    def test_aggregation(self, sample_aoi):
        from coffee_deforestation.data.validate_aoi import AOIValidationResult
        from coffee_deforestation.stats.summarize import build_summary

        val = AOIValidationResult(
            aoi_id="test", coffee_fraction=0.2,
            forest_2000_fraction=0.5, hansen_loss_pixels=1000, passed=True,
            messages=[],
        )
        hotspots = [
            {
                "properties": {
                    "hotspot_id": "h001", "area_ha": 10.0,
                    "centroid_lon": 108.3, "centroid_lat": 11.9,
                    "rank": 1, "loss_year": 2020,
                },
            },
            {
                "properties": {
                    "hotspot_id": "h002", "area_ha": 20.0,
                    "centroid_lon": 108.4, "centroid_lat": 11.8,
                    "rank": 2, "loss_year": 2020,
                },
            },
            {
                "properties": {
                    "hotspot_id": "h003", "area_ha": 15.0,
                    "centroid_lon": 108.5, "centroid_lat": 11.7,
                    "rank": 3, "loss_year": 2021,
                },
            },
        ]
        summary = build_summary(
            sample_aoi, val, hotspots,
            years_processed=[2019, 2020, 2021],
        )
        assert summary.change_detection.area_ha_by_loss_year["2020"] == 30.0
        assert summary.change_detection.area_ha_by_loss_year["2021"] == 15.0
        assert summary.change_detection.hotspots_by_loss_year["2020"] == 2
        assert summary.change_detection.hotspots_by_loss_year["2021"] == 1

    def test_hotspot_loss_year_in_top(self, sample_aoi):
        from coffee_deforestation.data.validate_aoi import AOIValidationResult
        from coffee_deforestation.stats.summarize import build_summary

        val = AOIValidationResult(
            aoi_id="test", coffee_fraction=0.2,
            forest_2000_fraction=0.5, hansen_loss_pixels=1000, passed=True,
            messages=[],
        )
        hotspots = [
            {
                "properties": {
                    "hotspot_id": "h001", "area_ha": 10.0,
                    "centroid_lon": 108.3, "centroid_lat": 11.9,
                    "rank": 1, "loss_year": 2022,
                },
            },
        ]
        summary = build_summary(sample_aoi, val, hotspots, years_processed=[2022])
        assert summary.top_hotspots[0].loss_year == 2022


# ---------------------------------------------------------------------------
# Compare periods with real data
# ---------------------------------------------------------------------------


class TestComparePeriodsSt4:
    """compare_periods uses real data from summary JSON."""

    def test_ndvi_metric(self, tmp_path):
        from coffee_deforestation.reporting.tools.compare_periods import compare_periods

        summary = _make_test_summary()
        _write_json(tmp_path, "lam_dong", summary)
        with patch("coffee_deforestation.reporting.tools.compare_periods.PROJECT_ROOT", tmp_path):
            result = compare_periods(2019, 2023, "ndvi_mean", "lam_dong")
        assert "value_a" in result
        assert result["value_a"] == 0.65
        assert result["value_b"] == 0.58

    def test_loss_cumulative(self, tmp_path):
        from coffee_deforestation.reporting.tools.compare_periods import compare_periods

        summary = _make_test_summary()
        _write_json(tmp_path, "lam_dong", summary)
        with patch("coffee_deforestation.reporting.tools.compare_periods.PROJECT_ROOT", tmp_path):
            result = compare_periods(2019, 2023, "loss_cumulative_ha", "lam_dong")
        assert "value_a" in result
        # Cumulative: 200 → 200+400+600+800+1000 = 3000
        assert result["value_b"] > result["value_a"]

    def test_missing_metric_returns_error(self, tmp_path):
        from coffee_deforestation.reporting.tools.compare_periods import compare_periods

        # Summary with no per-year data
        summary = {"metadata": {"aoi_id": "lam_dong"}, "change_detection": {}, "historical": {}}
        (tmp_path / "outputs" / "stats").mkdir(parents=True)
        import json
        (tmp_path / "outputs" / "stats" / "summary_lam_dong.json").write_text(json.dumps(summary))
        with patch("coffee_deforestation.reporting.tools.compare_periods.PROJECT_ROOT", tmp_path):
            result = compare_periods(2019, 2023, "ndvi_mean", "lam_dong")
        assert "error" in result


# ---------------------------------------------------------------------------
# Hotspot details with real data
# ---------------------------------------------------------------------------


class TestHotspotDetailsSt4:
    """hotspot_details uses real stats from JSON."""

    def test_returns_real_ndvi(self, tmp_path):
        from coffee_deforestation.reporting.tools.hotspot_details import get_hotspot_details

        _write_geojson(tmp_path, "lam_dong", "lam_dong_h001")
        summary = _make_test_summary()
        # Add ndvi_trajectory to first hotspot
        summary["top_hotspots"][0]["ndvi_trajectory"] = {
            "2019": 0.7, "2020": 0.65, "2021": 0.4,
        }
        _write_json(tmp_path, "lam_dong", summary)

        with patch("coffee_deforestation.reporting.tools.hotspot_details.PROJECT_ROOT", tmp_path):
            result = get_hotspot_details("lam_dong_h001", "lam_dong")
        assert "error" not in result
        assert result["ndvi_series"]["2019"] == 0.7
        assert result["ndvi_series"]["2021"] == 0.4

    def test_falls_back_to_aoi_ndvi(self, tmp_path):
        from coffee_deforestation.reporting.tools.hotspot_details import get_hotspot_details

        _write_geojson(tmp_path, "lam_dong", "lam_dong_h001")
        summary = _make_test_summary()
        # No per-hotspot trajectory, but AOI-wide exists
        _write_json(tmp_path, "lam_dong", summary)

        with patch("coffee_deforestation.reporting.tools.hotspot_details.PROJECT_ROOT", tmp_path):
            result = get_hotspot_details("lam_dong_h001", "lam_dong")
        assert "error" not in result
        # Falls back to AOI-wide NDVI
        assert "2019" in result["ndvi_series"]


# ---------------------------------------------------------------------------
# Viz: new figure functions
# ---------------------------------------------------------------------------


class TestNewFigures:
    """Test classification map, area-by-year, and NDVI trajectory figures."""

    def test_plot_classification_map(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_classification_map

        raster = np.array([[0, 1, 2], [3, 4, -1], [0, 1, 0]], dtype=np.int16)
        path = plot_classification_map(
            raster, sample_aoi, output_path=str(tmp_path / "class.png")
        )
        assert Path(path).exists()

    def test_plot_area_by_year(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_area_by_year

        data = {"2019": 100.0, "2020": 200.0, "2021": 350.0, "2022": 500.0}
        path = plot_area_by_year(
            data, sample_aoi, output_path=str(tmp_path / "area.png")
        )
        assert Path(path).exists()

    def test_plot_area_by_year_empty(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_area_by_year

        path = plot_area_by_year(
            {}, sample_aoi, output_path=str(tmp_path / "area_empty.png")
        )
        assert Path(path).exists()

    def test_plot_ndvi_trajectory(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_ndvi_trajectory

        data = {2019: 0.65, 2020: 0.63, 2021: 0.61, 2022: 0.59}
        path = plot_ndvi_trajectory(
            data, sample_aoi, output_path=str(tmp_path / "ndvi.png")
        )
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# ML evaluate ablation with clipped indices
# ---------------------------------------------------------------------------


def test_ablation_clips_indices():
    """Ablation handles feature data with fewer columns than expected."""
    from coffee_deforestation.ml.evaluate import run_ablation

    rng = np.random.default_rng(42)
    # 16-feature legacy data (not 19)
    X = rng.random((200, 16))
    y = rng.choice([0, 1, 2], 200)

    from sklearn.ensemble import RandomForestClassifier
    results = run_ablation(
        RandomForestClassifier, {"n_estimators": 10, "random_state": 42},
        X[:150], y[:150], X[150:], y[150:],
    )
    assert "s1_only" in results
    assert "s2_only" in results
    assert "s1_s2" in results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_summary() -> dict:
    """Create a summary dict with per-year fields."""
    return {
        "metadata": {"aoi_id": "lam_dong", "name": "Lam Dong", "country": "Vietnam"},
        "validation": {"coffee_fraction": 0.368, "forest_2000_fraction": 0.558},
        "change_detection": {
            "total_hotspots": 1985,
            "total_area_ha": 3603.0,
            "hotspots_by_loss_year": {
                "2019": 100, "2020": 200, "2021": 300, "2022": 400, "2023": 500,
            },
            "area_ha_by_loss_year": {
                "2019": 200.0, "2020": 400.0, "2021": 600.0, "2022": 800.0, "2023": 1000.0,
            },
        },
        "historical": {
            "ndvi_by_year": {
                "2019": 0.65, "2020": 0.63, "2021": 0.61, "2022": 0.59, "2023": 0.58,
            },
            "vv_mean_by_year": {
                "2019": -11.5, "2020": -11.8, "2021": -12.0, "2022": -12.2, "2023": -12.5,
            },
        },
        "top_hotspots": [
            {
                "hotspot_id": "lam_dong_h001",
                "area_ha": 116.0,
                "centroid_lon": 108.3,
                "centroid_lat": 11.93,
                "rank": 1,
            },
        ],
    }


def _write_json(tmp_path: Path, aoi_id: str, data: dict) -> None:
    import json
    stats_dir = tmp_path / "outputs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"summary_{aoi_id}.json").write_text(json.dumps(data))


def _write_geojson(tmp_path: Path, aoi_id: str, hotspot_id: str) -> None:
    import json
    vec_dir = tmp_path / "outputs" / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[108.3, 11.9], [108.31, 11.9], [108.31, 11.91], [108.3, 11.91], [108.3, 11.9]]
                    ],
                },
                "properties": {
                    "hotspot_id": hotspot_id,
                    "area_ha": 116.0,
                    "centroid_lon": 108.305,
                    "centroid_lat": 11.905,
                    "rank": 1,
                    "loss_year": 2021,
                },
            }
        ],
    }
    (vec_dir / f"hotspots_{aoi_id}.geojson").write_text(json.dumps(geojson))

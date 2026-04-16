"""Additional tests for Stage 2 modules to hit ≥70% coverage.

Covers: replacement.py aggregation, predict.py raster prediction,
stats schema Stage 2 models, and summarize.py Stage 2 data paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# change/replacement.py — aggregate_replacement_by_hotspot (pure numpy)
# ---------------------------------------------------------------------------

class TestAggregateReplacementByHotspot:
    """Tests for aggregate_replacement_by_hotspot."""

    def test_single_hotspot(self):
        from coffee_deforestation.change.replacement import aggregate_replacement_by_hotspot

        replacement = np.array([[0, 0, 1], [0, 2, 0]], dtype=np.int32)
        labels = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.int32)

        result = aggregate_replacement_by_hotspot(replacement, labels)
        assert 1 in result
        assert "coffee" in result[1]
        assert abs(sum(result[1].values()) - 1.0) < 0.01

    def test_multiple_hotspots(self):
        from coffee_deforestation.change.replacement import aggregate_replacement_by_hotspot

        rng = np.random.default_rng(0)
        replacement = rng.integers(0, 5, (20, 20)).astype(np.int32)
        labels = rng.integers(0, 4, (20, 20)).astype(np.int32)

        result = aggregate_replacement_by_hotspot(replacement, labels)
        # Background (0) is excluded
        assert 0 not in result
        for hid, dist in result.items():
            assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_empty_labels(self):
        from coffee_deforestation.change.replacement import aggregate_replacement_by_hotspot

        replacement = np.zeros((5, 5), dtype=np.int32)
        labels = np.zeros((5, 5), dtype=np.int32)  # all background
        result = aggregate_replacement_by_hotspot(replacement, labels)
        assert result == {}

    def test_custom_class_names(self):
        from coffee_deforestation.change.replacement import aggregate_replacement_by_hotspot

        replacement = np.array([[0, 1], [1, 0]], dtype=np.int32)
        labels = np.array([[1, 1], [1, 1]], dtype=np.int32)
        result = aggregate_replacement_by_hotspot(
            replacement, labels, class_names=["cls0", "cls1"]
        )
        assert "cls0" in result[1]
        assert "cls1" in result[1]


# ---------------------------------------------------------------------------
# ml/predict.py — predict_from_raster (local GeoTIFF)
# ---------------------------------------------------------------------------

def _make_raster(tmp_path: Path, n_bands: int = 16, size: int = 20) -> Path:
    """Create a synthetic multi-band GeoTIFF for prediction testing."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_bands, size, size)).astype(np.float32)
    # Add some NaN for nodata
    data[:, -2:, -2:] = np.nan

    transform = from_bounds(108.0, 11.5, 108.3, 11.8, size, size)
    path = tmp_path / "feature_stack.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=size, width=size,
        count=n_bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)

    return path


class TestPredictFromRaster:
    """Tests for raster-based prediction."""

    def test_predict_from_raster_rf(self, tmp_path):
        """RF prediction produces classification and probability rasters."""
        from sklearn.ensemble import RandomForestClassifier

        from coffee_deforestation.ml.predict import predict_from_raster

        raster_path = _make_raster(tmp_path)

        # Train a tiny RF
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 16)).astype(np.float32)
        y_train = rng.integers(0, 5, 100).astype(np.int32)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        class_path, prob_path = predict_from_raster(
            model, raster_path, tmp_path, "test_aoi"
        )

        assert class_path.exists()
        assert prob_path.exists()

        with rasterio.open(class_path) as src:
            cls_data = src.read(1)
            assert cls_data.shape == (20, 20)

        with rasterio.open(prob_path) as src:
            prob_data = src.read(1)
            # Valid pixels should be in [0, 1]
            valid = prob_data[~np.isnan(prob_data)]
            assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_predict_from_raster_nodata_preserved(self, tmp_path):
        """NaN pixels remain NaN in outputs."""
        from sklearn.ensemble import RandomForestClassifier

        from coffee_deforestation.ml.predict import predict_from_raster

        raster_path = _make_raster(tmp_path, size=10)

        rng = np.random.default_rng(0)
        model = RandomForestClassifier(n_estimators=3, random_state=0)
        model.fit(rng.standard_normal((50, 16)), rng.integers(0, 3, 50))

        class_path, prob_path = predict_from_raster(
            model, raster_path, tmp_path, "nodata_test"
        )
        with rasterio.open(class_path) as src:
            cls_data = src.read(1)
        # Nodata area (bottom-right 2x2) should be -1
        assert np.all(cls_data[-2:, -2:] == -1)


# ---------------------------------------------------------------------------
# stats/schema.py — Stage 2 models
# ---------------------------------------------------------------------------

class TestStage2Schema:
    """Tests for Stage 2 pydantic models."""

    def test_ablation_summary_defaults(self):
        from coffee_deforestation.stats.schema import AblationSummary

        a = AblationSummary()
        assert a.s1_only.f1_coffee == 0.0
        assert a.s2_only.f1_coffee == 0.0
        assert a.s1_s2.f1_coffee == 0.0

    def test_ablation_result_populated(self):
        from coffee_deforestation.stats.schema import AblationResult, AblationSummary

        a = AblationSummary(
            s1_only=AblationResult(f1_coffee=0.65, accuracy=0.70),
            s2_only=AblationResult(f1_coffee=0.72, accuracy=0.75),
            s1_s2=AblationResult(f1_coffee=0.81, accuracy=0.83),
        )
        assert a.s1_s2.f1_coffee == 0.81

    def test_historical_summary_defaults(self):
        from coffee_deforestation.stats.schema import HistoricalSummary

        h = HistoricalSummary()
        assert h.coffee_on_former_forest_fraction == 0.0
        assert h.mean_loss_year_offset is None
        assert h.replacement_class_distribution == {}

    def test_historical_summary_populated(self):
        from coffee_deforestation.stats.schema import HistoricalSummary

        h = HistoricalSummary(
            was_forest_2000_fraction=0.65,
            coffee_on_former_forest_fraction=0.23,
            mean_loss_year_offset=12.5,
            replacement_class_distribution={"coffee": 0.6, "forest": 0.3, "other": 0.1},
        )
        assert h.coffee_on_former_forest_fraction == 0.23
        assert h.mean_loss_year_offset == 12.5

    def test_aoi_summary_includes_stage2_fields(self):
        """AOISummary includes ablation and historical fields."""
        from coffee_deforestation.stats.schema import (
            AOIMetadata,
            AOISummary,
            BBoxSummary,
            ChangeDetectionSummary,
            DataCoverageSummary,
            ValidationSummary,
        )

        summary = AOISummary(
            metadata=AOIMetadata(
                aoi_id="test", name="Test", country="X",
                coffee_type="Robusta", role="primary",
                bbox=BBoxSummary(west=1, south=2, east=3, north=4),
                epsg_utm=32648,
            ),
            validation=ValidationSummary(
                coffee_fraction=0.1, forest_2000_fraction=0.4,
                hansen_loss_pixels=100, passed=True,
            ),
            data_coverage=DataCoverageSummary(years_processed=[2023]),
            change_detection=ChangeDetectionSummary(),
        )

        # Stage 2 fields exist and have defaults
        assert hasattr(summary, "ablation")
        assert hasattr(summary, "historical")
        assert summary.ablation.s1_s2.f1_coffee == 0.0
        assert summary.historical.coffee_on_former_forest_fraction == 0.0

    def test_aoi_summary_stage2_serializes(self):
        """AOISummary with Stage 2 data serializes/round-trips correctly."""
        import json

        from coffee_deforestation.stats.schema import (
            AblationResult,
            AblationSummary,
            AOIMetadata,
            AOISummary,
            BBoxSummary,
            ChangeDetectionSummary,
            DataCoverageSummary,
            HistoricalSummary,
            ValidationSummary,
        )

        summary = AOISummary(
            metadata=AOIMetadata(
                aoi_id="lam_dong", name="Lam Dong", country="Vietnam",
                coffee_type="Robusta", role="primary",
                bbox=BBoxSummary(west=108.0, south=11.5, east=108.3, north=11.8),
                epsg_utm=32648,
            ),
            validation=ValidationSummary(
                coffee_fraction=0.12, forest_2000_fraction=0.45,
                hansen_loss_pixels=2000, passed=True,
            ),
            data_coverage=DataCoverageSummary(years_processed=[2022, 2023, 2024]),
            change_detection=ChangeDetectionSummary(total_hotspots=1985, total_area_ha=3603.0),
            ablation=AblationSummary(
                s1_only=AblationResult(f1_coffee=0.65),
                s2_only=AblationResult(f1_coffee=0.72),
                s1_s2=AblationResult(f1_coffee=0.81),
            ),
            historical=HistoricalSummary(
                coffee_on_former_forest_fraction=0.23,
                mean_loss_year_offset=11.0,
            ),
        )

        json_str = summary.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["ablation"]["s1_s2"]["f1_coffee"] == 0.81
        assert parsed["historical"]["coffee_on_former_forest_fraction"] == 0.23

        restored = AOISummary.model_validate_json(json_str)
        assert restored.ablation.s1_s2.f1_coffee == 0.81


# ---------------------------------------------------------------------------
# stats/summarize.py — Stage 2 data paths
# ---------------------------------------------------------------------------

class TestSummarizeStage2:
    """Tests for build_summary with Stage 2 data."""

    def _make_hotspots(self, n: int = 3) -> list[dict]:
        return [
            {
                "properties": {
                    "hotspot_id": f"h{i:03d}",
                    "area_ha": float(10 + i * 5),
                    "centroid_lon": 108.1 + i * 0.01,
                    "centroid_lat": 11.6 + i * 0.01,
                    "rank": i + 1,
                }
            }
            for i in range(n)
        ]

    def _make_validation(self):
        from unittest.mock import MagicMock

        v = MagicMock()
        v.coffee_fraction = 0.12
        v.forest_2000_fraction = 0.45
        v.hansen_loss_pixels = 2000
        v.passed = True
        return v

    def test_build_summary_with_ml_metrics(self):
        from coffee_deforestation.config import load_aois
        from coffee_deforestation.stats.schema import ModelMetrics
        from coffee_deforestation.stats.summarize import build_summary

        aoi = list(load_aois().values())[0]
        validation = self._make_validation()
        hotspots = self._make_hotspots()

        metrics = ModelMetrics(
            model_type="random_forest",
            accuracy=0.88,
            f1_coffee=0.79,
            precision_coffee=0.82,
            recall_coffee=0.76,
        )

        summary = build_summary(
            aoi, validation, hotspots,
            years_processed=[2023, 2024],
            model_metrics=metrics,
        )

        assert summary.model_metrics.f1_coffee == 0.79
        assert summary.model_metrics.model_type == "random_forest"

    def test_build_summary_with_ablation(self):
        from coffee_deforestation.config import load_aois
        from coffee_deforestation.stats.summarize import build_summary

        aoi = list(load_aois().values())[0]
        validation = self._make_validation()

        ablation = {
            "s1_only": {"f1_coffee": 0.65, "accuracy": 0.70},
            "s2_only": {"f1_coffee": 0.72, "accuracy": 0.75},
            "s1_s2": {"f1_coffee": 0.81, "accuracy": 0.83},
        }

        summary = build_summary(
            aoi, validation, [],
            years_processed=[2023],
            ablation_results=ablation,
        )

        assert summary.ablation.s1_only.f1_coffee == 0.65
        assert summary.ablation.s1_s2.f1_coffee == 0.81

    def test_build_summary_with_historical_stats(self):
        from coffee_deforestation.config import load_aois
        from coffee_deforestation.stats.summarize import build_summary

        aoi = list(load_aois().values())[0]
        validation = self._make_validation()

        historical = {
            "was_forest_2000_mean": 0.65,
            "coffee_on_former_forest_mean": 0.23,
            "loss_year_before_coffee_mean": 11.5,
            "replacement_class_distribution": {"coffee": 0.6, "forest": 0.3},
        }

        summary = build_summary(
            aoi, validation, [],
            years_processed=[2023],
            historical_stats=historical,
        )

        assert summary.historical.was_forest_2000_fraction == 0.65
        assert summary.historical.coffee_on_former_forest_fraction == 0.23
        assert summary.historical.mean_loss_year_offset == 11.5

    def test_build_summary_defaults_when_no_stage2(self):
        from coffee_deforestation.config import load_aois
        from coffee_deforestation.stats.summarize import build_summary

        aoi = list(load_aois().values())[0]
        validation = self._make_validation()

        summary = build_summary(aoi, validation, [], years_processed=[2023])
        assert summary.model_metrics.model_type == "none"
        assert summary.ablation.s1_s2.f1_coffee == 0.0
        assert summary.historical.coffee_on_former_forest_fraction == 0.0

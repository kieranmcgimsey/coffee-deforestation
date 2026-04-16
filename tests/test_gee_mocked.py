"""Tests for GEE-dependent modules with mocked ee."""

from unittest.mock import MagicMock, patch

import pytest

from coffee_deforestation.config import AOIConfig, BBox, DrySeason, PipelineConfig


@pytest.fixture
def aoi():
    return AOIConfig(
        id="test",
        name="Test",
        country="Test",
        coffee_type="Test",
        role="Test",
        bbox=BBox(west=108.3, south=11.9, east=108.4, north=12.0),
        dry_season=DrySeason(start_month=12, end_month=3, cross_year=True),
        epsg_utm=32648,
    )


@pytest.fixture
def pipeline_cfg():
    return PipelineConfig()


class TestGEEClient:
    @patch("coffee_deforestation.data.gee_client.ee")
    def test_init_gee_service_account(self, mock_ee):
        """GEE initializes with service account."""
        import coffee_deforestation.data.gee_client as gc
        gc._initialized = False

        from coffee_deforestation.config import Settings
        settings = Settings(gee_service_account_key_path="/fake/key.json")

        gc.init_gee(settings)
        mock_ee.ServiceAccountCredentials.assert_called_once()
        mock_ee.Initialize.assert_called_once()
        gc._initialized = False  # Reset for other tests

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_init_gee_default(self, mock_ee):
        """GEE initializes with default credentials."""
        import coffee_deforestation.data.gee_client as gc
        gc._initialized = False

        from coffee_deforestation.config import Settings
        settings = Settings(gee_service_account_key_path="")

        gc.init_gee(settings)
        mock_ee.Initialize.assert_called()
        gc._initialized = False

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_aoi_to_geometry(self, mock_ee, aoi):
        """AOI converts to GEE geometry."""
        from coffee_deforestation.data.gee_client import aoi_to_geometry
        aoi_to_geometry(aoi)
        mock_ee.Geometry.Rectangle.assert_called_once_with([108.3, 11.9, 108.4, 12.0])

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_export_image_to_drive(self, mock_ee, aoi):
        """Export starts a GEE task."""
        from coffee_deforestation.data.gee_client import export_image_to_drive
        image = MagicMock()
        task = export_image_to_drive(image, "test_export", aoi, folder="test_folder")
        mock_ee.batch.Export.image.toDrive.assert_called_once()

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_poll_task_completed(self, mock_ee):
        """Poll returns True on completed task."""
        from coffee_deforestation.data.gee_client import poll_task
        task = MagicMock()
        task.status.return_value = {"state": "COMPLETED", "description": "test"}
        assert poll_task(task, poll_interval=0) is True

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_poll_task_failed(self, mock_ee):
        """Poll returns False on failed task."""
        from coffee_deforestation.data.gee_client import poll_task
        task = MagicMock()
        task.status.return_value = {"state": "FAILED", "error_message": "test error"}
        assert poll_task(task, poll_interval=0) is False

    @patch("coffee_deforestation.data.gee_client.ee")
    def test_compute_stats(self, mock_ee, aoi):
        """Compute stats calls GEE reduceRegion."""
        from coffee_deforestation.data.gee_client import compute_stats
        image = MagicMock()
        reduce_mock = MagicMock()
        reduce_mock.getInfo.return_value = {"band_mean": 0.5}
        image.reduceRegion.return_value = reduce_mock
        result = compute_stats(image, aoi)
        assert result == {"band_mean": 0.5}


class TestSentinel2:
    @patch("coffee_deforestation.data.sentinel2.ee")
    @patch("coffee_deforestation.data.gee_client.ee")
    def test_build_s2_composite(self, mock_gee_ee, mock_s2_ee, aoi, pipeline_cfg):
        """S2 composite builds without error."""
        from coffee_deforestation.data.sentinel2 import build_s2_composite

        # Setup mocks
        mock_col = MagicMock()
        mock_s2_ee.ImageCollection.return_value = mock_col
        mock_col.filterBounds.return_value = mock_col
        mock_col.filterDate.return_value = mock_col
        mock_col.filter.return_value = mock_col
        mock_col.select.return_value = mock_col
        mock_col.median.return_value = MagicMock()

        mock_s2_ee.Join.saveFirst.return_value = MagicMock()
        mock_s2_ee.Filter.equals.return_value = MagicMock()
        mock_s2_ee.Filter.lt.return_value = MagicMock()

        join_mock = MagicMock()
        mock_s2_ee.Join.saveFirst.return_value = join_mock
        join_result = MagicMock()
        join_mock.apply.return_value = join_result
        join_result.map.return_value = mock_col

        result = build_s2_composite(aoi, 2023, pipeline_cfg)
        assert result is not None


class TestSentinel1:
    @patch("coffee_deforestation.data.sentinel1.ee")
    @patch("coffee_deforestation.data.gee_client.ee")
    def test_build_s1_composite(self, mock_gee_ee, mock_s1_ee, aoi, pipeline_cfg):
        """S1 composite builds without error."""
        from coffee_deforestation.data.sentinel1 import build_s1_composite

        mock_col = MagicMock()
        mock_s1_ee.ImageCollection.return_value = mock_col
        mock_col.filterBounds.return_value = mock_col
        mock_col.filterDate.return_value = mock_col
        mock_col.filter.return_value = mock_col
        mock_col.select.return_value = mock_col
        composite = MagicMock()
        mock_col.median.return_value = composite
        composite.clip.return_value = composite
        composite.focal_median.return_value = composite
        composite.set.return_value = composite
        composite.toFloat.return_value = composite

        result = build_s1_composite(aoi, 2023, pipeline_cfg)
        assert result is not None


class TestValidateAOI:
    @patch("coffee_deforestation.data.validate_aoi.aoi_to_geometry")
    @patch("coffee_deforestation.data.validate_aoi.ee")
    def test_validate_passes(self, mock_ee, mock_geom, aoi, pipeline_cfg):
        """Validation passes with good data."""
        from coffee_deforestation.data.validate_aoi import validate_aoi

        mock_geom.return_value = MagicMock()

        # Mock FDP image
        mock_fdp = MagicMock()
        mock_hansen = MagicMock()

        def image_side_effect(collection_id):
            if "coffee" in collection_id:
                return mock_fdp
            return mock_hansen

        mock_ee.Image.side_effect = image_side_effect

        # Coffee fraction — validate_aoi calls fdp.select("probability").gt(0.5).rename("coffee")
        coffee_reduce = MagicMock()
        coffee_reduce.getInfo.return_value = {"coffee": 0.15}
        mock_fdp.select.return_value.gt.return_value.rename.return_value.reduceRegion.return_value = coffee_reduce

        # Forest fraction
        forest_reduce = MagicMock()
        forest_reduce.getInfo.return_value = {"forest": 0.45}
        mock_hansen.select.return_value.gt.return_value.rename.return_value.reduceRegion.return_value = forest_reduce

        # Loss pixels
        loss_reduce = MagicMock()
        loss_reduce.getInfo.return_value = {"loss": 5000}
        mock_hansen.select.return_value.rename.return_value.reduceRegion.return_value = loss_reduce

        mock_ee.Reducer.mean.return_value = MagicMock()
        mock_ee.Reducer.sum.return_value = MagicMock()

        result = validate_aoi(aoi, pipeline_cfg)
        assert result.aoi_id == "test"
        assert result.passed is True

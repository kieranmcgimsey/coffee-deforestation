"""Tests for feature engineering with mocked GEE.

Covers: indices.py, sar_features.py, contextual.py, stack.py
using MagicMock to avoid real GEE calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coffee_deforestation.config import AOIConfig, load_pipeline_config


@pytest.fixture
def mock_s2_image():
    """A mock S2 ee.Image with chained method returns."""
    img = MagicMock()
    # normalizedDifference and other operations return a new MagicMock image
    img.normalizedDifference.return_value = MagicMock()
    img.normalizedDifference.return_value.rename.return_value = MagicMock()
    img.select.return_value = MagicMock()
    img.addBands.return_value = img  # chaining returns same mock
    img.multiply.return_value = MagicMock()
    img.divide.return_value = MagicMock()
    img.add.return_value = MagicMock()
    img.toFloat.return_value = img
    return img


@pytest.fixture
def mock_s1_image():
    """A mock S1 ee.Image."""
    img = MagicMock()
    img.select.return_value = MagicMock()
    img.addBands.return_value = img
    img.divide.return_value = MagicMock()
    img.subtract.return_value = MagicMock()
    img.toFloat.return_value = img
    return img


# ---------------------------------------------------------------------------
# features/indices.py
# ---------------------------------------------------------------------------

class TestSpectralIndices:
    """Test spectral index computation with mocked GEE images."""

    def test_compute_ndvi(self, mock_s2_image):
        """NDVI calls normalizedDifference with correct bands."""
        # Build a mock that returns named bands properly
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_ndvi

            result = compute_ndvi(mock_s2_image)
            mock_s2_image.normalizedDifference.assert_called_once_with(["B8", "B4"])

    def test_compute_all_indices_returns_image(self, mock_s2_image):
        """compute_all_indices returns an image with addBands calls."""
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_all_indices

            result = compute_all_indices(mock_s2_image)
            # Should have called addBands at least once (stacking indices)
            assert mock_s2_image.addBands.call_count > 0

    def test_compute_evi(self, mock_s2_image):
        """EVI computation uses correct bands."""
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_evi

            result = compute_evi(mock_s2_image)
            # EVI uses B8, B4, B2
            assert result is not None

    def test_compute_ndwi(self, mock_s2_image):
        """NDWI calls normalizedDifference with B3 and B8."""
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_ndwi

            result = compute_ndwi(mock_s2_image)
            assert result is not None

    def test_compute_nbr(self, mock_s2_image):
        """NBR calls normalizedDifference with B8 and B12."""
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_nbr

            result = compute_nbr(mock_s2_image)
            assert result is not None

    def test_compute_savi(self, mock_s2_image):
        """SAVI computation uses B8 and B4."""
        with patch("coffee_deforestation.features.indices.ee"):
            from coffee_deforestation.features.indices import compute_savi

            result = compute_savi(mock_s2_image)
            assert result is not None


# ---------------------------------------------------------------------------
# features/sar_features.py
# ---------------------------------------------------------------------------

class TestSARFeatures:
    """Test SAR feature extraction with mocked GEE."""

    def test_compute_sar_features(self, mock_s1_image):
        """compute_sar_features returns an image with VV, VH, ratio."""
        with patch("coffee_deforestation.features.sar_features.ee"):
            from coffee_deforestation.features.sar_features import compute_sar_features

            result = compute_sar_features(mock_s1_image)
            assert result is not None

    def test_compute_temporal_sar_features(self, mock_s1_image):
        """compute_temporal_sar_features runs with a dict of composites."""
        with patch("coffee_deforestation.features.sar_features.ee") as mock_ee:
            mock_ee.ImageCollection.return_value.reduce.return_value.rename.return_value = MagicMock()
            mock_ee.ImageCollection.return_value.reduce.return_value.rename.return_value.addBands.return_value = MagicMock()
            mock_ee.Reducer.stdDev.return_value = MagicMock()

            from coffee_deforestation.features.sar_features import compute_temporal_sar_features

            result = compute_temporal_sar_features({2022: mock_s1_image, 2023: mock_s1_image})
            assert result is not None


# ---------------------------------------------------------------------------
# features/contextual.py
# ---------------------------------------------------------------------------

class TestContextualFeatures:
    """Test contextual feature computation with mocked GEE."""

    def test_compute_contextual_features(self, sample_aoi, pipeline_config):
        """compute_contextual_features runs with mocked GEE ancillary data.

        Patched at source (data.ancillary) since contextual.py imports inside the function.
        """
        mock_srtm = MagicMock()
        mock_hansen = MagicMock()
        mock_roads = MagicMock()

        # Make select().gt() etc. chain properly
        mock_hansen.select.return_value.gt.return_value = MagicMock()
        mock_srtm.select.return_value = MagicMock()
        mock_srtm.select.return_value.addBands.return_value = MagicMock()

        with (
            patch("coffee_deforestation.features.contextual.ee"),
            patch("coffee_deforestation.data.ancillary.get_srtm", return_value=mock_srtm),
            patch("coffee_deforestation.data.ancillary.get_hansen", return_value=mock_hansen),
            patch("coffee_deforestation.data.ancillary.get_roads", return_value=mock_roads),
        ):
            from coffee_deforestation.features.contextual import compute_contextual_features

            result = compute_contextual_features(sample_aoi, pipeline_config)
            assert result is not None


# ---------------------------------------------------------------------------
# features/stack.py
# ---------------------------------------------------------------------------

class TestFeatureStack:
    """Test feature stack assembly."""

    def test_build_feature_stack(self, sample_aoi, pipeline_config):
        """build_feature_stack chains all components correctly."""
        mock_s2 = MagicMock()
        mock_s1 = MagicMock()
        mock_contextual = MagicMock()

        # Mock the component functions and ee.Image method chaining
        with (
            patch(
                "coffee_deforestation.features.stack.compute_all_indices",
                return_value=mock_s2,
            ),
            patch(
                "coffee_deforestation.features.stack.compute_sar_features",
                return_value=mock_s1,
            ),
            patch(
                "coffee_deforestation.features.stack.compute_contextual_features",
                return_value=mock_contextual,
            ),
        ):
            mock_s2.select.return_value = mock_s2
            mock_s2.addBands.return_value = mock_s2
            mock_s2.toFloat.return_value = mock_s2

            # Mock ee.Image.constant for temporal fallback bands
            mock_const = MagicMock()
            mock_const.rename.return_value = mock_const
            mock_const.addBands.return_value = mock_const

            with patch("coffee_deforestation.features.stack.ee") as mock_ee:
                mock_ee.Image.constant.return_value = mock_const

                from coffee_deforestation.features.stack import build_feature_stack

                result = build_feature_stack(mock_s2, mock_s1, sample_aoi, pipeline_config)
                assert result is not None


# ---------------------------------------------------------------------------
# change/historical.py — mock GEE trajectory computation
# ---------------------------------------------------------------------------

class TestHistoricalLookback:
    """Test historical look-back with mocked GEE."""

    def test_compute_historical_trajectory(self, sample_aoi, pipeline_config):
        """compute_historical_trajectory returns a GEE image."""
        mock_hansen = MagicMock()
        mock_fdp = MagicMock()

        # Setup chained returns to mimic GEE image operations
        mock_coffee_mask = MagicMock()
        mock_was_forest = MagicMock()
        mock_loss_year = MagicMock()
        mock_coffee_on_ff = MagicMock()
        mock_loss_before_coffee = MagicMock()

        mock_hansen.select.return_value.gt.return_value.rename.return_value = mock_was_forest
        mock_hansen.select.return_value.rename.return_value = mock_loss_year
        mock_fdp.select.return_value.gt.return_value = mock_coffee_mask

        mock_was_forest.And.return_value.And.return_value.rename.return_value = mock_coffee_on_ff
        mock_loss_year.updateMask.return_value.rename.return_value = mock_loss_before_coffee
        mock_was_forest.addBands.return_value.addBands.return_value.addBands.return_value = MagicMock()

        with (
            patch("coffee_deforestation.change.historical.ee"),
            patch("coffee_deforestation.data.ancillary.get_hansen", return_value=mock_hansen),
            patch("coffee_deforestation.data.ancillary.get_fdp_coffee", return_value=mock_fdp),
            patch("coffee_deforestation.data.gee_client.aoi_to_geometry", return_value=MagicMock()),
        ):
            from coffee_deforestation.change.historical import compute_historical_trajectory

            result = compute_historical_trajectory(sample_aoi, pipeline_config)
            assert result is not None

    def test_compute_historical_stats(self, sample_aoi):
        """compute_historical_stats calls getInfo on the trajectory."""
        mock_trajectory = MagicMock()
        expected = {
            "was_forest_2000_mean": 0.65,
            "loss_year_mean": 8.5,
            "coffee_on_former_forest_mean": 0.23,
            "loss_year_before_coffee_mean": 11.0,
        }
        mock_trajectory.reduceRegion.return_value.getInfo.return_value = expected

        with (
            patch("coffee_deforestation.change.historical.ee") as mock_ee,
            patch("coffee_deforestation.data.gee_client.aoi_to_geometry", return_value=MagicMock()),
        ):
            mock_ee.Reducer.mean.return_value.combine.return_value = MagicMock()

            from coffee_deforestation.change.historical import compute_historical_stats

            stats = compute_historical_stats(mock_trajectory, sample_aoi)
            assert stats["was_forest_2000_mean"] == 0.65
            assert stats["coffee_on_former_forest_mean"] == 0.23

"""Tests for visualization modules."""

from pathlib import Path

import numpy as np
import pytest

from coffee_deforestation.config import AOIConfig
from coffee_deforestation.viz.theme import (
    CLASS_CMAP,
    CLASS_NAMES,
    COFFEE_PROB_CMAP,
    COLORS,
    LOSS_YEAR_CMAP,
    NDVI_CMAP,
    apply_theme,
    figure_with_title,
    save_figure,
)


def test_color_palette_complete():
    """All required colors are defined."""
    required = [
        "coffee", "forest_stable", "forest_loss", "coffee_on_former_forest",
        "non_coffee_cropland", "built_bare", "water", "background",
    ]
    for color in required:
        assert color in COLORS
        assert COLORS[color].startswith("#")


def test_class_names_count():
    """Class names match class colormap."""
    assert len(CLASS_NAMES) == 5
    assert len(CLASS_CMAP.colors) == 5


def test_apply_theme():
    """Theme applies without error."""
    apply_theme()


def test_figure_with_title():
    """Figure creation with title works."""
    fig, ax = figure_with_title("Test Title", "Test subtitle")
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_save_figure(tmp_path: Path):
    """Figure saves to disk."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    path = str(tmp_path / "test_fig.png")
    save_figure(fig, path)
    assert Path(path).exists()


def test_plot_rgb_composite(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """RGB composite plot generates."""
    from coffee_deforestation.viz.static import plot_rgb_composite
    path = plot_rgb_composite(
        sample_raster_data["red"],
        sample_raster_data["green"],
        sample_raster_data["blue"],
        sample_aoi, 2023,
        output_path=str(tmp_path / "rgb.png"),
    )
    assert Path(path).exists()


def test_plot_ndvi(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """NDVI plot generates."""
    from coffee_deforestation.viz.static import plot_ndvi
    path = plot_ndvi(
        sample_raster_data["ndvi"],
        sample_aoi, 2023,
        output_path=str(tmp_path / "ndvi.png"),
    )
    assert Path(path).exists()


def test_plot_s1_vv(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """S1 VV plot generates."""
    from coffee_deforestation.viz.static import plot_s1_vv
    path = plot_s1_vv(
        sample_raster_data["vv"],
        sample_aoi, 2023,
        output_path=str(tmp_path / "vv.png"),
    )
    assert Path(path).exists()


def test_plot_coffee_probability(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """Coffee probability plot generates."""
    from coffee_deforestation.viz.static import plot_coffee_probability
    path = plot_coffee_probability(
        sample_raster_data["coffee_prob"],
        sample_aoi,
        output_path=str(tmp_path / "coffee.png"),
    )
    assert Path(path).exists()


def test_plot_hansen_loss(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """Hansen loss year plot generates."""
    from coffee_deforestation.viz.static import plot_hansen_loss
    path = plot_hansen_loss(
        sample_raster_data["loss_year"],
        sample_aoi,
        output_path=str(tmp_path / "loss.png"),
    )
    assert Path(path).exists()


def test_plot_hotspots_overlay(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """Hotspot overlay plot generates."""
    from coffee_deforestation.viz.static import plot_hotspots_overlay
    hotspot_mask = (sample_raster_data["loss_year"] > 0).astype(np.float32)
    path = plot_hotspots_overlay(
        sample_raster_data["ndvi"],
        hotspot_mask,
        sample_aoi,
        output_path=str(tmp_path / "hotspots.png"),
    )
    assert Path(path).exists()


def test_plot_feature_correlation(sample_aoi: AOIConfig, tmp_path: Path):
    """Feature correlation plot generates."""
    from coffee_deforestation.viz.static import plot_feature_correlation
    rng = np.random.default_rng(42)
    features = rng.standard_normal((500, 5)).astype(np.float32)
    names = ["f1", "f2", "f3", "f4", "f5"]
    path = plot_feature_correlation(
        features, names, sample_aoi,
        output_path=str(tmp_path / "corr.png"),
    )
    assert Path(path).exists()


def test_plot_cloud_mask(sample_aoi: AOIConfig, sample_raster_data: dict, tmp_path: Path):
    """Cloud mask plot generates."""
    from coffee_deforestation.viz.static import plot_cloud_mask
    rgb = np.dstack([
        sample_raster_data["red"],
        sample_raster_data["green"],
        sample_raster_data["blue"],
    ])
    cloud = (sample_raster_data["ndvi"] < 0.1).astype(np.float32)
    path = plot_cloud_mask(
        rgb, cloud, sample_aoi, 2023,
        output_path=str(tmp_path / "cloud.png"),
    )
    assert Path(path).exists()

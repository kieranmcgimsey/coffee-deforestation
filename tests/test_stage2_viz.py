"""Tests for Stage 2 visualization additions."""

from pathlib import Path

import numpy as np

from coffee_deforestation.config import AOIConfig
from coffee_deforestation.viz.cloud_recovery import plot_cloud_recovery
from coffee_deforestation.viz.static import (
    plot_ablation_bar_chart,
    plot_confusion_matrix,
    plot_historical_lookback,
    plot_replacement_classes,
)


def test_plot_confusion_matrix(sample_aoi: AOIConfig, tmp_path: Path):
    """Confusion matrix plot generates."""
    cm = np.array([[80, 5, 3], [10, 70, 8], [2, 6, 85]])
    path = plot_confusion_matrix(
        cm, ["coffee", "forest", "cropland"], "test_rf",
        aoi=sample_aoi,
        output_path=str(tmp_path / "cm.png"),
    )
    assert Path(path).exists()


def test_plot_ablation_chart(sample_aoi: AOIConfig, tmp_path: Path):
    """Ablation bar chart generates."""
    results = {
        "s1_only": {"f1_coffee": 0.65},
        "s2_only": {"f1_coffee": 0.72},
        "s1_s2": {"f1_coffee": 0.81},
    }
    path = plot_ablation_bar_chart(
        results, aoi=sample_aoi,
        output_path=str(tmp_path / "ablation.png"),
    )
    assert Path(path).exists()


def test_plot_historical_lookback(sample_aoi: AOIConfig, tmp_path: Path):
    """Historical lookback plot generates."""
    rng = np.random.default_rng(42)
    loss_years = rng.choice([0, 0, 5, 10, 15, 20], (100, 100)).astype(np.int16)
    path = plot_historical_lookback(
        loss_years, sample_aoi,
        output_path=str(tmp_path / "lookback.png"),
    )
    assert Path(path).exists()


def test_plot_replacement_classes(sample_aoi: AOIConfig, tmp_path: Path):
    """Replacement class pie chart generates."""
    dist = {"coffee": 0.45, "forest": 0.20, "cropland": 0.25, "built_bare": 0.08, "water": 0.02}
    path = plot_replacement_classes(
        dist, sample_aoi,
        output_path=str(tmp_path / "replacement.png"),
    )
    assert Path(path).exists()


def test_plot_cloud_recovery(sample_aoi: AOIConfig, tmp_path: Path):
    """Cloud recovery figure generates."""
    rng = np.random.default_rng(42)
    shape = (100, 100)
    rgb = rng.uniform(0, 0.3, (*shape, 3)).astype(np.float32)
    vv = rng.uniform(-25, 0, shape).astype(np.float32)
    ndvi = rng.uniform(-0.1, 0.9, shape).astype(np.float32)
    path = plot_cloud_recovery(
        rgb, vv, ndvi, 0.65, sample_aoi,
        scene_date="2024-01-15",
        output_path=str(tmp_path / "cloud_recovery.png"),
    )
    assert Path(path).exists()


def test_plot_cloud_recovery_no_output_path(sample_aoi: AOIConfig, tmp_path: Path, monkeypatch):
    """Cloud recovery auto-generates output path when output_path=None."""
    from unittest.mock import patch
    rng = np.random.default_rng(99)
    shape = (50, 50)
    rgb = rng.uniform(0, 0.3, (*shape, 3)).astype(np.float32)
    vv = rng.uniform(-25, 0, shape).astype(np.float32)
    ndvi = rng.uniform(-0.1, 0.9, shape).astype(np.float32)

    with patch("coffee_deforestation.viz.cloud_recovery.PROJECT_ROOT", tmp_path):
        path = plot_cloud_recovery(rgb, vv, ndvi, 0.30, sample_aoi, output_path=None)

    assert path is not None
    assert Path(path).exists()

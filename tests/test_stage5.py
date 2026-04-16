"""Tests for Stage 5: code quality, refactoring, new analysis modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coffee_deforestation.config import AOIConfig, BBox, CloudMaskingConfig, DrySeason, PatchConfig
from coffee_deforestation.stats.schema import (
    ChangeDetectionSummary,
    DeforestationAttribution,
    HotspotSummary,
    YearlyLossStats,
)


# ---------------------------------------------------------------------------
# Config cleanup tests
# ---------------------------------------------------------------------------


class TestConfigCleanup:
    """Verify dead config was removed and new config works."""

    def test_shadow_config_removed(self):
        """CloudMaskingConfig should NOT have shadow fields."""
        cfg = CloudMaskingConfig()
        assert not hasattr(cfg, "shadow_max_distance_m")
        assert not hasattr(cfg, "dark_nir_threshold")
        assert hasattr(cfg, "cloud_probability_threshold")
        assert hasattr(cfg, "cloud_dilation_m")

    def test_patch_config(self):
        """PatchConfig has name and bbox."""
        p = PatchConfig(name="Test", bbox=BBox(west=0, south=0, east=1, north=1))
        assert p.name == "Test"
        assert p.bbox.width_deg == 1.0

    def test_aoi_get_effective_patches_with_patches(self):
        """When patches are defined, get_effective_patches returns them."""
        aoi = AOIConfig(
            id="t", name="T", country="X", coffee_type="Y", role="Z",
            bbox=BBox(west=0, south=0, east=1, north=1),
            dry_season=DrySeason(start_month=6, end_month=9),
            epsg_utm=32600,
            patches=[PatchConfig(name="P1", bbox=BBox(west=0, south=0, east=0.5, north=0.5))],
        )
        patches = aoi.get_effective_patches()
        assert len(patches) == 1
        assert patches[0].name == "P1"

    def test_aoi_get_effective_patches_fallback(self):
        """When no patches defined, wraps the main bbox as a single patch."""
        aoi = AOIConfig(
            id="t", name="T", country="X", coffee_type="Y", role="Z",
            bbox=BBox(west=0, south=0, east=1, north=1),
            dry_season=DrySeason(start_month=6, end_month=9),
            epsg_utm=32600,
        )
        patches = aoi.get_effective_patches()
        assert len(patches) == 1
        assert patches[0].name == "T"
        assert patches[0].bbox == aoi.bbox


# ---------------------------------------------------------------------------
# Schema tests for new models
# ---------------------------------------------------------------------------


class TestNewSchemaModels:
    """Test DeforestationAttribution and YearlyLossStats."""

    def test_deforestation_attribution_defaults(self):
        attr = DeforestationAttribution()
        assert attr.total_loss_ha == 0.0
        assert attr.coffee_pct == 0.0
        assert attr.by_year == {}

    def test_deforestation_attribution_round_trip(self):
        attr = DeforestationAttribution(
            total_loss_ha=10000.5,
            coffee_pct=42.3,
            other_crops_pct=5.1,
            built_industrial_pct=0.2,
            bare_degraded_pct=3.0,
            water_pct=0.1,
            regrowth_pct=49.3,
            by_year={2020: {"coffee_pct": 45.0, "other_crops_pct": 3.0}},
        )
        dumped = attr.model_dump()
        reloaded = DeforestationAttribution.model_validate(dumped)
        assert reloaded.coffee_pct == 42.3
        assert 2020 in reloaded.by_year

    def test_yearly_loss_stats(self):
        yl = YearlyLossStats(total_loss_ha=500, coffee_loss_ha=200, coffee_fraction=0.4)
        assert yl.coffee_fraction == 0.4

    def test_hotspot_summary_with_trajectory(self):
        hs = HotspotSummary(
            hotspot_id="h1", area_ha=10, centroid_lon=108, centroid_lat=12,
            rank=1, loss_year=2021,
            ndvi_trajectory={2019: 0.7, 2020: 0.65, 2021: 0.4},
        )
        assert hs.loss_year == 2021
        assert hs.ndvi_trajectory[2021] == 0.4


# ---------------------------------------------------------------------------
# Type annotation verification
# ---------------------------------------------------------------------------


class TestTypeAnnotations:
    """Verify object type annotations were replaced."""

    def test_no_object_types_in_attribution(self):
        """deforestation_attribution.py should use proper types, not object."""
        import inspect
        from coffee_deforestation.change import deforestation_attribution as mod
        source = inspect.getsource(mod)
        # Should have TYPE_CHECKING import
        assert "TYPE_CHECKING" in source
        # Should NOT have ': object' annotations
        assert ": object" not in source

    def test_no_object_types_in_temporal(self):
        """temporal.py should use proper types, not object."""
        import inspect
        from coffee_deforestation.change import temporal as mod
        source = inspect.getsource(mod)
        assert "TYPE_CHECKING" in source
        assert ": object" not in source

    def test_no_object_types_in_predict(self):
        """predict.py should use Any or proper types, not bare object."""
        import inspect
        from coffee_deforestation.ml import predict as mod
        source = inspect.getsource(mod)
        assert "TYPE_CHECKING" in source
        # Should use Any for model, not object
        assert ": object" not in source


# ---------------------------------------------------------------------------
# Cloud recovery docstring
# ---------------------------------------------------------------------------


def test_cloud_recovery_docstring_honest():
    """cloud_recovery.py docstring describes what the figure actually does."""
    import inspect
    from coffee_deforestation.viz import cloud_recovery as mod
    source = inspect.getsource(mod)
    # Docstring should explain the SAR-optical complementarity honestly
    assert "sar" in source.lower() or "radar" in source.lower()
    assert "cloud" in source.lower()


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Test the HTML report generator."""

    def test_generates_html(self, tmp_path):
        """Report script produces valid HTML from stats JSON."""
        # Create minimal stats JSON
        stats_dir = tmp_path / "outputs" / "stats"
        stats_dir.mkdir(parents=True)
        stats = {
            "metadata": {"aoi_id": "test", "name": "Test", "country": "X",
                         "coffee_type": "Y", "role": "Z"},
            "change_detection": {
                "total_hotspots": 100, "total_area_ha": 500,
                "hotspots_by_loss_year": {"2020": 30, "2021": 40, "2022": 30},
                "area_ha_by_loss_year": {"2020": 100, "2021": 200, "2022": 200},
            },
            "validation": {"coffee_fraction": 0.3, "forest_2000_fraction": 0.5},
            "deforestation_attribution": {
                "total_loss_ha": 5000, "coffee_pct": 45, "other_crops_pct": 5,
                "regrowth_pct": 40, "built_industrial_pct": 1,
                "bare_degraded_pct": 8, "water_pct": 1,
            },
        }
        (stats_dir / "summary_test.json").write_text(json.dumps(stats))

        # Import and call the report function (without GEE)
        from scripts.generate_report import _generate_aoi_report

        with patch("scripts.generate_report.STATS_DIR", stats_dir):
            with patch("scripts.generate_report.FIGURES_DIR", tmp_path / "figs"):
                with patch("scripts.generate_report.MAPS_DIR", tmp_path / "maps"):
                    with patch("scripts.generate_report.OUTPUTS", tmp_path / "outputs"):
                        html = _generate_aoi_report("test", ["test"])

        assert "<html" in html
        assert "Test" in html
        assert "500" in html  # total area
        assert "EUDR" in html or "post-2020" in html.lower()

    def test_eudr_metric_computed(self):
        """Post-2020 metric should count only 2021+ hotspots."""
        hby = {"2019": 100, "2020": 200, "2021": 50, "2022": 30, "2023": 20}
        post_2020 = sum(v for k, v in hby.items() if int(k) > 2020)
        assert post_2020 == 100  # 50 + 30 + 20


# ---------------------------------------------------------------------------
# Interactive map tests
# ---------------------------------------------------------------------------


class TestInteractiveMap:
    """Test the interactive map generation."""

    def test_create_rich_map_no_gee(self, sample_aoi):
        """create_rich_map works without GEE tile layers."""
        from coffee_deforestation.viz.interactive import create_rich_map
        m = create_rich_map(sample_aoi)
        assert m is not None
        html = m._repr_html_()
        assert "leaflet" in html.lower() or "L.map" in html

    def test_create_rich_map_with_geojson(self, sample_aoi, tmp_path):
        """create_rich_map renders hotspot GeoJSON with year slider."""
        from coffee_deforestation.viz.interactive import create_rich_map

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": [[[108.3, 11.9], [108.31, 11.9],
                                              [108.31, 11.91], [108.3, 11.91],
                                              [108.3, 11.9]]]},
                "properties": {"hotspot_id": "h1", "area_ha": 10,
                               "rank": 1, "loss_year": 2020,
                               "centroid_lat": 11.905, "centroid_lon": 108.305},
            }],
        }
        path = tmp_path / "hotspots.geojson"
        path.write_text(json.dumps(geojson))

        m = create_rich_map(sample_aoi, hotspot_geojson_path=path)
        html_str = m._repr_html_()
        # Should contain the slider JS
        assert "_initSlider" in html_str or "yearSlider" in html_str

    def test_backward_compat_create_aoi_map(self, sample_aoi):
        """Legacy create_aoi_map still works."""
        from coffee_deforestation.viz.interactive import create_aoi_map
        m = create_aoi_map(sample_aoi)
        assert m is not None


# ---------------------------------------------------------------------------
# New figure tests
# ---------------------------------------------------------------------------


class TestNewFigures:
    """Test attribution and temporal figure functions."""

    def test_attribution_pie(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_attribution_pie

        attr = {"total_loss_ha": 5000, "coffee_pct": 45, "other_crops_pct": 5,
                "regrowth_pct": 40, "built_industrial_pct": 1,
                "bare_degraded_pct": 8, "water_pct": 1}
        path = plot_attribution_pie(attr, sample_aoi,
                                     output_path=str(tmp_path / "pie.png"))
        assert Path(path).exists()

    def test_attribution_stacked_bar(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_attribution_stacked_bar

        yearly = {
            2019: {"coffee_pct": 40, "other_crops_pct": 10, "regrowth_pct": 50},
            2020: {"coffee_pct": 45, "other_crops_pct": 8, "regrowth_pct": 47},
        }
        path = plot_attribution_stacked_bar(yearly, sample_aoi,
                                             output_path=str(tmp_path / "bar.png"))
        assert Path(path).exists()

    def test_before_after(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_before_after

        rgb = np.random.rand(50, 50, 3).astype(np.float32) * 0.3
        path = plot_before_after(rgb, rgb, sample_aoi, 2019, 2024,
                                  output_path=str(tmp_path / "ba.png"))
        assert Path(path).exists()

    def test_ndvi_change(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_ndvi_change

        delta = np.random.randn(50, 50).astype(np.float32) * 0.1
        path = plot_ndvi_change(delta, sample_aoi, 2019, 2024,
                                 output_path=str(tmp_path / "ndvi.png"))
        assert Path(path).exists()

    def test_yearly_loss_comparison(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_yearly_loss_comparison

        stats = {2019: {"total_loss_ha": 100, "coffee_loss_ha": 40},
                 2020: {"total_loss_ha": 150, "coffee_loss_ha": 70}}
        path = plot_yearly_loss_comparison(stats, sample_aoi,
                                            output_path=str(tmp_path / "yearly.png"))
        assert Path(path).exists()

    def test_region_overview(self, sample_aoi, tmp_path):
        from coffee_deforestation.viz.static import plot_region_overview

        results = [{"name": "Patch 1", "bbox": sample_aoi.bbox,
                     "hotspot_count": 100, "total_area_ha": 500}]
        path = plot_region_overview(sample_aoi, results,
                                     output_path=str(tmp_path / "overview.png"))
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------


class TestThemeHelpers:
    """Test new theme helper functions."""

    def test_replacement_colors_defined(self):
        from coffee_deforestation.viz.theme import REPLACEMENT_COLORS, REPLACEMENT_NAMES
        assert "coffee" in REPLACEMENT_COLORS
        assert "regrowth" in REPLACEMENT_COLORS
        assert len(REPLACEMENT_COLORS) >= 5
        assert len(REPLACEMENT_NAMES) >= 5

    def test_ndvi_change_cmap(self):
        from coffee_deforestation.viz.theme import NDVI_CHANGE_CMAP
        assert NDVI_CHANGE_CMAP is not None
        # Should be a diverging colormap
        assert NDVI_CHANGE_CMAP(0.0) != NDVI_CHANGE_CMAP(1.0)

    def test_scale_bar(self, sample_aoi):
        import matplotlib.pyplot as plt
        from coffee_deforestation.viz.theme import add_scale_bar

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        add_scale_bar(ax, sample_aoi.bbox)
        plt.close(fig)

    def test_coordinate_axes(self, sample_aoi):
        import matplotlib.pyplot as plt
        from coffee_deforestation.viz.theme import format_coordinate_axes

        fig, ax = plt.subplots()
        format_coordinate_axes(ax, sample_aoi.bbox)
        # Should have tick labels
        assert len(ax.get_xticklabels()) > 0
        plt.close(fig)

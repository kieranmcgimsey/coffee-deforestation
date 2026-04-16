"""Smoke test: verify pipeline module imports and basic wiring."""

from unittest.mock import MagicMock, patch


def test_pipeline_imports():
    """Pipeline module imports without errors."""
    from coffee_deforestation.pipeline import run_aoi
    assert callable(run_aoi)


def test_pipeline_rejects_unknown_aoi():
    """Pipeline raises ValueError for unknown AOI names."""
    import pytest
    from coffee_deforestation.pipeline import run_aoi

    with pytest.raises(ValueError, match="Unknown AOI"):
        run_aoi("nonexistent_aoi")


def test_viz_theme_applies():
    """Visual theme applies without error."""
    from coffee_deforestation.viz.theme import apply_theme, COLORS
    apply_theme()
    assert "coffee" in COLORS
    assert "forest_stable" in COLORS
    assert len(COLORS) == 8

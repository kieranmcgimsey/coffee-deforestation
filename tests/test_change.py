"""Tests for change detection modules (mocked GEE)."""

from unittest.mock import MagicMock, patch


def test_detect_coffee_deforestation_imports():
    """Verify change detection module imports cleanly."""
    from coffee_deforestation.change.hansen_overlay import (
        detect_by_year_range,
        detect_coffee_deforestation_rule_based,
    )
    assert callable(detect_coffee_deforestation_rule_based)
    assert callable(detect_by_year_range)

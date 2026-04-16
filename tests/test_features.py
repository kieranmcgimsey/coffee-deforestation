"""Tests for feature engineering modules."""

from coffee_deforestation.features.stack import FEATURE_BAND_ORDER, get_feature_names


def test_feature_band_order():
    """Feature stack has expected number of bands."""
    assert len(FEATURE_BAND_ORDER) == 19  # 16 base + 3 temporal (ndvi_delta, vv_stddev, vh_stddev)


def test_get_feature_names():
    """get_feature_names returns consistent list."""
    names = get_feature_names()
    assert names == FEATURE_BAND_ORDER
    assert "ndvi" in names
    assert "vv_median" in names
    assert "elevation" in names


def test_feature_names_no_duplicates():
    """No duplicate feature names."""
    names = get_feature_names()
    assert len(names) == len(set(names))

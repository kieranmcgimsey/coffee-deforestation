"""Tests for label sampling and conversion."""

import numpy as np

from coffee_deforestation.ml.labels import (
    CLASS_MAP,
    WORLDCOVER_MAP,
    load_samples,
    samples_to_numpy,
    save_samples,
)


def test_class_map():
    """Class map has expected entries."""
    assert CLASS_MAP["coffee"] == 0
    assert CLASS_MAP["forest"] == 1
    assert CLASS_MAP["water"] == 4
    assert len(CLASS_MAP) == 5


def test_worldcover_map():
    """WorldCover mapping is complete for key classes."""
    assert WORLDCOVER_MAP[10] == 1  # Tree cover → forest
    assert WORLDCOVER_MAP[40] == 2  # Cropland
    assert WORLDCOVER_MAP[50] == 3  # Built-up
    assert WORLDCOVER_MAP[80] == 4  # Water


def test_samples_to_numpy():
    """Convert sample dicts to numpy arrays."""
    feature_names = ["ndvi", "evi", "vv"]
    samples = [
        {"ndvi": 0.5, "evi": 0.3, "vv": -15.0, "label": 0},
        {"ndvi": 0.8, "evi": 0.6, "vv": -12.0, "label": 1},
        {"ndvi": 0.2, "evi": 0.1, "vv": -18.0, "label": 2},
    ]
    X, y = samples_to_numpy(samples, feature_names)
    assert X.shape == (3, 3)
    assert y.shape == (3,)
    assert X[0, 0] == 0.5
    assert y[1] == 1


def test_samples_to_numpy_filters_invalid():
    """Invalid samples (negative label, None values) are filtered."""
    feature_names = ["f1", "f2"]
    samples = [
        {"f1": 1.0, "f2": 2.0, "label": 0},
        {"f1": None, "f2": 2.0, "label": 1},  # None value
        {"f1": 1.0, "f2": 2.0, "label": -1},   # Invalid label
    ]
    X, y = samples_to_numpy(samples, feature_names)
    assert X.shape == (1, 2)


def test_save_load_samples(tmp_path):
    """Save and load round-trips correctly."""
    X = np.random.rand(50, 10).astype(np.float32)
    y = np.random.randint(0, 5, 50).astype(np.int32)

    save_samples(X, y, tmp_path, "test_aoi")
    X_loaded, y_loaded = load_samples(tmp_path, "test_aoi")

    np.testing.assert_array_equal(X, X_loaded)
    np.testing.assert_array_equal(y, y_loaded)

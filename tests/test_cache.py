"""Tests for the content-addressed caching system."""

import json
from pathlib import Path

from coffee_deforestation.cache import (
    clear_cache,
    compute_hash,
    get_force,
    set_force,
)
from coffee_deforestation.config import AOIConfig, BBox, DrySeason


def test_compute_hash_deterministic():
    """Same inputs produce same hash."""
    inputs = {"stage": "test", "year": 2023, "aoi": "lam_dong"}
    h1 = compute_hash(inputs)
    h2 = compute_hash(inputs)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_hash_different_inputs():
    """Different inputs produce different hashes."""
    h1 = compute_hash({"year": 2023})
    h2 = compute_hash({"year": 2024})
    assert h1 != h2


def test_compute_hash_pydantic_model():
    """Pydantic models serialize correctly for hashing."""
    aoi = AOIConfig(
        id="test",
        name="Test",
        country="Test",
        coffee_type="Test",
        role="Test",
        bbox=BBox(west=1.0, south=2.0, east=3.0, north=4.0),
        dry_season=DrySeason(start_month=1, end_month=3),
        epsg_utm=32648,
    )
    h = compute_hash({"aoi": aoi})
    assert len(h) == 16


def test_compute_hash_order_independent():
    """Dict key order doesn't affect hash."""
    h1 = compute_hash({"a": 1, "b": 2})
    h2 = compute_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_force_flag():
    """Force flag defaults to False and can be toggled."""
    assert get_force() is False
    set_force(True)
    assert get_force() is True
    set_force(False)
    assert get_force() is False


def test_clear_cache_empty(tmp_path: Path):
    """Clearing a non-existent cache returns 0."""
    count = clear_cache(cache_dir=str(tmp_path / "nonexistent"))
    assert count == 0

"""Tests for the @cached decorator functionality."""

import json
from pathlib import Path
from unittest.mock import patch

from coffee_deforestation.cache import cached, compute_hash, set_force


def test_cached_decorator_creates_output(tmp_path: Path):
    """Cached function writes meta.json on first call."""
    cache_dir = str(tmp_path / "cache")

    @cached(stage="test_stage", cache_dir=cache_dir)
    def my_func(name: str, value: int) -> Path:
        output = tmp_path / f"{name}_{value}.txt"
        output.write_text(f"result: {value}")
        return output

    result = my_func("hello", 42)
    assert result.exists()
    assert result.read_text() == "result: 42"

    # Meta file should exist
    meta_files = list((tmp_path / "cache" / "test_stage").glob("*.meta.json"))
    assert len(meta_files) == 1

    meta = json.loads(meta_files[0].read_text())
    assert meta["stage"] == "test_stage"


def test_cached_decorator_returns_cached(tmp_path: Path):
    """Second call returns cached result without re-running."""
    cache_dir = str(tmp_path / "cache")
    call_count = 0

    @cached(stage="test_stage", cache_dir=cache_dir)
    def my_func(name: str) -> Path:
        nonlocal call_count
        call_count += 1
        output = tmp_path / f"{name}.txt"
        output.write_text(f"call {call_count}")
        return output

    result1 = my_func("test")
    assert call_count == 1
    assert result1.read_text() == "call 1"

    result2 = my_func("test")
    assert call_count == 1  # Not called again
    assert result2 == result1


def test_cached_decorator_force_bypass(tmp_path: Path):
    """Force flag bypasses cache."""
    cache_dir = str(tmp_path / "cache")
    call_count = 0

    @cached(stage="test_stage", cache_dir=cache_dir)
    def my_func(name: str) -> Path:
        nonlocal call_count
        call_count += 1
        output = tmp_path / f"{name}.txt"
        output.write_text(f"call {call_count}")
        return output

    my_func("test")
    assert call_count == 1

    set_force(True)
    try:
        my_func("test")
        assert call_count == 2
    finally:
        set_force(False)


def test_cached_decorator_different_args(tmp_path: Path):
    """Different arguments produce separate cache entries."""
    cache_dir = str(tmp_path / "cache")

    @cached(stage="test_stage", cache_dir=cache_dir)
    def my_func(year: int) -> Path:
        output = tmp_path / f"out_{year}.txt"
        output.write_text(str(year))
        return output

    r1 = my_func(2023)
    r2 = my_func(2024)
    assert r1 != r2
    assert r1.read_text() == "2023"
    assert r2.read_text() == "2024"

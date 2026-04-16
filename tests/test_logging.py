"""Tests for logging setup."""

from pathlib import Path

from coffee_deforestation.logging_setup import setup_logging


def test_setup_logging(tmp_path: Path):
    """Logging setup creates log files."""
    log_dir = str(tmp_path / "logs")
    setup_logging(level="DEBUG", log_dir=log_dir)
    assert (tmp_path / "logs").exists()

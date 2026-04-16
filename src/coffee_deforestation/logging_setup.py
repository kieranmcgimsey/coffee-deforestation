"""Logging configuration using loguru.

What: Configures a single loguru logger for the entire pipeline.
Why: Consistent, structured logging across all modules with file rotation.
Assumes: outputs/logs/ directory exists or will be created.
Produces: Configured loguru logger ready for import.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT


def setup_logging(level: str = "INFO", log_dir: str = "outputs/logs") -> None:
    """Configure loguru with console and file sinks."""
    log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console: concise format
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    )

    # File: detailed with rotation
    logger.add(
        str(log_path / "pipeline.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
    )

    # Error-only JSONL for structured error tracking
    logger.add(
        str(log_path / "pipeline_errors.jsonl"),
        level="ERROR",
        serialize=True,
        rotation="5 MB",
    )

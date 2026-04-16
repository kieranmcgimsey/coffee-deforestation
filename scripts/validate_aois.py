"""Validate all AOIs with cheap GEE queries before committing to expensive exports.

Usage: uv run python scripts/validate_aois.py [--aoi AOI_NAME]
"""

from __future__ import annotations

import sys

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    aoi: str = typer.Option("", help="Validate a single AOI (default: all)"),
) -> None:
    """Validate AOIs for sufficient coffee, forest, and loss coverage."""
    from coffee_deforestation.config import load_aois, load_pipeline_config
    from coffee_deforestation.data.gee_client import init_gee
    from coffee_deforestation.data.validate_aoi import validate_aoi
    from coffee_deforestation.logging_setup import setup_logging

    setup_logging()
    init_gee()

    aois = load_aois()
    pipeline_config = load_pipeline_config()

    if aoi:
        if aoi not in aois:
            logger.error(f"Unknown AOI: {aoi}. Available: {list(aois.keys())}")
            sys.exit(1)
        aois = {aoi: aois[aoi]}

    print("\n" + "=" * 80)
    print("AOI Validation Results")
    print("=" * 80)
    print(f"{'AOI':<15} | {'Coffee':>10} | {'Forest-2000':>12} | {'Loss pixels':>12} | Status")
    print("-" * 80)

    all_passed = True
    for aoi_id, aoi_config in aois.items():
        result = validate_aoi(aoi_config, pipeline_config)
        print(result.summary_row())
        for msg in result.messages:
            print(f"  → {msg}")
        if not result.passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("All AOIs PASSED validation.")
    else:
        print("Some AOIs FAILED validation. Check messages above.")
        sys.exit(1)


if __name__ == "__main__":
    app()

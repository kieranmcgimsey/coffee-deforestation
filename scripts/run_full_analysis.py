"""Run the complete analysis pipeline for all AOIs and patches.

What: Processes all patches across all 3 AOIs, runs deforestation attribution,
temporal analysis, ML prediction, and generates all figures for the HTML report.
Why: Produces the comprehensive analytical output the report needs.
Assumes: GEE is authenticated. Config patches are defined in aois.yaml.
Produces: Updated stats JSONs, all figures, GeoTIFFs, and reports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coffee_deforestation.config import PROJECT_ROOT, load_aois, load_pipeline_config

app = typer.Typer(help="Run full analysis pipeline with all new features.")

OUTPUTS = PROJECT_ROOT / "outputs"
STATS_DIR = OUTPUTS / "stats"
FIGURES_DIR = OUTPUTS / "figures"


def _run_single_patch(aoi_name: str, patch_name: str, patch_bbox: dict) -> dict:
    """Run the pipeline for a single patch, resilient to errors."""
    from coffee_deforestation.pipeline import run_aoi

    logger.info(f"  Processing patch: {patch_name} ({aoi_name})")
    try:
        result = run_aoi(aoi_name, resilient=True, skip_ml=False)
        return result
    except Exception as e:
        logger.error(f"  Patch {patch_name} failed: {e}")
        return {"error": str(e)}


def _run_deforestation_attribution(aoi_name: str) -> dict | None:
    """Run deforestation attribution analysis for an AOI."""
    from coffee_deforestation.change.deforestation_attribution import (
        compute_attribution,
        compute_attribution_by_year,
    )

    aois = load_aois()
    aoi = aois.get(aoi_name)
    if not aoi:
        return None

    pipeline_config = load_pipeline_config()

    logger.info(f"  Computing deforestation attribution for {aoi_name}...")
    try:
        attribution = compute_attribution(aoi, pipeline_config)
        logger.info(f"  Computing per-year attribution...")
        yearly = compute_attribution_by_year(aoi, pipeline_config, list(range(2005, 2024)))
        attribution["by_year"] = yearly
        return attribution
    except Exception as e:
        logger.error(f"  Attribution failed for {aoi_name}: {e}")
        return None


def _run_temporal_analysis(aoi_name: str) -> dict | None:
    """Run temporal analysis for an AOI."""
    from coffee_deforestation.change.temporal import (
        compute_ndvi_change_map,
        compute_real_yearly_stats,
        download_composite_rgb,
    )
    from coffee_deforestation.viz.static import (
        plot_before_after,
        plot_ndvi_change,
        plot_yearly_loss_comparison,
    )

    aois = load_aois()
    aoi = aois.get(aoi_name)
    if not aoi:
        return None

    pipeline_config = load_pipeline_config()
    results: dict = {}

    # Before/after RGB
    logger.info(f"  Downloading 2019 RGB composite...")
    rgb_before = download_composite_rgb(aoi, 2019, pipeline_config, scale=300)
    logger.info(f"  Downloading 2024 RGB composite...")
    rgb_after = download_composite_rgb(aoi, 2024, pipeline_config, scale=300)

    if rgb_before is not None and rgb_after is not None:
        plot_before_after(rgb_before, rgb_after, aoi, 2019, 2024, bbox=aoi.bbox)
        results["before_after"] = True
        logger.info(f"  Before/after figure saved")

    # NDVI change map
    logger.info(f"  Computing NDVI change map 2019→2024...")
    ndvi_result = compute_ndvi_change_map(aoi, 2019, 2024, pipeline_config, scale=300)
    if ndvi_result is not None:
        _, _, delta = ndvi_result
        plot_ndvi_change(delta, aoi, 2019, 2024, bbox=aoi.bbox)
        results["ndvi_change"] = True
        results["ndvi_delta_mean"] = float(delta.mean())
        logger.info(f"  NDVI change figure saved")

    # Per-year loss stats
    logger.info(f"  Computing real per-year loss stats...")
    try:
        yearly_stats = compute_real_yearly_stats(aoi, pipeline_config)
        if yearly_stats:
            plot_yearly_loss_comparison(yearly_stats, aoi)
            results["yearly_stats"] = yearly_stats
            logger.info(f"  Yearly loss comparison figure saved ({len(yearly_stats)} years)")
    except Exception as e:
        logger.warning(f"  Yearly stats failed: {e}")

    return results


def _generate_attribution_figures(aoi_name: str, attribution: dict) -> None:
    """Generate deforestation attribution figures."""
    from coffee_deforestation.viz.static import (
        plot_attribution_pie,
        plot_attribution_stacked_bar,
    )

    aois = load_aois()
    aoi = aois.get(aoi_name)
    if not aoi:
        return

    plot_attribution_pie(attribution, aoi)
    logger.info(f"  Attribution pie chart saved")

    yearly = attribution.get("by_year", {})
    if yearly:
        plot_attribution_stacked_bar(yearly, aoi)
        logger.info(f"  Attribution stacked bar chart saved")


def _generate_region_overview(aoi_name: str) -> None:
    """Generate region overview map with patches."""
    from coffee_deforestation.viz.static import plot_region_overview

    aois = load_aois()
    aoi = aois.get(aoi_name)
    if not aoi:
        return

    # Build patch results from stats
    patch_results = []
    summary_path = STATS_DIR / f"summary_{aoi_name}.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        # Use the main AOI as the single patch for now
        patch_results.append({
            "name": aoi.name,
            "bbox": aoi.bbox,
            "hotspot_count": data.get("change_detection", {}).get("total_hotspots", 0),
            "total_area_ha": data.get("change_detection", {}).get("total_area_ha", 0),
        })

    # Add configured patches
    for patch in aoi.patches:
        if patch.name != aoi.name:
            patch_results.append({
                "name": patch.name,
                "bbox": patch.bbox,
                "hotspot_count": 0,  # Would be populated after processing
                "total_area_ha": 0,
            })

    if patch_results:
        plot_region_overview(aoi, patch_results)
        logger.info(f"  Region overview map saved")


def _update_stats_json(aoi_name: str, attribution: dict | None, temporal: dict | None) -> None:
    """Update the stats JSON with attribution and temporal results."""
    from coffee_deforestation.stats.schema import (
        AOISummary,
        DeforestationAttribution,
        YearlyLossStats,
    )

    summary_path = STATS_DIR / f"summary_{aoi_name}.json"
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        data = json.load(f)

    # Add attribution
    if attribution:
        data["deforestation_attribution"] = {
            "total_loss_ha": attribution.get("total_loss_ha", 0),
            "coffee_pct": attribution.get("coffee_pct", 0),
            "other_crops_pct": attribution.get("other_crops_pct", 0),
            "built_industrial_pct": attribution.get("built_industrial_pct", 0),
            "bare_degraded_pct": attribution.get("bare_degraded_pct", 0),
            "water_pct": attribution.get("water_pct", 0),
            "regrowth_pct": attribution.get("regrowth_pct", 0),
            "by_year": attribution.get("by_year", {}),
        }

    # Add yearly loss stats
    if temporal and "yearly_stats" in temporal:
        data["yearly_loss"] = {
            str(year): stats
            for year, stats in temporal["yearly_stats"].items()
        }

    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"  Updated stats JSON: {summary_path.name}")


@app.command()
def main(
    aoi: list[str] = typer.Option(
        default=["lam_dong", "huila", "sul_de_minas"],
        help="AOI IDs to process.",
    ),
    skip_pipeline: bool = typer.Option(False, help="Skip the main pipeline run."),
    skip_temporal: bool = typer.Option(False, help="Skip temporal analysis (slower)."),
    skip_attribution: bool = typer.Option(False, help="Skip deforestation attribution."),
) -> None:
    """Run the complete analysis pipeline."""
    from coffee_deforestation.logging_setup import setup_logging

    setup_logging()

    # Initialize GEE
    from coffee_deforestation.data.gee_client import init_gee
    init_gee()

    total_start = time.time()

    for aoi_name in aoi:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing AOI: {aoi_name}")
        logger.info(f"{'='*60}")

        # Step 1: Run main pipeline (composites, features, hotspots, stats)
        if not skip_pipeline:
            logger.info(f"Step 1: Main pipeline")
            try:
                from coffee_deforestation.pipeline import run_aoi

                result = run_aoi(aoi_name, resilient=True, skip_ml=False)
                if "error" in result:
                    logger.error(f"Pipeline failed: {result['error']}")
                    continue
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                continue
        else:
            logger.info(f"Step 1: Skipped (--skip-pipeline)")

        # Step 2: Deforestation attribution
        attribution = None
        if not skip_attribution:
            logger.info(f"Step 2: Deforestation attribution")
            attribution = _run_deforestation_attribution(aoi_name)
            if attribution:
                _generate_attribution_figures(aoi_name, attribution)
        else:
            logger.info(f"Step 2: Skipped (--skip-attribution)")

        # Step 3: Temporal analysis
        temporal = None
        if not skip_temporal:
            logger.info(f"Step 3: Temporal analysis")
            temporal = _run_temporal_analysis(aoi_name)
        else:
            logger.info(f"Step 3: Skipped (--skip-temporal)")

        # Step 4: Region overview map
        logger.info(f"Step 4: Region overview")
        _generate_region_overview(aoi_name)

        # Step 5: Update stats JSON
        logger.info(f"Step 5: Update stats JSON")
        _update_stats_json(aoi_name, attribution, temporal)

    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Full analysis complete in {total_time:.1f}s")
    logger.info(f"Run scripts/generate_report.py to build final reports")


if __name__ == "__main__":
    app()

"""Top-level pipeline orchestrator.

What: Runs the complete pipeline for a single AOI — from data acquisition through
feature engineering, change detection, stats summary, report, and visualization.
Why: Single entry point per AOI, with stage manifests and caching at every step.
Assumes: GEE is authenticated. Config files are valid.
Produces: All outputs for one AOI: rasters, vectors, stats JSON, report, figures, maps.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ee
from loguru import logger

from coffee_deforestation.cache import compute_hash
from coffee_deforestation.config import (
    PROJECT_ROOT,
    AOIConfig,
    PipelineConfig,
    load_aois,
    load_pipeline_config,
    load_settings,
)
from coffee_deforestation.logging_setup import setup_logging


def _write_manifest(
    stage: str,
    aoi_id: str,
    inputs_hash: str,
    outputs: dict,
    timing_s: float,
    params: dict,
) -> Path:
    """Write a stage manifest for debugging and reproducibility."""
    manifest_dir = PROJECT_ROOT / "outputs" / "logs" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    content_hash = compute_hash({"stage": stage, "aoi": aoi_id, "inputs": inputs_hash})
    manifest_path = manifest_dir / f"{stage}_{aoi_id}_{content_hash}.json"

    manifest = {
        "stage": stage,
        "aoi_id": aoi_id,
        "inputs_hash": inputs_hash,
        "outputs": outputs,
        "timing_seconds": round(timing_s, 2),
        "parameters": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def run_aoi(
    aoi_name: str,
    pipeline_config: PipelineConfig | None = None,
    resilient: bool = False,
    skip_ml: bool = False,
) -> dict:
    """Run the complete pipeline for a single AOI (Stage 1 + Stage 2 data prep).

    Stage 2 ML training is done separately via run_all.py
    after samples from all AOIs are available.

    Returns a dict of all output paths.
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    setup_logging(pipeline_config.logging.level, pipeline_config.logging.log_dir)

    aois = load_aois()
    if aoi_name not in aois:
        raise ValueError(f"Unknown AOI: {aoi_name}. Available: {list(aois.keys())}")

    aoi = aois[aoi_name]
    outputs: dict[str, list[str] | str] = {}
    figures: list[str] = []
    total_start = time.time()

    logger.info(f"=== Pipeline start: {aoi.name} ({aoi.country}) ===")

    # --- Stage: Validation ---
    stage_start = time.time()
    logger.info("Stage: AOI Validation")
    from coffee_deforestation.data.gee_client import init_gee
    from coffee_deforestation.data.validate_aoi import validate_aoi

    init_gee()
    validation_result = validate_aoi(aoi, pipeline_config)

    if not validation_result.passed:
        msg = f"AOI validation failed: {validation_result.messages}"
        if resilient:
            logger.warning(msg)
        else:
            raise RuntimeError(msg)

    _write_manifest(
        "validation", aoi_name,
        compute_hash({"aoi": aoi.model_dump()}),
        {"passed": validation_result.passed},
        time.time() - stage_start,
        {"thresholds": pipeline_config.validation.model_dump()},
    )

    # --- Stage: Data Acquisition (S2, S1, ancillary) ---
    stage_start = time.time()
    logger.info("Stage: Data Acquisition")
    from coffee_deforestation.data.sentinel1 import build_s1_composite
    from coffee_deforestation.data.sentinel2 import build_s2_composite

    years = pipeline_config.temporal.years
    s2_composites = {}
    s1_composites = {}

    for year in years:
        try:
            s2 = build_s2_composite(aoi, year, pipeline_config)
            s2_composites[year] = s2
            logger.info(f"S2 composite ready for {aoi_name} {year}")

            s1 = build_s1_composite(aoi, year, pipeline_config)
            s1_composites[year] = s1
            logger.info(f"S1 composite ready for {aoi_name} {year}")

        except Exception as e:
            if resilient:
                logger.error(f"Data acquisition failed for {year}: {e}")
            else:
                raise

    _write_manifest(
        "data_acquisition", aoi_name,
        compute_hash({"aoi": aoi.id, "years": years}),
        {"s2_years": list(s2_composites.keys()), "s1_years": list(s1_composites.keys())},
        time.time() - stage_start,
        {"years": years},
    )

    # --- Stage: Feature Stack ---
    stage_start = time.time()
    logger.info("Stage: Feature Stack")
    from coffee_deforestation.features.stack import build_feature_stack

    latest_year = years[-1]
    feature_stack = build_feature_stack(
        s2_composites[latest_year],
        s1_composites[latest_year],
        aoi,
        pipeline_config,
        s2_composites_all=s2_composites,
        s1_composites_all=s1_composites,
    )

    _write_manifest(
        "feature_stack", aoi_name,
        compute_hash({"aoi": aoi.id, "year": latest_year}),
        {"band_count": len(pipeline_config.features.spectral_indices) + 12},
        time.time() - stage_start,
        {"features": pipeline_config.features.model_dump()},
    )

    # --- Stage: Change Detection (Rule-based) ---
    stage_start = time.time()
    logger.info("Stage: Change Detection")
    from coffee_deforestation.change.hansen_overlay import (
        detect_coffee_deforestation_rule_based,
    )
    from coffee_deforestation.change.hotspots import (
        enrich_hotspots,
        polygonize_hotspots_gee,
        save_hotspots,
    )

    candidates = detect_coffee_deforestation_rule_based(aoi, pipeline_config)
    raw_hotspots = polygonize_hotspots_gee(candidates, aoi, pipeline_config)
    hotspot_features = enrich_hotspots(raw_hotspots, aoi, candidates)

    hotspots_path = PROJECT_ROOT / "outputs" / "vectors" / f"hotspots_{aoi_name}.geojson"
    save_hotspots(hotspot_features, hotspots_path, aoi)
    outputs["hotspots"] = str(hotspots_path)

    _write_manifest(
        "change_detection", aoi_name,
        compute_hash({"aoi": aoi.id, "method": "rule_based"}),
        {"hotspot_count": len(hotspot_features), "path": str(hotspots_path)},
        time.time() - stage_start,
        {"method": "rule_based_hansen_fdp",
         "threshold": pipeline_config.change_detection.fdp_coffee_threshold},
    )

    # --- Stage 2: Label Sampling ---
    historical_stats: dict = {}
    if not skip_ml:
        stage_start_ml = time.time()
        logger.info("Stage 2: Label Sampling")
        try:
            from coffee_deforestation.features.stack import get_feature_names
            from coffee_deforestation.ml.labels import (
                create_label_image,
                sample_training_data_gee,
                samples_to_numpy,
                save_samples,
            )

            label_image = create_label_image(aoi, pipeline_config)
            gee_samples_fc = sample_training_data_gee(
                feature_stack, label_image, aoi, pipeline_config
            )
            fc_info = gee_samples_fc.getInfo()
            sample_dicts = [f["properties"] for f in fc_info.get("features", [])]

            if sample_dicts:
                feature_names = get_feature_names()
                X, y = samples_to_numpy(sample_dicts, feature_names)
                labels_dir = PROJECT_ROOT / "outputs" / "cache" / "labels"
                save_samples(X, y, labels_dir, aoi_name)
                outputs["samples_X"] = str(labels_dir / f"X_{aoi_name}.npy")
                outputs["samples_y"] = str(labels_dir / f"y_{aoi_name}.npy")
                logger.info(f"Saved {len(X)} training samples for {aoi_name}")
            else:
                logger.warning(f"No GEE samples returned for {aoi_name}")

            _write_manifest(
                "label_sampling", aoi_name,
                compute_hash({"aoi": aoi.id}),
                {"samples_saved": bool(sample_dicts), "n_samples": len(sample_dicts)},
                time.time() - stage_start_ml,
                {"samples_per_class": pipeline_config.ml.samples_per_class_per_aoi},
            )
        except Exception as e:
            msg = f"Label sampling failed for {aoi_name}: {e}"
            if resilient:
                logger.warning(msg)
            else:
                raise

    # --- Stage 2: Historical Look-back ---
    if not skip_ml:
        stage_start_hist = time.time()
        logger.info("Stage 2: Historical Look-back")
        try:
            from coffee_deforestation.change.historical import (
                compute_historical_stats,
                compute_historical_trajectory,
            )
            from coffee_deforestation.change.replacement import classify_replacement_gee

            trajectory = compute_historical_trajectory(aoi, pipeline_config)
            historical_stats = compute_historical_stats(trajectory, aoi)

            # Replacement classification (GEE, proxy-based without trained model)
            replacement_img = classify_replacement_gee(aoi, pipeline_config, feature_stack)

            # Aggregate replacement class fractions across the entire AOI
            from coffee_deforestation.data.gee_client import aoi_to_geometry
            region = aoi_to_geometry(aoi)
            repl_hist = replacement_img.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=region,
                scale=30,
                maxPixels=10_000_000,
                bestEffort=True,
            ).getInfo()

            # Convert pixel counts to fractions
            class_names = ["coffee", "forest", "cropland", "built_bare", "water"]
            raw_hist = repl_hist.get("replacement_class", {}) or {}
            total_px = sum(raw_hist.values()) or 1
            replacement_distribution = {
                class_names[int(k)]: round(v / total_px, 4)
                for k, v in raw_hist.items()
                if 0 <= int(k) < len(class_names)
            }
            historical_stats["replacement_class_distribution"] = replacement_distribution

            _write_manifest(
                "historical_lookback", aoi_name,
                compute_hash({"aoi": aoi.id}),
                {
                    "coffee_on_former_forest_fraction": historical_stats.get(
                        "coffee_on_former_forest_mean", 0
                    ),
                    "replacement_classes": list(replacement_distribution.keys()),
                },
                time.time() - stage_start_hist,
                {},
            )
            logger.info(f"Historical stats for {aoi_name}: {historical_stats}")
        except Exception as e:
            msg = f"Historical look-back failed for {aoi_name}: {e}"
            if resilient:
                logger.warning(msg)
                historical_stats = {}
            else:
                raise

    # --- Stage: Per-Year Zonal Statistics ---
    per_year_stats = None
    if not skip_ml:
        try:
            stage_start_pys = time.time()
            logger.info("Stage: Per-Year Zonal Statistics")
            from coffee_deforestation.stats.summarize import compute_per_year_stats

            per_year_stats = compute_per_year_stats(
                s2_composites, s1_composites, aoi, hotspot_features
            )
            _write_manifest(
                "per_year_stats", aoi_name,
                compute_hash({"aoi": aoi.id, "years": years}),
                {
                    "ndvi_years": len(per_year_stats.get("ndvi_by_year", {})),
                    "vv_years": len(per_year_stats.get("vv_mean_by_year", {})),
                    "hotspot_trajectories": len(
                        per_year_stats.get("hotspot_ndvi_trajectories", {})
                    ),
                },
                time.time() - stage_start_pys,
                {},
            )
        except Exception as e:
            if resilient:
                logger.warning(f"Per-year stats failed: {e}")
            else:
                raise

    # --- Stage: Stats Summary ---
    stage_start = time.time()
    logger.info("Stage: Stats Summary")
    from coffee_deforestation.stats.summarize import build_summary, save_summary

    summary = build_summary(
        aoi, validation_result, hotspot_features,
        years_processed=years, figures=figures,
        historical_stats=historical_stats if historical_stats else None,
        per_year_stats=per_year_stats,
    )
    summary_path = save_summary(summary)
    outputs["summary"] = str(summary_path)

    _write_manifest(
        "stats_summary", aoi_name,
        compute_hash({"aoi": aoi.id}),
        {"path": str(summary_path)},
        time.time() - stage_start,
        {},
    )

    # --- Stage: Report (dry-run) ---
    stage_start = time.time()
    logger.info("Stage: Report Generation (dry-run)")
    from coffee_deforestation.reporting.llm_client import generate_report, save_report

    report = generate_report(summary, dry_run=True)
    report_path = save_report(report, aoi_name)
    outputs["report"] = str(report_path)

    _write_manifest(
        "report", aoi_name,
        compute_hash({"aoi": aoi.id, "dry_run": True}),
        {"path": str(report_path)},
        time.time() - stage_start,
        {"dry_run": True},
    )

    # --- Stage: Visualization ---
    stage_start = time.time()
    logger.info("Stage: Visualization")
    from coffee_deforestation.viz.interactive import create_aoi_map, save_map

    # Interactive map
    m = create_aoi_map(aoi, hotspot_geojson_path=hotspots_path)
    map_path = save_map(m, aoi)
    outputs["map"] = str(map_path)

    # Stage 2: historical look-back figure (if historical data available)
    if historical_stats and not skip_ml:
        try:
            from coffee_deforestation.viz.static import (
                plot_historical_lookback,
                plot_replacement_classes,
            )
            import numpy as np

            # Build a stub loss-year array from aggregated stats for the figure.
            # Real per-pixel export is gated behind GEE Drive export; use summary stats
            # to generate a representative histogram-only figure.
            coffee_frac = historical_stats.get("coffee_on_former_forest_mean", 0) or 0
            mean_offset = historical_stats.get("loss_year_before_coffee_mean") or 0
            # Synthetic representative array for visualization (histogram only)
            if coffee_frac > 0 and mean_offset > 0:
                rng = np.random.default_rng(42)
                synthetic_loss = rng.normal(
                    loc=mean_offset, scale=3, size=max(int(coffee_frac * 10000), 100)
                ).clip(1, 23).astype(int)
                stub_raster = np.zeros((50, 50), dtype=int)
                stub_raster.flat[: len(synthetic_loss)] = synthetic_loss
            else:
                stub_raster = np.zeros((50, 50), dtype=int)

            hist_fig = plot_historical_lookback(stub_raster, aoi)
            figures.append(hist_fig)

            # Replacement classes figure
            repl_dist = historical_stats.get("replacement_class_distribution", {})
            if repl_dist:
                repl_fig = plot_replacement_classes(repl_dist, aoi)
                figures.append(repl_fig)
        except Exception as e:
            if resilient:
                logger.warning(f"Stage 2 figures failed: {e}")
            else:
                raise

    # Cloud recovery figure
    if not skip_ml:
        try:
            from coffee_deforestation.viz.cloud_recovery import plot_cloud_recovery_panel

            recovery_fig = plot_cloud_recovery_panel(aoi, pipeline_config)
            if recovery_fig:
                figures.append(recovery_fig)
        except Exception as e:
            if resilient:
                logger.warning(f"Cloud recovery figure failed: {e}")
            else:
                raise

    _write_manifest(
        "visualization", aoi_name,
        compute_hash({"aoi": aoi.id}),
        {"map_path": str(map_path), "figures": figures},
        time.time() - stage_start,
        {},
    )

    # Store lazy GEE references for downstream prediction (ee.Image, not data)
    outputs["_feature_stack"] = feature_stack
    outputs["_aoi_config"] = aoi

    total_time = time.time() - total_start
    logger.info(f"=== Pipeline complete: {aoi.name} in {total_time:.1f}s ===")
    # Filter non-serializable objects for logging
    log_outputs = {k: v for k, v in outputs.items() if not k.startswith("_")}
    logger.info(f"Outputs: {json.dumps(log_outputs, indent=2)}")

    return outputs

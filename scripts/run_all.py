"""Run the pipeline for all AOIs, then train models with cross-AOI holdout.

Usage: uv run python scripts/run_all.py [--force] [--resilient] [--skip-ml]

Stage 2 flow:
  1. run_aoi() for each AOI  — GEE composites, hotspots, label sampling, historical stats
  2. train_models()           — RF + XGBoost trained on pooled samples, cross-AOI eval
  3. plot_ml_figures()        — confusion matrices, ablation chart, feature importance
  4. save updated stats JSONs — including ML metrics and ablation results
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(
    force: bool = typer.Option(False, help="Bypass cache"),
    resilient: bool = typer.Option(False, help="Continue on errors"),
    skip_ml: bool = typer.Option(False, help="Skip ML training (data-prep only)"),
) -> None:
    """Run the complete pipeline for all AOIs, then train and evaluate models."""
    import numpy as np

    from coffee_deforestation.cache import set_force
    from coffee_deforestation.config import PROJECT_ROOT, load_aois, load_pipeline_config
    from coffee_deforestation.logging_setup import setup_logging
    from coffee_deforestation.pipeline import run_aoi

    setup_logging()

    if force:
        set_force(True)

    config = load_pipeline_config()
    aois = load_aois()
    results: dict[str, dict] = {}

    # --- Step 1: Per-AOI pipeline (data acquisition + sampling + historical) ---
    for aoi_name in aois:
        print(f"\n{'='*60}")
        print(f"Running pipeline for: {aoi_name}")
        print(f"{'='*60}")
        try:
            results[aoi_name] = run_aoi(
                aoi_name,
                pipeline_config=config,
                resilient=resilient,
                skip_ml=skip_ml,
            )
        except Exception as e:
            if resilient:
                print(f"ERROR: {aoi_name} failed: {e}")
                results[aoi_name] = {"error": str(e)}
            else:
                raise

    if skip_ml:
        print("\nSkipping ML training (--skip-ml).")
        _print_results(results)
        return

    # --- Step 2: Train models on pooled samples from all AOIs ---
    print(f"\n{'='*60}")
    print("Training ML models (RF + XGBoost)")
    print(f"{'='*60}")

    from coffee_deforestation.ml.evaluate import (
        evaluate_model,
        run_ablation,
        run_cross_aoi_evaluation,
    )
    from coffee_deforestation.ml.explain import get_feature_importance, plot_feature_importance
    from coffee_deforestation.ml.labels import load_samples
    from coffee_deforestation.ml.train import save_model, split_data, train_both

    data_dir = PROJECT_ROOT / "outputs" / "cache" / "labels"
    model_dir = PROJECT_ROOT / "outputs" / "cache" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load samples for all AOIs that have data
    aoi_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for aoi_name in aois:
        if "error" in results.get(aoi_name, {"error": True}):
            continue
        try:
            X, y = load_samples(data_dir, aoi_name)
            aoi_data[aoi_name] = (X, y)
            print(f"  Loaded {X.shape[0]} samples for {aoi_name}")
        except FileNotFoundError:
            print(f"  No training data for {aoi_name} — skipping")

    if not aoi_data:
        print("No training data found. Run the pipeline first (without --skip-ml).")
        _print_results(results)
        return

    # Pool all samples for main model training
    X_all = np.concatenate([d[0] for d in aoi_data.values()])
    y_all = np.concatenate([d[1] for d in aoi_data.values()])
    X_train, X_test, y_train, y_test = split_data(X_all, y_all, config.ml.test_split)
    print(f"  Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    models = train_both(X_train, y_train, config)

    # Evaluate and save each model
    model_metrics_all: dict[str, dict] = {}
    ablation_results: dict = {}

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        model_metrics_all[name] = metrics
        save_model(model, model_dir / f"{name}.pkl")
        importances = get_feature_importance(model)
        plot_feature_importance(importances, name)
        print(
            f"  {name}: accuracy={metrics['accuracy']:.3f}, "
            f"coffee F1={metrics['f1_coffee']:.3f}"
        )

    # --- Step 3: Ablation (S1-only / S2-only / S1+S2) using RF ---
    from sklearn.ensemble import RandomForestClassifier

    rf_config = config.ml.random_forest
    rf_kwargs = {
        "n_estimators": rf_config.n_estimators,
        "max_depth": rf_config.max_depth,
        "min_samples_leaf": rf_config.min_samples_leaf,
        "random_state": rf_config.random_state,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    ablation_results = run_ablation(
        RandomForestClassifier, rf_kwargs,
        X_train, y_train, X_test, y_test,
    )
    print("\nAblation results:")
    for name, m in ablation_results.items():
        print(f"  {name}: F1={m['f1_coffee']:.3f}")

    # --- Cross-AOI holdout (if 2+ AOIs have data) ---
    cross_results: dict = {}
    if len(aoi_data) >= 2:
        cross_results = run_cross_aoi_evaluation(models, aoi_data)
        print("\nCross-AOI holdout:")
        for holdout, model_res in cross_results.items():
            for model_name, m in model_res.items():
                print(f"  {holdout}/{model_name}: F1={m['f1_coffee']:.3f}")

    # --- Step 4: Stage 2 ML figures + update stats JSONs ---
    from coffee_deforestation.ml.labels import CLASS_MAP
    from coffee_deforestation.stats.schema import AblationResult, AblationSummary, ModelMetrics
    from coffee_deforestation.stats.summarize import save_summary
    from coffee_deforestation.viz.static import plot_ablation_bar_chart, plot_confusion_matrix

    aois_config = load_aois()
    import json

    for aoi_name, aoi in aois_config.items():
        if "error" in results.get(aoi_name, {"error": True}):
            continue

        # Load existing summary and update with ML metrics
        summary_path = PROJECT_ROOT / "outputs" / "stats" / f"summary_{aoi_name}.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary_dict = json.load(f)

        # Re-load summary as pydantic model
        from coffee_deforestation.stats.schema import AOISummary

        summary = AOISummary.model_validate(summary_dict)

        # Update model metrics (use RF as primary)
        rf_metrics = model_metrics_all.get("random_forest", {})
        summary.model_metrics = ModelMetrics(
            model_type="random_forest",
            accuracy=rf_metrics.get("accuracy", 0.0),
            f1_coffee=rf_metrics.get("f1_coffee", 0.0),
            precision_coffee=rf_metrics.get("precision_coffee", 0.0),
            recall_coffee=rf_metrics.get("recall_coffee", 0.0),
        )

        # Update ablation
        summary.ablation = AblationSummary(
            s1_only=AblationResult(
                f1_coffee=ablation_results.get("s1_only", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s1_only", {}).get("accuracy", 0.0),
            ),
            s2_only=AblationResult(
                f1_coffee=ablation_results.get("s2_only", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s2_only", {}).get("accuracy", 0.0),
            ),
            s1_s2=AblationResult(
                f1_coffee=ablation_results.get("s1_s2", {}).get("f1_coffee", 0.0),
                accuracy=ablation_results.get("s1_s2", {}).get("accuracy", 0.0),
            ),
        )

        save_summary(summary)

        # Confusion matrix figure
        cm = rf_metrics.get("confusion_matrix")
        if cm:
            class_names = list(CLASS_MAP.keys())
            n_classes = len(np.unique(y_test))
            plot_confusion_matrix(
                np.array(cm), class_names[:n_classes],
                model_name="random_forest", aoi=aoi,
            )

        # Per-AOI cross-AOI confusion matrix
        if aoi_name in cross_results:
            cross_cm = cross_results[aoi_name].get("random_forest", {}).get(
                "confusion_matrix"
            )
            if cross_cm:
                n_classes_cross = len(np.array(cross_cm))
                plot_confusion_matrix(
                    np.array(cross_cm), class_names[:n_classes_cross],
                    model_name="random_forest_cross_aoi", aoi=aoi,
                )

    # Ablation chart (global, not per-AOI)
    plot_ablation_bar_chart(ablation_results)

    # --- Step 5: Apply ML predictions spatially via sampleRectangle ---
    rf_model = models.get("random_forest")
    if rf_model:
        from coffee_deforestation.ml.predict import predict_from_gee

        raster_dir = PROJECT_ROOT / "outputs" / "rasters"
        for aoi_name, aoi in aois_config.items():
            out = results.get(aoi_name, {})
            feature_stack = out.get("_feature_stack")
            aoi_cfg = out.get("_aoi_config")
            if feature_stack and aoi_cfg:
                print(f"\nApplying RF model to {aoi_name} via sampleRectangle...")
                try:
                    pred_result = predict_from_gee(
                        rf_model, feature_stack, aoi_cfg, raster_dir, scale=300
                    )
                    if pred_result:
                        results[aoi_name]["classification"] = str(pred_result[0])
                        results[aoi_name]["probability"] = str(pred_result[1])
                        print(f"  Classification saved: {pred_result[0]}")
                        # Generate classification map figure
                        try:
                            import rasterio as rio

                            from coffee_deforestation.viz.static import (
                                plot_classification_map,
                            )

                            with rio.open(pred_result[0]) as src:
                                class_data = src.read(1)
                            plot_classification_map(class_data, aoi_cfg)
                            print(f"  Classification figure saved for {aoi_name}")
                        except Exception as fig_err:
                            print(f"  Classification figure failed: {fig_err}")
                    else:
                        print(f"  Prediction returned None for {aoi_name}")
                except Exception as e:
                    print(f"  Prediction failed for {aoi_name}: {e}")

    _print_results(results)
    print("\nML training complete. Models saved to outputs/cache/models/")

    # --- Validate Stage 2 success criteria ---
    _validate_stage2(ablation_results, cross_results, aoi_data, aois_config)


def _print_results(results: dict) -> None:
    print(f"\n{'='*60}")
    print("All AOIs complete.")
    for name, output in results.items():
        status = "ERROR" if "error" in output else "OK"
        print(f"  {name}: {status}")


def _validate_stage2(
    ablation_results: dict,
    cross_results: dict,
    aoi_data: dict,
    aois_config: dict,
) -> None:
    """Log Stage 2 success criteria checks."""
    from loguru import logger

    # Criterion 2: cross-AOI F1 >= 0.75
    for holdout, model_res in cross_results.items():
        rf_f1 = model_res.get("random_forest", {}).get("f1_coffee", 0.0)
        status = "PASS" if rf_f1 >= 0.75 else "WARN (below 0.75)"
        logger.info(f"Cross-AOI F1 [{holdout}]: {rf_f1:.3f} — {status}")

    # Criterion 3: S1+S2 > max(S1-only, S2-only)
    s1s2_f1 = ablation_results.get("s1_s2", {}).get("f1_coffee", 0)
    s2_f1 = ablation_results.get("s2_only", {}).get("f1_coffee", 0)
    s1_f1 = ablation_results.get("s1_only", {}).get("f1_coffee", 0)
    fusion_wins = s1s2_f1 > max(s2_f1, s1_f1)
    logger.info(
        f"Ablation — S1+S2={s1s2_f1:.3f} > max(S2={s2_f1:.3f}, S1={s1_f1:.3f}): "
        f"{'PASS' if fusion_wins else 'FAIL (log as surprising finding)'}"
    )

    # Criterion 4: Brazil negative control (<5% of Vietnam's area)
    # This is checked visually via the output stats; log the reminder
    if "sul_de_minas" in aois_config and "lam_dong" in aois_config:
        logger.info(
            "Negative control (sul_de_minas vs lam_dong) — "
            "check outputs/stats/ for area comparison"
        )


if __name__ == "__main__":
    app()

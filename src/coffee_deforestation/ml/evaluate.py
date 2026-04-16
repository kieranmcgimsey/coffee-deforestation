"""Model evaluation: per-AOI, cross-AOI holdout, and S1/S2 ablation.

What: Evaluates trained classifiers with confusion matrices, F1 scores, and
ablation studies comparing S1-only, S2-only, and S1+S2 feature sets.
Why: Rigorous evaluation demonstrates model quality and the value of multi-sensor fusion.
Assumes: Trained models and labeled test data are available.
Produces: Evaluation metrics dict, confusion matrices, ablation results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from coffee_deforestation.ml.labels import CLASS_MAP

# Feature indices for ablation
# Based on FEATURE_BAND_ORDER from features/stack.py (19 bands):
# 0-4: spectral indices (S2), 5-8: S2 bands, 9-11: SAR (S1), 12-15: contextual,
# 16: ndvi_delta (temporal, S2-derived), 17-18: vv_stddev/vh_stddev (temporal, S1-derived)
S2_FEATURE_INDICES = list(range(0, 9))     # spectral indices + S2 bands
S1_FEATURE_INDICES = list(range(9, 12))    # SAR features
CONTEXTUAL_INDICES = list(range(12, 16))   # contextual
TEMPORAL_S2_INDICES = [16]                 # ndvi_delta
TEMPORAL_S1_INDICES = [17, 18]             # vv_stddev, vh_stddev

S2_ONLY_INDICES = S2_FEATURE_INDICES + CONTEXTUAL_INDICES + TEMPORAL_S2_INDICES
S1_ONLY_INDICES = S1_FEATURE_INDICES + CONTEXTUAL_INDICES + TEMPORAL_S1_INDICES
ALL_INDICES = list(range(19))


CLASS_NAMES = list(CLASS_MAP.keys())


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Evaluate a model on test data.

    Returns a dict with accuracy, per-class metrics, confusion matrix, and
    coffee-specific F1/precision/recall.
    """
    y_pred = model.predict(X_test)  # type: ignore[union-attr]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES[:len(np.unique(y_test))],
        output_dict=True,
        zero_division=0,
    )

    # Coffee-specific metrics (class 0)
    coffee_class = CLASS_MAP["coffee"]
    f1_coffee = f1_score(y_test, y_pred, labels=[coffee_class], average="macro", zero_division=0)
    prec_coffee = precision_score(y_test, y_pred, labels=[coffee_class], average="macro", zero_division=0)
    rec_coffee = recall_score(y_test, y_pred, labels=[coffee_class], average="macro", zero_division=0)

    logger.info(
        f"{model_name}: accuracy={acc:.3f}, coffee F1={f1_coffee:.3f}, "
        f"precision={prec_coffee:.3f}, recall={rec_coffee:.3f}"
    )

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "f1_coffee": float(f1_coffee),
        "precision_coffee": float(prec_coffee),
        "recall_coffee": float(rec_coffee),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def run_ablation(
    model_class: type,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """Run S1-only / S2-only / S1+S2 ablation study.

    Trains separate models on each feature subset and evaluates.
    Returns a dict mapping subset name to evaluation metrics.
    """
    subsets = {
        "s2_only": S2_ONLY_INDICES,
        "s1_only": S1_ONLY_INDICES,
        "s1_s2": ALL_INDICES,
    }

    n_features = X_train.shape[1]

    results = {}
    for name, indices in subsets.items():
        # Clip indices to actual feature count (handles 16-band legacy data)
        valid_indices = [i for i in indices if i < n_features]
        logger.info(f"Ablation: training {name} ({len(valid_indices)} features)")
        model = model_class(**model_kwargs)
        model.fit(X_train[:, valid_indices], y_train)
        metrics = evaluate_model(model, X_test[:, valid_indices], y_test, model_name=name)
        results[name] = metrics

    # Log comparison
    logger.info("Ablation results:")
    for name, metrics in results.items():
        logger.info(f"  {name}: F1={metrics['f1_coffee']:.3f}")

    s1s2_f1 = results["s1_s2"]["f1_coffee"]
    s2_f1 = results["s2_only"]["f1_coffee"]
    s1_f1 = results["s1_only"]["f1_coffee"]
    if s1s2_f1 > max(s2_f1, s1_f1):
        logger.info("S1+S2 outperforms both single modalities — multi-sensor fusion confirmed")
    else:
        logger.warning(
            "S1+S2 does NOT outperform single modalities — "
            "logging as surprising finding"
        )

    return results


def run_cross_aoi_evaluation(
    models: dict[str, object],
    aoi_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, dict]]:
    """Run cross-AOI holdout evaluation for all models.

    For each AOI, trains on the other AOIs and tests on the held-out one.
    Returns: dict[holdout_aoi][model_name] = metrics
    """
    from coffee_deforestation.ml.train import prepare_cross_aoi_holdout

    results: dict[str, dict[str, dict]] = {}

    for holdout_aoi in aoi_data:
        logger.info(f"Cross-AOI holdout: testing on {holdout_aoi}")
        X_train, y_train, X_test, y_test = prepare_cross_aoi_holdout(
            aoi_data, holdout_aoi
        )

        results[holdout_aoi] = {}
        for model_name, model in models.items():
            # Retrain on combined non-holdout data
            model.fit(X_train, y_train)  # type: ignore[union-attr]
            metrics = evaluate_model(
                model, X_test, y_test,
                model_name=f"{model_name}_holdout_{holdout_aoi}",
            )
            results[holdout_aoi][model_name] = metrics

    return results

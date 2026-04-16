"""Model training: Random Forest + XGBoost.

What: Trains RF and XGBoost classifiers on the labeled feature stack using
scikit-learn interfaces. Supports per-AOI and cross-AOI holdout training.
Why: Two model families provide comparison and ensemble potential. Both are
interpretable and fast on CPU.
Assumes: Labeled training data (X, y) is available as numpy arrays.
Produces: Trained model objects (picklable) and training metadata.
"""

from __future__ import annotations

from typing import Any

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from coffee_deforestation.config import PipelineConfig


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    rf_config = config.ml.random_forest
    logger.info(
        f"Training Random Forest: n_estimators={rf_config.n_estimators}, "
        f"max_depth={rf_config.max_depth}"
    )

    model = RandomForestClassifier(
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        min_samples_leaf=rf_config.min_samples_leaf,
        random_state=rf_config.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    logger.info(f"RF training complete. OOB score not available (oob_score not enabled)")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    xgb_config = config.ml.xgboost
    n_classes = len(np.unique(y_train))
    logger.info(
        f"Training XGBoost: n_estimators={xgb_config.n_estimators}, "
        f"max_depth={xgb_config.max_depth}, n_classes={n_classes}"
    )

    model = XGBClassifier(
        n_estimators=xgb_config.n_estimators,
        max_depth=xgb_config.max_depth,
        learning_rate=xgb_config.learning_rate,
        random_state=xgb_config.random_state,
        n_jobs=-1,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    logger.info("XGBoost training complete")
    return model


def train_both(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> dict[str, RandomForestClassifier | XGBClassifier]:
    """Train both RF and XGBoost. Returns dict keyed by model name."""
    return {
        "random_forest": train_random_forest(X_train, y_train, config),
        "xgboost": train_xgboost(X_train, y_train, config),
    }


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_cross_aoi_holdout(
    aoi_data: dict[str, tuple[np.ndarray, np.ndarray]],
    holdout_aoi: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare cross-AOI holdout: train on all AOIs except holdout, test on holdout.

    aoi_data: dict mapping aoi_id to (X, y) tuples.
    holdout_aoi: the AOI to hold out for testing.

    Returns: (X_train, y_train, X_test, y_test)
    """
    train_X_parts = []
    train_y_parts = []

    for aoi_id, (X, y) in aoi_data.items():
        if aoi_id == holdout_aoi:
            X_test, y_test = X, y
        else:
            train_X_parts.append(X)
            train_y_parts.append(y)

    X_train = np.concatenate(train_X_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)

    logger.info(
        f"Cross-AOI holdout: train={X_train.shape[0]} samples from "
        f"{len(train_X_parts)} AOIs, test={X_test.shape[0]} from {holdout_aoi}"
    )

    return X_train, y_train, X_test, y_test


def save_model(model: Any, path: Path) -> Path:
    """Save a trained model to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {path}")
    return path


def load_model(path: Path) -> object:
    """Load a trained model from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)

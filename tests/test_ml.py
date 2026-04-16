"""Tests for ML training, prediction, and evaluation."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from coffee_deforestation.config import PipelineConfig, load_pipeline_config
from coffee_deforestation.ml.evaluate import (
    ALL_INDICES,
    S1_FEATURE_INDICES,
    S1_ONLY_INDICES,
    S2_ONLY_INDICES,
    evaluate_model,
    run_ablation,
)
from coffee_deforestation.ml.explain import get_feature_importance, plot_feature_importance
from coffee_deforestation.ml.predict import predict_array
from coffee_deforestation.ml.train import (
    load_model,
    prepare_cross_aoi_holdout,
    save_model,
    split_data,
    train_both,
    train_random_forest,
    train_xgboost,
)


@pytest.fixture
def synthetic_data():
    """Synthetic training data with 5 classes, 16 features."""
    rng = np.random.default_rng(42)
    n_per_class = 200
    n_features = 16
    X_parts = []
    y_parts = []
    for cls in range(5):
        # Each class has a slightly different mean
        X_cls = rng.standard_normal((n_per_class, n_features)) + cls * 0.5
        X_parts.append(X_cls)
        y_parts.append(np.full(n_per_class, cls))
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int32)
    return X, y


@pytest.fixture
def config():
    return load_pipeline_config()


def test_split_data(synthetic_data):
    """Stratified split produces correct proportions."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200


def test_train_random_forest(synthetic_data, config):
    """RF trains and predicts."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, config)
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_train_xgboost(synthetic_data, config):
    """XGBoost trains and predicts."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_xgboost(X_train, y_train, config)
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape


def test_train_both(synthetic_data, config):
    """Both models train successfully."""
    X, y = synthetic_data
    X_train, _, y_train, _ = split_data(X, y)
    models = train_both(X_train, y_train, config)
    assert "random_forest" in models
    assert "xgboost" in models


def test_evaluate_model(synthetic_data, config):
    """Evaluation produces expected metrics."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test, model_name="test_rf")

    assert "accuracy" in metrics
    assert "f1_coffee" in metrics
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_coffee"] <= 1


def test_predict_array(synthetic_data, config):
    """predict_array returns predictions and probabilities."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, config)

    preds, probs = predict_array(model, X_test)
    assert preds.shape == (X_test.shape[0],)
    assert probs is not None
    assert probs.shape == (X_test.shape[0], 5)


def test_feature_importance(synthetic_data, config):
    """Feature importance extraction works."""
    X, y = synthetic_data
    X_train, _, y_train, _ = split_data(X, y)
    model = train_random_forest(X_train, y_train, config)
    importances = get_feature_importance(model)
    assert len(importances) == 16
    assert all(v >= 0 for v in importances.values())


def test_plot_feature_importance(synthetic_data, config, sample_aoi, tmp_path):
    """Feature importance plot generates."""
    X, y = synthetic_data
    model = train_random_forest(X, y, config)
    importances = get_feature_importance(model)
    path = plot_feature_importance(
        importances, "test_rf", aoi=sample_aoi,
        output_path=str(tmp_path / "importance.png"),
    )
    assert Path(path).exists()


def test_save_load_model(synthetic_data, config, tmp_path):
    """Model saves and loads correctly."""
    X, y = synthetic_data
    model = train_random_forest(X, y, config)
    path = save_model(model, tmp_path / "model.pkl")
    loaded = load_model(path)
    np.testing.assert_array_equal(model.predict(X[:10]), loaded.predict(X[:10]))


def test_ablation(synthetic_data):
    """Ablation study runs for all subsets."""
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = run_ablation(
        RandomForestClassifier,
        {"n_estimators": 10, "random_state": 42},
        X_train, y_train, X_test, y_test,
    )
    assert "s1_only" in results
    assert "s2_only" in results
    assert "s1_s2" in results


def test_cross_aoi_holdout(synthetic_data):
    """Cross-AOI holdout splits correctly."""
    X, y = synthetic_data
    half = len(X) // 2
    aoi_data = {
        "aoi_a": (X[:half], y[:half]),
        "aoi_b": (X[half:], y[half:]),
    }
    X_train, y_train, X_test, y_test = prepare_cross_aoi_holdout(aoi_data, "aoi_b")
    assert X_train.shape[0] == half
    assert X_test.shape[0] == len(X) - half


def test_feature_indices_coverage():
    """Feature subset indices cover all 19 features (16 base + 3 temporal)."""
    assert len(ALL_INDICES) == 19
    assert len(S2_ONLY_INDICES) > 0
    assert len(S1_ONLY_INDICES) > 0
    # S1+S2 includes everything
    assert set(ALL_INDICES) == set(range(19))

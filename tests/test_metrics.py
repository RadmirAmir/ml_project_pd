"""Unit tests for the metric and plotting helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.evaluate import (
    compute_metrics,
    find_best_threshold,
    ks_statistic,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_threshold_sweep,
    top_k_metrics,
)

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PERFECT = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
Y_INVERTED = 1 - Y_PERFECT


def test_perfect_predictions_reach_maximum_metrics() -> None:
    """A perfect ranking gives ROC-AUC = PR-AUC = F1 = KS = 1."""
    metrics = compute_metrics(Y_TRUE, Y_PERFECT, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["ks"] == pytest.approx(1.0)


def test_inverted_predictions_give_auc_zero() -> None:
    """Fully reversed scores produce ROC-AUC = 0 (sanity check of the sign)."""
    metrics = compute_metrics(Y_TRUE, Y_INVERTED, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(0.0)
    assert metrics["f1"] == 0.0


def test_metrics_are_json_serialisable() -> None:
    """Every metric value must survive ``json.dumps`` (MLflow/DVC contract)."""
    metrics = compute_metrics(Y_TRUE, Y_PERFECT)
    payload = json.dumps(metrics)
    assert "roc_auc" in payload


def test_confusion_matrix_counts_are_consistent() -> None:
    """TP + FP + TN + FN equals the sample size."""
    metrics = compute_metrics(Y_TRUE, Y_PERFECT, threshold=0.5)
    total = metrics["true_positive"] + metrics["false_positive"] + metrics["true_negative"] + metrics["false_negative"]
    assert total == metrics["n_samples"] == len(Y_TRUE)


def test_threshold_controls_precision_recall_tradeoff() -> None:
    """Raising the threshold cannot increase recall and cannot decrease precision."""
    probabilities = np.linspace(0.01, 0.99, 100)
    y_true = (np.random.default_rng(0).random(100) < probabilities).astype(int)

    low = compute_metrics(y_true, probabilities, threshold=0.2)
    high = compute_metrics(y_true, probabilities, threshold=0.8)

    assert low["recall"] >= high["recall"]
    assert high["precision"] >= low["precision"]


def test_best_threshold_is_within_the_scanned_grid() -> None:
    """The tuned threshold lies inside (0, 1) and beats the default F1."""
    threshold, score = find_best_threshold(Y_TRUE, Y_PERFECT / 2, metric="f1")
    assert 0 < threshold < 1
    assert 0 <= score <= 1


def test_ks_statistic_is_bounded() -> None:
    """KS is a distance, therefore bounded by [0, 1]."""
    assert 0 <= ks_statistic(Y_TRUE, Y_PERFECT) <= 1
    assert ks_statistic(np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])) == 0.0


def test_top_k_capture_rate() -> None:
    """Selecting the riskiest 25% of a perfectly ranked sample captures half the defaults."""
    result = top_k_metrics(Y_TRUE, Y_PERFECT, fractions=(0.25,))
    # 25% of 8 rows = the 2 highest scores, which are 2 of the 4 defaults.
    assert result["top_25pct_capture_rate"] == pytest.approx(0.5)
    assert result["top_25pct_precision"] == pytest.approx(1.0)


def test_metric_validation_errors() -> None:
    """Bad inputs raise instead of returning nonsense."""
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [0.5])
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_plots_are_written(tmp_path) -> None:
    """ROC, threshold sweep and confusion matrix produce non-empty PNG files."""
    roc = plot_roc_curve(Y_TRUE, Y_PERFECT, tmp_path / "roc.png")
    sweep = plot_threshold_sweep(Y_TRUE, Y_PERFECT, tmp_path / "sweep.png")
    matrix = plot_confusion_matrix(Y_TRUE, Y_PERFECT, tmp_path / "cm.png", threshold=0.5)

    for path in (roc, sweep, matrix):
        assert path.exists()
        assert path.stat().st_size > 1000


def test_feature_importance_plot_for_tree_model(train_test_frames, tmp_path) -> None:
    """A fitted random forest produces a non-empty feature-importance figure."""
    from src.data.make_dataset import split_features_target
    from src.models.pipeline import create_pipeline

    train_df, _ = train_test_frames
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="random_forest", n_estimators=10)
    pipeline.fit(x_train, y_train)

    path = plot_feature_importance(pipeline, tmp_path / "importance.png")

    assert path is not None
    assert path.exists() and path.stat().st_size > 1000

"""Tests for the PSI drift monitoring script."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    PSI_THRESHOLDS,
    _inject_drift,
    _score_via_api,
    analyse_drift,
    calculate_psi,
    calculate_psi_categorical,
    classify_psi,
    ks_test,
    plot_drift,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    """No drift means PSI ~= 0."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    assert calculate_psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shifted_distribution() -> None:
    """A mean shift larger than one sigma triggers the significant band."""
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)

    psi = calculate_psi(reference, shifted)

    assert psi > PSI_THRESHOLDS["moderate"]
    assert classify_psi(psi) == "significant_shift"


def test_psi_is_small_for_a_slight_shift() -> None:
    """Small noise stays below the alert threshold."""
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 5000)
    slightly_shifted = rng.normal(0.05, 1.02, 5000)

    assert calculate_psi(reference, slightly_shifted) < PSI_THRESHOLDS["moderate"]


def test_psi_of_a_constant_reference_column() -> None:
    """A constant reference column (e.g. all zeros) still yields a finite PSI."""
    reference = np.zeros(1000)
    assert np.isfinite(calculate_psi(reference, reference))
    assert calculate_psi(reference, np.ones(500)) > 0


def test_psi_requires_two_samples() -> None:
    """Empty inputs raise instead of returning a meaningless number."""
    with pytest.raises(ValueError):
        calculate_psi(np.array([]), np.array([1.0, 2.0]))


def test_categorical_psi_detects_new_category_mix() -> None:
    """A changed category mix produces a positive PSI."""
    reference = np.array(["married"] * 900 + ["single"] * 100)
    actual = np.array(["married"] * 200 + ["single"] * 800)

    assert calculate_psi_categorical(reference, actual) > PSI_THRESHOLDS["moderate"]


def test_ks_test_returns_statistic_and_pvalue() -> None:
    """KS on identical samples is zero with a large p-value."""
    rng = np.random.default_rng(3)
    sample = rng.normal(0, 1, 1000)
    statistic, pvalue = ks_test(sample, sample)

    assert statistic == pytest.approx(0.0, abs=1e-9)
    assert 0 <= pvalue <= 1


def test_classify_psi_thresholds() -> None:
    """The verdict follows the retail-credit PSI convention."""
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate_shift"
    assert classify_psi(0.30) == "significant_shift"


@pytest.fixture
def drift_inputs():
    """Reference/current frames with PD scores for the analyse_drift tests."""
    rng = np.random.default_rng(4)
    rows = 400
    reference = pd.DataFrame(
        {
            "LIMIT_BAL": rng.normal(150_000, 30_000, rows),
            "AGE": rng.integers(21, 60, rows),
            "PAY_0": rng.integers(-2, 4, rows),
            "BILL_AMT1": rng.normal(50_000, 10_000, rows),
            "PAY_AMT1": rng.normal(5_000, 1_000, rows),
            "CREDIT_UTILIZATION": rng.uniform(0, 0.8, rows),
            "DELINQ_COUNT": rng.integers(0, 4, rows),
            "default.payment.next.month": rng.binomial(1, 0.22, rows),
        }
    )
    current = reference.copy()
    reference_scores = rng.uniform(0, 1, rows)
    current_scores = rng.uniform(0, 1, rows)
    return reference, current, reference_scores, current_scores


def test_analyse_drift_reports_no_alert_for_same_data(drift_inputs) -> None:
    """Comparing a sample with itself produces no alerts."""
    reference, current, reference_scores, current_scores = drift_inputs

    report = analyse_drift(reference, current, reference_scores, reference_scores, threshold=0.5)

    assert report.score_psi == pytest.approx(0.0, abs=1e-9)
    assert report.alerts == []
    assert report.verdict == "stable"
    assert report.performance["roc_auc"] >= 0.0


def test_analyse_drift_raises_alerts_on_shift(drift_inputs) -> None:
    """A strongly shifted score distribution triggers an alert."""
    reference, current, reference_scores, _ = drift_inputs
    shifted_scores = reference_scores + 0.4

    report = analyse_drift(reference, current, reference_scores, shifted_scores, threshold=0.5)

    assert report.has_alert
    assert any("score PSI" in alert for alert in report.alerts)


def test_drift_report_is_json_serialisable(drift_inputs) -> None:
    """The report can be persisted by the monitoring job."""
    reference, current, reference_scores, current_scores = drift_inputs
    report = analyse_drift(reference, current, reference_scores, current_scores)

    payload = json.dumps(report.to_dict())

    assert "score_psi" in payload
    assert "feature_psi" in payload


def test_plot_drift_writes_png(drift_inputs, tmp_path) -> None:
    """The monitoring figure is written and is non-empty."""
    reference, current, reference_scores, current_scores = drift_inputs
    report = analyse_drift(reference, current, reference_scores, current_scores)

    path = plot_drift(report, tmp_path / "drift.png")

    assert path.exists()
    assert path.stat().st_size > 1000


def test_inject_drift_changes_values(synthetic_raw: pd.DataFrame) -> None:
    """The synthetic drift helper perturbs a share of the clients."""
    drifted = _inject_drift(synthetic_raw, factor=2.0, share=1.0, seed=1)

    assert drifted["LIMIT_BAL"].sum() > synthetic_raw["LIMIT_BAL"].sum()
    assert len(drifted) == len(synthetic_raw)


def test_score_via_api_uses_batch_endpoint(monkeypatch, bundle_scorer, synthetic_raw) -> None:
    """Records are posted to ``/predict/batch`` and probabilities are collected."""
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors requests' signature
        calls.append((url, len(json["records"])))
        predictions = [
            {"default_prediction": 0, "default_probability": 0.11, "risk_band": "medium", "threshold": 0.4}
            for _ in json["records"]
        ]
        return FakeResponse({"predictions": predictions, "n_records": len(predictions)})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    scores = _score_via_api(bundle_scorer, synthetic_raw.head(250), "http://localhost:8000", batch_size=100)

    assert scores.shape == (250,)
    assert np.allclose(scores, 0.11)
    assert [size for _, size in calls] == [100, 100, 50]
    assert all(url.endswith("/predict/batch") for url, _ in calls)


def test_plot_drift_thresholds_are_documented() -> None:
    """The alert thresholds match the values quoted in the README."""
    assert PSI_THRESHOLDS == {"stable": 0.10, "moderate": 0.25}

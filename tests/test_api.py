"""Tests for the FastAPI scoring service."""

from __future__ import annotations

import joblib
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.data.make_dataset import split_features_target
from src.features.build_features import get_feature_lists
from src.models.pipeline import create_pipeline


@pytest.fixture(scope="module")
def model_bundle_path(tmp_path_factory, featurised_frame) -> str:
    """Fit a tiny model and persist a bundle in the pytest tmp directory."""
    directory = tmp_path_factory.mktemp("model")
    path = directory / "credit_default_model.joblib"

    from src.data.make_dataset import split_data

    train_df, _ = split_data(featurised_frame, test_size=0.25, random_state=42)
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    numeric_features, categorical_features = get_feature_lists()
    joblib.dump(
        {
            "model": pipeline,
            "model_name": "logistic_regression",
            "threshold": 0.5,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": numeric_features + categorical_features,
            "trained_at": "2026-01-01T00:00:00+00:00",
            "dataset_hash": "test",
            "sklearn_version": "test",
            "test_metrics": {"roc_auc": 0.7},
        },
        path,
    )
    return str(path)


@pytest.fixture
def client(monkeypatch, model_bundle_path: str) -> TestClient:
    """TestClient with the model loaded through the lifespan hook."""
    monkeypatch.setenv("MODEL_PATH", model_bundle_path)
    from src.api.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_banner(client: TestClient) -> None:
    """The root endpoint confirms the service is alive."""
    response = client.get("/")
    assert response.status_code == 200
    assert "alive" in response.json()["message"]


def test_health_reports_loaded_model(client: TestClient) -> None:
    """``/health`` exposes the model state for orchestrators."""
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["threshold"] == 0.5


def test_model_info_exposes_feature_schema(client: TestClient) -> None:
    """``/model/info`` returns the schema and metrics stored in the bundle."""
    payload = client.get("/model/info").json()

    assert payload["model_name"] == "logistic_regression"
    assert payload["features"]["numeric"]
    assert payload["features"]["categorical"]
    assert payload["test_metrics"]["roc_auc"] == 0.7


def test_predict_compact_payload(client: TestClient, numeric_payload: dict) -> None:
    """The eight-field payload from the assignment brief is accepted."""
    response = client.post("/predict", json=numeric_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_prediction"] in (0, 1)
    assert 0.0 <= payload["default_probability"] <= 1.0
    assert payload["risk_band"] in {"low", "medium", "high", "critical"}
    assert payload["threshold"] == 0.5
    assert payload["model_name"] == "logistic_regression"


def test_predict_full_payload(client: TestClient) -> None:
    """The full 23-column UCI record is scored as well."""
    record = {
        "LIMIT_BAL": 120_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 26,
        "PAY_0": -1,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 2,
        "BILL_AMT1": 2682,
        "BILL_AMT2": 1725,
        "BILL_AMT3": 2682,
        "BILL_AMT4": 3272,
        "BILL_AMT5": 3455,
        "BILL_AMT6": 3261,
        "PAY_AMT1": 0,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 0,
        "PAY_AMT6": 2000,
    }

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert 0.0 <= response.json()["default_probability"] <= 1.0


def test_predict_rejects_out_of_range_values(client: TestClient, numeric_payload: dict) -> None:
    """Pydantic validators reject impossible ages before the model is called."""
    broken = {**numeric_payload, "AGE": 150}

    response = client.post("/predict", json=broken)

    assert response.status_code == 422


def test_predict_requires_core_fields(client: TestClient) -> None:
    """A payload without the core fields is rejected with a 422."""
    response = client.post("/predict", json={"LIMIT_BAL": 20000})

    assert response.status_code == 422


def test_predict_batch(client: TestClient, numeric_payload: dict) -> None:
    """Batch scoring returns one prediction per record and aggregate stats."""
    payload = {"records": [numeric_payload, numeric_payload, {**numeric_payload, "PAY_0": 5}]}

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["n_records"] == 3
    assert len(body["predictions"]) == 3
    assert 0.0 <= body["mean_probability"] <= 1.0
    assert 0 <= body["n_flagged"] <= 3


def test_batch_rejects_empty_list(client: TestClient) -> None:
    """An empty batch is a client error, not an empty success."""
    response = client.post("/predict/batch", json={"records": []})

    assert response.status_code == 422


def test_service_returns_503_without_model(monkeypatch, tmp_path) -> None:
    """A missing artifact yields 503 with an actionable message."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    import src.api.app as api_module

    api_module.load_scorer(tmp_path / "missing.joblib")
    with pytest.raises(HTTPException) as error:
        api_module.get_scorer()

    assert error.value.status_code == 503
    assert "Train the model first" in error.value.detail


def test_threshold_is_applied_consistently(client: TestClient) -> None:
    """The returned label is exactly ``probability >= threshold`` for that payload."""
    payload = {
        "LIMIT_BAL": 20_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "BILL_AMT1": 3913,
        "PAY_AMT1": 0,
    }
    body = client.post("/predict", json=payload).json()

    expected = int(body["default_probability"] >= body["threshold"])
    assert body["default_prediction"] == expected
    assert body["threshold"] == pytest.approx(0.5)

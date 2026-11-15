"""Tests for the persisted model bundle and the scoring helpers."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import split_features_target
from src.features.build_features import get_feature_lists
from src.models.pipeline import create_pipeline
from src.models.predict import RISK_BANDS, CreditDefaultScorer, ModelNotFoundError, score_file


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, featurised_frame):
    """Fitted bundle written to disk (mirrors what the train stage produces)."""
    from src.data.make_dataset import split_data

    train_df, test_df = split_data(featurised_frame, test_size=0.25, random_state=42)
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    numeric_features, categorical_features = get_feature_lists()
    path = tmp_path_factory.mktemp("bundle") / "model.joblib"
    joblib.dump(
        {
            "model": pipeline,
            "model_name": "logistic_regression",
            "threshold": 0.4,
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
    return path, train_df, test_df


def test_load_reads_bundle_metadata(bundle) -> None:
    """Loading restores the model, threshold and metadata."""
    path, _, _ = bundle
    scorer = CreditDefaultScorer.load(path)

    assert scorer.threshold == pytest.approx(0.4)
    assert scorer.metadata["model_name"] == "logistic_regression"
    assert scorer.metadata["test_metrics"]["roc_auc"] == 0.7


def test_load_missing_bundle_raises(tmp_path) -> None:
    """A missing artifact produces an explicit, actionable error."""
    with pytest.raises(ModelNotFoundError, match="Run the `train` stage"):
        CreditDefaultScorer.load(tmp_path / "nope.joblib")


def test_predict_proba_on_featurised_frame(bundle) -> None:
    """A featurised frame is scored without recomputing the features."""
    path, _, test_df = bundle
    scorer = CreditDefaultScorer.load(path)
    x_test, _ = split_features_target(test_df)

    probabilities = scorer.predict_proba(x_test)

    assert probabilities.shape == (len(x_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_predict_proba_on_raw_frame(bundle) -> None:
    """A raw frame is featurised on the fly (no train/serve skew)."""
    path, _, _ = bundle
    scorer = CreditDefaultScorer.load(path)
    raw = pd.DataFrame(
        [
            {
                "LIMIT_BAL": 20_000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 0,
                "BILL_AMT5": 0,
                "BILL_AMT6": 0,
                "PAY_AMT1": 0,
                "PAY_AMT2": 689,
                "PAY_AMT3": 0,
                "PAY_AMT4": 0,
                "PAY_AMT5": 0,
                "PAY_AMT6": 0,
            }
        ]
    )

    probabilities = scorer.predict_proba(raw)

    assert probabilities.shape == (1,)
    assert 0 <= probabilities[0] <= 1


def test_predict_proba_rejects_incomplete_raw_frame(bundle) -> None:
    """A raw frame missing mandatory columns is rejected with a clear message."""
    path, _, _ = bundle
    scorer = CreditDefaultScorer.load(path)

    with pytest.raises(ValueError, match="missing raw columns"):
        scorer.predict_proba(pd.DataFrame([{"AGE": 30}]))


def test_score_records_returns_business_fields(bundle, numeric_payload: dict) -> None:
    """Single-record scoring returns class, PD, band and threshold."""
    path, _, _ = bundle
    scorer = CreditDefaultScorer.load(path)

    result = scorer.score_records([numeric_payload])[0]

    assert set(result) == {"default_prediction", "default_probability", "risk_band", "threshold"}
    assert result["default_prediction"] in (0, 1)
    assert result["threshold"] == pytest.approx(0.4)
    assert result["risk_band"] in {"low", "medium", "high", "critical"}


def test_score_records_empty_input(bundle) -> None:
    """An empty list is an empty result, not an exception."""
    path, _, _ = bundle
    assert CreditDefaultScorer.load(path).score_records([]) == []


def test_score_frame_adds_prediction_columns(bundle) -> None:
    """Batch scoring appends probability, label and risk band."""
    path, _, test_df = bundle
    scorer = CreditDefaultScorer.load(path)
    x_test, _ = split_features_target(test_df)

    scored = scorer.score_frame(x_test.head(5))

    assert {"default_probability", "default_prediction", "risk_band"}.issubset(scored.columns)
    assert len(scored) == 5


@pytest.mark.parametrize(
    "probability,expected",
    [(0.05, "low"), (0.20, "medium"), (0.40, "high"), (0.75, "critical")],
)
def test_risk_band_boundaries(bundle, probability: float, expected: str) -> None:
    """Risk bands follow the documented PD cut-offs."""
    path, _, _ = bundle
    scorer = CreditDefaultScorer.load(path)
    assert scorer.risk_band(probability) == expected


def test_risk_bands_are_ordered() -> None:
    """The band table is sorted and covers the whole [0, 1] interval."""
    assert RISK_BANDS[0][0] == 0.0
    assert RISK_BANDS[-1][1] > 1.0
    for (_, upper, _), (lower, _, _) in zip(RISK_BANDS, RISK_BANDS[1:]):
        assert upper == lower


def test_score_csv_file_roundtrip(bundle, tmp_path) -> None:
    """The CLI helper writes one row per client with the prediction columns."""
    path, _, test_df = bundle
    raw_like = test_df.head(10).copy()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    raw_like.to_csv(input_path, index=False)

    written = score_file(input_path, output_path, path)

    assert written == 10
    scored = pd.read_csv(output_path)
    assert {"default_probability", "default_prediction", "risk_band"}.issubset(scored.columns)

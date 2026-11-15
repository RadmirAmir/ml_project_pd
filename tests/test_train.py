"""End-to-end tests for the training stage (dataset -> metrics -> artifact).

These tests use a small synthetic dataset written to a temporary directory, so
they exercise the full ``prepare -> train`` path without touching the real
2.8 MB CSV and without polluting the MLflow store. The artifact paths
(``FIGURES_DIR`` / ``MODEL_CARD_FILE``) are redirected into the pytest temp
directory, so running the tests never modifies the DVC-tracked outputs in
``reports/`` and ``models/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from src.data.make_dataset import prepare_dataset, split_features_target
from src.models.train import fit_and_evaluate

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def isolate_repo_artifacts(monkeypatch, tmp_path):
    """Point every training-stage artifact at the pytest temp directory."""
    import src.models.train as train_module

    monkeypatch.setattr(train_module, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(train_module, "MODEL_CARD_FILE", tmp_path / "model_card.json")
    return tmp_path


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory, synthetic_raw):
    """Write a synthetic dataset and run the prepare stage on it."""
    directory = tmp_path_factory.mktemp("prepared")
    raw_path = directory / "UCI_Credit_Card.csv"
    synthetic_raw.to_csv(raw_path, index=False)

    processed_dir = directory / "processed"
    train_df, test_df = prepare_dataset(
        input_path=raw_path,
        output_dir=processed_dir,
        test_size=0.25,
        random_state=42,
    )
    return {
        "train": processed_dir / "train.csv",
        "test": processed_dir / "test.csv",
        "metadata": processed_dir / "feature_metadata.json",
        "train_df": train_df,
        "test_df": test_df,
    }


def test_prepare_dataset_writes_artifacts(prepared_dataset) -> None:
    """The prepare stage produces both splits and the feature metadata."""
    for key in ("train", "test", "metadata"):
        assert Path(prepared_dataset[key]).exists()

    metadata = json.loads(Path(prepared_dataset["metadata"]).read_text(encoding="utf-8"))
    assert metadata["train_rows"] > metadata["test_rows"] > 0
    assert metadata["target"] == "default.payment.next.month"
    assert metadata["numeric_features"]
    assert metadata["categorical_features"]
    assert 0.1 <= metadata["default_rate_train"] <= 0.4


def test_train_produces_a_loadable_bundle(prepared_dataset, tmp_path) -> None:
    """A quick fit writes a bundle with metrics and a usable pipeline."""
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    result = fit_and_evaluate(
        model_name="logistic_regression",
        search_method="none",
        log_to_mlflow=False,
        save_artifacts=True,
        output_model_path=model_path,
        metrics_path=metrics_path,
        train_path=prepared_dataset["train"],
        test_path=prepared_dataset["test"],
    )

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["model_name"] == "logistic_regression"

    bundle = joblib.load(model_path)
    x_test, _ = split_features_target(prepared_dataset["test_df"])
    probabilities = bundle["model"].predict_proba(x_test)[:, 1]
    assert len(probabilities) == len(x_test)
    assert result["test_metrics"]["roc_auc"] == pytest.approx(metrics["roc_auc"])


def test_train_result_contains_all_required_metrics(prepared_dataset, tmp_path) -> None:
    """ROC-AUC, Precision, Recall and F1 are reported on the hold-out split."""
    result = fit_and_evaluate(
        model_name="logistic_regression",
        search_method="none",
        log_to_mlflow=False,
        save_artifacts=False,
        train_path=prepared_dataset["train"],
        test_path=prepared_dataset["test"],
    )

    for metric in ("roc_auc", "precision", "recall", "f1", "ks", "brier_score"):
        assert metric in result["test_metrics"]
    assert result["test_metrics"]["n_samples"] == len(prepared_dataset["test_df"])


def test_model_card_is_written(prepared_dataset, tmp_path) -> None:
    """The model card documents the artifact for reviewers and auditors."""
    fit_and_evaluate(
        model_name="logistic_regression",
        search_method="none",
        log_to_mlflow=False,
        save_artifacts=True,
        output_model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        train_path=prepared_dataset["train"],
        test_path=prepared_dataset["test"],
    )

    card_path = tmp_path / "model_card.json"  # redirected by the autouse fixture
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["name"] == "credit_default_model"
    assert "hyperparameters" in card
    assert "environment" in card
    assert card["dataset_hash_sha256"]

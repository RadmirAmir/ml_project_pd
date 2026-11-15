"""Shared pytest fixtures.

The suite is intentionally fast: most tests use a small synthetic dataset with
the exact UCI schema, so they run in a couple of seconds and do not depend on
the 2.8 MB raw file being present. Tests that need the real data are skipped
automatically when ``data/raw/UCI_Credit_Card.csv`` is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BILL_COLUMNS,
    PAYMENT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)
from src.data.io import coerce_numeric_dtypes, load_raw_csv  # noqa: E402
from src.data.make_dataset import split_data  # noqa: E402
from src.features.build_features import build_features  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path of the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def has_raw_data() -> bool:
    """Whether the real UCI dataset is available on disk."""
    return RAW_DATA_FILE.exists()


@pytest.fixture(scope="session")
def raw_frame(has_raw_data: bool) -> pd.DataFrame:
    """A 500-row slice of the real dataset (session scoped for speed)."""
    if not has_raw_data:
        pytest.skip("raw dataset not available")
    return load_raw_csv(RAW_DATA_FILE).head(500).reset_index(drop=True)


def make_synthetic_raw(rows: int = 1200, seed: int = 7, default_rate: float = 0.22) -> pd.DataFrame:
    """Build a synthetic dataframe with the exact UCI schema and dtypes."""
    rng = np.random.default_rng(seed)
    limit_bal = rng.choice([20_000, 50_000, 120_000, 200_000, 500_000], size=rows)
    age = rng.integers(21, 60, size=rows)
    pay_status = rng.integers(-2, 6, size=(rows, len(PAY_STATUS_COLUMNS)))
    bills = rng.integers(0, 200_000, size=(rows, len(BILL_COLUMNS)))
    payments = rng.integers(0, 50_000, size=(rows, len(PAYMENT_COLUMNS)))

    # Make the target depend on the features so models can learn something.
    logit = (
        -1.5 + 0.35 * (bills[:, 0].mean() / 100_000) - 0.4 * (payments[:, 0].mean() / 50_000) + 0.25 * pay_status[:, 0]
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    probability = np.clip(probability * (default_rate / max(probability.mean(), 1e-6)), 0.01, 0.99)
    target = rng.binomial(1, probability)

    data = {
        "ID": np.arange(1, rows + 1),
        "LIMIT_BAL": limit_bal,
        "SEX": rng.choice([1, 2], size=rows),
        "EDUCATION": rng.choice([1, 2, 3, 4], size=rows),
        "MARRIAGE": rng.choice([1, 2, 3], size=rows),
        "AGE": age,
    }
    for index, column in enumerate(PAY_STATUS_COLUMNS):
        data[column] = pay_status[:, index]
    for index, column in enumerate(BILL_COLUMNS):
        data[column] = bills[:, index]
    for index, column in enumerate(PAYMENT_COLUMNS):
        data[column] = payments[:, index]
    data[TARGET_COLUMN] = target.astype(int)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """Session-scoped synthetic raw dataframe (1 200 rows)."""
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def featurised_frame(synthetic_raw: pd.DataFrame) -> pd.DataFrame:
    """Synthetic dataframe after feature engineering."""
    return build_features(synthetic_raw)


@pytest.fixture(scope="session")
def train_test_frames(featurised_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the synthetic featurised frame."""
    return split_data(featurised_frame, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def bundle_scorer(featurised_frame: pd.DataFrame, tmp_path_factory):
    """A fitted ``CreditDefaultScorer`` persisted in the session temp directory."""
    import joblib

    from src.data.make_dataset import split_data, split_features_target
    from src.features.build_features import get_feature_lists
    from src.models.pipeline import create_pipeline
    from src.models.predict import CreditDefaultScorer

    train_df, _ = split_data(featurised_frame, test_size=0.25, random_state=42)
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    numeric_features, categorical_features = get_feature_lists()
    path = tmp_path_factory.mktemp("scorer") / "model.joblib"
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
    return CreditDefaultScorer.load(path)


@pytest.fixture
def numeric_payload() -> dict:
    """A compact JSON payload compatible with ``POST /predict``."""
    return {
        "LIMIT_BAL": 20_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "BILL_AMT1": 3913,
        "PAY_AMT1": 0,
    }


@pytest.fixture(scope="session")
def coerced_frame() -> pd.DataFrame:
    """Frame with scientific-notation amounts as produced by the raw CSV."""
    frame = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": ["5e+05", "1e+05"],
            "SEX": [1, 2],
            "EDUCATION": [2, 2],
            "MARRIAGE": [1, 2],
            "AGE": [24, 26],
            "PAY_0": [2, -1],
            "PAY_2": [2, 2],
            "PAY_3": [-1, 0],
            "PAY_4": [-1, 0],
            "PAY_5": [-2, 0],
            "PAY_6": [-2, 2],
            "BILL_AMT1": ["1e+05", "2682"],
            "BILL_AMT2": ["3102", "1725"],
            "BILL_AMT3": ["689", "2682"],
            "BILL_AMT4": ["0", "3272"],
            "BILL_AMT5": ["0", "3455"],
            "BILL_AMT6": ["0", "3261"],
            "PAY_AMT1": ["0", "0"],
            "PAY_AMT2": ["689", "1000"],
            "PAY_AMT3": ["0", "1000"],
            "PAY_AMT4": ["0", "1000"],
            "PAY_AMT5": ["0", "0"],
            "PAY_AMT6": ["0", "2000"],
            TARGET_COLUMN: [1, 1],
        }
    )
    return coerce_numeric_dtypes(frame)

"""Feature engineering for the credit default (PD) model.

The module is intentionally free of I/O: every function takes a dataframe and
returns a new one. That makes the transformations easy to unit-test and lets
the exact same code run inside the training pipeline and inside the API.

Two public entry points are used by the rest of the project:

``clean_data``      – fixes the known dirty categories of the UCI dataset
                      (``EDUCATION``/``MARRIAGE`` codes ``0``, ``5``, ``6``),
                      clips the delinquency history and coerces dtypes.
``build_features``  – applies ``clean_data`` and adds the engineered columns
                      (credit utilisation, aggregated bill/payment history,
                      delinquency counters, age binning, ...).

Both are idempotent and defensive: ``build_features`` can be called twice on
the same frame without duplicating columns, which keeps batch inference safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BILL_COLUMNS,
    PAYMENT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    TARGET_COLUMN,
)

#: Engineered numeric columns produced by :func:`build_features`.
ENGINEERED_NUMERIC_FEATURES = [
    "CREDIT_UTILIZATION",
    "BILL_AMT_MEAN",
    "BILL_AMT_STD",
    "BILL_AMT_MAX",
    "BILL_AMT_MIN",
    "BILL_AMT_TREND",
    "BILL_AMT_CHANGE",
    "PAY_AMT_MEAN",
    "PAY_AMT_STD",
    "PAY_AMT_MAX",
    "PAY_AMT_SUM",
    "PAY_AMT_TREND",
    "PAYMENT_RATIO",
    "BILL_PAYMENT_GAP",
    "DELINQ_COUNT",
    "SEVERE_DELINQ_COUNT",
    "MAX_DELINQ",
    "DELINQ_TREND",
    "MONTHS_WITHOUT_PAYMENT",
    "LIMIT_PER_AGE",
]

#: Engineered categorical columns produced by :func:`build_features`.
ENGINEERED_CATEGORICAL_FEATURES = ["AGE_BIN", "LIMIT_BAL_BIN", "PAY_0_STATUS"]

#: Columns of the raw dataset that are treated as categorical by the model.
RAW_CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

#: Nominal numeric columns of the raw dataset.
RAW_NUMERIC_FEATURES = [col for col in RAW_FEATURE_COLUMNS if col not in RAW_CATEGORICAL_FEATURES]

AGE_BINS = [0, 25, 30, 35, 40, 45, 50, 55, 200]
AGE_LABELS = ["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55+"]

LIMIT_BAL_BINS = [-1, 50_000, 100_000, 200_000, 300_000, 500_000, 10_000_000]
LIMIT_BAL_LABELS = ["<=50k", "50-100k", "100-200k", "200-300k", "300-500k", ">500k"]

#: ``PAY_*`` codes are documented in the UCI dataset description:
#: -2 = no consumption, -1 = paid in full, 0 = revolving credit, 1..9 = months
#: of delay. Extreme delays (>8) are clipped because they are rare.
PAY_STATUS_CLIP = (-2, 8)


class FeatureEngineeringError(ValueError):
    """Raised when a dataframe cannot be featurised (missing/blank columns)."""


def get_feature_lists() -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` used by the model."""
    numeric = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    categorical = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
    return numeric, categorical


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise FeatureEngineeringError("input dataframe is missing required columns: " + ", ".join(sorted(missing)))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that maps ``x / 0`` to ``0`` instead of ``inf``."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _row_trend(values: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row against the month index ``0..n-1``.

    A negative slope means the value is decreasing over time (bills shrinking or
    payments shrinking), which is what the model needs to see.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("_row_trend expects a 2-D array")
    n_periods = values.shape[1]
    index = np.arange(n_periods, dtype=float)
    index_centered = index - index.mean()
    denominator = float(np.sum(index_centered**2))
    centered = values - values.mean(axis=1, keepdims=True)
    return centered @ index_centered / denominator


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw UCI credit card dataframe.

    The UCI dataset is documented as "clean", yet it contains undocumented
    category codes that hurt a linear model and break schema validation:

    * ``EDUCATION`` uses 1-4, but 0, 5 and 6 appear in the data (they are folded
      into the "others" category ``4``);
    * ``MARRIAGE`` uses 1-3, but ``0`` appears (folded into "others" ``3``);
    * ``PAY_*`` values outside ``[-2, 8]`` are clipped.

    The function does not mutate its input and keeps the target column intact.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_data expects a pandas DataFrame")
    if df.empty:
        raise FeatureEngineeringError("input dataframe is empty")

    _require_columns(df, RAW_FEATURE_COLUMNS)
    cleaned = df.copy()

    for column in RAW_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "EDUCATION" in cleaned.columns:
        education = cleaned["EDUCATION"].fillna(4).astype(int)
        cleaned["EDUCATION"] = education.where(education.isin([1, 2, 3]), 4)

    if "MARRIAGE" in cleaned.columns:
        marriage = cleaned["MARRIAGE"].fillna(3).astype(int)
        cleaned["MARRIAGE"] = marriage.where(marriage.isin([1, 2, 3]), 3)

    if "SEX" in cleaned.columns:
        sex = cleaned["SEX"].fillna(0).astype(int)
        cleaned["SEX"] = sex.where(sex.isin([1, 2]), 0)

    if "AGE" in cleaned.columns:
        cleaned["AGE"] = cleaned["AGE"].clip(lower=18, upper=100)

    for column in PAY_STATUS_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].clip(*PAY_STATUS_CLIP)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned


def _pay_status_bucket(value: float) -> str:
    """Map a ``PAY_0`` code to a coarse, human-readable status bucket."""
    if pd.isna(value):
        return "unknown"
    if value <= -2:
        return "no_consumption"
    if value == -1:
        return "paid_in_full"
    if value == 0:
        return "revolving_ok"
    if value <= 2:
        return "delay_1_2m"
    return "delay_3m_plus"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned dataframe enriched with engineered features.

    Added numeric columns are listed in :data:`ENGINEERED_NUMERIC_FEATURES`,
    added categorical columns in :data:`ENGINEERED_CATEGORICAL_FEATURES`.
    The target column (when present) is passed through untouched.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_features expects a pandas DataFrame")
    if df.empty:
        raise FeatureEngineeringError("input dataframe is empty")

    _require_columns(df, RAW_FEATURE_COLUMNS)
    features = clean_data(df)

    bills = features[BILL_COLUMNS].to_numpy(dtype=float)
    payments = features[PAYMENT_COLUMNS].to_numpy(dtype=float)
    pay_status = features[PAY_STATUS_COLUMNS].to_numpy(dtype=float)
    limit_balance = features["LIMIT_BAL"].to_numpy(dtype=float)
    age = features["AGE"].to_numpy(dtype=float)

    bill_mean = bills.mean(axis=1)
    bill_std = bills.std(axis=1)
    payment_mean = payments.mean(axis=1)
    payment_sum = payments.sum(axis=1)
    bill_sum = bills.sum(axis=1)

    features["CREDIT_UTILIZATION"] = _safe_divide(bill_mean, limit_balance)
    features["BILL_AMT_MEAN"] = bill_mean
    features["BILL_AMT_STD"] = bill_std
    features["BILL_AMT_MAX"] = bills.max(axis=1)
    features["BILL_AMT_MIN"] = bills.min(axis=1)
    features["BILL_AMT_TREND"] = _row_trend(bills)
    features["BILL_AMT_CHANGE"] = bills[:, 0] - bills[:, -1]

    features["PAY_AMT_MEAN"] = payment_mean
    features["PAY_AMT_STD"] = payments.std(axis=1)
    features["PAY_AMT_MAX"] = payments.max(axis=1)
    features["PAY_AMT_SUM"] = payment_sum
    features["PAY_AMT_TREND"] = _row_trend(payments)

    # Share of the outstanding balance that was actually repaid.
    features["PAYMENT_RATIO"] = _safe_divide(payment_sum, bill_sum + 1.0)
    features["BILL_PAYMENT_GAP"] = bills[:, 0] - payments[:, 0]

    features["DELINQ_COUNT"] = (pay_status >= 1).sum(axis=1)
    features["SEVERE_DELINQ_COUNT"] = (pay_status >= 2).sum(axis=1)
    features["MAX_DELINQ"] = pay_status.max(axis=1)
    features["DELINQ_TREND"] = features["PAY_0"] - features["PAY_6"]
    features["MONTHS_WITHOUT_PAYMENT"] = (payments <= 0).sum(axis=1)
    features["LIMIT_PER_AGE"] = _safe_divide(limit_balance, age)

    # Categorical buckets. ``astype(str)`` keeps a stable dtype between the
    # training frame and single-row API payloads and avoids pandas' category
    # dtype mismatch when unseen values arrive.
    features["AGE_BIN"] = (
        pd.cut(features["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str).replace("nan", "unknown")
    )
    features["LIMIT_BAL_BIN"] = (
        pd.cut(features["LIMIT_BAL"], bins=LIMIT_BAL_BINS, labels=LIMIT_BAL_LABELS)
        .astype(str)
        .replace("nan", "unknown")
    )
    features["PAY_0_STATUS"] = features["PAY_0"].map(_pay_status_bucket).fillna("unknown")

    numeric_features, categorical_features = get_feature_lists()
    for column in categorical_features:
        features[column] = features[column].astype(str)

    # Guarantee the declared schema is present even for exotic inputs.
    _require_columns(features, numeric_features + categorical_features)

    return features


def select_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the model features (declared order) from a featurised frame."""
    numeric_features, categorical_features = get_feature_lists()
    columns = numeric_features + categorical_features
    _require_columns(df, columns)
    return df[columns].copy()


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Raise :class:`FeatureEngineeringError` when mandatory columns are absent."""
    _require_columns(df, REQUIRED_RAW_COLUMNS)


def build_inference_frame(record: dict) -> pd.DataFrame:
    """Turn a single JSON-like record into a one-row featurised dataframe.

    Missing raw columns are back-filled with documented neutral defaults so the
    API accepts both the compact payload from the assignment brief and the full
    23-column payload.
    """
    defaults = {
        "LIMIT_BAL": 0.0,
        "SEX": 0,
        "EDUCATION": 4,
        "MARRIAGE": 3,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 0.0,
        "BILL_AMT2": 0.0,
        "BILL_AMT3": 0.0,
        "BILL_AMT4": 0.0,
        "BILL_AMT5": 0.0,
        "BILL_AMT6": 0.0,
        "PAY_AMT1": 0.0,
        "PAY_AMT2": 0.0,
        "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0,
        "PAY_AMT5": 0.0,
        "PAY_AMT6": 0.0,
    }
    row = {column: record.get(column, default) for column, default in defaults.items()}
    return build_features(pd.DataFrame([row]))
